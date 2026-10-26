import pandas as pd
import numpy as np
import torch
import re
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

class context_processing:
    def __init__(self):
        self.wanted=[]
        self.unwanted=[]
        self.current_type=None
        self.current_token=[]
        self.token_indices=[]
        self.sentence_context_vector=None
        self.model_name=Path(__file__).resolve().parent / "ner_model"
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name)
        self.model=AutoModelForTokenClassification.from_pretrained(
            self.model_name, 
            num_labels=5, 
            ignore_mismatched_sizes=True
        )
        self.id2lb={0 : "O", 1 : "B-want", 2 : "B-unwant", 3 : "I-want", 4: "I-unwant" }
    def context_extraction(self,text):
        self.wanted = []
        self.unwanted = []
        current_type = None
        token_indices = []
        span_start = None
        span_end = None
        inputs=self.tokenizer(text,return_offsets_mapping=True, return_tensors="pt")
        offset_mapping=inputs.pop("offset_mapping")[0]
        with torch.no_grad():
            outputs=self.model(**inputs,output_hidden_states=True)
            logits=outputs.logits[0]
            pred_label_id=torch.argmax(logits,dim=-1)
            last_hidden_state=outputs.hidden_states[-1][0]
        for idx, (label_id,offset) in  enumerate(zip(pred_label_id,offset_mapping)):
            if offset[0]== offset[1]:
                continue
            label=self.id2lb[label_id.item()]
            if label.startswith("B-"):
                if current_type is not None:
                    span_text = text[span_start:span_end]
                    span_vector = torch.mean(last_hidden_state[token_indices], dim=0)
                    entry = {"text": span_text, "embedding_vector": span_vector}
                    if current_type == "want":
                        self.wanted.append(entry)
                    elif current_type == "unwant": 
                        self.unwanted.append(entry)
                current_type = label.split("-")[1]
                token_indices = [idx]
                span_start = offset[0].item()
                span_end = offset[1].item()
            elif label.startswith("I-") and current_type == label.split('-')[1]:
                token_indices.append(idx)
                span_end = offset[1].item() 
            elif label == "O": 
                if current_type is not None:
                    span_text = text[span_start:span_end]
                    span_vector = torch.mean(last_hidden_state[token_indices], dim=0)
                    entry = {"text": span_text, "embedding_vector": span_vector}
                    if current_type == "want":
                        self.wanted.append(entry) 
                    elif current_type == "unwant":
                        self.unwanted.append(entry) 
                current_type = None
                token_indices = []
                span_start = None
                span_end = None
        if current_type is not None:
            span_text = text[span_start:span_end]
            span_vector = torch.mean(last_hidden_state[token_indices], dim=0)
            entry = {"text": span_text, "embedding_vector": span_vector}
            if current_type == "want":
                self.wanted.append(entry)
            elif current_type == "unwant":
                self.unwanted.append(entry)
        attention_mask = inputs["attention_mask"][0].unsqueeze(-1)  
        self.sentence_context_vector = torch.sum(last_hidden_state * attention_mask, dim=0) / torch.clamp(attention_mask.sum(), min=1e-9)
    def extract_budget(self, text):
        pattern = r'\b(\d+(?:\.\d+)?)\s*(k|nghìn|ngàn|triệu|tr|M)\b'
        matches = re.findall(pattern, text.lower())
        budgets = []
        for amount, unit in matches:
            val = float(amount)
            if unit in ['k', 'nghìn', 'ngàn']:
                budgets.append(int(val * 1000))
            elif unit in ['triệu', 'tr','m']:
                budgets.append(int(val * 1000000))
        return budgets
    def get_want_list(self):
        return self.wanted
    def get_unwant_list(self):
        return self.unwanted
    def get_vector(self):
        return self.sentence_context_vector
    def process(self, text):
        text = text or ""
        out = {"text": text, "want_text": [], "unwant_text": [], "budget": None,
               "sentence": None, "want": [], "unwant": []}
        if not text.strip():
            return out
        self.context_extraction(text)
        budgets = self.extract_budget(text)
        out.update(
            want_text=[e["text"] for e in self.wanted],
            unwant_text=[e["text"] for e in self.unwanted],
            budget=min(budgets) if budgets else None,          
            sentence=self._np(self.sentence_context_vector),
            want=[self._np(e["embedding_vector"]) for e in self.wanted],
            unwant=[self._np(e["embedding_vector"]) for e in self.unwanted],
        )
        return out
 
    def embed(self, texts, batch_size=32):
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                 max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc, output_hidden_states=True).hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(vec, dim=-1).cpu().numpy())
        return np.concatenate(out)
 
    def embed_meals(self, meals):
        ids = list(meals)
        texts = [f"{meals[m]['name']}. {meals[m].get('description') or ''}".strip() for m in ids]
        return dict(zip(ids, self.embed(texts)))
 
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
 
    @classmethod
    def load(cls, path, device=None):
        return cls(model_path=path, device=device)
 
    @staticmethod
    def _np(t):
        if t is None:
            return None
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)




