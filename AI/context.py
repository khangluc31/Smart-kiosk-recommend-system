import pandas as pd
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification

class context_processing:
    def __init__(self):
        self.wanted=[]
        self.unwanted=[]
        self.current_type=None
        self.current_token=[]
        self.token_indices=[]
        self.sentence_context_vector=None
        self.model_name="bert-base-multilingual-cased"
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




