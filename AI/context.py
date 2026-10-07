import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name="bert-base-multilingual-cased"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForTokenClassification.from_pretrained(model_name)
id2lb={0 : 0, 1 : "B-want", 2 : "B-unwant", 3 : "I-want", 4: "I-unwant" }
wanted=[]
unwanted=[]
current_type=None
current_token=[]
token_indices=[]
text=''
inputs=tokenizer(text,return_offsets_mapping=True, return_tensors="pt")
offset_mapping=inputs.pop("offset_mapping")[0]
with torch.no_grad():
    outputs=model(**inputs,output_hidden_state=True)
    logits=outputs.logits[0]
    pred_label_id=torch.argmax(logits,dim=-1)
    last_hidden_state=outputs.last_hidden_state[-1][0]
for idx, (label_id,offset) in  enumerate(zip(pred_label_id,offset_mapping)):
    if offset[0]== offset[1]:
        continue
    label=id2lb[label_id.item()]
    word_text=text[offset[0]:offset[1]]
    if label.startswith("B-"):
        if current_token:
            span_text=" ".join(current_token)
            span_vector=torch.mean(last_hidden_state[token_indices],dim=0)
            entry={"text:" : span_text, "embedding vector: " : span_vector}
            if current_type == "want":
                wanted.append(entry)
            elif current_type== "unwanted":
                unwanted.append(entry)
        current_type = label.split("-")[1]
        current_token = [word_text]
        token_indices = [idx]

    elif label.startswith("I-") and current_type== label.split('-')[1]:
        current_token.append(word_text)
        token_indices.append(idx)

    elif label.startswith("O-"):
        if current_token:
            span_text="".join(current_token)
            span_vector=torch.mean(last_hidden_state[token_indices],dim=0)
            entry= {"text:" : span_text, "embedding vector: " : span_vector}
            if current_type == "want":
                wanted.append(current_token)
            else:
                unwanted.append(current_token)
        current_type = None
        current_token = []
        token_indices= []
if current_token:
    span_text="".join(current_token)
    span_vector=torch.mean(last_hidden_state[token_indices],dim=0)
    entry= {"text:" : span_text, "embedding vector: " : span_vector}
    if current_type == "want":
        wanted.append(current_token)
    else:
        unwanted.append(current_token)
attention_mask = inputs["attention_mask"][0].unsqueeze(-1)  
sentence_context_vector = torch.sum(last_hidden_state * attention_mask, dim=0) / torch.clamp(attention_mask.sum(), min=1e-9)




