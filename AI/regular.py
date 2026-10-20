import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD=0
N_HOUR_BUCKETS = 5 
N_USER_NUM = 3              # num_buy, avg_spent, combo_ratio
N_ITEM_NUM = 5              # price, combo, hot, best, seasonal

def hour_bucket(hour):
    if 7<=hour<=10:
        return 0 # sang
    elif hour <=13:
        return 1 # trua
    elif hour <= 18:
        return 2 # chieu
    elif hour <= 22:
        return 3 #toi
    return 4  

class Vocab():
    def __init__(self, items=()):
        self.intostr= ['<pad>'] +sorted({str(x) for x in items})
        self.strtoin ={s : i for i,s in enumerate(self.intostr)}
    def __len__(self):
        return len(self.intostr)
    def encode(self,x):
        return self.strtoin.get(str(x),PAD) if x is not None else PAD
    def encode_list(self,values,max_len):
        ids = [self.encode(v) for v in (values or [])][:max_len]
        return ids + [PAD] * (max_len - len(ids))
    @classmethod
    def from_itos(cls, itos):
        v = cls()
        v.intostr = list(itos)
        v.strtoin = {s: i for i, s in enumerate(v.intostr)}
        return v

def masked_mean(emb, ids):
    # emb (b,l,d), ids (b,l)
    # b so khach, l so o cua danh sach mon, d so chieu cua vector moi mon
    mask= (ids!=PAD).unsqueeze(-1).float()
    total=(emb*mask).sum(1)
    num= mask.sum(1).clamp(min=1)
    return total/num

class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_meals, n_cats, emb_dim=32, hidden=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.meal_emb=nn.Embedding(n_meals,emb_dim,PAD)
        self.user_emb=nn.Embedding(n_users,emb_dim,PAD)
        self.cat_emb=nn.Embedding(n_cats,emb_dim,PAD)
        self.hour_emb=nn.Embedding(N_HOUR_BUCKETS,8)
        self.weekday_emb = nn.Embedding(7, 4)
        self.user_mlp=nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 4 + 8 + 4 + N_USER_NUM,hidden),
            nn.ReLU(),
            nn.Linear(hidden,out_dim)
        )
        self.item_mlp=nn.Sequential(
            nn.Linear(emb_dim * 2 + N_ITEM_NUM, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def item_tower(self, meal_idx, cat_idx, num):
        x=torch.cat([self.meal_emb(meal_idx),self.cat_emb(cat_idx),num],-1)
        return F.normalize(self.item_mlp(x), dim=-1)
    def user_tower(self, user_idx, fav_meal, recent_meal, fav_cat, hour, weekday, num):
        x=torch.cat([
            self.user_emb(user_idx),
            masked_mean(self.meal_emb(fav_meal),fav_meal),
            masked_mean(self.meal_emb(recent_meal),recent_meal),
            masked_mean(self.cat_emb(fav_cat),fav_cat),
            self.hour_emb(hour),
            self.weekday_emb(weekday),
            num

        ],dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)




