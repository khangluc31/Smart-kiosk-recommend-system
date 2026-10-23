import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD=0
N_HOUR_BUCKETS = 5 
N_USER_NUM = 3              # num_buy, avg_spent, combo_ratio
N_ITEM_NUM = 5              # price, combo, hot, best, seasonal
N_FAV_MEAL = 5              # 5 mon khach mua nhieu nhat
N_FAV_CAT = 3               # 3 danh muc mua nhieu nhat
N_RECENT_MEAL = 10          # 10 mon mua gan nhat

def hour_bucket(hour):
    if hour >= 23 or hour < 7: 
        return 4
    if 7<=hour<=10:
        return 0 # sang
    elif hour <=13:
        return 1 # trua
    elif hour <= 18:
        return 2 # chieu
    elif hour <= 22:
        return 3 #toi

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
class TwoTowerRecommender:
    def __init__(self, emb_dim=32, hidden=128, out_dim=64, dropout=0.3,temperature=0.1, user_dropout=0.2):
        self.cfg = dict(emb_dim=emb_dim, hidden=hidden, out_dim=out_dim, dropout=dropout)
        self.temperature = temperature
        self.user_dropout = user_dropout     
        self.model = None
        self.meals = {}
        self.index = None
        self.index_ids = []
    def _build_vocabs(self, samples, meals):
        self.meals=meals
        self.meal_vocab=Vocab(meals)
        self.cat_vocab=Vocab(m['cat_id'] for m in meals.values())
        self.user_vocab=Vocab(u['user_id'] for u, _ in samples if u.get("user_id"))
    def _encode_user(self, users, train=False, rng=None):
        cols = {k: [] for k in ("user", "fav_meal", "recent_meal", "fav_cat",
                                "hour", "weekday", "num")}
        for u in users:
            uid = self.user_vocab.encode(u.get("user_id"))
            if train and rng is not None and rng.random() < self.user_dropout:
                uid = PAD                     # buộc model dựa vào hành vi, không học thuộc id
            hb = u["hour_bucket"] if "hour_bucket" in u else hour_bucket(u.get("hour", 12))
            cols["user"].append(uid)
            cols["fav_meal"].append(self.meal_vocab.encode_list(u.get("fav_meal"), N_FAV_MEAL))
            cols["recent_meal"].append(self.meal_vocab.encode_list(u.get("recent_meal"), N_RECENT_MEAL))
            cols["fav_cat"].append(self.cat_vocab.encode_list(u.get("fav_cat"), N_FAV_CAT))
            cols["hour"].append(hb)
            cols["weekday"].append(int(u.get("weekday", 0)))
            cols["num"].append([
                np.log1p(u.get("num_buy", 0)) / 4.0,
                np.log1p(u.get("avg_spent", 0.0)) / 14.0,
                float(u.get("combo_ratio", 0.0)),
            ])
        t = {k: torch.tensor(v, dtype=torch.long) for k, v in cols.items() if k != "num"}
        t["num"] = torch.tensor(cols["num"], dtype=torch.float32)
        return (t["user"], t["fav_meal"], t["recent_meal"], t["fav_cat"],
                t["hour"], t["weekday"], t["num"])
 
    def _encode_items(self, meal_ids):
        meals = self.meals
        return (
            torch.tensor([self.meal_vocab.encode(m) for m in meal_ids], dtype=torch.long),
            torch.tensor([self.cat_vocab.encode(meals[m]["cat_id"]) for m in meal_ids],
                         dtype=torch.long),
            torch.tensor([[np.log1p(float(meals[m]["price"])) / 14.0,
                           float(meals[m].get("is_combo", False)),
                           float(meals[m].get("is_hotsell", False)),
                           float(meals[m].get("is_bestsell", False)),
                           float(meals[m].get("is_seasonal", False))] for m in meal_ids],
                         dtype=torch.float32),
        )

    def similarity(self, u, v):
        return u @ v.T / self.temperature
    def fit(self, samples, meals, epochs=25, batch_size=256, lr=2e-3, weight_decay=1e-5,
            seed=42, verbose=True):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self._build_vocabs(samples, meals)          
        self.model = TwoTowerModel(len(self.user_vocab), len(self.meal_vocab),
                                   len(self.cat_vocab), **self.cfg)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
 
        menu = list(meals)
        pos = {m: i for i, m in enumerate(menu)}
        samples = [(u, m) for u, m in samples if m in pos]
        if not samples:
            raise ValueError("Không có mẫu nào có meal_id nằm trong menu")
        menu_tensors = self._encode_items(menu)
        losses = []
        for epoch in range(epochs):
            self.model.train()
            order = rng.permutation(len(samples))
            total = 0.0
            for start in range(0, len(samples), batch_size):
                batch = [samples[i] for i in order[start:start + batch_size]]
                u = self.model.user_tower(*self._encode_user([s[0] for s in batch], True, rng))
                v = self.model.item_tower(*menu_tensors)           # cả menu làm ứng viên
                labels = torch.tensor([pos[s[1]] for s in batch])
                loss = F.cross_entropy(self.similarity(u, v), labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            losses.append(total / len(samples))
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  epoch {epoch + 1}/{epochs}  loss={losses[-1]:.4f}")
        self.build_index(meals)
        return losses
    @torch.no_grad()
    def build_index(self, meals=None):
        if meals is not None:
            self.meals = meals
        self.model.eval()
        self.index_ids = list(self.meals)
        v = self.model.item_tower(*self._encode_items(self.index_ids)).numpy().astype("float32")
        self.index = faiss.IndexFlatIP(v.shape[1])      # vector đã chuẩn hoá -> inner product = cosine
        self.index.add(np.ascontiguousarray(v))
    @torch.no_grad()
    def user_vector(self, user):
        self.model.eval()
        u = self.model.user_tower(*self._encode_user([user]))
        return np.ascontiguousarray(u.numpy().astype("float32"))
    def recommend(self, user, k=20, exclude=(), only_available=True):
        scores, idx = self.index.search(self.user_vector(user), len(self.index_ids))
        exclude = set(exclude)
        out_ids, out_scores = [], []
        for s, i in zip(scores[0], idx[0]):
            m = self.index_ids[i]
            if m in exclude or (only_available and not self.meals[m].get("available", True)):
                continue
            out_ids.append(m)
            out_scores.append(float(s))
            if len(out_ids) == k:
                break
        return out_ids, out_scores
    def evaluate(self, queries, k=20):
        hit = total = 0
        for q in queries:
            user, targets = (q["user"], q["targets"]) if isinstance(q, dict) else q
            top, _ = self.recommend(user, k=k, only_available=False)
            hit += len(set(targets) & set(top))
            total += len(targets)
        return hit / max(total, 1)
    def save(self, path):
        torch.save({
            "cfg": self.cfg, "temperature": self.temperature, "user_dropout": self.user_dropout,
            "state": self.model.state_dict(),
            "vocabs": [self.user_vocab.intostr, self.meal_vocab.intostr, self.cat_vocab.intostr],
        }, path)
 
    @classmethod
    def load(cls, path, meals):
        ckpt = torch.load(path, weights_only=False)
        rec = cls(**ckpt["cfg"], temperature=ckpt["temperature"], user_dropout=ckpt["user_dropout"])
        rec.user_vocab, rec.meal_vocab, rec.cat_vocab = (Vocab.from_itos(v) for v in ckpt["vocabs"])
        rec.model = TwoTowerModel(len(rec.user_vocab), len(rec.meal_vocab), len(rec.cat_vocab),
                                  **ckpt["cfg"])
        rec.model.load_state_dict(ckpt["state"])
        rec.build_index(meals)
        return rec



