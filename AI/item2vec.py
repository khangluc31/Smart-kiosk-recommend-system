
import pickle
from collections import Counter
from itertools import permutations

import faiss
import numpy as np
import torch


def _l2(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(min=1e-9)


class Item2Vec:
    def __init__(self, dim=32, epochs=100, negative=5, min_count=2, seed=42):
        self.dim = dim
        self.epochs = epochs
        self.negative = negative
        self.min_count = min_count      
        self.seed = seed
        self.ids = []                   
        self.in_vecs = None             
        self.out_vecs = None           
        self.n_baskets = 0

    def fit(self, baskets, batch_size=1024, lr=0.01):
        baskets = [list(dict.fromkeys(b)) for b in baskets]
        baskets = [b for b in baskets if len(b) >= 2]
        if not baskets:
            raise ValueError("Không có giỏ nào từ 2 món trở lên để học")
        freq = Counter(m for b in baskets for m in b)
        self.ids = sorted((m for m, c in freq.items() if c >= self.min_count), key=lambda m: -freq[m])
        idx = {m: i for i, m in enumerate(self.ids)}

        pairs = [(idx[a], idx[b]) for bk in baskets for a, b in permutations(bk, 2)
                 if a in idx and b in idx]
        pairs = torch.tensor(pairs, dtype=torch.long)
        noise = torch.tensor([freq[m] for m in self.ids], dtype=torch.float) ** 0.75

        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        n, d = len(self.ids), self.dim
        emb_in = torch.nn.Parameter((torch.rand(n, d, generator=g) - 0.5) / d)   
        emb_out = torch.nn.Parameter(torch.zeros(n, d))
        opt = torch.optim.Adam([emb_in, emb_out], lr=lr)
        logsig = torch.nn.functional.logsigmoid

        for _ in range(self.epochs):
            order = torch.randperm(len(pairs), generator=g)
            for s in range(0, len(pairs), batch_size):
                batch = pairs[order[s:s + batch_size]]
                a, b = batch[:, 0], batch[:, 1]
                neg = torch.multinomial(noise, len(batch) * self.negative, replacement=True,
                                        generator=g).view(len(batch), self.negative)
                va = emb_in[a]                                             # (B, d)
                pos = (va * emb_out[b]).sum(-1)                            # (B,)
                negs = torch.bmm(emb_out[neg], va.unsqueeze(-1)).squeeze(-1)   # (B, k)
                loss = -(logsig(pos) + logsig(-negs).sum(-1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.in_vecs = emb_in.detach().numpy().astype("float32")
        self.out_vecs = emb_out.detach().numpy().astype("float32")
        self.n_baskets = len(baskets)
        self.build_index()
        return self

    def build_index(self):
        self.pos = {m: i for i, m in enumerate(self.ids)}
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(self.out_vecs))
        self.sub_index = faiss.IndexFlatIP(self.dim)
        self.sub_index.add(np.ascontiguousarray(_l2(self.in_vecs)))

    def has(self, meal_id):
        return meal_id in self.pos

    @staticmethod
    def _search(index, ids, query, k, exclude):
        scores, idx = index.search(np.ascontiguousarray(query[None, :], dtype="float32"), len(ids))
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i >= 0 and ids[i] not in exclude:
                out.append((ids[i], float(s)))
                if len(out) == k:
                    break
        return out

    def recommend_for_cart(self, cart_meal_ids, k=10, exclude=()):
        known = [m for m in cart_meal_ids if self.has(m)]
        if not known:
            return []
        query = self.in_vecs[[self.pos[m] for m in known]].mean(axis=0)
        return self._search(self.index, self.ids, query, k, set(cart_meal_ids) | set(exclude))

    def scores_for_cart(self, cart_meal_ids, exclude=()):
        shift = float(np.log(self.negative))
        return {m: s + shift for m, s in
                self.recommend_for_cart(cart_meal_ids, k=len(self.ids), exclude=exclude)}

    def complements(self, meal_id, k=5):
        return self.recommend_for_cart([meal_id], k)

    def substitutes(self, meal_id, k=5):
        if not self.has(meal_id):
            return []
        q = _l2(self.in_vecs[self.pos[meal_id]])
        return self._search(self.sub_index, self.ids, q, k, {meal_id})

   
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"cfg": dict(dim=self.dim, epochs=self.epochs, negative=self.negative,
                                     min_count=self.min_count, seed=self.seed),
                         "ids": self.ids, "in": self.in_vecs, "out": self.out_vecs,
                         "n_baskets": self.n_baskets}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            st = pickle.load(f)
        obj = cls(**st["cfg"])
        obj.ids, obj.in_vecs, obj.out_vecs = st["ids"], st["in"], st["out"]
        obj.n_baskets = st.get("n_baskets", 0)
        obj.build_index()
        return obj