
import os
import numpy as np
import lightgbm as lgb
FEATURES = [
    "retrieval_score", "retrieval_rank", "from_two_tower",
    "log_price", "is_hotsell", "is_bestsell", "is_seasonal", "is_combo",
    "has_budget", "price_over_budget", "over_budget",
    "log_num_buy", "times_ordered_before", "in_fav_meal", "in_fav_cat", "in_recent",
    "hour_bucket", "is_weekend",
    "sim_sentence", "max_sim_want", "max_sim_unwant",
]


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_features(candidates, scores, source, meals, user, text_ctx=None, meal_vecs=None):
    text_ctx = text_ctx or {}
    meal_vecs = meal_vecs or {}
    budget = user.get("budget")
    fav_meal, fav_cat = set(user["fav_meal"]), set(user["fav_cat"])
    recent = set(user["recent_meal"])
    rows = []
    for rank, (m, s) in enumerate(zip(candidates, scores)):
        meal = meals[m]
        vec = meal_vecs.get(m)
        sent = text_ctx.get("sentence")
        sim_sent = _cos(sent, vec) if vec is not None and sent is not None else 0.0
        sim_want = max((_cos(w, vec) for w in text_ctx.get("want", [])), default=0.0) if vec is not None else 0.0
        sim_unwant = max((_cos(w, vec) for w in text_ctx.get("unwant", [])), default=0.0) if vec is not None else 0.0
        rows.append([
            s, rank, float(source == "two_tower"),
            np.log1p(meal["price"]), float(meal["is_hotsell"]), float(meal["is_bestsell"]),
            float(meal["is_seasonal"]), float(meal["is_combo"]),
            float(budget is not None),
            meal["price"] / budget if budget else 0.0,
            float(bool(budget) and meal["price"] > budget),
            np.log1p(user["num_buy"]), user["meal_orders"].get(m, 0),
            float(m in fav_meal), float(meal["cat_id"] in fav_cat), float(m in recent),
            user["hour_bucket"], float(user["weekday"] >= 5),
            sim_sent, sim_want, sim_unwant,
        ])
    return np.array(rows, dtype=float).reshape(-1, len(FEATURES))


class Ranker:
    def __init__(self, booster=None):
        self.booster = booster

    def fit(self, groups, seed=42):
        groups = [(X, y) for X, y in groups if len(y) and y.sum() > 0]
        X = np.vstack([g[0] for g in groups])
        y = np.concatenate([g[1] for g in groups])
        model = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                               num_leaves=15, min_child_samples=20, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                               verbose=-1)
        model.fit(X, y, group=[len(g[1]) for g in groups], feature_name=FEATURES)
        self.booster = model.booster_
        return self

    def rank(self, candidates, X, k=5):
        if self.booster is None or len(candidates) == 0:
            return list(candidates[:k]), [None] * min(k, len(candidates))
        pred = self.booster.predict(X)
        order = np.argsort(-pred)[:k]
        return [candidates[i] for i in order], [float(pred[i]) for i in order]

    def feature_importance(self):
        if self.booster is None:
            return {}
        gain = self.booster.feature_importance("gain")
        return dict(sorted(zip(FEATURES, gain), key=lambda x: -x[1]))

    def save(self, path):
        self.booster.save_model(path)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return cls()
        return cls(lgb.Booster(model_file=path))