
from collections import Counter
from datetime import timedelta, timezone

import numpy as np

VN_TZ = timezone(timedelta(hours=7))
N_FAV_MEAL = 5
N_FAV_CAT = 3
N_RECENT_ORDERS = 3
N_RECENT_MEAL = 10
N_HOUR_BUCKETS = 5


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


def context_features(ts, budget=None):
    local = ts.astimezone(VN_TZ)
    return {
        "hour_bucket": hour_bucket(local.hour),
        "weekday": local.weekday(),
        "budget": budget,
    }


def user_features(past_orders, meals, ts, budget=None, user_id=None):
    meal_orders, cat_orders, meal_count = Counter(), Counter(), Counter()
    n_combo = 0
    for o in past_orders:
        ids = {m for m, _ in o["items"] if m in meals}
        meal_orders.update(ids)
        cat_orders.update(meals[m]["cat_id"] for m in ids)
        meal_count.update({m: q for m, q in o["items"] if m in meals})
        n_combo += any(meals[m]["is_combo"] for m in ids)

    recent = []
    for o in reversed(past_orders[-N_RECENT_ORDERS:]):
        recent += [m for m, _ in o["items"] if m in meals]

    n = len(past_orders)
    return {
        "user_id": user_id,
        "fav_meal": [m for m, _ in meal_orders.most_common(N_FAV_MEAL)],
        "fav_cat": [c for c, _ in cat_orders.most_common(N_FAV_CAT)],
        "recent_meal": list(dict.fromkeys(recent))[:N_RECENT_MEAL],
        "meal_orders": dict(meal_orders),        # số đơn có món đó (cho ranker)
        "num_buy": n,
        "avg_spent": float(np.mean([o["total"] for o in past_orders])) if n else 0.0,
        "combo_ratio": n_combo / n if n else 0.0,
        **context_features(ts, budget),
    }


def build_queries(user_orders, meals, min_history=1):
    queries = []
    for uid, orders in user_orders.items():
        for i, o in enumerate(orders):
            if i < min_history:
                continue
            targets = {m for m, _ in o["items"] if m in meals}
            if not targets:
                continue
            feats = user_features(orders[:i], meals, o["created_at"], o["budget"], uid)
            queries.append({"user": feats, "targets": targets, "ts": o["created_at"], "order": o})
    queries.sort(key=lambda q: q["ts"])
    return queries


def queries_to_samples(queries):
    return [(q["user"], m) for q in queries for m in sorted(q["targets"])]