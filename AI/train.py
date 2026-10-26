
import os
from collections import Counter
from itertools import combinations

import numpy as np

from db import Database
from decide import (ITEM2VEC_PATH, MODEL_DIR, N_CONTEXT, RANKER_PATH, REGULAR_THRESHOLD,
                    TOP_K_RETRIEVE, TOP_K_SHOW, TWO_TOWER_PATH, linucb_history, make_x,
                    save_linucb, to_two_tower_user)
from features import VN_TZ, build_queries, user_features
from item2vec import Item2Vec
from ranker import Ranker, build_features
from regular import TwoTowerRecommender, hour_bucket
from unregular import linUCB

EPOCHS = int(os.getenv("EPOCHS", "25"))
TRAIN_FRAC, RANKER_FRAC = 0.70, 0.15          
class Recall:
    """recall@k cộng dồn: (số món đúng nằm trong top k) / (tổng số món khách thật sự mua)."""
    def __init__(self):
        self.hit = self.total = 0

    def add(self, top, targets):
        self.hit += len(set(top) & set(targets))
        self.total += len(targets)

    @property
    def value(self):
        return self.hit / max(self.total, 1)


def print_table(title, rows, ks):
    print(f"\n{title}")
    print(f"  {'':<34}" + "".join(f"{'@' + str(k):>9}" for k in ks))
    for name, recs in rows:
        print(f"  {name:<34}" + "".join(f"{r.value:9.3f}" for r in recs))


def popular_top(counts, k, exclude=()):
    return [m for m, _ in counts.most_common() if m not in exclude][:k]


def guest_queries(orders, meals):
    out = []
    for o in orders:
        if o["user_id"]:
            continue
        targets = {m for m, _ in o["items"] if m in meals}
        if targets:
            out.append({"user": user_features([], meals, o["created_at"], o["budget"]),
                        "targets": targets, "ts": o["created_at"], "order": o})
    return out


class Retriever:
    def __init__(self, two_tower, linucb, meal_ids):
        self.two_tower, self.linucb, self.meal_ids = two_tower, linucb, meal_ids

    def __call__(self, q, k=TOP_K_RETRIEVE):
        user = to_two_tower_user(q["user"], q["ts"])
        if user["num_buy"] >= REGULAR_THRESHOLD:
            ids, scores = self.two_tower.recommend(user, k=k, only_available=False)
            return list(ids), [float(s) for s in scores], "two_tower", user
        x = make_x(q["ts"], q["user"]["budget"], has_account=q["user"]["num_buy"] > 0)
        arms, scores = self.linucb.select_arm(k,np.tile(x, (len(self.meal_ids), 1)))
        return [self.meal_ids[a] for a in arms], [float(s) for s in scores], "linucb", user


def train_two_tower(queries, meals, verbose=True):
    samples = []
    for q in queries:
        user = to_two_tower_user(q["user"], q["ts"])
        samples += [(user, m) for m in sorted(q["targets"])]
    rec = TwoTowerRecommender()
    rec.fit(samples, meals, epochs=EPOCHS, verbose=verbose)       # fit trả về list loss
    return rec


def train_linucb(orders, meal_ids):
    model = linUCB(len(meal_ids), N_CONTEXT, alpha=1.0)
    model.warm_start(linucb_history(orders, {m: a for a, m in enumerate(meal_ids)}))
    return model


def baskets_of(orders):
    return [[m for m, _ in o["items"]] for o in orders]


def main():
    with Database() as db:
        meals = db.get_meals()
        orders = sorted(db.get_orders(), key=lambda o: o["created_at"])
    meal_ids = list(meals)
    times = [o["created_at"] for o in orders]
    t1 = times[int(len(times) * TRAIN_FRAC)]
    t2 = times[int(len(times) * (TRAIN_FRAC + RANKER_FRAC))]
    o_train = [o for o in orders if o["created_at"] < t1]
    o_before_test = [o for o in orders if o["created_at"] < t2]

    queries = build_queries(Database.group_by_user(orders), meals) + guest_queries(orders, meals)
    q_train = [q for q in queries if q["ts"] < t1]
    q_rank = [q for q in queries if t1 <= q["ts"] < t2]
    q_test = [q for q in queries if q["ts"] >= t2]
    is_reg = lambda q: q["user"]["num_buy"] >= REGULAR_THRESHOLD  # noqa: E731
    print(f"{len(meals)} món | {len(orders)} đơn | train {len(q_train)} / ranker {len(q_rank)} "
          f"/ test {len(q_test)} lượt (test: {sum(map(is_reg, q_test))} khách quen)")

    pop = Counter(m for o in o_train for m, _ in o["items"])
    pop_hour = {}
    for o in o_train:
        hb = hour_bucket(o["created_at"].astimezone(VN_TZ).hour)
        pop_hour.setdefault(hb, Counter()).update(m for m, _ in o["items"])
    ks = (5, TOP_K_RETRIEVE)

    print("\n[1] Train two tower ...")
    two_tower = train_two_tower([q for q in q_train if q["user"]["num_buy"] > 0], meals)
    rows = {n: [Recall() for _ in ks] for n in ("Bán chạy nhất", "Món ruột + bán chạy", "Two tower")}
    for q in filter(is_reg, q_test):
        user = to_two_tower_user(q["user"], q["ts"])
        tt = list(two_tower.recommend(user, k=max(ks), only_available=False)[0])
        fav = (q["user"]["fav_meal"] + popular_top(pop, max(ks), q["user"]["fav_meal"]))
        for i, k in enumerate(ks):
            rows["Bán chạy nhất"][i].add(popular_top(pop, k), q["targets"])
            rows["Món ruột + bán chạy"][i].add(fav[:k], q["targets"])
            rows["Two tower"][i].add(tt[:k], q["targets"])
    print_table("Khách quen — recall (test)", rows.items(), ks)

    linucb = train_linucb(o_train, meal_ids)
    retriever = Retriever(two_tower, linucb, meal_ids)
    rows = {n: [Recall() for _ in ks] for n in ("Bán chạy nhất", "Bán chạy theo khung giờ", "LinUCB")}
    for q in (q for q in q_test if not is_reg(q)):
        ids = retriever(q, k=max(ks))[0]
        by_hour = pop_hour.get(hour_bucket(q["ts"].astimezone(VN_TZ).hour), pop)
        for i, k in enumerate(ks):
            rows["Bán chạy nhất"][i].add(popular_top(pop, k), q["targets"])
            rows["Bán chạy theo khung giờ"][i].add(popular_top(by_hour, k), q["targets"])
            rows["LinUCB"][i].add(ids[:k], q["targets"])
    print_table("[2] Khách mới / vãng lai — recall (test)", rows.items(), ks)

   
    print("\n[3] Train LightGBM ranker ...")
    groups = []
    for q in q_rank:
        ids, scores, src, user = retriever(q)
        X = build_features(ids, scores, src, meals, user)
        groups.append((X, np.array([float(m in q["targets"]) for m in ids])))
    ranker = Ranker().fit(groups)
    rows = {}
    for name, qs in (("khách quen", list(filter(is_reg, q_test))),
                     ("khách mới", [q for q in q_test if not is_reg(q)])):
        base, ranked = Recall(), Recall()
        for q in qs:
            ids, scores, src, user = retriever(q)
            base.add(ids[:TOP_K_SHOW], q["targets"])
            ranked.add(ranker.rank(ids, build_features(ids, scores, src, meals, user),
                                   TOP_K_SHOW)[0], q["targets"])
        rows[f"{name}: thứ tự retrieval"] = [base]
        rows[f"{name}: LightGBM xếp lại"] = [ranked]
    print_table(f"Top {TOP_K_SHOW} hiển thị — recall (test)", rows.items(), (TOP_K_SHOW,))
    print("  feature quan trọng nhất:", ", ".join(list(ranker.feature_importance())[:6]))

    i2v = Item2Vec().fit(baskets_of(o_train + [o for o in orders if t1 <= o["created_at"] < t2]))
    pair = Counter()
    for b in baskets_of(o_before_test):
        pair.update(combinations(sorted(set(b)), 2))

    def cooc_top(cart, k):
        sc = Counter()
        for c in cart:
            for (a, b), n in pair.items():
                if a == c and b not in cart:
                    sc[b] += n
                elif b == c and a not in cart:
                    sc[a] += n
        return [m for m, _ in sc.most_common(k)]

    k3 = (1, 3)
    rows = {n: [Recall() for _ in k3] for n in ("Bán chạy nhất", "Đếm cặp mua cùng", "Item2vec")}
    for o in orders:
        if o["created_at"] < t2:
            continue
        b = list(dict.fromkeys(m for m, _ in o["items"]))
        if len(b) < 2:
            continue
        for hidden in b:
            cart = [m for m in b if m != hidden]
            for i, k in enumerate(k3):
                rows["Bán chạy nhất"][i].add(popular_top(pop, k, cart), [hidden])
                rows["Đếm cặp mua cùng"][i].add(cooc_top(cart, k), [hidden])
                rows["Item2vec"][i].add([m for m, _ in i2v.recommend_for_cart(cart, k)], [hidden])
    print_table("[4] Gợi ý mua kèm — giấu 1 món trong giỏ, đoán lại (test)", rows.items(), k3)

    print("\n[5] Train lại trên toàn bộ dữ liệu và lưu ...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_two_tower([q for q in queries if q["user"]["num_buy"] > 0], meals,
                    verbose=False).save(TWO_TOWER_PATH)
    save_linucb(train_linucb(orders, meal_ids), meal_ids)
    ranker.save(RANKER_PATH)
    Item2Vec().fit(baskets_of(orders)).save(ITEM2VEC_PATH)
    print(f"Đã lưu two tower, LinUCB, ranker, Item2vec vào {MODEL_DIR}/")


if __name__ == "__main__":
    main()
