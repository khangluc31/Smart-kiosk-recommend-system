"""
decide.py — phân luồng khi phục vụ khách (theo sơ đồ), dùng các model của bạn:

  câu khách nói ─► context (ngân sách, thứ không muốn, vector câu)
  có tài khoản & >= 11 đơn ─► regular.TwoTowerRecommender ─┐
  còn lại                  ─► unregular.linUCB            ─┴─► top 20 ─► ranker (LightGBM) ─► top 5
  khách chọn món ─► ilp.plan: giỏ rẻ nhất + nâng cấp + gợi ý mua thêm
  khách đặt đơn  ─► lưu orders + rec_logs ─► linUCB.update (reward 1/0 cho món đã hiển thị)

Chạy:
  python decide.py train   # train two tower + warm start LinUCB từ lịch sử đơn (chạy offline)
  python decide.py demo    # thử 1 phiên phục vụ
"""
import os
import pickle
import sys
from datetime import datetime

import numpy as np

import ilp
from db import Database
from features import VN_TZ, build_queries, user_features
from ids import create_user_id
from ranker import Ranker, build_features
from regular import TwoTowerRecommender, hour_bucket
from unregular import linUCB

REGULAR_THRESHOLD = 11        # >= 11 đơn = khách quen -> two tower
TOP_K_RETRIEVE = 20           # số ứng viên đưa cho ranker
TOP_K_SHOW = 5                # số món hiển thị cho khách
MODEL_DIR = os.getenv("MODEL_DIR", "models")
TWO_TOWER_PATH = os.path.join(MODEL_DIR, "two_tower.pt")
LINUCB_PATH = os.path.join(MODEL_DIR, "linucb.pkl")
RANKER_PATH = os.path.join(MODEL_DIR, "ranker.txt")


# --------------------------------------------------------------------------- #
# Vector bối cảnh x cho LinUCB — DÙNG CHUNG cho warm start và lúc phục vụ
# --------------------------------------------------------------------------- #
N_CONTEXT = 10


def make_x(ts, budget=None, has_account=False):
    """
    [1, trưa, chiều, tối, khuya, cuối tuần, có ngân sách, log(ngân sách)/13, ngân sách<100k, có tài khoản]
    (sáng là mốc nên không có cột riêng)
    """
    local = ts.astimezone(VN_TZ)
    hb = hour_bucket(local.hour)                     # cùng hàm với two tower (regular.py)
    return np.array([
        1.0,
        float(hb == 1), float(hb == 2), float(hb == 3), float(hb == 4),
        float(local.weekday() >= 5),
        float(budget is not None),
        np.log1p(budget) / 13.0 if budget else 0.0,
        float(budget is not None and budget < 100_000),
        float(has_account),
    ])


def to_two_tower_user(feats, ts):
    """features.user_features -> dict cho two tower, khung giờ theo hour_bucket của regular.py."""
    user = dict(feats)
    user["hour_bucket"] = hour_bucket(ts.astimezone(VN_TZ).hour)
    return user


# --------------------------------------------------------------------------- #
# Lưu / tải LinUCB kèm danh sách món (arm số mấy là món nào)
# --------------------------------------------------------------------------- #
def save_linucb(model, meal_ids, path=LINUCB_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "meal_ids": list(meal_ids)}, f)


def load_linucb(meal_ids, path=LINUCB_PATH):
    """Menu đổi: món cũ giữ A, b đã học; món mới bắt đầu từ cold start (A = I, b = 0)."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        state = pickle.load(f)
    old_model, old_ids = state["model"], state["meal_ids"]
    if list(old_ids) == list(meal_ids):
        return old_model
    new = linUCB(len(meal_ids), old_model.n_feature, old_model.alpha)
    old_arm = {m: a for a, m in enumerate(old_ids)}
    for a, m in enumerate(meal_ids):
        if m in old_arm:
            new.A[a] = old_model.A[old_arm[m]].copy()
            new.b[a] = old_model.b[old_arm[m]].copy()
    return new


def linucb_history(orders, arm_of):
    """Đơn cũ -> history cho linUCB.warm_start: [(x, [arm đã mua]), ...]"""
    history = []
    for o in orders:
        arms = [arm_of[m] for m, _ in o["items"] if m in arm_of]
        if arms:
            x = make_x(o["created_at"], o["budget"], has_account=o["user_id"] is not None)
            history.append((x, arms))
    return history


# --------------------------------------------------------------------------- #
# Hiểu câu nói: nhận được cả context_processing (của bạn) lẫn ContextProcessor
# --------------------------------------------------------------------------- #
def understand(cp, text):
    out = {"budget": None, "want_text": [], "unwant_text": [],
           "sentence": None, "want": [], "unwant": []}
    if cp is None or not text:
        return out
    if hasattr(cp, "process"):                                   # ContextProcessor
        return {**out, **cp.process(text)}
    if hasattr(cp, "context_extraction"):                        # context_processing của bạn
        cp.context_extraction(text)

        def np_vec(t):
            return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

        wants, unwants = cp.get_want_list(), cp.get_unwant_list()
        budgets = cp.extract_budget(text) if hasattr(cp, "extract_budget") else []
        sentence = cp.get_vector()
        out.update(
            budget=min(budgets) if budgets else None,            # nhiều con số -> lấy mức thấp nhất
            want_text=[e["text"] for e in wants], unwant_text=[e["text"] for e in unwants],
            want=[np_vec(e["embedding_vector"]) for e in wants],
            unwant=[np_vec(e["embedding_vector"]) for e in unwants],
            sentence=np_vec(sentence) if sentence is not None else None,
        )
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class RecommendationPipeline:
    def __init__(self, db, context_processor=None):
        self.db = db
        self.cp = context_processor
        self.meals = db.get_meals()
        self.meal_ids = list(self.meals)
        self.arm_of = {m: a for a, m in enumerate(self.meal_ids)}
        self.menu_model = ilp.MenuModel.from_db(db, self.meals)

        self.two_tower = (TwoTowerRecommender.load(TWO_TOWER_PATH, self.meals)
                          if os.path.exists(TWO_TOWER_PATH) else None)
        self.linucb = load_linucb(self.meal_ids)
        if self.linucb is None:                       # chưa có -> warm start từ lịch sử rồi lưu
            self.linucb = linUCB(len(self.meal_ids), N_CONTEXT, alpha=1.0)
            self.linucb.warm_start(linucb_history(db.get_orders(), self.arm_of))
            save_linucb(self.linucb, self.meal_ids)
        self.ranker = Ranker.load(RANKER_PATH)        # chưa train -> giữ thứ tự retrieval
        self.meal_vecs = {}                           # vector món cho ranker (khi có model embedding)
        self._sessions = {}                           # session_id -> bối cảnh để update / ILP

    # ---------------- bước 1: gợi ý ban đầu ----------------
    def start_session(self, transcript="", name=None, phone=None, now=None):
        now = now or datetime.now(VN_TZ)
        user_id = create_user_id(name, phone) if name and phone else None
        if user_id:
            self.db.ensure_user(user_id)

        text = understand(self.cp, transcript)
        budget = text["budget"]
        session_id, _ = self.db.create_session(user_id, transcript, budget)

        history = self.db.get_orders(user_id=user_id) if user_id else []
        user = to_two_tower_user(user_features(history, self.meals, now, budget, user_id), now)
        is_regular = user["num_buy"] >= REGULAR_THRESHOLD and self.two_tower is not None
        banned = ilp.banned_from_text(self.menu_model, text["unwant_text"])

        x = None
        if is_regular:
            source = "two_tower"
            cand, scores = self.two_tower.recommend(user, k=len(self.meal_ids))
        else:
            source = "linucb"
            x = make_x(now, budget, has_account=user_id is not None)
            arms, arm_scores = self.linucb.select_arm(np.tile(x, (len(self.meal_ids), 1)),
                                                      k=len(self.meal_ids))
            cand = [self.meal_ids[a] for a in arms]
            scores = [float(s) for s in arm_scores]

        # bỏ món hết hàng và món chứa thứ khách không muốn, giữ top 20
        kept = [(m, s) for m, s in zip(cand, scores)
                if ilp.meal_allowed(self.menu_model, m, banned)][:TOP_K_RETRIEVE]
        cand, scores = [m for m, _ in kept], [s for _, s in kept]

        X = build_features(cand, scores, source, self.meals, user, text, self.meal_vecs)
        top, top_scores = self.ranker.rank(cand, X, k=TOP_K_SHOW)
        self.db.log_recommendations(session_id, "initial", source, top, top_scores)

        self._sessions[session_id] = {
            "ts": now, "budget": budget, "user_id": user_id, "source": source, "x": x,
            "unwant": text["unwant_text"], "scores": dict(zip(cand, scores)),
        }
        return {"session_id": session_id, "user_id": user_id, "route": source,
                "is_regular": is_regular, "budget": budget, "unwant": text["unwant_text"],
                "candidates": cand, "top": top,
                "top_names": [self.meals[m]["name"] for m in top]}

    # ---------------- bước 2: khách chọn món -> ILP ----------------
    def plan_cart(self, session_id, cart):
        """cart = [(meal_id, qty)] hoặc [{"meal_id", "qty", "choices": {group_id: [item_id]}}]"""
        info = self._sessions.get(session_id, {})
        result = ilp.plan(self.menu_model, cart, budget=info.get("budget"),
                          unwant_texts=info.get("unwant", []), scores=info.get("scores") or None)
        upsell = [a["meal_id"] for a in result["addons"]]
        up = result["upgrade"]
        if up and up["status"] == "ok" and up["extra_cost"] > 0:
            upsell += [i["meal_id"] for i in up["items"] if i["meal_id"] not in upsell]
        if upsell:
            self.db.log_recommendations(session_id, "upsell", "ilp", upsell)
        result["messages"] = ilp.describe(result)
        return result

    # ---------------- bước 3: đặt đơn -> lưu + LinUCB học ----------------
    def checkout(self, session_id, items):
        """items = [(meal_id, qty)]"""
        info = self._sessions.pop(session_id, {})
        order_id = self.db.save_order(session_id, info.get("user_id"), items)
        if info.get("source") == "linucb" and info.get("x") is not None:
            bought = {m for m, _ in items}
            for meal_id, stage, source, _ in self.db.get_session_recs(session_id):
                if stage == "initial" and source == "linucb" and meal_id in self.arm_of:
                    reward = 1.0 if meal_id in bought else 0.0
                    self.linucb.update(self.arm_of[meal_id], info["x"], reward)
            save_linucb(self.linucb, self.meal_ids)
        return order_id


# --------------------------------------------------------------------------- #
# Offline: train two tower + warm start LinUCB
# --------------------------------------------------------------------------- #
def train_all(db, epochs=25):
    meals = db.get_meals()
    orders = db.get_orders()

    # mỗi món trong mỗi đơn của khách = 1 mẫu (user tính từ các đơn TRƯỚC đơn đó)
    samples = []
    for q in build_queries(Database.group_by_user(orders), meals):
        user = to_two_tower_user(q["user"], q["ts"])
        samples += [(user, m) for m in sorted(q["targets"])]
    print(f"Two tower: {len(samples)} mẫu từ {len(orders)} đơn")
    rec = TwoTowerRecommender()
    rec.fit(samples, meals, epochs=epochs)
    os.makedirs(MODEL_DIR, exist_ok=True)
    rec.save(TWO_TOWER_PATH)

    meal_ids = list(meals)
    model = linUCB(len(meal_ids), N_CONTEXT, alpha=1.0)
    history = linucb_history(orders, {m: a for a, m in enumerate(meal_ids)})
    model.warm_start(history)
    save_linucb(model, meal_ids)
    print(f"LinUCB: warm start từ {len(history)} đơn. Đã lưu vào {MODEL_DIR}/")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    with Database() as db:
        if cmd == "train":
            train_all(db)
        else:
            pipe = RecommendationPipeline(db)                 # truyền context_processing() khi có NER
            r = pipe.start_session("cho mình 1 phần, tầm 120k")
            print(r["route"], r["budget"], r["top_names"])
            plan = pipe.plan_cart(r["session_id"], [(r["top"][0], 1)])
            print(*plan["messages"], sep="\n")