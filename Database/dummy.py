"""
dummy.py — sinh dữ liệu giả THEO ĐƠN HÀNG, có quy luật để model học được.

Chạy sau menu.py. Ghi vào: users, sessions, orders, order_items,
rồi đặt cờ is_bestsell / is_hotsell theo doanh số giả.
Xuất thêm dummy_users.csv (tên, SĐT, user_id, persona...) để test đăng nhập
và kiểm tra model sau này có học ra đúng quy luật không.

Quy luật được cài vào dữ liệu:
  1. Mỗi khách có persona: ngân sách, giờ hay ăn, số người đi cùng.
  2. Mỗi khách thích 1–2 danh mục món chính, có "món ruột" KHÁC NHAU giữa
     ban ngày (trước 17h) và buổi tối -> cần biết giờ đặt mới đoán đúng.
  3. 35% khách ĐỔI KHẨU VỊ ở một mốc thời gian -> cần biết đơn gần đây,
     đếm "món mua nhiều nhất từ trước tới nay" sẽ sai.
  4. Mỗi khách có 1 loại nước quen.
  5. Mỗi món chính có 1 "món hợp" (nước hoặc món phụ) chung cho mọi khách
     -> tín hiệu "hay mua cùng nhau" cho Item2vec.
  6. 30% khách là khách quen (>= 11 đơn), còn lại 1–10 đơn; thêm đơn khách vãng lai.
"""
import csv
import os
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2.extras import execute_values

from ids import create_user_id

SEED = 42
N_USERS = 200
N_GUEST_ORDERS = 400
DAYS = 365
REGULAR_RATIO = 0.3
REGULAR_THRESHOLD = 11
DRIFT_RATIO = 0.35
RESET = True        # xoá users/sessions/orders cũ trước khi sinh (KHÔNG đụng tới menu)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "recsys_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

# Danh mục -> vai trò. Sửa từ khoá cho khớp tên danh mục trong DB của bạn.
ROLE_KEYWORDS = {
    "drink": ["uống", "tráng miệng", "nước", "kem"],
    "side": ["ăn nhẹ", "ăn kèm", "khoai"],
}

PERSONAS = {
    #             ngân sách (VND)         giờ hay ăn          số người   P(chọn combo)
    "student": dict(budget=(50_000, 120_000), hours=[11, 12, 17, 18, 21], party=(1, 1), p_combo=0.3),
    "office":  dict(budget=(80_000, 180_000), hours=[11, 12, 13],         party=(1, 2), p_combo=0.4),
    "family":  dict(budget=(200_000, 500_000), hours=[18, 19, 20],        party=(3, 5), p_combo=0.7),
    "night":   dict(budget=(70_000, 200_000), hours=[21, 22, 23],         party=(1, 2), p_combo=0.3),
}

FIRST = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Vũ", "Đặng", "Bùi", "Đỗ", "Ngô"]
MIDDLE = ["Văn", "Thị", "Minh", "Ngọc", "Đức", "Thu", "Hoàng", "Anh"]
LAST = ["An", "Bình", "Châu", "Dũng", "Giang", "Hà", "Khánh", "Linh", "Nam", "Phúc", "Quân", "Trang"]



def load_menu(cur):
    cur.execute("""
        SELECT m.id, m.cat_id, c.name, m.price, m.is_combo
        FROM meals m JOIN categories c ON c.id = m.cat_id
        WHERE m.available
    """)
    meals = {}
    for meal_id, cat_id, cat_name, price, is_combo in cur.fetchall():
        name = cat_name.lower()
        role = "main"
        for r, keywords in ROLE_KEYWORDS.items():
            if any(k in name for k in keywords):
                role = r
                break
        if is_combo and role == "main":
            role = "combo"
        meals[meal_id] = {"cat_id": cat_id, "price": float(price), "role": role}
    return meals


def by_role(meals, role):
    return [m for m, v in meals.items() if v["role"] == role]



def make_user(rng, meals, mains, combos, drinks, used_phones):
    while True:
        phone = "09" + "".join(rng.choice("0123456789") for _ in range(8))
        if phone not in used_phones:
            used_phones.add(phone)
            break
    name = f"{rng.choice(FIRST)} {rng.choice(MIDDLE)} {rng.choice(LAST)}"
    persona = rng.choice(list(PERSONAS))

    main_cats = sorted({meals[m]["cat_id"] for m in mains + combos})

    def new_taste():
        """Một 'giai đoạn khẩu vị': danh mục thích + món ruột ban ngày / buổi tối."""
        cats = rng.sample(main_cats, k=min(len(main_cats), rng.randint(1, 2)))
        pool = [m for m in mains + combos if meals[m]["cat_id"] in cats]
        return {
            "fav_cats": cats,
            "day": rng.sample(pool, k=min(len(pool), rng.randint(2, 4))),     # trước 17h
            "night": rng.sample(pool, k=min(len(pool), rng.randint(2, 4))),   # từ 17h
        }

    tastes = [new_taste()]
    # một số khách đổi khẩu vị: từ mốc drift_day (số ngày trước hiện tại) dùng khẩu vị mới
    drift_day = rng.randint(30, DAYS // 2) if rng.random() < DRIFT_RATIO else None
    if drift_day is not None:
        tastes.append(new_taste())

    is_regular = rng.random() < REGULAR_RATIO
    return {
        "user_id": create_user_id(name, phone),
        "name": name,
        "phone": phone,
        "persona": persona,
        "tastes": tastes,
        "drift_day": drift_day,
        "fav_drink": rng.choice(drinks) if drinks else None,
        "n_orders": rng.randint(REGULAR_THRESHOLD, 40) if is_regular else rng.randint(1, 10),
    }


def current_taste(user, ts, now):
    if user["drift_day"] is not None and (now - ts).days < user["drift_day"]:
        return user["tastes"][1]
    return user["tastes"][0]



def pick_anchor(rng, user, meals, mains, combos, persona, ts, now):
    """Món đầu tiên của đơn: ưu tiên món ruột (theo buổi + giai đoạn), rồi danh mục thích."""
    if user:
        taste = current_taste(user, ts, now)
        fav = taste["night"] if ts.hour >= 17 else taste["day"]
        r = rng.random()
        if r < 0.65:
            return rng.choice(fav)
        in_cat = [m for m in mains + combos if meals[m]["cat_id"] in taste["fav_cats"]]
        if in_cat and r < 0.85:
            return rng.choice(in_cat)
    want_combo = rng.random() < persona["p_combo"] and combos
    return rng.choice(combos if want_combo else mains)


def make_basket(rng, user, meals, groups, pairs, persona, ts, now):
    mains, combos, drinks, sides = groups
    budget = rng.uniform(*persona["budget"])
    party = rng.randint(*persona["party"])
    basket = Counter()

    anchor = pick_anchor(rng, user, meals, mains, combos, persona, ts, now)
    is_combo = meals[anchor]["role"] == "combo"
    basket[anchor] += 1 if is_combo else party

    # món hợp đi kèm (quy luật chung cho mọi khách)
    partner = pairs.get(anchor)
    if partner and rng.random() < (0.3 if is_combo else 0.6):
        basket[partner] += 1 if is_combo else party

    # nước: combo thường đã có nước nên ít gọi thêm
    if drinks and rng.random() < (0.2 if is_combo else 0.7):
        drink = user["fav_drink"] if user and rng.random() < 0.7 else rng.choice(drinks)
        basket[drink] += 1 if is_combo else party

    # món phụ
    if sides and rng.random() < (0.2 if is_combo else 0.4):
        basket[rng.choice(sides)] += 1

    # vượt ngân sách quá nhiều thì bỏ bớt món phụ/nước (giữ món chính)
    def total():
        return sum(meals[m]["price"] * q for m, q in basket.items())

    for m in [m for m in list(basket) if m != anchor]:
        if total() <= budget * 1.2:
            break
        del basket[m]
    return basket, round(budget, -3)


def random_time(rng, persona, now):
    day = now - timedelta(days=rng.randint(0, DAYS - 1))
    return day.replace(hour=rng.choice(persona["hours"]), minute=rng.randint(0, 59),
                       second=rng.randint(0, 59), microsecond=0)


def main():
    rng = random.Random(SEED)
    now = datetime.now(timezone(timedelta(hours=7)))

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn, conn.cursor() as cur:
            meals = load_menu(cur)
            groups = mains, combos, drinks, sides = [by_role(meals, r) for r in
                                                     ("main", "combo", "drink", "side")]
            print("Vai trò món:", {r: len(g) for r, g in
                                   zip(("main", "combo", "drink", "side"), groups)})
            if not mains:
                raise SystemExit("Không có món chính nào — kiểm tra ROLE_KEYWORDS hoặc chạy menu.py trước.")

            # món hợp đi kèm của từng món chính/combo
            partners = drinks + sides
            pairs = {m: rng.choice(partners) for m in mains + combos} if partners else {}

            if RESET:
                cur.execute("TRUNCATE rec_logs, cart_items, order_items, orders, sessions, users "
                            "RESTART IDENTITY CASCADE")

            used_phones = set()
            users = [make_user(rng, meals, mains, combos, drinks, used_phones)
                     for _ in range(N_USERS)]

            sessions, orders, baskets = [], [], {}
            jobs = [(u, PERSONAS[u["persona"]]) for u in users for _ in range(u["n_orders"])]
            jobs += [(None, PERSONAS[rng.choice(list(PERSONAS))]) for _ in range(N_GUEST_ORDERS)]
            for user, persona in jobs:
                created = random_time(rng, persona, now)
                basket, budget = make_basket(rng, user, meals, groups, pairs, persona, created, now)
                sid = str(uuid.UUID(int=rng.getrandbits(128), version=4))
                uid = user["user_id"] if user else None
                total = sum(meals[m]["price"] * q for m, q in basket.items())
                sessions.append((sid, uid, created - timedelta(minutes=rng.randint(1, 5)), None, budget))
                orders.append((sid, uid, created, total))
                baskets[sid] = basket

            execute_values(cur, "INSERT INTO users (id, created_at) VALUES %s",
                           [(u["user_id"], now - timedelta(days=DAYS)) for u in users])
            execute_values(cur, "INSERT INTO sessions (id, user_id, started_at, transcript, budget) "
                                "VALUES %s", sessions)
            rows = execute_values(cur, "INSERT INTO orders (session_id, user_id, created_at, total) "
                                       "VALUES %s RETURNING id, session_id", orders, fetch=True)
            order_items = [(oid, m, q, meals[m]["price"])
                           for oid, sid in rows for m, q in baskets[str(sid)].items()]
            execute_values(cur, "INSERT INTO order_items (order_id, meal_id, quantity, unit_price) "
                                "VALUES %s", order_items)

            # cờ bán chạy theo doanh số giả: top 10% toàn thời gian / 30 ngày gần nhất
            top = max(1, len(meals) // 10)
            cur.execute("""
                UPDATE meals SET
                  is_bestsell = id IN (SELECT meal_id FROM meal_stats ORDER BY num_sold DESC LIMIT %s),
                  is_hotsell  = id IN (SELECT meal_id FROM meal_stats ORDER BY num_sold_30d DESC LIMIT %s)
            """, (top, top))

        with open("dummy_users.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "phone", "user_id", "persona", "n_orders", "fav_drink",
                        "fav_cats", "fav_day", "fav_night",
                        "drift_days_ago", "new_fav_cats", "new_fav_day", "new_fav_night"])

            def ids(xs):
                return " ".join(map(str, xs))

            for u in users:
                t0 = u["tastes"][0]
                t1 = u["tastes"][1] if len(u["tastes"]) > 1 else None
                w.writerow([u["name"], u["phone"], u["user_id"], u["persona"], u["n_orders"],
                            u["fav_drink"], ids(t0["fav_cats"]), ids(t0["day"]), ids(t0["night"]),
                            u["drift_day"] or "",
                            ids(t1["fav_cats"]) if t1 else "", ids(t1["day"]) if t1 else "",
                            ids(t1["night"]) if t1 else ""])

        n_regular = sum(u["n_orders"] >= REGULAR_THRESHOLD for u in users)
        print(f"{len(users)} khách ({n_regular} khách quen), {len(orders)} đơn, "
              f"{len(order_items)} dòng order_items. Đã ghi dummy_users.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
