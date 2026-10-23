"""
db.py — mọi truy cập database nằm ở đây (theo schema.sql).

Các model (regular, unregular, ranker) không biết gì về SQL;
chúng chỉ nhận dict/list mà file này trả về.
"""
import os
from collections import defaultdict

import psycopg2
from psycopg2.extras import execute_values

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "recsys_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}


class Database:
    def __init__(self, config=None):
        self.conn = psycopg2.connect(**(config or DB_CONFIG))

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _all(self, sql, params=None):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _one(self, sql, params=None):
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def _write(self, sql, params=None, fetch=False):
        with self.conn, self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone() if fetch else None

    # ------------------------------------------------------------------ #
    # Menu
    # ------------------------------------------------------------------ #
    def get_meals(self):
        """{meal_id: {...}} — chỉ đọc, gồm cả món hết hàng (cờ available)."""
        rows = self._all("""
            SELECT m.id, m.cat_id, m.name, COALESCE(m.description, ''), m.price,
                   m.available, m.is_hotsell, m.is_bestsell, m.is_seasonal, m.is_combo,
                   s.num_sold
            FROM meals m JOIN meal_stats s ON s.meal_id = m.id
            ORDER BY m.id
        """)
        return {
            r[0]: {
                "cat_id": r[1], "name": r[2], "description": r[3], "price": float(r[4]),
                "available": r[5], "is_hotsell": r[6], "is_bestsell": r[7],
                "is_seasonal": r[8], "is_combo": r[9], "num_sold": int(r[10]),
            }
            for r in rows
        }

    def get_menu_structure(self):
        """
        Cấu trúc món cho ILP:
          fixed[meal_id]  = [(item_id, item_name, item_type, quantity)]        thành phần cố định
          groups[meal_id] = [{"id","name","min","max","options":[(item_id, name, type, qty, extra_price)]}]
        """
        fixed, groups = defaultdict(list), defaultdict(dict)
        for meal_id, item_id, name, itype, qty, gid, gname, gmin, gmax, extra in self._all("""
            SELECT mi.meal_id, i.id, i.name, i.item_type, mi.quantity,
                   og.id, og.name, og.min_choices, og.max_choices, mi.extra_price
            FROM meals_item mi
            JOIN items i ON i.id = mi.item_id
            LEFT JOIN option_groups og ON og.id = mi.group_id
            ORDER BY mi.meal_id, og.id NULLS FIRST, mi.id
        """):
            if gid is None:
                fixed[meal_id].append((item_id, name, itype, qty))
            else:
                g = groups[meal_id].setdefault(
                    gid, {"id": gid, "name": gname, "min": gmin, "max": gmax, "options": []})
                g["options"].append((item_id, name, itype, qty, float(extra)))
        return dict(fixed), {m: list(g.values()) for m, g in groups.items()}

    def get_category_names(self):
        return dict(self._all("SELECT id, name FROM categories"))

    # ------------------------------------------------------------------ #
    # Lịch sử đơn hàng
    # ------------------------------------------------------------------ #
    def get_orders(self, user_id=None, before=None, include_guests=True):
        """
        Danh sách đơn, sắp theo thời gian tăng dần:
            {"order_id", "session_id", "user_id", "created_at", "total", "budget",
             "items": [(meal_id, quantity), ...]}
        user_id: chỉ lấy đơn của khách này.  before: chỉ lấy đơn trước thời điểm này.
        """
        where, params = [], []
        if user_id is not None:
            where.append("o.user_id = %s")
            params.append(user_id)
        elif not include_guests:
            where.append("o.user_id IS NOT NULL")
        if before is not None:
            where.append("o.created_at < %s")
            params.append(before)
        sql = """
            SELECT o.id, o.session_id, o.user_id, o.created_at, o.total, s.budget,
                   oi.meal_id, oi.quantity
            FROM orders o
            JOIN sessions s     ON s.id = o.session_id
            JOIN order_items oi ON oi.order_id = o.id
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY o.created_at, o.id"

        orders = {}
        for oid, sid, uid, ts, total, budget, meal_id, qty in self._all(sql, params):
            if oid not in orders:
                orders[oid] = {
                    "order_id": oid, "session_id": str(sid),
                    "user_id": str(uid) if uid else None, "created_at": ts,
                    "total": float(total), "budget": float(budget) if budget else None,
                    "items": [],
                }
            orders[oid]["items"].append((meal_id, qty))
        return list(orders.values())

    @staticmethod
    def group_by_user(orders):
        by_user = defaultdict(list)
        for o in orders:
            if o["user_id"]:
                by_user[o["user_id"]].append(o)
        return dict(by_user)

    # ------------------------------------------------------------------ #
    # Khách hàng
    # ------------------------------------------------------------------ #
    def user_exists(self, user_id):
        return self._one("SELECT 1 FROM users WHERE id = %s", (user_id,)) is not None

    def ensure_user(self, user_id):
        self._write("INSERT INTO users (id) VALUES (%s) ON CONFLICT (id) DO NOTHING", (user_id,))

    def get_num_buy(self, user_id):
        row = self._one("SELECT num_buy FROM user_stats WHERE user_id = %s", (user_id,))
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------ #
    # Phiên gọi món, log gợi ý, đơn hàng
    # ------------------------------------------------------------------ #
    def create_session(self, user_id=None, transcript=None, budget=None):
        row = self._write(
            "INSERT INTO sessions (user_id, transcript, budget) VALUES (%s, %s, %s) "
            "RETURNING id, started_at",
            (user_id, transcript, budget), fetch=True,
        )
        return str(row[0]), row[1]

    def log_recommendations(self, session_id, stage, source, meal_ids, scores=None):
        scores = scores or [None] * len(meal_ids)
        with self.conn, self.conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO rec_logs (session_id, meal_id, stage, source, position, score) VALUES %s",
                [(session_id, m, stage, source, i, s)
                 for i, (m, s) in enumerate(zip(meal_ids, scores))],
            )

    def get_session_recs(self, session_id):
        return self._all(
            "SELECT meal_id, stage, source, chosen FROM rec_logs WHERE session_id = %s",
            (session_id,),
        )

    def save_order(self, session_id, user_id, items):
        """items: [(meal_id, quantity)]. Giá lấy từ bảng meals tại thời điểm đặt."""
        with self.conn, self.conn.cursor() as cur:
            ids = [m for m, _ in items]
            cur.execute("SELECT id, price FROM meals WHERE id = ANY(%s)", (ids,))
            price = {mid: p for mid, p in cur.fetchall()}
            total = sum(price[m] * q for m, q in items)
            cur.execute(
                "INSERT INTO orders (session_id, user_id, total) VALUES (%s, %s, %s) RETURNING id",
                (session_id, user_id, total),
            )
            order_id = cur.fetchone()[0]
            execute_values(
                cur,
                "INSERT INTO order_items (order_id, meal_id, quantity, unit_price) VALUES %s",
                [(order_id, m, q, price[m]) for m, q in items],
            )
            cur.execute(
                "UPDATE rec_logs SET chosen = true WHERE session_id = %s AND meal_id = ANY(%s)",
                (session_id, ids),
            )
        return order_id