
import os
import re

import psycopg2
import requests

MENU_URL = "https://api.kfcvietnam.com.vn/menu/kfcvn-generic-menu"

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "recsys_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "123456"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

SEASONAL_TYPES = {"Hot Deal", "Value Deals", "New Product"}
COMBO_TYPES = {"For One", "Sharing"}



def vn(field, default=""):
    for d in field or []:
        if d.get("lang") == "vn" and d.get("value"):
            return d["value"].strip()
    return default


def parse_item(text):
    text = text.strip()
    match = re.match(r"^(\d+)\s*(.*)$", text)
    if match:
        return match.group(2).strip(), int(match.group(1))
    return text, 1


def parse_components(description):
    """
    'A + 2 B + C / D' -> fixed = [('A',1), ('B',2)], n_parts = 3
    Phần có '/' là lựa chọn (đã có trong modgrps) nên không coi là thành phần cố định.
    """
    fixed, n_parts = [], 0
    for part in description.split("+"):
        part = part.strip()
        if not part:
            continue
        n_parts += 1
        if "/" not in part:
            fixed.append(parse_item(part))
    return fixed, n_parts


# --------------------------------------------------------------------------- #
# Ghi DB
# --------------------------------------------------------------------------- #
def upsert_category(cur, name):
    cur.execute(
        """
        INSERT INTO categories (name) VALUES (%s)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        (name,),
    )
    return cur.fetchone()[0]


def get_or_create_item(cur, name):
    cur.execute(
        """
        INSERT INTO items (name) VALUES (%s)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        (name,),
    )
    return cur.fetchone()[0]


def upsert_meal(cur, cat_id, item, product):
    name = vn(item.get("dname"), item.get("name", ""))
    description = vn(item.get("description")) or vn(product.get("description"))
    price = item.get("price")
    if price is None or float(price) <= 0:
        return None, []
    strike_price = item.get("strikePrice") or 0
    item_type = item.get("categoryName")

    is_seasonal = item_type in SEASONAL_TYPES or strike_price > price
    fixed, n_parts = parse_components(description)
    is_combo = (
        item_type in COMBO_TYPES
        or n_parts > 1
        or (len(fixed) == 1 and fixed[0][1] > 1)
    )

    cur.execute(
        """
        INSERT INTO meals (source_id, cat_id, name, description, price,
                           available, is_seasonal, is_combo)
        VALUES (%s, %s, %s, %s, %s, true, %s, %s)
        ON CONFLICT (source_id) DO UPDATE SET
            cat_id = EXCLUDED.cat_id, name = EXCLUDED.name,
            description = EXCLUDED.description, price = EXCLUDED.price,
            available = true, is_seasonal = EXCLUDED.is_seasonal,
            is_combo = EXCLUDED.is_combo
        RETURNING id
        """,
        (str(item["id"]), cat_id, name, description, price, is_seasonal, is_combo),
    )
    return cur.fetchone()[0], fixed


def insert_meal_item(cur, meal_id, item_id, quantity, group_id, extra_price=0):
    cur.execute(
        """
        INSERT INTO meals_item (meal_id, item_id, quantity, group_id, extra_price)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (meal_id, item_id, group_id) DO NOTHING
        """,
        (meal_id, item_id, quantity, group_id, extra_price),
    )


def insert_option_group(cur, meal_id, grp, prefix=""):
    """Một nhóm lựa chọn + các option bên trong. Nhóm lồng nhau xử lý bằng đệ quy."""
    name = prefix + vn(grp.get("dname"), grp.get("name", ""))
    cur.execute(
        """
        INSERT INTO option_groups (meal_id, name, min_choices, max_choices)
        VALUES (%s, %s, %s, %s) RETURNING id
        """,
        (meal_id, name, grp.get("min", 1), grp.get("max", 1)),
    )
    group_id = cur.fetchone()[0]

    for mod in grp.get("modifiers") or []:
        item_name, quantity = parse_item(vn(mod.get("dname"), mod.get("name", "")))
        item_id = get_or_create_item(cur, item_name)
        insert_meal_item(cur, meal_id, item_id, quantity, group_id, mod.get("price") or 0)
        for sub in mod.get("modgrps") or []:
            insert_option_group(cur, meal_id, sub, prefix=f"{item_name} - ")


def load_menu(cur, data):
    counts = {"categories": 0, "meals": 0}
    for cat in data["categories"][0]["categories"]:
        cat_name = vn(cat.get("dname"), cat.get("name", ""))
        EXCLUDED_CATEGORIES = {
            "upsell_2",
        }
        if not cat_name:
            continue
        if cat_name.strip().lower() in EXCLUDED_CATEGORIES:
            print(f"Bỏ qua danh mục: {cat_name}")
            continue
        cat_id = upsert_category(cur, cat_name)
        counts["categories"] += 1

        for product in cat.get("products") or []:
            for item in product.get("items") or []:
                meal_id, fixed = upsert_meal(cur, cat_id, item, product)
                counts["meals"] += 1

                # xoá chi tiết cũ rồi nạp lại -> chạy lại không bị nhân đôi option
                cur.execute("DELETE FROM meals_item WHERE meal_id = %s", (meal_id,))
                cur.execute("DELETE FROM option_groups WHERE meal_id = %s", (meal_id,))

                for item_name, quantity in fixed:
                    insert_meal_item(cur, meal_id, get_or_create_item(cur, item_name),
                                     quantity, None)
                for grp in item.get("modgrps") or []:
                    insert_option_group(cur, meal_id, grp)
    return counts


def main():
    response = requests.get(MENU_URL, timeout=30)
    response.raise_for_status()
    data = response.json()

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:                       # tự commit nếu ổn, rollback nếu lỗi
            with conn.cursor() as cur:
                counts = load_menu(cur, data)
                cur.execute("SELECT COUNT(*) FROM meals")
                total_meals = cur.fetchone()[0]
        print(f"Đã nạp {counts['categories']} danh mục, {counts['meals']} lượt món "
              f"({total_meals} món khác nhau trong DB).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
