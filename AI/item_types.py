"""
item_types.py — gom item theo LOẠI (gà, nước, khoai...).

Khách nói "2 miếng gà, 1 nước" chứ không nói "Gà Giòn Cay + Pepsi (M)".
ILP vì vậy ràng buộc theo loại, không theo tên item cụ thể.
Luật xét theo THỨ TỰ: "Burger Gà" -> burger (vì burger đứng trước gà).
Sửa từ khoá cho khớp tên món trong menu thật nếu thấy phân loại sai.
"""
import re
import unicodedata

TYPE_RULES = [
    ("sot",         ["sốt", "tương ớt", "tương cà", "mayonnaise", "mayo"]),
    ("burger",      ["burger"]),
    ("com",         ["cơm"]),
    ("mi",          ["mì ý", "mì", "spaghetti"]),
    ("ga_vien",     ["popcorn", "gà viên", "gà lắc", "nugget", "phi lê", "fillet", "tender"]),
    ("trang_mieng", ["kem", "bánh trứng", "tart", "sundae", "flan", "pudding", "bánh"]),
    ("nuoc",        ["pepsi", "7up", "7 up", "mirinda", "sting", "aquafina", "lipton", "trà",
                     "nước", "coca", "sprite", "fanta", "cà phê", "milo", "sữa", "soda", "drink"]),
    ("khoai",       ["khoai", "fries"]),
    ("salad",       ["salad", "bắp cải"]),
    ("an_vat",      ["phô mai", "xúc xích", "tôm", "mực", "bắp", "ngô"]),
    ("ga",          ["gà", "cánh", "đùi", "chicken", "cob", "miếng", "bucket"]),
]

# Loại mà khách thật sự "đếm" khi gọi món. sot / khac không tính vào nhu cầu.
VALUED_TYPES = [t for t, _ in TYPE_RULES if t != "sot"]

TYPE_LABELS = {
    "ga": "miếng gà", "ga_vien": "phần gà viên/phi lê", "burger": "burger", "com": "phần cơm",
    "mi": "phần mì", "khoai": "phần khoai", "nuoc": "nước", "trang_mieng": "tráng miệng",
    "salad": "salad", "an_vat": "ăn vặt", "sot": "sốt", "khac": "khác",
}


def normalize(text):
    return unicodedata.normalize("NFC", (text or "").lower()).strip()


def tokens(text):
    return set(re.findall(r"\w+", normalize(text)))


def classify(name, fallback=None):
    """Tên item/món -> loại. Không khớp luật nào -> fallback (thường là tên danh mục) -> 'khac'."""
    n = normalize(name)
    for item_type, keywords in TYPE_RULES:
        for kw in keywords:
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", n):
                return item_type
    if fallback:
        return classify(fallback)
    return "khac"