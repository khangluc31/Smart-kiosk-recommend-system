
import os
import re
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import ilp
from db import Database
from decide import MODEL_DIR, RecommendationPipeline
from pathlib import Path
from fastapi.responses import FileResponse
import edge_tts
BASE_DIR = Path(__file__).resolve().parent
NER_PATH = os.getenv("NER_PATH", os.path.join(MODEL_DIR, "ner_model"))
IMAGE_TEMPLATE = os.getenv("KFC_IMAGE_URL")      
STATE = {}
LOCK = threading.Lock()          
EMOJI = [("combo", "🍱"), ("bucket", "🍗"), ("burger", "🍔"), ("cơm", "🍚"), ("mì", "🍝"),
         ("súp", "🥣"), ("salad", "🥗"), ("bắp cải", "🥗"), ("khoai", "🍟"), ("kem", "🍦"),
         ("bánh trứng", "🥧"), ("pepsi", "🥤"), ("7up", "🥤"), ("lipton", "🧋"), ("trà", "🧋"),
         ("nước", "🥤"), ("gà", "🍗")]



def make_context_processor():
    import context
    has_ner = os.path.isdir(NER_PATH)
    if hasattr(context, "context_processing"):
        if not has_ner:         
            print(f"CẢNH BÁO: chưa có {NER_PATH}")
            return None
        return context.context_processing(model_path=NER_PATH)
    return context.ContextProcessor(ner_model_path=NER_PATH if has_ner else None,
                                    use_bert=os.getenv("USE_BERT_EMBED", "0") == "1")


@asynccontextmanager
async def lifespan(app):
    db = Database()
    cp = make_context_processor()
    STATE["db"] = db
    STATE["pipe"] = RecommendationPipeline(db, cp)
    STATE["categories"] = db.get_category_names()
    print(f"NER: {'BERT ' + NER_PATH if getattr(cp, 'model_name', None) else 'CHƯA CÓ models/ner_model'} | "
          f"two tower: {'có' if STATE['pipe'].two_tower else 'CHƯA TRAIN'} | "
          f"item2vec: {'có' if STATE['pipe'].item2vec else 'CHƯA TRAIN'}")
    yield
    db.close()


app = FastAPI(title="KBuddy API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[os.getenv("FRONTEND_ORIGIN", "*")],
                   allow_methods=["*"], allow_headers=["*"])


class Answers(BaseModel):
    want: str = ""
    avoid: str = ""
    party: str = ""
    budget: str = ""


class SessionIn(BaseModel):
    name: str = ""
    phone: str = ""
    answers: Answers = Answers()


class CartLine(BaseModel):
    id: int
    quantity: int = Field(1, ge=1)


class CartIn(BaseModel):
    session_id: str
    cart: list[CartLine]


def emoji(name):
    low = name.lower()
    return next((e for k, e in EMOJI if k in low), "🍗")


def image_url(m):
    if IMAGE_TEMPLATE and m.get("image_name"):
        return IMAGE_TEMPLATE.replace("{name}", m["image_name"])
    return m.get("image_url")


def meal_out(meal_id, quantity=None):
    m = STATE["pipe"].meals[meal_id]
    out = {"id": meal_id, "name": m["name"], "price": m["price"], "description": m["description"],
           "emoji": emoji(m["name"]), "image": image_url(m), "category": STATE["categories"].get(m["cat_id"], ""),
           "is_combo": m["is_combo"], "is_hot": m["is_hotsell"] or m["is_bestsell"]}
    if quantity is not None:
        out["quantity"] = quantity
    return out


def items_out(items):
    return [meal_out(i["meal_id"], i["qty"]) for i in items]


def to_transcript(a: Answers):
    parts = []
    want = a.want.strip()
    if want:
        has_pos = re.search(
            r"\b(muốn|thích|cho|lấy|ăn|uống)\b",
            want.lower()
        )
        if has_pos:
            parts.append(want)
        else:
            parts.append(f"Tôi muốn ăn {want}")
    avoid = a.avoid.strip()
    skip_values = {
        "không có",
        "không",
        "ko",
        "không có gì",
        "không dị ứng",
        "không bị dị ứng"
    }

    if avoid and avoid.lower() not in skip_values:
        has_neg = re.search(
            r"\b(không|đừng|bỏ|khỏi|trừ|dị ứng|chẳng)\b",
            avoid.lower()
        )

        if has_neg:
            parts.append(avoid)
        else:
            parts.append(
                f"Tôi không muốn ăn {avoid}"
            )
    if a.party.strip():
        n = re.search(r"\d+", a.party)

        if n:
            parts.append(f"Cho {n.group()} người")
        else:
            parts.append(a.party.strip())
    if a.budget.strip():
        b = a.budget.strip()

        if re.search(r"[a-zA-Zđ]", b):
            parts.append(f"Ngân sách khoảng {b}")
        else:
            parts.append(f"Ngân sách khoảng {b} đồng")

    return ". ".join(parts)


def check_cart(cart):
    meals = STATE["pipe"].meals
    bad = [c.id for c in cart if c.id not in meals]
    if bad:
        raise HTTPException(400, f"Món không tồn tại: {bad}")
    if not cart:
        raise HTTPException(400, "Giỏ hàng trống")
    return [(c.id, c.quantity) for c in cart]


def check_session(session_id):
    if session_id not in STATE["pipe"]._sessions:
        raise HTTPException(404, "Phiên đã hết hạn hoặc server vừa khởi động lại. Vui lòng bắt đầu lại.")


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    pipe = STATE["pipe"]
    return {"ok": True, "meals": len(pipe.meals), "two_tower": pipe.two_tower is not None,
            "item2vec": pipe.item2vec is not None,
            "ner": getattr(pipe.cp, "ner_model_path", None) is not None}


@app.get("/api/meals")
def meals():
    pipe = STATE["pipe"]
    ids = sorted((m for m, v in pipe.meals.items() if v["available"]),
                 key=lambda m: (pipe.meals[m]["cat_id"], pipe.meals[m]["price"]))
    return {"meals": [meal_out(m) for m in ids]}


@app.post("/api/session")
def session(body: SessionIn):
    name, phone = body.name.strip(), body.phone.strip()
    if bool(name) != bool(phone):
        raise HTTPException(400, "Cần nhập cả họ tên và số điện thoại (hoặc bỏ trống cả hai).")
    if phone and not re.fullmatch(r"(0|\+84)\d{9}", phone):
        raise HTTPException(400, "Số điện thoại không hợp lệ")
    transcript = to_transcript(body.answers)
    with LOCK:
        r = STATE["pipe"].start_session(transcript, name or None, phone or None)
    return {"session_id": r["session_id"], "route": r["route"], "is_regular": r["is_regular"],
            "budget": r["budget"], "unwant": r["unwant"], "transcript": transcript,
            "meals": [meal_out(m) for m in r["top"]]}


@app.post("/api/optimize-cart")
def optimize_cart(body: CartIn):
    """Chỉ ILP1: giỏ rẻ nhất có đúng những gì khách chọn (chưa gợi ý mua thêm)."""
    check_session(body.session_id)
    cart = check_cart(body.cart)
    pipe = STATE["pipe"]
    info = pipe._sessions[body.session_id]
    with LOCK:
        d = ilp.build_data(pipe.menu_model, cart, info.get("budget"), info.get("unwant", []))
        best = ilp.optimize_ilp(d["feasible_set"], d["price_set"], d["composition"],
                                d["customer_choice_set"]) or dict(d["chosen"])
    original = sum(d["price_set"][j] * n for j, n in d["chosen"].items())
    cost = sum(d["price_set"][j] * n for j, n in best.items())
    budget = info.get("budget")
    return {"cart": [meal_out(j, n) for j, n in sorted(best.items(), key=lambda x: -d["price_set"][x[0]])],
            "original_cost": original, "cost": cost, "saving": original - cost,
            "changed": best != dict(d["chosen"]), "budget": budget,
            "over_budget": bool(budget) and cost > budget}


@app.post("/api/upsell")
def upsell(body: CartIn):
    """Giỏ đã chốt -> ilp.plan đầy đủ: ILP2 (lên combo) + Item2vec + ILP3 (món mua thêm)."""
    check_session(body.session_id)
    cart = check_cart(body.cart)
    with LOCK:
        plan = STATE["pipe"].plan_cart(body.session_id, cart)
    up = plan["upgrade"]
    budget = (STATE["pipe"]._sessions.get(body.session_id) or {}).get("budget")
    upgrade = None
    if up.get("status") == "ok" and up["extra_cost"] > 0:
        upgrade = {"cart": items_out(up["items"]), "cost": up["cost"],
                   "extra_cost": up["extra_cost"], "gained": up["gained"],
                   "over_budget": bool(budget) and up["cost"] > budget}
    cart_cost = sum(STATE["pipe"].meals[m]["price"] * q for m, q in cart)
    return {"upgrade": upgrade, "budget": budget,
            "left": budget - cart_cost if budget else None,
            "meals": [meal_out(a["meal_id"]) for a in plan["addons"]],
            "fit_source": plan["fit_source"], "messages": plan["messages"]}


@app.post("/api/checkout")
def checkout(body: CartIn):
    check_session(body.session_id)
    cart = check_cart(body.cart)
    with LOCK:
        order_id = STATE["pipe"].checkout(body.session_id, cart)
    meals = STATE["pipe"].meals
    return {"order_id": order_id, "total": sum(meals[m]["price"] * q for m, q in cart)}

@app.get("/api/tts")
async def text_to_speech(text: str):
    file_name = f"temp_{hash(text)}.mp3"
    if not os.path.exists(file_name):
        communicate = edge_tts.Communicate(text, "vi-VN-HoaiMyNeural")
        await communicate.save(file_name)
    return FileResponse(file_name, media_type="audio/mpeg")