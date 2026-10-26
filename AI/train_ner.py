
import argparse
import json
import os
import random
import re
import unicodedata
from collections import Counter

import numpy as np
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

try:
    from context import LABELS                 
except ImportError:
    LABELS = {0: "O", 1: "B-want", 2: "B-unwant", 3: "I-want", 4: "I-unwant"}

LABEL2ID = {v: k for k, v in LABELS.items()}
TOKEN_KEYS = ("tokens", "words")
TAG_KEYS = ("labels", "tags", "ner_tags", "iob")

CONT = {LABEL2ID["B-want"]: LABEL2ID["I-want"], LABEL2ID["B-unwant"]: LABEL2ID["I-unwant"]}



def parse_marked_line(line):
    """Convert `[W gà rán]`, `[U khoai tây]` into a BIO example."""
    tokens, labels = [], []
    for match in re.finditer(r"\[([WU]) ([^\]]+)\]|([^\[]+)", line):
        kind = match.group(1)
        segment = match.group(2) if kind else match.group(3)
        pieces = re.findall(r"\w+|[^\w\s]", segment)
        for i, token in enumerate(pieces):
            tokens.append(token)
            labels.append(0 if not kind else (1 if kind == "W" else 2) + (2 if i else 0))
    if not tokens:
        raise ValueError("Empty annotated line")
    return {
        "text": re.sub(r"\[([WU]) ([^\]]+)\]", r"\2", line),
        "tokens": tokens,
        "labels": labels,
    }


def generate_marked_test(source, output):
    """Convert a UTF-8 text file containing one annotated sentence per line."""
    with open(source, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    with open(output, "w", encoding="utf-8") as f:
        for line in lines:
            if not re.fullmatch(r"(?:\[[WU] [^\]]+\]|[^\[])+", line):
                raise ValueError(f"Malformed [W ...] / [U ...] notation: {line}")
            f.write(json.dumps(parse_marked_line(line), ensure_ascii=False) + "\n")
    print(f"Đã chuyển {len(lines)} câu có nhãn thành {output}")
    return output


def _norm_label(tag):
    if isinstance(tag, int) or str(tag).isdigit():
        return LABELS.get(int(tag), f"?{tag}")
    tag = str(tag).strip()
    if tag.upper() == "O":
        return "O"
    prefix, _, ent = tag.partition("-")
    return f"{prefix.upper()}-{ent.lower()}"


def spans(tags):

    out, cur = set(), None
    for i, t in enumerate(list(tags) + ["O"]):
        if cur and not (t.startswith("I-") and t[2:] == cur[0]):
            out.add((cur[0], cur[1], i))
            cur = None
        if t.startswith("B-"):
            cur = (t[2:], i)
    return out


def entity_strings(tokens, tags, kind=None):
    return {" ".join(tokens[a:b]).lower() for e, a, b in spans(tags) if kind in (None, e)}


def load_data(path, drop_conflicts=True, verbose=True):
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    rows = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    if isinstance(rows, dict):
        rows = next(v for v in rows.values() if isinstance(v, list))

    data, problems, seen = [], Counter(), set()
    for r in rows:
        tokens = next((r[k] for k in TOKEN_KEYS if k in r), None)
        if tokens is None and "text" in r:
            tokens = r["text"].split()
        tags = next((r[k] for k in TAG_KEYS if k in r), None)
        if tokens is None or tags is None:
            problems["thiếu tokens/labels"] += 1
            continue
        if len(tokens) != len(tags):
            problems["số từ != số nhãn"] += 1
            continue
        tags = [_norm_label(t) for t in tags]
        if any(t not in LABEL2ID for t in tags):
            problems["nhãn lạ"] += 1
            continue
        for i, t in enumerate(tags):                   # I- đứng đầu thực thể -> B-
            if t.startswith("I-") and (i == 0 or tags[i - 1][2:] != t[2:]):
                problems["I- mồ côi (đã sửa thành B-)"] += 1
                tags[i] = "B-" + t[2:]
        tokens = [str(t) for t in tokens]
        key = " ".join(tokens).lower()
        if key in seen:
            problems["câu trùng (bỏ)"] += 1
            continue
        seen.add(key)
        if drop_conflicts and entity_strings(tokens, tags, "want") & entity_strings(tokens, tags, "unwant"):
            problems["câu tự mâu thuẫn: 1 món vừa want vừa unwant (bỏ)"] += 1
            continue
        data.append((tokens, tags))

    if verbose:
        print(f"Đọc {len(rows)} câu từ {os.path.basename(path)}, dùng được {len(data)}")
        for p, c in problems.items():
            print(f"  ! {p}: {c}")
        print("  phân bố nhãn:", dict(Counter(t for _, tags in data for t in tags)))
    return data


def split(data, val_frac, unseen_frac, seed):
    rng = random.Random(seed)
    names = sorted({s for t, l in data for s in entity_strings(t, l)})
    held = set(rng.sample(names, int(len(names) * unseen_frac))) if unseen_frac > 0 else set()
    unseen = [d for d in data if entity_strings(*d) & held]
    rest = [d for d in data if not entity_strings(*d) & held]
    rng.shuffle(rest)
    n_val = max(1, int(len(rest) * val_frac))
    print(f"  {len(names)} tên món khác nhau; giữ riêng {len(held)} món cho val_unseen: {sorted(held)}")
    return rest[n_val:], rest[:n_val], unseen



def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def augment(tokens, rng, p_lower, p_noaccent):
    if rng.random() < p_noaccent:
        tokens = [strip_accents(t) for t in tokens]
    if rng.random() < p_lower:
        tokens = [t.lower() for t in tokens]
    return tokens

def encode(tokenizer, batch, max_len=128):
    enc = tokenizer([t for t, _ in batch], is_split_into_words=True, truncation=True,
                    max_length=max_len, padding=True, return_tensors="pt")
    labels = []
    for i, (_, tags) in enumerate(batch):
        prev, row = None, []
        for w in enc.word_ids(batch_index=i):
            if w is None:
                row.append(-100)                       
            else:
                lid = LABEL2ID[tags[w]]
                row.append(CONT.get(lid, lid) if w == prev else lid)
            prev = w
        labels.append(row)
    enc["labels"] = torch.tensor(labels)
    return enc


@torch.no_grad()
def evaluate(model, tokenizer, data, device, bs=64):
    model.eval()
    tp, n_pred, n_gold = Counter(), Counter(), Counter()
    loss_sum, n = 0.0, 0
    for s in range(0, len(data), bs):
        batch = data[s:s + bs]
        enc = encode(tokenizer, batch).to(device)
        out = model(**enc)
        loss_sum += out.loss.item() * len(batch)
        n += len(batch)
        pred_ids = out.logits.argmax(-1).cpu()
        for i, (tokens, gold) in enumerate(batch):
            pred, prev = ["O"] * len(tokens), None
            for j, w in enumerate(enc.word_ids(batch_index=i)):
                if w is not None and w != prev:
                    pred[w] = LABELS[int(pred_ids[i, j])]
                prev = w
            g, p = spans(gold), spans(pred)
            for e in ("want", "unwant"):
                gs, ps = {x for x in g if x[0] == e}, {x for x in p if x[0] == e}
                tp[e] += len(gs & ps)
                n_pred[e] += len(ps)
                n_gold[e] += len(gs)
    model.train()
    res = {"loss": loss_sum / max(n, 1)}
    for e in ("want", "unwant", "all"):
        t = sum(tp.values()) if e == "all" else tp[e]
        p = sum(n_pred.values()) if e == "all" else n_pred[e]
        g = sum(n_gold.values()) if e == "all" else n_gold[e]
        prec, rec = t / max(p, 1), t / max(g, 1)
        res[e] = 2 * prec * rec / max(prec + rec, 1e-9)
    return res


def row(name, r):
    return (f"  {name:<12} loss {r['loss']:.4f} | F1 want {r['want']:.3f}  "
            f"unwant {r['unwant']:.3f}  tổng {r['all']:.3f}")


def diagnose(tr, va, un, history):
    print("\nĐánh giá overfit (epoch tốt nhất):")
    gap = tr["all"] - va["all"]
    print(f"  train F1 - val F1 = {gap:+.3f}  ->",
          "ổn, chưa học thuộc câu train" if gap < 0.02 else "có dấu hiệu học thuộc câu train")
    if un:
        gap2 = va["all"] - un["all"]
        print(f"  val F1 - val_unseen F1 = {gap2:+.3f}  ->",
              "nhận được cả món chưa thấy" if gap2 < 0.05 else
              "đang dựa nhiều vào việc THUỘC TÊN MÓN; cần thêm dữ liệu đa dạng tên món / câu thật")
    vl = [h["val_loss"] for h in history]
    if len(vl) >= 2 and vl[-1] > min(vl) * 1.2:
        print(f"  val loss tăng từ {min(vl):.4f} lên {vl[-1]:.4f} ở các epoch cuối -> overfit; "
              "đã giữ epoch tốt nhất, lần sau có thể giảm --epochs")
    print("  Lưu ý: val/val_unseen vẫn là câu do AI sinh cùng kiểu với train. Muốn biết chạy thật "
          "ra sao, tự viết ~50 câu như khách nói rồi chạy lại với --test.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+", help="1 hoặc nhiều file dữ liệu, gộp lại khi train")
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--out", default=os.path.join(os.getenv("MODEL_DIR", "models"), "ner"))
    ap.add_argument("--epochs", type=int, default=4, help="tối đa; dừng sớm nếu không cải thiện")
    ap.add_argument("--patience", type=int, default=1, help="số epoch không cải thiện thì dừng")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--val", type=float, default=0.05, help="tỉ lệ câu làm validation")
    ap.add_argument("--unseen", type=float, default=0.1,
                    help="tỉ lệ TÊN MÓN giữ riêng để đo khả năng nhận món chưa từng thấy. "
                         "Bản train CUỐI dùng --unseen 0 để model học đủ mọi tên món")
    ap.add_argument("--aug-lower", type=float, default=0.15, help="xác suất viết thường cả câu")
    ap.add_argument("--aug-noaccent", type=float, default=0.15, help="xác suất bỏ dấu cả câu")
    ap.add_argument("--test", default=None, help="file JSONL có nhãn để đánh giá")
    ap.add_argument("--test-marked", default=None,
                    help="file TXT: mỗi dòng chứa nhãn [W món muốn] / [U món tránh]")
    ap.add_argument("--max-train", type=int, default=0, help="chỉ dùng N câu train (chạy thử)")
    ap.add_argument("--keep-conflicts", action="store_true", help="giữ câu 1 món vừa want vừa unwant")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.test and args.test_marked:
        ap.error("Dùng một trong --test hoặc --test-marked, không dùng đồng thời")
    if args.test_marked:
        test_path = os.path.splitext(args.test_marked)[0] + ".jsonl"
        args.test = generate_marked_test(args.test_marked, test_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data, seen = [], set()
    for path in args.data:                         # gộp nhiều file, bỏ câu trùng giữa các file
        for tokens, tags in load_data(path, drop_conflicts=not args.keep_conflicts):
            key = " ".join(tokens).lower()
            if key not in seen:
                seen.add(key)
                data.append((tokens, tags))
    print(f"Tổng cộng {len(data)} câu")
    train, val, unseen = split(data, args.val, args.unseen, args.seed)
    if args.max_train:
        train = train[:args.max_train]
        val, unseen = val[:max(200, args.max_train // 5)], unseen[:max(200, args.max_train // 5)]
    train_probe = rng.sample(train, min(len(train), max(len(val), 500)))   # đo F1 trên train
    test = load_data(args.test, verbose=True) if args.test else []
    print(f"train {len(train)} | val {len(val)} | val_unseen {len(unseen)} | test {len(test)} câu")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("thiết bị:", device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=LABELS, label2id=LABEL2ID).to(device)

    steps = args.epochs * ((len(train) + args.bs - 1) // args.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device == "cuda"                       # GPU: tính bằng float16 cho nhanh
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    select_on = "val_unseen" if unseen else "val"
    print(f"chọn epoch tốt nhất theo F1 trên {select_on} (bằng nhau thì loss thấp hơn)\n")
    best, best_ep, bad, step, history = (-1.0, 0.0), 0, 0, 0, []
    best_res = None
    model.train()
    for ep in range(1, args.epochs + 1):
        order = list(range(len(train)))
        random.shuffle(order)
        total = 0.0
        for s in range(0, len(order), args.bs):
            batch = [(augment(train[i][0], rng, args.aug_lower, args.aug_noaccent), train[i][1])
                     for i in order[s:s + args.bs]]
            enc = encode(tokenizer, batch).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(**enc).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            total += loss.item() * len(batch)
            step += 1
            if step % 200 == 0:
                print(f"  bước {step}/{steps}  loss {loss.item():.4f}")

        tr = evaluate(model, tokenizer, train_probe, device)
        va = evaluate(model, tokenizer, val, device)
        un = evaluate(model, tokenizer, unseen, device) if unseen else None
        print(f"epoch {ep}  (loss lúc train, có tăng cường: {total / len(train):.4f})")
        print(row("train", tr)); print(row("val", va))
        if un:
            print(row("val_unseen", un))
        history.append({"val_loss": va["loss"]})
        score = (round((un or va)["all"], 4), -(un or va)["loss"])
        if score > best:
            best, best_ep, bad, best_res = score, ep, 0, (tr, va, un)
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            print(f"  -> tốt nhất đến giờ, đã lưu vào {args.out}/")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  -> {select_on} không cải thiện {bad} epoch, dừng sớm")
                break

    print(f"\nModel lưu ở {args.out}/ là epoch {best_ep} ({select_on} F1 {best[0]:.3f})")
    diagnose(*best_res, history)

    if test:
        model = AutoModelForTokenClassification.from_pretrained(args.out).to(device)
        print(row("test (thật)", evaluate(model, tokenizer, test, device)))

    try:
        from context import context_processing as CP
        cp = CP(model_path=args.out)
    except Exception:                                # noqa: BLE001
        try:
            from context import ContextProcessor
            cp = ContextProcessor(ner_model_path=args.out, use_bert=False)
        except Exception as e:                       # noqa: BLE001
            print("  (bỏ qua bước thử context.py:", e, ")")
            return
    print("\nThử với câu ngoài dữ liệu:")
    for text in ("cho mình 2 miếng gà rán với pepsi, không lấy khoai tây nhé",
                 "tầm 100k, đừng cho tương ớt, lấy burger zinger",
                 "pepsi thì thôi khỏi, cho anh combo nhóm",
                 "cho minh combo ga ran, khong uong 7up, nhung uong pepsi",
                 "không uống 7up nhưng uống pepsi"):
        r = cp.process(text)
        print(f"  {text!r}\n     want={r['want_text']}  unwant={r['unwant_text']}  budget={r['budget']}")


if __name__ == "__main__":
    main()
