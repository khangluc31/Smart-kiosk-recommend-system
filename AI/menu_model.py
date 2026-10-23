
import re
from collections import Counter, defaultdict
import pulp
from item_types import VALUED_TYPES, classify, normalize, tokens

VALUED = set(VALUED_TYPES)
EPS_MEAL = 1.0       
EPS_UNIT = 0.01
PACKAGING = {"lon", "chai", "ly", "cốc", "hộp", "phần"}
DRINK_SIZES = {"m", "l", "s", "vừa", "lớn", "nhỏ", "size"}   
EMPTY_BAN = {"items": set(), "meals": set(), "types": set()}


def _var(prob, name, cat="Integer", low=0, up=None):
    """PuLP >= 4 tạo biến qua prob.add_variable; PuLP 2.x/3.x dùng pulp.LpVariable."""
    if hasattr(prob, "add_variable"):
        return prob.add_variable(name, lowBound=low, upBound=up, cat=cat)
    return pulp.LpVariable(name, lowBound=low, upBound=up, cat=cat)


def _solver():
    available = pulp.listSolvers(onlyAvailable=True)
    if "PULP_CBC_CMD" in available:
        return pulp.PULP_CBC_CMD(msg=False)
    if "HiGHS" in available:
        return pulp.HiGHS(msg=False)
    raise RuntimeError("Không có solver cho PuLP. Cài: pip install 'pulp<4' (kèm sẵn CBC)")


def base_name(name, item_type):
    n = re.sub(r"^\d+\s+", "", normalize(name))
    drop = PACKAGING | (DRINK_SIZES if item_type == "nuoc" else set())
    toks = [t for t in re.findall(r"\w+", n) if t not in drop]
    return " ".join(toks) or n

class MenuModel:
    def __init__(self, meals, fixed, groups, category_names=None):
        self.meals = meals
        category_names = category_names or {}
        parent, self.label, self.key_type = {}, {}, {}
        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def unit(item_id, name, t, qty):
            e = base_name(name, t)
            find(e)
            self.key_type.setdefault(e, t)
            if e not in self.label or len(name) < len(self.label[e]):
                self.label[e] = name
            return {"item_id": item_id, "name": name, "type": t, "qty": qty, "exact": e}
        self.fixed, self.groups = {}, {}
        for j, meal in meals.items():
            gs = []
            for g in groups.get(j, []):
                opts = [dict(unit(o[0], o[1], o[2], o[3]), extra=o[4]) for o in g["options"]]
                types = Counter(o["type"] for o in opts)
                gs.append({**g, "options": opts,
                           "type": types.most_common(1)[0][0] if types else "khac"})
                for t in types:                
                    same = [o["exact"] for o in opts if o["type"] == t]
                    for e in same[1:]:
                        parent[find(e)] = find(same[0])
            required = {g["type"] for g in gs if g["min"] > 0}
            fx = [unit(*c) for c in fixed.get(j, []) if c[2] not in required]
            has_unit = any(u["type"] in VALUED for u in fx) or \
                any(g["type"] in VALUED and g["min"] > 0 for g in gs)
            if not has_unit:    
                t = classify(meal["name"], category_names.get(meal["cat_id"]))
                fx.append(unit(None, meal["name"], t, 1))
            self.fixed[j], self.groups[j] = fx, gs

        self.cls = {e: find(e) for e in parent}
        members = defaultdict(list)
        for e, c in self.cls.items():
            members[c].append(self.label[e])
        self.class_label = {c: " / ".join(sorted(set(names))) for c, names in members.items()}
        self.unit_value = self._unit_values()

    @classmethod
    def from_db(cls, db, meals=None):
        fixed, groups = db.get_menu_structure()
        return cls(meals or db.get_meals(), fixed, groups, db.get_category_names())

    def group_class(self, g):
        for o in g["options"]:
            if o["type"] == g["type"]:
                return self.cls[o["exact"]]
        return None

    def min_units(self, j):
        u = Counter()
        for x in self.fixed[j]:
            if x["type"] in VALUED:
                u[self.cls[x["exact"]]] += x["qty"]
        for g in self.groups[j]:
            if g["type"] in VALUED and g["min"] > 0:
                u[self.group_class(g)] += g["min"]
        return u

    def main_type(self, j):
        u = self.min_units(j)
        return self.key_type[u.most_common(1)[0][0]] if u else "khac"

    def _unit_values(self):
        best = {}
        for j, meal in self.meals.items():
            u = self.min_units(j)
            if len(u) == 1 and meal["available"]:
                (c, n), = u.items()
                key = (n, meal["price"])
                if c not in best or key < best[c][0]:
                    best[c] = (key, meal["price"] / n)
        values = {c: v for c, (_, v) in best.items()}
        by_type = defaultdict(list)
        for c, v in values.items():
            by_type[self.key_type[c]].append(v)
        for j, meal in self.meals.items():       
            u = self.min_units(j)
            for c in u:
                if c not in values:
                    same = by_type.get(self.key_type[c])
                    values[c] = sum(same) / len(same) if same else meal["price"] / sum(u.values())
        return values


def _cart_lines(cart):
    for line in cart:
        if isinstance(line, dict):
            yield line["meal_id"], line.get("qty", 1), line.get("choices") or {}
        else:
            yield line[0], line[1], {}


def demand_from_cart(model, cart):
    d_class, d_exact = Counter(), Counter()
    for j, n, choices in _cart_lines(cart):
        for u in model.fixed[j]:
            if u["type"] in VALUED:
                d_class[model.cls[u["exact"]]] += u["qty"] * n
                d_exact[u["exact"]] += u["qty"] * n
        for g in model.groups[j]:
            chosen = choices.get(g["id"])
            if chosen:
                for item_id in chosen:
                    o = next(o for o in g["options"] if o["item_id"] == item_id)
                    if o["type"] in VALUED:
                        d_class[model.cls[o["exact"]]] += o["qty"] * n
                        d_exact[o["exact"]] += o["qty"] * n
            elif g["min"] > 0 and g["type"] in VALUED:
                d_class[model.group_class(g)] += g["min"] * n
    return d_class, d_exact


def cart_cost(model, cart):
    total = 0.0
    for j, n, choices in _cart_lines(cart):
        total += model.meals[j]["price"] * n
        for g in model.groups[j]:
            for item_id in choices.get(g["id"], []):
                total += next(o["extra"] for o in g["options"] if o["item_id"] == item_id) * n
    return total


def banned_from_text(model, unwant_texts):
    banned = {"items": set(), "meals": set(), "types": set()}
    names = {}
    for j in model.meals:
        for u in model.fixed[j]:
            if u["item_id"] is not None:
                names[u["item_id"]] = u["name"]
        for g in model.groups[j]:
            for o in g["options"]:
                names[o["item_id"]] = o["name"]
    for text in unwant_texts or []:
        tk = tokens(text)
        if not tk:
            continue
        hit = False
        for j, meal in model.meals.items():
            if tk <= tokens(meal["name"]):
                banned["meals"].add(j)
                hit = True
        for item_id, name in names.items():
            if tk <= tokens(name):
                banned["items"].add(item_id)
                hit = True
        if not hit and classify(text) != "khac":
            banned["types"].add(classify(text))
    return banned


def _exclude_cart_from_ban(model, banned, cart, d_class, d_exact):
    meals, items = set(), set()
    for j, _, choices in _cart_lines(cart):
        meals.add(j)
        items.update(u["item_id"] for u in model.fixed[j])
        for ids in choices.values():
            items.update(ids)
    for j in model.meals:              
        for u in model.fixed[j] + [o for g in model.groups[j] for o in g["options"]]:
            if u["exact"] in d_exact:
                items.add(u["item_id"])
    types = {model.key_type[c] for c in d_class}
    return {"meals": banned["meals"] - meals, "items": banned["items"] - items,
            "types": banned["types"] - types}

def _banned_unit(banned, u):
    return (u["item_id"] is not None and u["item_id"] in banned["items"]) or u["type"] in banned["types"]

def meal_allowed(model, j, banned):
    if not model.meals[j]["available"] or j in banned["meals"]:
        return False
    return not any(_banned_unit(banned, u) for u in model.fixed[j])


_meal_allowed = meal_allowed