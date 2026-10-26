
from collections import Counter
import pulp
from menu_model import MenuModel, _cart_lines, banned_from_text, meal_allowed  
SCORE_WEIGHTS = {"hot": 0.8, "best": 1.0, "seasonal": 0.5, "best_fit": 2.0, "popular": 0.5}
MIN_PMI = 0.0
def _solve(prob):
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.LpStatus[prob.status] == "Optimal"


def build_data(model, cart, budget=None, unwant_texts=(), fit_scores=None):
    meals = model.meals
    chosen = Counter()
    for j, n, _ in _cart_lines(cart):
        chosen[j] += n

    composition = {j: dict(model.min_units(j)) for j in meals}
    price_set = {j: meals[j]["price"] for j in meals}

    customer_choice_set = Counter()
    for j, n in chosen.items():
        for i, q in composition[j].items():
            customer_choice_set[i] += q * n

    banned = banned_from_text(model, unwant_texts)
    banned["meals"] -= set(chosen)
    feasible_set = {j for j in meals if meal_allowed(model, j, banned)} | set(chosen)

    upsell_set, score_set = build_upsell(model, feasible_set, chosen, fit_scores)
    return {
        "chosen": chosen, "composition": composition, "price_set": price_set,
        "customer_choice_set": customer_choice_set, "feasible_set": feasible_set,
        "upsell_set": upsell_set, "score_set": score_set, "budget": budget,
    }


def build_upsell(model, feasible_set, chosen, fit_scores=None, min_fit=None):
    meals = model.meals
    hot_set = {j for j in feasible_set if meals[j].get("is_hotsell")}
    best_set = {j for j in feasible_set if meals[j].get("is_bestsell")}
    seasonal_set = {j for j in feasible_set if meals[j].get("is_seasonal")}
    fit = {j: s for j, s in (fit_scores or {}).items() if j in feasible_set and j not in chosen}
    rejected = {j for j, s in fit.items() if min_fit is not None and s <= min_fit}
    if fit:                                               
        lo, hi = min(fit.values()), max(fit.values())
        fit = {j: (s - lo) / (hi - lo) if hi > lo else 1.0 for j, s in fit.items()}
    best_fit = {j for j, s in sorted(fit.items(), key=lambda x: -x[1])[:10]}
    max_sold = max((meals[j].get("num_sold", 0) for j in meals), default=0) or 1

    upsell_set = (hot_set | best_set | best_fit | seasonal_set) - set(chosen)
    upsell_set -= rejected                              
    score_set = {
        j: SCORE_WEIGHTS["hot"] * (j in hot_set)
        + SCORE_WEIGHTS["best"] * (j in best_set)
        + SCORE_WEIGHTS["seasonal"] * (j in seasonal_set)
        + SCORE_WEIGHTS["best_fit"] * fit.get(j, 0.0)
        + SCORE_WEIGHTS["popular"] * meals[j].get("num_sold", 0) / max_sold
        for j in upsell_set
    }
    return upsell_set, score_set


def optimize_ilp(feasible_set, price_set, composition, customer_choice_set):
    prob = pulp.LpProblem("optimize_cart", pulp.LpMinimize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, cat="Integer") for j in feasible_set}
    prob += pulp.lpSum(price_set[j] * x[j] for j in feasible_set)
    for i in customer_choice_set:
        prob += pulp.lpSum(composition[j].get(i, 0) * x[j]
                           for j in feasible_set) == customer_choice_set[i]
    if not _solve(prob):
        return None
    return {j: int(round(v.value())) for j, v in x.items() if v.value() and v.value() > 0.5}


def upsell1(feasible_set, price_set, composition, customer_choice_set,
            combo_set=None, chosen=None, base_cost=None, base_combo_units=0,
            max_extra_abs=25000, max_extra_ratio=0.3):
    allowed = set(feasible_set)
    if combo_set is not None and chosen is not None:
        allowed &= set(combo_set) | set(chosen)                                  
    prob = pulp.LpProblem("upsell_1", pulp.LpMinimize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, cat="Integer") for j in allowed}
    prob += pulp.lpSum(price_set[j] * x[j] for j in allowed)
    for i in customer_choice_set:
        prob += pulp.lpSum(composition[j].get(i, 0) * x[j]
                           for j in allowed) >= customer_choice_set[i]
    customer_num_of_meal = sum(customer_choice_set.values())
    prob += pulp.lpSum(sum(composition[j].values()) * x[j]
                       for j in allowed) >= customer_num_of_meal + 1
    if combo_set is not None and chosen is not None:
        for j in allowed - set(combo_set):
            prob += x[j] <= chosen[j]                                            
        prob += pulp.lpSum(sum(composition[j].values()) * x[j]
                           for j in allowed & set(combo_set)) >= base_combo_units + 1   
    if base_cost is not None:
        cap = base_cost + max(max_extra_abs, base_cost * max_extra_ratio)
        prob += pulp.lpSum(price_set[j] * x[j] for j in allowed) <= cap         
    if not _solve(prob):
        return None
    return {j: int(round(v.value())) for j, v in x.items() if v.value() and v.value() > 0.5}


def upsell2(upsell_set, price_set, score_set, left=None, k=2):
    if not upsell_set:
        return []
    prob = pulp.LpProblem("upsell_2", pulp.LpMaximize)
    x = {j: pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in upsell_set}
    prob += pulp.lpSum(score_set[j] * x[j] for j in upsell_set)
    if left is not None:                             
        prob += pulp.lpSum(price_set[j] * x[j] for j in upsell_set) <= left
    prob += pulp.lpSum(x[j] for j in upsell_set) <= k
    if not _solve(prob):
        return []
    return sorted((j for j, v in x.items() if v.value() and v.value() > 0.5),
                  key=lambda j: -score_set[j])

def _cost(cart, price_set):
    return sum(price_set[j] * n for j, n in cart.items())


def _items(cart, meals):
    return [{"meal_id": j, "name": meals[j]["name"], "qty": n, "options": {}}
            for j, n in sorted(cart.items(), key=lambda x: -meals[x[0]]["price"])]


def plan(model, cart, budget=None, unwant_texts=(), scores=None, k_addons=2, fit_fn=None):
    meals = model.meals
    d = build_data(model, cart, budget, unwant_texts, scores)
    original_cost = _cost(d["chosen"], d["price_set"])
    best = optimize_ilp(d["feasible_set"], d["price_set"], d["composition"], d["customer_choice_set"])
    if best is None:                                  
        best = dict(d["chosen"])
    best_cost = _cost(best, d["price_set"])
    combo_set = {j for j in meals if meals[j].get("is_combo")}
    base_combo_units = sum(sum(d["composition"][j].values()) * n
                           for j, n in best.items() if j in combo_set)
    up = upsell1(d["feasible_set"], d["price_set"], d["composition"], d["customer_choice_set"],
                 combo_set=combo_set, chosen=d["chosen"], base_cost=best_cost,
                 base_combo_units=base_combo_units)
    upgrade = {"status": "none"}
    if up is not None and up != best:
        gained = Counter()
        for j, n in up.items():
            for i, q in d["composition"][j].items():
                gained[i] += q * n
        gained.subtract(d["customer_choice_set"])
        upgrade = {"status": "ok", "items": _items(up, meals),
                   "cost": _cost(up, d["price_set"]),
                   "extra_cost": _cost(up, d["price_set"]) - best_cost,
                   "gained": {model.class_label[i]: n for i, n in gained.items() if n > 0}}
    fit_source = "scores" if scores else "none"
    if fit_fn is not None:
        fit_scores = fit_fn(list(best) + [j for j in d["chosen"] if j not in best])
        if fit_scores:
            d["upsell_set"], d["score_set"] = build_upsell(model, d["feasible_set"], d["chosen"],
                                                           fit_scores, min_fit=MIN_PMI)
            fit_source = "item2vec"

    left = budget - best_cost if budget else None
    ilp2_added = set(up or {}) - set(best)            # món ILP2 đã gợi ý -> ILP3 không lặp lại
    addon_ids = upsell2(d["upsell_set"] - set(best) - ilp2_added, d["price_set"],
                        d["score_set"], left, k_addons)
    addons = [{"meal_id": j, "name": meals[j]["name"], "price": meals[j]["price"],
               "score": round(d["score_set"][j], 3)} for j in addon_ids]

    return {
        "demand": {model.class_label[i]: n for i, n in d["customer_choice_set"].items()},
        "original_cost": original_cost,
        "optimized": {"status": "ok", "items": _items(best, meals), "cost": best_cost,
                      "over_budget": bool(budget) and best_cost > budget},
        "saving": original_cost - best_cost,
        "upgrade": upgrade,
        "addons": addons,
        "fit_source": fit_source,
    }


def _fmt(items):
    return " + ".join(f"{i['qty']}× {i['name']}" for i in items)


def describe(result):
    best = result["optimized"]
    lines = []
    if result["saving"] > 0:
        lines.append(f"Đổi sang: {_fmt(best['items'])} — {best['cost']:,.0f}đ, "
                     f"tiết kiệm {result['saving']:,.0f}đ.")
    else:
        lines.append(f"Giỏ hiện tại đã rẻ nhất: {best['cost']:,.0f}đ.")
    if best["over_budget"]:
        lines.append("Lưu ý: giỏ vượt ngân sách khách nói.")
    up = result["upgrade"]
    if up["status"] == "ok":
        gained = ", ".join(f"{n} {name}" for name, n in up["gained"].items())
        lines.append(f"Thêm {up['extra_cost']:,.0f}đ: {_fmt(up['items'])} (được thêm {gained}).")
    if result["addons"]:
        lines.append("Gợi ý thêm: " + ", ".join(f"{a['name']} ({a['price']:,.0f}đ)"
                                               for a in result["addons"]) + ".")
    return lines