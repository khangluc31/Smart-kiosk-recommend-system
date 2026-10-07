import pulp
import context as ct
import re

customer_choice_set={}
universe_set={}
price_set={}
composition={}
hot_set={}
best_set={}
seasonal_set={}
score_set={}
best_fit={}

wanted_set=set(ct.wanted)
unwanted_set=set(ct.unwanted)
max_price_text=wanted_set.pop()
people=wanted_set.pop()

feasible_set= (universe_set | wanted_set) - unwanted_set
upsell_set=(hot_set | best_set | best_fit | seasonal_set )

pay=None
max_price=None
left=max_price-pay
def optimize_ilp():
    prob = pulp.LpProblem("optimize cart",pulp.LpMinize)
    x= {
        j : pulp.LpVariable(
            f"x_{j}",
            lowBound=0,
            cat="Integer"
        )
        for j in feasible_set
    }
    prob += pulp.lpSum(
        price_set[j]*x[j]
        for j in feasible_set
    )
    for i in customer_choice_set:
        prob+= pulp.lpSum(
            composition[j].get(i,0)*x[j]
            for j in feasible_set
        ) == customer_choice_set[i]
    prob.solve()
def upsell1():
    prob=pulp.LpProblem("upsell_1", pulp.LpMinimize)
    x={
        j : pulp.LpVariable(
            f"x_{j}",
            lowBound=0,
            cat="Integer"
        )
        for j in feasible_set
    }
    prob += pulp.lpSum(
        price_set[j]*x[j]
        for j in feasible_set
    )
    for i in customer_choice_set:
        prob+= pulp.lpSum(
            composition[j].get(i,0)*x[j]
            for j in feasible_set
        ) >= customer_choice_set[i]
    customer_num_of_meal=sum(customer_choice_set.values())
    prob+=pulp.lpsum(
        sum(composition[j].values()) * x[j]
        for j in feasible_set
    ) >= customer_num_of_meal +1
    prob.solve()
    

def upsell2():
    prob =pulp.LpProblem("upsell_2",pulp.LpMaximize)
    x = {
        j : pulp.LpVariable(
            f"x_{j}",
            lowBound=0,
            cat=pulp.LpBinary
        )
        for j in upsell_set
    }
    prob += pulp.lpSum(
        score_set[j]*x[j]
        for j in upsell_set
    )
    
    prob += pulp.lpSum(
        price_set[j]*x[j]
        for j in upsell_set
    ) <= left

    prob += pulp.lpSum(
        x[j]
        for j in upsell_set
    ) <= 2
    prob.solve()