import pulp
import pandas as pd

# ============================
# 1. Data (replace with actual Excel)
# ============================

periods = list(range(1, 13))
items = [1, 2]
resources = [1, 2]
T = len(periods)

demand = {
    1: [400, 400, 450, 500, 550, 700, 900, 1000, 900, 600, 450, 400],
    2: [800, 600, 500, 500, 400, 300, 200, 200, 400, 600, 900, 1000],
}

prod_cost = {1: 5, 2: 6}
labor_req = {1: 0.4, 2: 0.3}
labor_cost = 2
inv_cost = {1: 1, 2: 2}
setup_cost = {1: 1000, 2: 500}
price = {1: 8, 2: 10}

labor_cost_per_unit = {i: labor_req[i] * labor_cost for i in items}
total_unit_cost = {i: prod_cost[i] + labor_cost_per_unit[i] for i in items}

resource_usage = {
    1: {1: 5, 2: 4},
    2: {1: 3, 2: 2},
}

capacity = {
    1: {t: 15000 for t in periods},
    2: {t: 10000 for t in periods},
}

initial_inventory = {1: 0, 2: 0}

# ============================
# 2. MIP Model
# ============================

model = pulp.LpProblem("MultiItem_LotSizing_MIP", pulp.LpMinimize)

Q = pulp.LpVariable.dicts("Q", (items, periods), lowBound=0)
I = pulp.LpVariable.dicts("I", (items, periods), lowBound=0)
Y = pulp.LpVariable.dicts("Y", (items, periods), lowBound=0, upBound=1, cat="Binary")

M = 10_000_000

# Inventory balance
for i in items:
    for idx, t in enumerate(periods):
        if idx == 0:
            model += I[i][t] == initial_inventory[i] + Q[i][t] - demand[i][idx]
        else:
            model += I[i][t] == I[i][periods[idx - 1]] + Q[i][t] - demand[i][idx]

# Capacity
for r in resources:
    for t in periods:
        model += sum(resource_usage[r][i] * Q[i][t] for i in items) <= capacity[r][t]

# Setup logic
for i in items:
    for t in periods:
        model += Q[i][t] <= M * Y[i][t]

# Objective
model += sum(
    setup_cost[i] * Y[i][t] + total_unit_cost[i] * Q[i][t] + inv_cost[i] * I[i][t]
    for i in items
    for t in periods
)

# Solve
model.solve(pulp.PULP_CBC_CMD(msg=0))

# ============================
# 3. Build Full Result Table
# ============================

rows = []

for idx, t in enumerate(periods):
    row = {"Period": t}

    for i in items:
        q = Q[i][t].value()
        setup = int(Y[i][t].value())
        inv = I[i][t].value()
        workforce = q * labor_req[i]

        prod_c = prod_cost[i] * q
        work_c = workforce * labor_cost
        inv_c = inv_cost[i] * max(inv, 0)
        setup_c = setup_cost[i] * setup

        total_c = prod_c + work_c + inv_c + setup_c
        revenue = demand[i][idx] * price[i]
        profit = revenue - total_c

        row[f"Q_item{i}"] = round(q, 2)
        row[f"Setup_item{i}"] = setup
        row[f"Inv_item{i}"] = round(inv, 2)
        row[f"Workforce_item{i}"] = round(workforce, 2)

        row[f"ProdCost_item{i}"] = round(prod_c, 2)
        row[f"WorkCost_item{i}"] = round(work_c, 2)
        row[f"InvCost_item{i}"] = round(inv_c, 2)
        row[f"SetupCost_item{i}"] = round(setup_c, 2)

        row[f"TotalCost_item{i}"] = round(total_c, 2)
        row[f"Revenue_item{i}"] = round(revenue, 2)
        row[f"Profit_item{i}"] = round(profit, 2)

    rows.append(row)

df = pd.DataFrame(rows)
print(df)

# Overall totals
total_cost = sum(df[f"TotalCost_item{i}"].sum() for i in items)
total_revenue = sum(df[f"Revenue_item{i}"].sum() for i in items)
total_profit = total_revenue - total_cost

print("\nTotal Cost =", round(total_cost, 2))
print("Total Revenue =", round(total_revenue, 2))
print("Total Profit =", round(total_profit, 2))
