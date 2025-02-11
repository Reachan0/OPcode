import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyomo.environ import *
from Model import model, solver  

if not os.path.exists("results_alpha"):
    os.makedirs("results_alpha")

# Error tolerance (to prevent floating point errors)
tolerance = 1e-6  

# Define different alpha_1 and alpha_2 values to test
alpha_1_values = [0.25, 0.3, 0.35]
alpha_2_values = [25, 27, 29]

results = []

for alpha_1 in alpha_1_values:
    for alpha_2 in alpha_2_values:
        print(f"\n🚀 Running model with α1 = {alpha_1}, α2 = {alpha_2}...")

        # Reset model parameters
        model_instance = model.clone()
        model_instance.alpha_1.set_value(alpha_1)
        model_instance.alpha_2.set_value(alpha_2)

        # Run the solver
        start_time = time.time()
        results_obj = solver.solve(model, tee=False, options={
            'Seed': 42, 'Presolve': 2, 'MIPGap': 0.03, 'TimeLimit': 600
        })
        end_time = time.time()
        solve_time = end_time - start_time
        termination_condition = results_obj.solver.termination_condition
        obj_value = value(model.obj)

        emission_over_time = [value(model.Q[t]) for t in model.T]
        print(f"📊 Emission over time for α1={alpha_1}, α2={alpha_2}: {emission_over_time}")

        #  Record and store data
        results.append({
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "termination": str(termination_condition),
            "solve_time": solve_time,
            "objective_value": obj_value,
            "emission_over_time": emission_over_time
        })

# Convert all results into a DataFrame
df_results = pd.DataFrame(results)
df_results = df_results.explode("emission_over_time")
df_results["Year"] = df_results.groupby(["alpha_1", "alpha_2"]).cumcount()
print(df_results.groupby(["alpha_1", "alpha_2"])["emission_over_time"].nunique())


# Plot a histogram of solution time versus α1 and α2
plt.figure(figsize=(10, 5))
sns.barplot(data=df_results, x="alpha_1", y="solve_time", hue="alpha_2")
plt.xlabel("Alpha 1")
plt.ylabel("Solve Time (s)")
plt.title("Solve Time vs. Alpha_1, Alpha_2")
plt.legend(title="Alpha 2")
plt.savefig("results_alpha/solve_time_vs_alpha.png")
plt.show()

# Plot a histogram of the objective function value versus α1 and α2
plt.figure(figsize=(10, 5))
sns.barplot(data=df_results, x="alpha_1", y="objective_value", hue="alpha_2")
plt.xlabel("Alpha 1")
plt.ylabel("Objective Value (Total Cost)")
plt.title("Objective Value vs. Alpha_1, Alpha_2")
plt.legend(title="Alpha 2")
plt.savefig("results_alpha/objective_value_vs_alpha.png")
plt.show()

# 不同 α 组合下的 CO₂ 排放变化趋势（折线图）
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_results, x="Year", y="emission_over_time", hue="alpha_1", style="alpha_2")
plt.xlabel("Year")
plt.ylabel("CO₂ Emission")
plt.title("CO₂ Emission Reduction Over Time")
plt.legend(title="Alpha 1, Alpha 2")
plt.grid(True)
plt.savefig("results_alpha/co2_emission_trend.png")
plt.show()

# 输出检查
for alpha_1, alpha_2 in zip(df_results["alpha_1"].unique(), df_results["alpha_2"].unique()):
    print(f"\n🔹 Alpha_1: {alpha_1}, Alpha_2: {alpha_2}")
    for t in range(26):
        print(f"Year {t}: Q[t] = {df_results[(df_results['alpha_1'] == alpha_1) & (df_results['alpha_2'] == alpha_2) & (df_results['Year'] == t)]['emission_over_time'].values[0]}")

