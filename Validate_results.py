import pandas as pd
import matplotlib.pyplot as plt

from Model import model  

# # Error tolerance (to prevent false positives caused by floating point errors) 不添加这个容错，会有浮点误差
tolerance = 1e-6  

# 1. Exporting Transportation Flow Data
f_values = []
for (i, j, l, t) in model.A * model.L * model.T:
    f_values.append([i, j, l, t, model.f[i, j, l, t].value])

df_transport = pd.DataFrame(f_values, columns=["From", "To", "TransportMode", "Year", "Flow"])
df_transport.to_csv("results/transportation_flows_0329.csv", index=False)
print(" Traffic flow data has been exported to `results/transportation_flows_0329.csv`")

#  2. Check whether CO₂ emission targets are being achieved
emission_check = []
for t in model.T:
    actual_emission = model.Q[t].value
    expected_emission = model.Q_init - t / 25 * (model.Q_init - model.Q_T)
    emission_check.append([t, actual_emission, expected_emission])

df_emissions = pd.DataFrame(emission_check, columns=["Year", "Actual Emissions", "Target Emissions"])
df_emissions.to_csv("results/emission_validation_0329.csv", index=False)
print(" Emission target verification data has been exported to `results/emission_validation_0325.csv`")

#  3. Visualizing CO₂ transport trends
df_grouped = df_transport.groupby("Year")["Flow"].sum()
plt.figure(figsize=(8, 5))
plt.plot(df_grouped.index, df_grouped.values, marker="o", linestyle="-", color="b", label="Total Transported CO₂")
plt.xlabel("Year")
plt.ylabel("CO₂ Transported (tons)")
plt.title("CO₂ Transportation Over Time")
plt.grid(True)
plt.legend()
plt.savefig("results/CO2_transport_trend_0329.png")
plt.show()
print(" Transport trend graph has generated!!! `results/CO2_transport_trend.png`")

# 4. Check if pipeline capacity meets constraints
pipeline_check = []
for (i, j) in model.A:
    pipeline_capacity = model.S[i, j].value
    pipeline_check.append([i, j, pipeline_capacity])

df_pipeline = pd.DataFrame(pipeline_check, columns=["From", "To", "Pipeline Capacity"])
df_pipeline.to_csv("results/pipeline_capacity_0329.csv", index=False)
print(" Pipeline capacity data has been exported to `results/pipeline_capacity.csv`")

# 5. Final inspection: if there are any violations of constraints
violations = []
for (i, j, t) in model.A * model.T:
    allowed_capacity = model.S[i, j].value * model.s[i, j, t].value + 0.05 * model.S[i, j].value  # 允许 5% 超载
    actual_flow = model.f[i, j, "pipeline", t].value

    # Only record violations when the actual traffic is higher than the allowed traffic by tolerance
    if actual_flow > allowed_capacity + tolerance:  
        violations.append([i, j, t, actual_flow, allowed_capacity])

df_violations = pd.DataFrame(violations, columns=["From", "To", "Year", "Transported", "Allowed (5% Relaxed)"])

if not df_violations.empty:
    df_violations.to_csv("results/constraint_violations.csv", index=False)
    print(" Violations of pipeline transport constraints were found (5% capacity slack was considered), see\
           `results/constraint_violations.csv`")
else:
    print(" No violations of pipeline transportation constraints were found and all constraints were satisfied \
          (5% capacity slack was considered).")

print(" Verify your results! Please check the CSV files in the `results/` directory for further analysis.")
