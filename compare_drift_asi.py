import pandas as pd
import matplotlib.pyplot as plt

# Load the summaries

llama = pd.read_csv("category_drift_summary.csv")
gemma = pd.read_csv("gemma_category_drift_summary.csv")

# Merge on category

comparison = pd.merge(
llama,
gemma,
on="category",
suffixes=("_llama", "_gemma")
)

# Print table

print("\nCross-Model Comparison:\n")
print(
comparison[
[
"category",
"behavior_drift_llama",
"behavior_drift_gemma",
"alignment_stability_index_llama",
"alignment_stability_index_gemma"
]
]
)

# ---------------------------

# Drift Comparison Chart

# ---------------------------

drift_df = comparison.set_index("category")[
[
"behavior_drift_llama",
"behavior_drift_gemma"
]
]

drift_df.columns = ["Llama", "Gemma"]

drift_df.plot(
kind="bar",
figsize=(12, 6)
)

plt.title("Behavior Drift Comparison by Category")
plt.xlabel("Category")
plt.ylabel("Behavior Drift")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("drift_comparison_chart.png")
plt.close()

# ---------------------------

# ASI Comparison Chart

# ---------------------------

asi_df = comparison.set_index("category")[
[
"alignment_stability_index_llama",
"alignment_stability_index_gemma"
]
]

asi_df.columns = ["Llama", "Gemma"]

asi_df.plot(
kind="bar",
figsize=(12, 6)
)

plt.title("Alignment Stability Index Comparison by Category")
plt.xlabel("Category")
plt.ylabel("ASI")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("asi_comparison_chart.png")
plt.close()

print("\nSaved drift_comparison_chart.png")
print("Saved asi_comparison_chart.png")

