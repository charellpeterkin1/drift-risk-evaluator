import pandas as pd
import matplotlib.pyplot as plt

# Load classified responses
llama = pd.read_csv("classified_llama_responses.csv")
gemma = pd.read_csv("classified_gemma_responses.csv")
qwen = pd.read_csv("classified_qwen_responses.csv")

# Load drift summaries
llama_drift = pd.read_csv("category_drift_summary.csv")
gemma_drift = pd.read_csv("gemma_category_drift_summary.csv")
qwen_drift = pd.read_csv("qwen_category_drift_summary.csv")

# ---------------------------
# Chart 1: Safety classification counts
# ---------------------------
classification_comparison = pd.DataFrame({
    "Llama": llama["safety_classification"].value_counts(),
    "Gemma": gemma["safety_classification"].value_counts(),
    "Qwen": qwen["safety_classification"].value_counts()
}).fillna(0)

print("\nSafety Classification Comparison\n")
print(classification_comparison)

classification_comparison.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Safety Classifications Across Three Models")
plt.xlabel("Safety Classification")
plt.ylabel("Number of Responses")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("three_model_classification_chart.png")
plt.close()

# ---------------------------
# Chart 2: Behavior drift by category
# ---------------------------
drift_comparison = pd.DataFrame({
    "category": llama_drift["category"],
    "Llama": llama_drift["behavior_drift"],
    "Gemma": gemma_drift["behavior_drift"],
    "Qwen": qwen_drift["behavior_drift"]
})

print("\nBehavior Drift Comparison\n")
print(drift_comparison)

drift_comparison.set_index("category").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Behavior Drift by Category Across Three Models")
plt.xlabel("Adversarial Category")
plt.ylabel("Behavior Drift")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("three_model_drift_chart.png")
plt.close()

# ---------------------------
# Chart 3: Alignment Stability Index by category
# ---------------------------
asi_comparison = pd.DataFrame({
    "category": llama_drift["category"],
    "Llama": llama_drift["alignment_stability_index"],
    "Gemma": gemma_drift["alignment_stability_index"],
    "Qwen": qwen_drift["alignment_stability_index"]
})

print("\nASI Comparison\n")
print(asi_comparison)

asi_comparison.set_index("category").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Alignment Stability Index by Category Across Three Models")
plt.xlabel("Adversarial Category")
plt.ylabel("Alignment Stability Index")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("three_model_asi_chart.png")
plt.close()

print("\nSaved three_model_classification_chart.png")
print("Saved three_model_drift_chart.png")
print("Saved three_model_asi_chart.png")
