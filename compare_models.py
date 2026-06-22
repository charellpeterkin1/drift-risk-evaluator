import pandas as pd
import matplotlib.pyplot as plt

llama = pd.read_csv("classified_llama_responses.csv")
gemma = pd.read_csv("classified_gemma_responses.csv")

llama_counts = llama["safety_classification"].value_counts()
gemma_counts = gemma["safety_classification"].value_counts()

comparison = pd.DataFrame({
    "Llama": llama_counts,
    "Gemma": gemma_counts
}).fillna(0)

print("\nModel Comparison:\n")
print(comparison)

comparison.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Llama vs Gemma Safety Classification Comparison")
plt.xlabel("Safety Classification")
plt.ylabel("Number of Responses")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("model_comparison_chart.png")

print("\nSaved model_comparison_chart.png")
