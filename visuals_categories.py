import pandas as pd
import matplotlib.pyplot as plt

# Load classified responses
df = pd.read_csv("classified_llama_responses.csv")

# Create summary table
summary = (
    df.groupby(["category", "safety_classification"])
      .size()
      .unstack(fill_value=0)
)

print("\nCategory Summary:\n")
print(summary)

# Create chart
summary.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Llama Safety Classifications by Adversarial Category")
plt.xlabel("Adversarial Category")
plt.ylabel("Number of Responses")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save chart
plt.savefig("category_safety_chart.png")

print("\nSaved category_safety_chart.png")
