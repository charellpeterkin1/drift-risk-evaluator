import pandas as pd

df = pd.read_csv("classified_llama_responses.csv")

summary = (
    df.groupby(["category", "safety_classification"])
      .size()
      .unstack(fill_value=0)
)

print("\nCategory-Level Safety Results\n")
print(summary)

summary.to_csv("category_analysis.csv")

print("\nSaved category_analysis.csv")
