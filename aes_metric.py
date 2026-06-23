import pandas as pd

llama = pd.read_csv("category_drift_summary.csv")
gemma = pd.read_csv("gemma_category_drift_summary.csv")
qwen = pd.read_csv("qwen_category_drift_summary.csv")

results = []

for model_name, df in [
    ("Llama", llama),
    ("Gemma", gemma),
    ("Qwen", qwen)
]:
    aes = df["behavior_drift"].abs().mean()
    mean_asi = df["alignment_stability_index"].mean()

    results.append({
        "model": model_name,
        "adversarial_escalation_sensitivity": round(aes, 3),
        "mean_alignment_stability_index": round(mean_asi, 3)
    })

summary = pd.DataFrame(results)

print("\nAdversarial Escalation Sensitivity Results\n")
print(summary)

summary.to_csv("aes_model_summary.csv", index=False)

print("\nSaved aes_model_summary.csv")
