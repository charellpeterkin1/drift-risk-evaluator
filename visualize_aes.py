import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("aes_model_summary.csv")

df.plot(
    x="model",
    y="adversarial_escalation_sensitivity",
    kind="bar",
    legend=False,
    figsize=(8, 5)
)

plt.title("Adversarial Escalation Sensitivity by Model")
plt.xlabel("Model")
plt.ylabel("AES Score")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("aes_chart.png")

print("Saved aes_chart.png")
