import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("behavior_drift_summary.csv")

# Chart 1
plt.figure()
plt.bar(
    df["conversation_id"],
    df["behavior_drift"]
)
plt.xlabel("Conversation ID")
plt.ylabel("Behavior Drift Score")
plt.title("Behavior Drift Across Conversations")
plt.savefig("behavior_drift_chart.png")

# Chart 2
plt.figure()
plt.bar(
    df["conversation_id"],
    df["alignment_stability_index"]
)
plt.xlabel("Conversation ID")
plt.ylabel("Alignment Stability Index")
plt.title("Alignment Stability Across Conversations")
plt.savefig("asi_chart.png")

print("Charts created successfully!")
