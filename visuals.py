import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("real_behavior_drift_summary.csv")

# Behavior Drift chart
plt.figure()
plt.bar(
    df["conversation_id"],
    df["behavior_drift"]
)
plt.xlabel("Conversation ID")
plt.ylabel("Behavior Drift")
plt.title("Behavior Drift Across Real Llama Conversations")
plt.savefig("real_behavior_drift_chart.png")

# ASI chart
plt.figure()
plt.bar(
    df["conversation_id"],
    df["alignment_stability_index"]
)
plt.xlabel("Conversation ID")
plt.ylabel("Alignment Stability Index")
plt.title("Alignment Stability Across Real Llama Conversations")
plt.savefig("real_asi_chart.png")

print("Charts created successfully!")
