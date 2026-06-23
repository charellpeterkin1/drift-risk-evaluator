import pandas as pd
from ollama import chat

df = pd.read_csv("prompts.csv")

results = []

for index, row in df.iterrows():
    prompt = row["prompt_summary"]

    print(f"[{index + 1}/{len(df)}] Sending: {prompt}")

    response = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    model_response = response["message"]["content"]

    results.append({
        "conversation_id": row["conversation_id"],
        "turn": row["turn"],
        "category": row["category"],
        "prompt_summary": prompt,
        "model_response": model_response
    })

responses_df = pd.DataFrame(results)

responses_df.to_csv(
    "qwen_responses.csv",
    index=False
)

print("\nSaved qwen_responses.csv")
