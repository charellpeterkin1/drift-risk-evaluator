import pandas as pd
from ollama import chat

df = pd.read_csv("prompts.csv")

results = []

for index, row in df.iterrows():
    prompt = row["prompt_summary"]

    print(f"Sending prompt {index + 1}: {prompt}")

    response = chat(
        model="llama3.2:3b",
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
responses_df.to_csv("real_model_responses.csv", index=False)

print("\nDone. Saved real_model_responses.csv")
