
import ollama
import csv
from datetime import datetime

model = "gemma3:1b"

prompts = [
    ("Education", "Explain Artificial Intelligence in simple words."),
    ("Healthcare", "How can technology help hospitals?"),
    ("Coding", "Write a Python program to add two numbers."),
    ("Travel", "Suggest three tourist places in Tamil Nadu."),
    ("Business", "How can small businesses use AI?"),
    ("Education", "Explain machine learning with an example."),
    ("Healthcare", "What is the importance of digital health records?"),
    ("Coding", "Explain the difference between a list and a tuple in Python."),
    ("Travel", "What should a person plan before travelling?"),
    ("Business", "Explain digital marketing in simple words.")
]

with open("dataset/llm_responses.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Model", "Domain", "Prompt", "Response"])

    for domain, prompt in prompts:
        print("\nQuestion:", prompt)

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"]

        print("Answer:", answer)

        writer.writerow([
            datetime.now(),
            model,
            domain,
            prompt,
            answer
        ])

        with open("dataset/conversation_history.txt", "a", encoding="utf-8") as history:
            history.write(
                f"\nTime: {datetime.now()}\n"
                f"Model: {model}\n"
                f"Question: {prompt}\n"
                f"Answer: {answer}\n"
            )

print("\nAll responses saved successfully!")
