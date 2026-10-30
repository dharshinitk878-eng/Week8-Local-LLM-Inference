
import ollama
import csv
from transformers import pipeline

# Read the article
with open("dataset/article.txt", "r", encoding="utf-8") as file:
    article = file.read()

print("Article word count:", len(article.split()))

if not 500 <= len(article.split()) <= 800:
    raise ValueError("The article must contain 500 to 800 words.")

# Load summarization model
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

# Compare different max_length values
summary_1 = summarizer(
    article,
    max_length=60,
    min_length=20,
    do_sample=False,
    truncation=True
)[0]["summary_text"]

summary_2 = summarizer(
    article,
    max_length=100,
    min_length=30,
    do_sample=False,
    truncation=True
)[0]["summary_text"]

summary_3 = summarizer(
    article,
    max_length=150,
    min_length=40,
    do_sample=False,
    truncation=True
)[0]["summary_text"]

# Ask Ollama for sentiment
sentiment = ollama.chat(
    model="gemma3:1b",
    messages=[{
        "role": "user",
        "content": (
            "Analyse the sentiment of this article. "
            "Classify it as Positive, Negative or Neutral. "
            "Explain the reason briefly.\n\n" + article
        )
    }]
)["message"]["content"]

# Translate the article into two languages
translation_tamil = ollama.chat(
    model="gemma3:1b",
    messages=[{
        "role": "user",
        "content": "Translate this article into Tamil. Preserve its meaning:\n\n" + article
    }]
)["message"]["content"]

translation_hindi = ollama.chat(
    model="gemma3:1b",
    messages=[{
        "role": "user",
        "content": "Translate this article into Hindi. Preserve its meaning:\n\n" + article
    }]
)["message"]["content"]

# Create five questions from the article
questions = [
    "What is the main topic of the article?",
    "What important issue is discussed?",
    "What facts or examples are mentioned?",
    "What are the main benefits or effects described?",
    "What is the overall conclusion of the article?"
]

answers = []

for question in questions:
    response = ollama.chat(
        model="gemma3:1b",
        messages=[{
            "role": "user",
            "content": (
                "Answer using only the information in this article. "
                "If the answer is not available, say so.\n\n"
                + article + "\n\nQuestion: " + question
            )
        }]
    )

    answers.append(response["message"]["content"])

# Save all results
with open(
    "dataset/article_analysis.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(["Analysis_Type", "Output"])

    writer.writerow(["Summary max_length 60", summary_1])
    writer.writerow(["Summary max_length 100", summary_2])
    writer.writerow(["Summary max_length 150", summary_3])
    writer.writerow(["Sentiment", sentiment])
    writer.writerow(["Tamil Translation", translation_tamil])
    writer.writerow(["Hindi Translation", translation_hindi])

    for question, answer in zip(questions, answers):
        writer.writerow(["Question: " + question, answer])

print("\n--- SUMMARY 1 ---")
print(summary_1)

print("\n--- SUMMARY 2 ---")
print(summary_2)

print("\n--- SUMMARY 3 ---")
print(summary_3)

print("\n--- SENTIMENT ---")
print(sentiment)

print("\n--- TAMIL TRANSLATION ---")
print(translation_tamil)

print("\n--- HINDI TRANSLATION ---")
print(translation_hindi)

print("\n--- QUESTIONS AND ANSWERS ---")

for question, answer in zip(questions, answers):
    print("\nQuestion:", question)
    print("Answer:", answer)

print("\nAll article analysis results saved successfully!")
