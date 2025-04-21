from sentence_transformers import SentenceTransformer
import faiss
import pickle
import pandas as pd
import requests

from proxy.query_enhancer import enhance_query_basic_english
# Load model + index + data
model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("embeddings/question_index.faiss")

with open("embeddings/questions.pkl", "rb") as f:
    q_data = pickle.load(f)

context_df = pd.read_csv("data/questions.csv")
context_map = {
    row["id"]: row["context"]
    for _, row in context_df.iterrows()
}


# Input
query = "where did beyonce gave birth to her first child?"
query = enhance_query_basic_english(query)
query_vec = model.encode([query])

# Top 5 search
_, top_indices = index.search(query_vec, 5)
# Get top Q+A pairs
qa_blocks = []
for idx in top_indices[0]:
    q_id = ids[idx]
    question_text = questions[idx]
    answer_text = answer_map.get(q_id, "[No answer]")
    qa_blocks.append(f"Q: {question_text}\nA: {answer_text}")


# Merge into one context block
qa_context = "\n\n".join(qa_blocks)

# Final prompt
final_prompt = f"""You are provided with a set of example questions and their correct answers. These are the **only** references you may use. You must not use outside knowledge, assumptions, or inferred facts. Your task is to analyze the example questions and answers to determine the correct answer to a new question based **only** on the provided context.

Do **not** hallucinate, do **not** infer beyond the given data, do **not** repeat answers unless supported by the references. Stay completely within the scope of the examples. Respect the data and answer only based on what is directly supported by it.

Below are the verified question-answer pairs for reference:
{qa_context}

Now, based strictly on the above context, answer the following question:

question = {query}

Return the response in this exact JSON format:
"question": "...", "answer": "..."

Do **not** add anything else. Do **not** change the format. Do **not** include explanations, notes, or extra content. Follow instructions precisely.
""" 


print(final_prompt)
