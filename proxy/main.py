from fastapi import FastAPI, Request
import requests
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load everything once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("embeddings/question_index.faiss")
with open("embeddings/questions.pkl", "rb") as f:
    question_data = pickle.load(f)

context_df = pd.read_csv("data/questions.csv")
context_map = {
    row["id"]: row["context"]
    for _, row in context_df.iterrows()
}

def build_prompt(context, original_question):
    return f"""Use the following context to answer the question.

Context:
{context}

Question:
{original_question}

Answer:"""

def embed_query(query):
    return embed_model.encode([query])

def search_question_index(query_vec, top_k=1):
    _, indices = faiss_index.search(query_vec, top_k)
    return indices[0][0]

@app.post("/api/generate")
async def reroute(request: Request):
    body = await request.json()
    user_prompt = body.get("prompt", "")

    # Simplify question (optional)
    simplified_query = enhance_query_basic_english(user_prompt)

    # Vector search
    query_vec = embed_query(simplified_query)
    match_idx = search_question_index(query_vec)
    matched_id = question_data["ids"][match_idx]
    matched_context = context_map.get(matched_id, "")

    # Build prompt
    final_prompt = build_prompt(matched_context, user_prompt)
    body["prompt"] = final_prompt

    # Forward to Ollama
    response = requests.post("http://localhost:11434/api/generate", json=body)
    return response.json()
