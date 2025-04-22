import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import numpy as np

print("Loading data...")
# Load questions from CSV file
df = pd.read_csv("data/questions.csv")

# Initialize sentence transformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract questions and IDs from dataframe
questions = df["question"].tolist()
ids = df["id"].tolist()

print("Generating embeddings...")
# Generate embeddings as NumPy array
embeddings = model.encode(questions, convert_to_numpy=True)

print("Creating FAISS index...")
# Create FAISS index and add embeddings
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("Indexing complete.")
# Save FAISS index to file
os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, "embeddings/question_index.faiss")

# Save ID and question mappings to pickle file
with open("embeddings/questions.pkl", "wb") as f:
    pickle.dump({"ids": ids, "questions": questions}, f)
print("ID and question mappings saved.")
