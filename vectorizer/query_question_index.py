import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_index_and_data():
    """
    Loads the FAISS index and question data from disk.

    Returns:
        index (faiss.Index): FAISS index for question embeddings.
        ids (list): List of question IDs.
        questions (list): List of question texts.
    """
    index_path = "embeddings/question_index.faiss"
    mapping_path = "embeddings/questions.pkl"

    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError("Index or mapping file not found in 'embeddings/'")

    index = faiss.read_index(index_path)

    with open(mapping_path, "rb") as f:
        data = pickle.load(f)
        ids = data["ids"]
        questions = data["questions"]

    return index, ids, questions

def embed_query(query, model):
    """
    Converts a query string into an embedding using the given model.

    Args:
        query (str): Input query.
        model (SentenceTransformer): Embedding model.

    Returns:
        np.ndarray: Query embedding.
    """
    return np.array([model.encode(query)])

def search_similar_questions(model, query, top_k=3):
    """
    Searches for top-k most similar questions to the input query.

    Args:
        query (str): Input question string.
        top_k (int): Number of similar questions to return.

    Returns:
        list: List of dictionaries containing matched question IDs and texts.
    """
    index, ids, questions = load_index_and_data()
    query_embedding = embed_query(query, model)

    # Perform FAISS search
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        results.append({
            "id": ids[idx],
            "question": questions[idx],
            "distance": float(distances[0][i])
        })

    return results

# Example usage
if __name__ == "__main__":
    query = "Who did Mary appear to in France?"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    results = search_similar_questions(model, query, top_k=3)
    for res in results:
        print(res)
