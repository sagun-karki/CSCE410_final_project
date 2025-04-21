def load_index_and_data():
    """
    Loads the FAISS index and question data from disk.

    Returns:
        index (faiss.Index): FAISS index for question embeddings.
        ids (list): List of question IDs.
        questions (list): List of question texts.
    """
    ...

def embed_query(query, model):
    """
    Converts a query string into an embedding using the given model.

    Args:
        query (str): Input query.
        model (SentenceTransformer): Embedding model.

    Returns:
        np.ndarray: Query embedding.
    """
    ...

def search_similar_questions(model, query, top_k=3):
    """
    Searches for top-k most similar questions to the input query.

    Args:
        query (str): Input question string.
        top_k (int): Number of similar questions to return.

    Returns:
        list: List of dictionaries containing matched question IDs and texts.
    """
    ...

# Example usage
if __name__ == "__main__":
    query = "Who did Mary appear to in France?"
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    results = search_similar_questions(model, query, top_k=3)
    for res in results:
        print(res)
