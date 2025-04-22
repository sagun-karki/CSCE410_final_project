import os
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer

def convert_to_index(output_path="embeddings/"):
    """
    Converts a list of questions from 'data/questions.csv' into vector embeddings using a 
    SentenceTransformer model and builds a FAISS index for similarity search.

    The function performs the following:
    - Loads questions and their IDs (ids = row_number) from a CSV file.
    - Encodes the questions into dense vector embeddings.
    - Builds and saves a FAISS index for efficient similarity search.
    - Saves a mapping of question IDs to questions as a pickle file.

    Parameters:
    output_path (str): Directory path to save the FAISS index and question mapping. 
                       Defaults to 'embeddings/'.

    Output:
    - FAISS index file saved as '<output_path>/question_index.faiss'
    - Pickle file saved as '<output_path>/questions.pkl' containing question IDs and texts.
    """
    # Step 1: Load questions from CSV
    csv_path = "data/questions.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    if "question" not in df.columns:
        raise ValueError("CSV must contain a 'question' column.")
    
    questions = df["question"].tolist()
    ids = list(range(len(questions)))

    # Step 2: Encode questions using Sentence-BERT
    model = SentenceTransformer("all-MiniLM-L6-v2")  # or any other model you prefer
    embeddings = model.encode(questions, show_progress_bar=True)

    # Step 3: Build FAISS index
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Step 4: Save index and mapping
    os.makedirs(output_path, exist_ok=True)

    index_path = os.path.join(output_path, "question_index.faiss")
    mapping_path = os.path.join(output_path, "questions.pkl")

    faiss.write_index(index, index_path)

    with open(mapping_path, "wb") as f:
        pickle.dump({"ids": ids, "questions": questions}, f)

    print(f"✅ Saved FAISS index to: {index_path}")
    print(f"✅ Saved question mapping to: {mapping_path}")
