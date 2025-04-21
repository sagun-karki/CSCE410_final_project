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
    pass
