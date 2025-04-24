"""
Python script to retrieve top-k Q&A pairs from a FAISS index and query an LLM for answers.
"""
import faiss
from sentence_transformers import SentenceTransformer
from utils.query_enhancer import enhance_query_basic_english, build_prompt
from vectorizer.get_context_from_id import load_context_map
from llm.ask_ollama import ask_ollama

# ================================
# Initialization
# ================================

# Load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("embeddings/question_index.faiss")

# Load context mapping
context_map = load_context_map("data/questions.csv")

# Prepare ID and question lists
ids = []
questions_list = []

for qid, values in context_map.items():
    ids.append(qid)
    questions_list.append(values["question"])

# ================================
# Core Function
# ================================


def get_top_k_answers(query, questions: list, enhance_query: bool = False, k=5):
    """
    Retrieves top-k most relevant Q&A pairs based on the input query.

    Parameters:
        query (str): The input query string.
        k (int): Number of top results to return.
        questions (list): List of question texts.
        enhance_query (bool): If True, enhance the query using basic English.

    Returns:
        list: List of top-k Q&A blocks as strings.
    """
    if enhance_query:
        query = enhance_query_basic_english(query)

    query_vec = model.encode([query])
    _, top_indices = index.search(query_vec, k)

    qa_blocks = []
    for idx in top_indices[0]:
        q_id = ids[idx]
        question_text = questions[idx]
        answer_text = context_map.get(q_id, "[No answer]")
        qa_blocks.append(f"Q: {question_text}\nA: {answer_text}")

    return qa_blocks

# ================================
# Query Test + LLM Call
# ================================


if __name__ == "__main__":
    TEST_QUERY = "where is kathmandu?"
    QA_BLOCKS_FINAL = get_top_k_answers(
        TEST_QUERY, k=5, questions=questions_list, enhance_query=True)
    QA_CONTEXT_FINAL = "\n\n".join(QA_BLOCKS_FINAL)
    
    # print(QA_CONTEXT_FINAL)

    FINAL_PROMPT = build_prompt(QA_CONTEXT_FINAL, TEST_QUERY)

    print("original query:", TEST_QUERY)
    response = ask_ollama(FINAL_PROMPT)
    print("Response from LLM:")
    print(response)
