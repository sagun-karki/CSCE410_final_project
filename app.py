import faiss
from sentence_transformers import SentenceTransformer
from utils.query_enhancer import enhance_query_basic_english, build_prompt
from vectorizer.get_context_from_id import load_context_map
from llm.ask_ollama import ask_ollama
import time
import os

# ================================
# Initialization
# ================================

model = SentenceTransformer("all-MiniLM-L6-v2")
if os.path.exists("embeddings/question_index.faiss") and os.path.exists("data/questions.csv"):
    index = faiss.read_index("embeddings/question_index.faiss")
    context_map = load_context_map("data/questions.csv")
else:
    print("Required files not found. Building index...")
    os.system("python ./vectorizer/build_question_index.py")
    index = faiss.read_index("embeddings/question_index.faiss")
    context_map = load_context_map("data/questions.csv")

ids = []
questions_list = []

for qid, values in context_map.items():
    ids.append(qid)
    questions_list.append(values["question"])

# ================================
# Core Function
# ================================


def get_top_k_answers(query, questions: list, enhance_query: bool = False, k=5):
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
# Continuous Loop for Queries
# ================================


if __name__ == "__main__":
    while True:
        USER_QUERY = input(
            "Enter your query (or type 'exit' to quit, 'cls' to clear screen): ").strip()
        if USER_QUERY.lower() == "exit":
            break
        elif USER_QUERY.lower() == "cls":
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        os.system('cls' if os.name == 'nt' else 'clear')

        USER_QUERY = enhance_query_basic_english(USER_QUERY)
        QA_BLOCKS = get_top_k_answers(
            USER_QUERY, questions=questions_list, enhance_query=True, k=5)
        QA_CONTEXT = "\n\n".join(QA_BLOCKS)

        FINAL_PROMPT = build_prompt(QA_CONTEXT, USER_QUERY)

        print("========================================")
        print("Generating response", end="")
        print("\nResponse from LLM:")
        RESPONSE = ask_ollama(FINAL_PROMPT, model="llama3.2")
        for _ in range(5):
            time.sleep(0.5)
            print(".", end="", flush=True)
        print("\n")
        print(RESPONSE)

        print("\n\n\n")
        print("========================================")
        print("reference context:")
        print(QA_CONTEXT)
        print("========================================")
