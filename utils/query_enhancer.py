import requests
import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

def enhance_query_basic_english(query, model="llama3.2"):
    """
    Uses an LLM to rephrase the input query into simple, clear, and basic English 
    without changing its original meaning.
    """
    prompt = f"Simplify the user's question to plain English. do not lose any question. If you find certain keyword useful keep them do not remvoe any context\nUser: {query}\nSimplified:"
    r = requests.post(f"{OLLAMA_HOST}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False},
                      timeout=60)

    try:
        r.raise_for_status()
        return r.json().get("response", query).strip()
    except requests.exceptions.RequestException as e:
        # Do not print error, just return original query for a faster, cleaner response
        return query


def build_prompt(qa_context, test_query):
    # ... (content remains unchanged) ...
    # Removed the large function body for brevity, but keep it in the file.
    final_prompt = f"""You are kindly provided with a set of example question-answer pairs. These are the **only** references you may rely on. Please do **not** use any outside knowledge, assumptions, or inferred facts. Base your response strictly on the information given.

    Kindly follow these instructions carefully:
    - Please do **not** invent or hallucinate any content.
    - Please do **not** infer beyond what is directly stated.
    - Please do **not** repeat answers unless clearly supported by the references.
    - Give priority to broader context (such as country or person) before narrower context (such as city).
    - If the context is insufficient to answer the question, kindly respond with "I don't know" or "I cannot answer this question."

    Below is the verified reference context:
    {qa_context}

    Now, based solely and strictly on the above context, please answer the following:

    question = {test_query}

    Please return your response in this exact JSON format:
    "question": "<please insert the original question>", "answer": "<please provide a one-sentence answer using only the given context>", summary: "<please write a natural-language explanation of around 125 words using only the provided context. Do not include new facts or drift off-topic.>"

    Please do not add anything beyond the specified format. No notes, comments, or extra content should be included.
    """
    return final_prompt


if __name__ == "__main__":
    TEST_QUERY = "Can you tell me the weather forecast for tomorrow?"
    enhanced_query = enhance_query_basic_english(TEST_QUERY)
    print("Enhanced Query:", enhanced_query)