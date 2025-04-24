import requests

def enhance_query_basic_english(query, model="llama3.2"):
    """
    Uses an LLM to rephrase the input query into simple, clear, and basic English 
    without changing its original meaning.

    Args:
        query (str): Input query string.

    Returns:
        str: Rewritten query in basic English.
    """

    prompt = f"Simplify the user's question to plain English. do not lose any question. If you find certain keyword useful keep them do not remvoe any context\nUser: {query}\nSimplified:"
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False},
                      timeout=60)

    try:
        r.raise_for_status()
        return r.json().get("response", query).strip()
    except requests.exceptions.RequestException as e:
        print("API error:", e)
        return query


def build_prompt(qa_context, test_query):
    final_prompt = f"""You are provided with a set of example questions and their correct answers. These are the **only** references you may use. You must not use outside knowledge, assumptions, or inferred facts. Your task is to analyze the example questions and answers to determine the correct answer to a new question based **only** on the provided context.

    Do **not** hallucinate, do **not** infer beyond the given data, do **not** repeat answers unless supported by the references. Stay completely within the scope of the examples. Respect the data and answer only based on what is directly supported by it. Give prority to biggest possible context like country or persons name and then go to smaller context like city or other things.

    Below are the verified question-answer pairs for reference:
    {qa_context}

    Now, based strictly on the above context, answer the following question:

    question = {test_query}

    Return the response in this exact JSON format:
    "question": "...", "answer": "...", summary: "<make natural language summary of the answer using the context> + <use other information available in the context to provide a summary of the answer>"

    Do **not** add anything else. Do **not** change the format. Do **not** include explanations, notes, or extra content. Follow instructions precisely.
    """ 
    return final_prompt

if __name__ == "__main__":
    TEST_QUERY = "Can you tell me the weather forecast for tomorrow?"
    enhanced_query = enhance_query_basic_english(TEST_QUERY)
    print("Enhanced Query:", enhanced_query)
