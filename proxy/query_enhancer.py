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


if __name__ == "__main__":
    TEST_QUERY = "Can you tell me the weather forecast for tomorrow?"
    enhanced_query = enhance_query_basic_english(TEST_QUERY)
    print("Enhanced Query:", enhanced_query)
