import pandas as pd

def load_context_map(df: pd.DataFrame) -> dict:
    """
    Loads a context map from a DataFrame with 'id', 'context', 'question', and 'answers' columns.

    Args:
        df (pd.DataFrame): DataFrame containing the columns.

    Returns:
        dict: Mapping from question ID to a dictionary with context, question, and answer.
    """
    required_columns = {"id", "context", "question", "answers"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    context_map = {}
    for _, row in df.iterrows():
        qid = row["id"]
        context_map[qid] = {
            "context": row["context"],
            "question": row["question"],
            "answer": row["answers"]
        }

    return context_map

def build_prompt_from_id(match_id, user_question, context_map) -> str | None:
    """
    Builds a prompt using the context from a matched question ID.

    Args:
        match_id (str or int): ID of the matched question.
        user_question (str): Original user question.
        context_map (dict): Mapping from ID to context data.

    Returns:
        str or None: Formatted prompt string or None if ID not found.
    """
    entry = context_map.get(match_id)
    if entry is None:
        return None

    prompt = f"""
You are a helpful assistant. A user asked a question: "{user_question}"

Here is a related passage and Q&A pair that may help you:

Context:
{entry['context']}

Related Question:
{entry['question']}

Answer:
{entry['answer']}

Using only this information, generate a helpful and concise answer to the user's question. 
If the context is not relevant, respond with: "Sorry, I couldn’t find relevant information."
"""
    return prompt.strip()
