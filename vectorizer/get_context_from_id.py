def load_context_map(df: pd.DataFrame) -> dict:
    """
    Loads a context map from a CSV file with 'id', 'context', 'question', and 'answers' columns.

    Args:
        csv_path (str): Path to the CSV file. Defaults to 'data/questions.csv'.

    Returns:
        dict: Mapping from question ID to a dictionary with context, question, and answer.
    """
    ...

def build_prompt_from_id(match_id, user_question, context_map):
    """
    Builds a prompt using the context from a matched question ID.

    Args:
        match_id (str or int): ID of the matched question.
        user_question (str): Original user question.
        context_map (dict): Mapping from ID to context data.

    Returns:
        str or None: Formatted prompt string or None if ID not found.
    """
    ...
