import pandas as pd
import ast


def load_context_map(csv_path="data/questions.csv"):
    df = pd.read_csv(csv_path)
    context_map = {}

    for _, row in df.iterrows():
        question = row["question"]
        answers = row["answers"]
        context = row["context"]
        title = row["title"]

        parsed = ast.literal_eval(answers)
        parsed_answer = parsed["text"][0]

        context_map[row["id"]] = {
            "question": question,
            "answers": parsed_answer,
            "context": context,
            "title": title
        }

    return context_map

