from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
In recent years, artificial intelligence has advanced rapidly, with applications in fields ranging from healthcare and finance to education and transportation. One area that has seen significant growth is natural language processing, which enables machines to understand and generate human language. Models like GPT, BERT, and T5 have made it possible to build systems that can translate languages, answer questions, summarize documents, and even write stories.
"""

summary = summarizer(text, max_length=60, min_length=10, do_sample=False)
print(summary[0]['summary_text'])
