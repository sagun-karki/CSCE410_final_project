import faiss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Example dataset
questions = [
    "How do I reset my password?",
    "What is the process to change my password?",
    "How can I recover a forgotten password?",
    "Where can I update my account information?",
    "How to delete my account?",
    "What is the refund policy?",
    "How do I contact customer support?",
    "Steps to upgrade my subscription?",
    "How can I cancel my membership?",
    "Methods to change email address on file?",
    "I want to permanently close my account.",
    "How to enable two-factor authentication?",
    "What to do if my account was hacked?",
    "How can I stop receiving promotional emails?",
    "I want to update my billing information.",
    "Can I transfer my account to another user?",
    "Is there a way to pause my subscription?",
    "How do I view past payment transactions?",
    "How can I get help with a technical issue?",
    "Where is the form to request a refund?",
    "How do I unsubscribe from the service?",
    "What are the steps to change my username?",
    "I need help updating my phone number.",
    "How do I file a complaint or report a bug?",
    "Can I recover a deleted account?",
    "What if I’m being charged wrongly?",
    "Can I speak to a real person for help?",
    "How do I add a new payment method?",
    "Where is the FAQ for account management?",
    "How to disable auto-renewal of a plan?",
    "I want to know about trial period conditions.",
    "What happens when my subscription expires?",
    "Can I access support on weekends?",
    "Is it possible to merge two accounts?",
    "Where do I set my security questions?",
    "Can I preview my billing before payment?",
    "How do I report suspicious activity?",
    "Can I see login activity on my account?",
    "How can I find the app's changelog or updates?",
    "How to manage push notification settings?"
]

# Multiple User Queries
queries = [
    "I forgot my password. How do I fix it?",
    "How can I recover my account?",
    "How do I delete my account?",
    "Where can I find information on billing?",
    "How do I contact customer support?"
]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode with SBERT
question_embeddings_norm = model.encode(questions, normalize_embeddings=True)

# Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# Search FAISS
def search_faiss(index, query_embedding, k):
    scores, indices = index.search(query_embedding, k)
    return scores[0], indices[0]

# TF-IDF baseline
def tfidf_baseline(questions, query, k):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions + [query])
    query_vec = tfidf_matrix[-1]
    corpus_vecs = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vec, corpus_vecs)[0]
    top_k_idx = np.argsort(similarities)[::-1][:k]
    top_k_scores = similarities[top_k_idx]
    return top_k_scores, top_k_idx

# Different experiments
experiments = {}

# 1. Baseline 1: TF-IDF keyword matching
tfidf_scores = []
for query in queries:
    scores, _ = tfidf_baseline(questions, query, k=5)
    tfidf_scores.append(np.mean(scores))
experiments['TF-IDF'] = np.mean(tfidf_scores)

# 2. Our main version: SBERT + FAISS
index_norm = build_faiss_index(question_embeddings_norm)
sbert_faiss_scores = []
for query in queries:
    query_embedding_norm = model.encode([query], normalize_embeddings=True)
    scores, _ = search_faiss(index_norm, query_embedding_norm, k=5)
    sbert_faiss_scores.append(np.mean(scores))
experiments['SBERT'] = np.mean(sbert_faiss_scores)

# 3. FAISS with noisy query
np.random.seed(42)
faiss_noisy_scores = []
for query in queries:
    query_embedding_norm = model.encode([query], normalize_embeddings=True)
    query_embedding_noisy = query_embedding_norm + np.random.normal(0, 0.01, query_embedding_norm.shape)
    scores, _ = search_faiss(index_norm, query_embedding_noisy, k=5)
    faiss_noisy_scores.append(np.mean(scores))
experiments['+ Noisy Query'] = np.mean(faiss_noisy_scores)

# 4. FAISS with different model
model2 = SentenceTransformer('bert-base-nli-mean-tokens')
question_embeddings_norm2 = model2.encode(questions, normalize_embeddings=True)
index_norm2 = build_faiss_index(question_embeddings_norm2)
faiss_bigger_model_scores = []
for query in queries:
    query_embedding_norm2 = model2.encode([query], normalize_embeddings=True)
    scores, _ = search_faiss(index_norm2, query_embedding_norm2, k=5)
    faiss_bigger_model_scores.append(np.mean(scores))
experiments['More Advanced Model'] = np.mean(faiss_bigger_model_scores)

# Plotting
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']

plt.figure(figsize=(10, 6))
bars = plt.bar(experiments.keys(), experiments.values(), color=colors, width=0.5)

plt.xlabel('Method', fontsize=14)
plt.ylabel('Average Top-5 Similarity', fontsize=14)
plt.title('Performance Comparison: TF-IDF vs SBERT', fontsize=16)
plt.xticks(rotation=25, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 1)  # Force full scale 0–1 for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values above bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 6),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()
