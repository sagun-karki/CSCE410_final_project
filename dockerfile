FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the entire application source code and data (MUST include data/questions.csv)
COPY . .

# 3. Create the directories if they don't exist (build script will create them, but safer to ensure)
RUN mkdir -p embeddings data

# 4. Pre-download and cache the Sentence Transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 5. Build the FAISS index during the image build process
# This creates embeddings/question_index.faiss and data/questions.csv is read here.
RUN python ./vectorizer/build_question_index.py

# 6. Command to run the application
# Use the --network=host flag when running the container to allow access to the host's Ollama service.
CMD ["python", "app.py"]