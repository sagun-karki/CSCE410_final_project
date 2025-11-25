# CSCE 410 Final Project Spring 2025
This project is a local question-answering system built on top of a pretrained LLM. It uses a preprocessed version of the SQuAD Augmented v2 dataset and performs semantic search using FAISS and SentenceTransformers to retrieve the most relevant context for a user query. The retrieved context is then used to build a prompt for a locally hosted LLM to generate final answers.

## Example Results

LLM will use only the given context to answer the question. The context is retrieved using semantic search, and the LLM generates an answer based on that context. The current implementation is a CLI application.

![Example Results](/assets/example.png)

---

## Team Members

- Ryan Argo  
- Jason DiBraccio  
- Victor Knapp  
- Sagun Karki  
- Midia Yousif  

---

## Dataset

Download the simplified dataset: [SQuAD Augmented v2](https://huggingface.co/datasets/christti/squad-augmented-v2).

---

## Language Model

- Use any OpenAI-compatible LLM.
- **Recommended:** Llama 3.2 from Ollama.

---

## Setup Instructions

### Environment Setup

#### Step 1: Create a Virtual Environment

```bash
python3 -m venv .venv
```

#### Step 2: Activate the Virtual Environment

```bash
source .venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Install Ollama

We are using Ollama for this project. While ChatGPT or other OpenAI-compatible LLMs can be used, ensure to replace the `ask_ollama` function with the corresponding function for your chosen LLM. The rest of the setup should remain the same.

Visit [Ollama's website](https://ollama.com/) for installation instructions.

> After installing Ollama, it should start running a REST API. By default, it serves at `http://localhost:11434/`. Once confirmed, proceed to run [./app.py](./app.py). Before running the application, ensure you create question embeddings and an index.

#### Step 5: Get Data

Download the simplified dataset: [SQuAD Augmented v2](https://huggingface.co/datasets/christti/squad-augmented-v2) and place it in the [./data/](./data/) directory. The data should be in CSV format.

Alternatively, you can download the data (and also full code) from this [Google Drive link](https://drive.google.com/drive/folders/1oxnIDOYypX323P2906ndabekJ1ogt7N7?usp=sharing).

#### Step 6: Run the App

On startup, [./app.py](./app.py) will search for the embeddings in [./embeddings/question_index.faiss](./embeddings/question_index.faiss) and [./embeddings/questions.pkl](./embeddings/questions.pkl). If these files are not found, it will automatically run [./vectorizer/build_question_index.py](./vectorizer/build_question_index.py) to generate them.

```bash
# Be sure to be in the root of the repo.
python3 app.py
```

---

## Docker Steps

### 1. Build the Docker Image

The build process will automatically install dependencies, download the Sentence Transformer model, and generate the necessary FAISS index/embeddings inside the image.

```bash
docker build -t csce410-qa-app .
```

### 2. Run the Docker Container

The container needs to connect to the Ollama API running on your host machine. We use an environment variable to specify the host URL.

```bash
docker run -it --rm \
    -e OLLAMA_HOST="http://host.docker.internal:11434" \
    csce410-qa-app
```

**Note on Connectivity:**  
The `-e OLLAMA_HOST="http://host.docker.internal:11434"` setting is required for Docker Desktop (Mac/Windows) to allow the container to reach the Ollama service on the host machine.

For native Linux Docker installations, you might need to use `--network=host` and set `OLLAMA_HOST="http://localhost:11434"`.