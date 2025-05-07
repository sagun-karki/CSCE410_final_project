# CSCE 410 Final Project

## Dataset

Download the simplified dataset: [SQuAD Augmented v2](https://huggingface.co/datasets/christti/squad-augmented-v2).

## Language Model

- Use any OpenAI-compatible LLM.
- Recommended: Llama 3.2 from Ollama.

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

#### Step 6: Run the APP

On startup, [./app.py](./app.py) will search for the embeddings in [./embeddings/question_index.faiss](./embeddings/question_index.faiss) and [./embeddings/questions.pkl](./embeddings/questions.pkl). If these files are not found, it will automatically run [./vectorizer/build_question_index.py](./vectorizer/build_question_index.py) to generate them.

```bash
# be sure to be in the root of the repo. 
python3 app.py
```

