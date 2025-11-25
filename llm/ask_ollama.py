import requests
import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

def ask_ollama(prompt, model="llama3.2"):
    res = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=60
    )
    return res.json().get("response", "[No response]")