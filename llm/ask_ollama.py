import requests


def ask_ollama(prompt, model="llama3.2"):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=60
    )
    return res.json().get("response", "[No response]")
