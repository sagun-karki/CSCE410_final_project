# CSCE 410 Final Project

## Download Dataset

use this simplified dataset [https://huggingface.co/datasets/christti/squad-augmented-v2](https://huggingface.co/datasets/christti/squad-augmented-v2)


## LLM to use
- Any llm that is openai compatible
- for now llama3.2 from ollama

## UI
- open-webui

## Process
- Run open-webui
- When user inputs a question, we will intercept the input using a proxy and add additional context from our dataset.
- The context will be added to the input as a prompt.
- The model will then generate a response based on the input and the additional context.
- The response will be displayed to the user in the UI.
