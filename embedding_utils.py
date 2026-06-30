import requests

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "bge-m3"

def create_embedding(text_list):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "input": text_list
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"]