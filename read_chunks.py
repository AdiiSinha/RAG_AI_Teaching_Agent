import os
import json
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
# ==========================
# Configuration
# ==========================

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "bge-m3"
JSON_FOLDER = "jsons"

# Number of chunks sent to Ollama in one request
BATCH_SIZE = 32

# ==========================
# Embedding Function
# ==========================

def create_embedding(text_list):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "input": text_list
        }
    )

    if response.status_code != 200:
        raise Exception(
            f"HTTP Error {response.status_code}\n{response.text}"
        )

    data = response.json()

    if "embeddings" not in data:

        print("\n================ ERROR FROM OLLAMA ================")
        print(data)
        print("===================================================\n")

        raise Exception("No 'embeddings' key found in response.")

    return data["embeddings"]


# ==========================
# Read JSON Files
# ==========================

json_files = sorted(os.listdir(JSON_FOLDER))

all_chunks = []

chunk_id = 0

for json_file in json_files:

    print(f"\nProcessing : {json_file}")

    with open(
        os.path.join(JSON_FOLDER, json_file),
        encoding="utf-8"
    ) as f:

        content = json.load(f)

    chunks = content["chunks"]

    print(f"Total Chunks : {len(chunks)}")

    # Remove empty chunks
    texts = []

    valid_chunks = []

    for chunk in chunks:

        text = chunk["text"].strip()

        if len(text) == 0:
            continue

        texts.append(text)

        valid_chunks.append(chunk)

    print(f"Valid Chunks : {len(valid_chunks)}")

    embeddings = []

    # ==========================
    # Create Embeddings Batchwise
    # ==========================

    for start in range(0, len(texts), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(texts))

        batch = texts[start:end]

        print(
            f"Embedding Batch "
            f"{start+1}-{end}"
        )

        batch_embeddings = create_embedding(batch)

        embeddings.extend(batch_embeddings)

    # ==========================
    # Safety Check
    # ==========================

    if len(embeddings) != len(valid_chunks):

        raise Exception(
            f"""
Embedding count mismatch.

Chunks      : {len(valid_chunks)}
Embeddings  : {len(embeddings)}
"""
        )

    # ==========================
    # Store Everything
    # ==========================

    for chunk, embedding in zip(valid_chunks, embeddings):

        chunk["chunk_id"] = chunk_id

        chunk["embedding"] = embedding

        chunk_id += 1

        all_chunks.append(chunk)

print("\nCreating DataFrame...")

df = pd.DataFrame(all_chunks)

print(df.head())

print("\n")
print(f"Total Chunks Embedded : {len(df)}")

# Optional
df.to_pickle("embedded_chunks.pkl")

print("\nSaved embeddings to embedded_chunks.pkl")

joblib.dump(df , 'embeddings.joblib')
 