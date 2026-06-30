import os
import json
import requests
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from google import genai
from embedding_utils import create_embedding
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

def inference(prompt):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embeddings'].values))
# print(np.vstack(df['embeddings'].shape))

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
print("\nTop 5 Similar Chunks:\n")

# print what those chunks contain.
for idx in max_indx:

    print("=" * 80)

    print(f"Similarity : {similarities[idx]:.4f}")

    print(f"Chunk ID   : {df.iloc[idx]['chunk_id']}")

    print(f"Title      : {df.iloc[idx]['title']}")

    print(f"Number      : {df.iloc[idx]['number']}")

    print(f"Time       : {df.iloc[idx]['start']} - {df.iloc[idx]['end']}")

    print(f"Text       : {df.iloc[idx]['text']}")

    print()
    
new_df = df.loc[max_indx]

prompt = f"""
You are an AI Teaching Assistant for the Sigma Web Development Course.

Use ONLY the provided course chunks to answer.

If the answer is present in the chunks:
- Answer the question clearly.
- Mention the video number.
- Mention the video title.
- Mention the approximate timestamp.( in minutes and seconds both)
- Recommend the user watch that section.

If the answer is NOT present in the chunks:
Reply:
"I couldn't find this topic in the available course videos."

Course Chunks:
{new_df[['title','number','start','end','text']].to_json(orient='records', indent=2)}

User Question:
{incoming_query}

Answer:
"""       
with open("prompt.txt" , "w") as f:
    f.write(prompt)
      
      
response = inference(prompt)
print(response)

print("LENGTH OF PROMPT : ")
print(len(prompt))

with open("response.txt" , "w" ) as f:
    f.write(response)
# for index , item in new_df.iterrows():
#     print(index , item["title"] , item["number"] , item["text"])

