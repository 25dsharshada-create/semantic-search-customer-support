from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "data", "documents.json"), "r") as f:
    documents = json.load(f)

with open(os.path.join(BASE_DIR, "data", "embeddings.json"), "r") as f:
    embedding_data = json.load(f)

embeddings = np.array(embedding_data["embeddings"])

HF_TOKEN = os.environ.get("HF_TOKEN")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


def get_embedding(text):
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        headers=headers,
        json={"inputs": text}
    )

    response.raise_for_status()

    result = response.json()

    return np.array(result)


@app.get("/")
def home():
    return {
        "message": "Semantic Search API is running!"
    }


@app.post("/search")
def semantic_search(request: QueryRequest):

    query_embedding = get_embedding(request.query)

    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    scores = np.dot(embeddings, query_embedding)

    indices = np.argsort(scores)[::-1][:request.top_k]

    results = []

    for index in indices:
        results.append({
            "text": documents[index]["text"],
            "category": documents[index]["category"],
            "importance": documents[index]["importance"],
            "similarity": round(float(scores[index]), 4)
        })

    best = results[0]

    return {
        "query": request.query,
        "predicted_category": best["category"],
        "importance": best["importance"],
        "confidence": (
            "High" if best["similarity"] >= 0.65
            else "Medium" if best["similarity"] >= 0.45
            else "Low"
        ),
        "results": results
    }
