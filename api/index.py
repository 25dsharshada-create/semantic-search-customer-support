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



from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Support Semantic Search</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 850px;
            margin: 40px auto;
            padding: 20px;
            background: #f7f7f7;
        }

        h1 {
            color: #222;
        }

        .sub {
            color: #666;
        }

        textarea {
            width: 100%;
            min-height: 100px;
            padding: 12px;
            font-size: 16px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 8px;
        }

        button {
            padding: 12px 25px;
            margin-top: 10px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
            background: #111;
            color: white;
            font-size: 16px;
        }

        .card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 18px;
            margin-top: 20px;
        }

        .tag {
            display: inline-block;
            padding: 5px 9px;
            border-radius: 12px;
            background: #eee;
            margin-right: 6px;
        }

        .result {
            border-top: 1px solid #eee;
            padding: 12px 0;
        }

        #error {
            color: #b00020;
            margin-top: 15px;
        }
    </style>
</head>

<body>

<h1>Customer Support Semantic Search</h1>

<p class="sub">
Enter a customer message and the system finds the closest meaning
from the document corpus.
</p>

<textarea id="query"
placeholder="Example: My card payment did not go through"></textarea>

<br>

<button onclick="search()">Search</button>

<div id="error"></div>
<div id="output"></div>

<script>

async function search() {

    const query = document.getElementById("query").value.trim();

    const output = document.getElementById("output");
    const error = document.getElementById("error");

    output.innerHTML = "";
    error.textContent = "";

    if (!query) {
        error.textContent = "Please enter a customer message.";
        return;
    }

    try {

        const res = await fetch("/search", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                query: query,
                top_k: 3
            })

        });

        const data = await res.json();

        if (!res.ok) {
            error.textContent = data.detail || "Search failed.";
            return;
        }

        let html = `
        <div class="card">

            <h2>Prediction</h2>

            <p>
                <b>Category:</b>
                ${data.predicted_category}
            </p>

            <p>
                <b>Importance:</b>
                <span class="tag">
                    ${data.importance}
                </span>
            </p>

            <p>
                <b>Confidence:</b>
                <span class="tag">
                    ${data.confidence}
                </span>
            </p>

        </div>

        <div class="card">

            <h2>Top Semantic Matches</h2>
        `;

        data.results.forEach((r, index) => {

            html += `
            <div class="result">

                <b>#${index + 1}</b>

                <p>${r.text}</p>

                <span class="tag">
                    ${r.category}
                </span>

                <span class="tag">
                    Similarity: ${r.similarity}
                </span>

            </div>
            `;

        });

        html += `</div>`;

        output.innerHTML = html;

} catch (e) {

    error.textContent = "Error: " + e.message;

}

}

</script>

</body>
</html>
"""
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
