# Semantic Search Customer Support Chatbot

A simple NLP case-study project for an e-commerce customer-support corpus.

## Pipeline
1. Build a small document corpus.
2. Generate semantic embeddings in Google Colab using Sentence Transformers.
3. Save embeddings + metadata as JSON. The deployed API uses the same MiniLM model through Hugging Face Inference so Vercel does not need to package PyTorch.
5. The API embeds a user query, compares cosine similarity with the stored corpus, and returns the best category, matched message, confidence, importance, and response.

## Project structure
```text
semantic-search-project/
├── api/
│   └── index.py
├── data/
│   ├── documents.json
│   └── embeddings.json
├── index.html
├── requirements.txt
├── vercel.json
└── README.md
```

## Google Colab
Run the Colab cells supplied with the project. The final cell creates `data/embeddings.json`.

Copy the generated `data/embeddings.json` and `data/documents.json` into this repository.

## Vercel
1. Create a GitHub repository and upload these files.
2. Import the repository into Vercel.
3. Deploy.
4. Open the Vercel URL. The browser UI calls `/api/search`.

## Example queries
- Where is my parcel?
- I need to send this item back.
- My card payment failed.
- When will my refund arrive?

## Hugging Face setup for Vercel
Create a Hugging Face access token and add it in Vercel as an environment variable named `HF_TOKEN`.
The token is used only by the server-side API to create the query embedding.

Do not put the token in `index.html` or in GitHub.
