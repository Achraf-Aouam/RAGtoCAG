# File: /home/your_username/rag-lab/src/query_advanced.py

import os

import pickle
from rank_bm25 import BM25Okapi
import numpy as np
import jsonlines
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from qdrant_client import models
from sentence_transformers import CrossEncoder  # Reranking

# Initialize clients
client = QdrantClient(host="localhost", port=6333)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Changed


with open("bm25_model.pkl", "rb") as f:
    bm25 = pickle.load(f)

def advanced_rag(question):
    # Generate dense vector
    dense_vector = embeddings.embed_query(question)
    
    # Generate BM25 sparse vector (indices = token IDs, values = weights)
    tokenized_query = question.split()  # Simple tokenization
    sparse_indices = [hash(token) % 10000 for token in tokenized_query]  # Map tokens to indices
    sparse_values = [1.0] * len(sparse_indices)  # Example weights (replace with BM25 logic)
    
    # Create sparse vector
    sparse_vector = models.SparseVector(
        indices=sparse_indices,
        values=sparse_values
    )
    
    # Hybrid search
    results = client.search(
        collection_name="advanced_corpus",
        query_vector=dense_vector,
        sparse_vector=sparse_vector,  # Pass directly (no `name` needed here)
        limit=10
    )
    # Rerank with cross-encoder
    pairs = [[question, hit.payload["text"]] for hit in results]
    rerank_scores = reranker.predict(pairs)
    reranked_results = sorted(
        zip(results, rerank_scores),
        key=lambda x: x[1], reverse=True
    )[:3]  # Keep top 3

    contexts = [hit[0].payload["text"] for hit in reranked_results]

    # Generate with Gemini Flash
    model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
    prompt = f"Answer using ONLY the context below. Cite sources like [doc1].\n\nContext:\n{contexts}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    answer = response.text

    # Log to JSONL
    with jsonlines.open("results_advanced.jsonl", mode="a") as writer:
        writer.write({
            "question": question,
            "contexts": contexts,
            "answer": answer
        })

    return answer

if __name__ == "__main__":
    question = input("Enter question: ")
    print(advanced_rag(question))