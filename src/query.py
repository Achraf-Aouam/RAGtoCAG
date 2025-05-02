

import os
import jsonlines
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai  

# Initialize clients
client = QdrantClient(host="localhost", port=6333)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  

def naive_rag(question):
   
    query_vector = embeddings.embed_query(question)

    
    results = client.search(
        collection_name="naive_corpus",
        query_vector=query_vector,
        limit=3
    )
    contexts = [hit.payload["text"] for hit in results]

    
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""Answer using ONLY the context below. Be concise.
    
    Context:
    {contexts}
    
    Question: {question}"""
    
    response = model.generate_content(prompt)
    answer = response.text  

  
    with jsonlines.open("results_naive.jsonl", mode="a") as writer:
        writer.write({
            "question": question,
            "contexts": contexts,
            "answer": answer
        })
    
    return answer


if __name__ == "__main__":
    question = input("Enter your question: ")
    print(naive_rag(question))