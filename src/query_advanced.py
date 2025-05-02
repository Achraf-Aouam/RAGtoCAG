import os
import jsonlines
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from sentence_transformers import CrossEncoder
from time import time
import tiktoken
import json
from qdrant_client.http import models


client = QdrantClient(host="localhost", port=6333)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")




genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def query_rewriter(question):
    """Use Gemini to rewrite the query for better retrieval"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""Rewrite the following question to make it more specific and searchable.
        Keep the rewritten question focused and concise.
        
        Original Question: {question}
        
        Rewritten Question:"""
        
        response = model.generate_content(prompt)
        rewritten_question = response.text.strip()
        print(f"Original: {question}\nRewritten: {rewritten_question}")
        return rewritten_question
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return question  

def metadata_filter(question):
    """Extract metadata filters from question if present"""
    filters = None
    
    
    if "from document" in question.lower() or "in document" in question.lower():
        
        for phrase in ["from document", "in document"]:
            if phrase in question.lower():
                parts = question.lower().split(phrase)
                if len(parts) > 1:
                    doc_name = parts[1].strip().split()[0].strip('."\'?!')
                    from qdrant_client.http import models
                    filters = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="title",
                                match=models.MatchText(text=doc_name)
                            )
                        ]
                    )
                    
                    question = parts[0].strip()
                    break
    
    return question, filters

def advanced_rag(question):
    start_time = time()
    
    try:
        
        rewritten_question = query_rewriter(question)
        
        
        clean_question, filters = metadata_filter(rewritten_question)
        
        
        query_vector = embeddings.embed_query(clean_question)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        
        clean_question = question
        query_vector = embeddings.embed_query(clean_question)
        filters = None
    
    
    search_params = {
        "collection_name": "advanced_corpus",
        "query_vector": query_vector,
        "limit": 10  
    }
    
    
    if filters:
        search_params["filter"] = filters
        
    results = client.search(**search_params)
    
    
    pairs = [[clean_question, hit.payload["text"]] for hit in results]
    rerank_scores = reranker.predict(pairs)
    
    
    reranked_results = sorted(
        zip(results, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:3]  
    
    
    contexts = [hit[0].payload["text"] for hit in reranked_results]
    sources = [f"[{hit[0].payload.get('title', 'doc')}:{hit[0].id}]" for hit in reranked_results]
    
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    
    prompt = f"""Answer the following question using ONLY the provided context.
    Be concise and to the point. Cite your sources using the reference numbers like [1], [2], etc.
    
    Context:
    {' '.join([f'[{i+1}] {ctx}' for i, ctx in enumerate(contexts)])}
    
    Question: {question}
    
    Answer with citations:"""
    
    
    print(f"\nPrompt for generation:\n{prompt[:300]}...(truncated)")
    print(f"Using {len(contexts)} context chunks for generation")
    
    response = model.generate_content(prompt)
    answer = response.text
    
    
    end_time = time()
    elapsed_time = end_time - start_time
    
    
    log_entry = {
        "question": question,
        "rewritten_question": rewritten_question,
        "contexts": contexts,
        "sources": sources,
        "answer": answer,
        "retrieval_time": elapsed_time
    }
    
    with jsonlines.open("results_advanced.jsonl", mode="a") as writer:
        writer.write(log_entry)
    
    return answer, elapsed_time

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer, time_taken = advanced_rag(question)
    print(f"\nAnswer (took {time_taken:.2f} seconds):\n{answer}")