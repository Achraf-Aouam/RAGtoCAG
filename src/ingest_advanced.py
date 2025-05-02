import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
import tiktoken


loader = DirectoryLoader("data/raw", glob="**/*.txt")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200,  
    chunk_overlap=50  
)
chunks = text_splitter.split_documents(documents)


processed_chunks = []
for i, chunk in enumerate(chunks):
    
    source = chunk.metadata.get("source", "unknown")
    title = os.path.basename(source) if source != "unknown" else f"document_{i//5}"
    
    processed_chunks.append({
        "id": i,
        "text": chunk.page_content,
        "metadata": {
            "source": source,
            "title": title,
            "chunk_id": i
        }
    })


client = QdrantClient("localhost", port=6333)


try:
    client.delete_collection("advanced_corpus")
    print("Deleted existing advanced_corpus collection")
except Exception as e:
    print(f"Collection doesn't exist yet: {e}")


client.create_collection(
    collection_name="advanced_corpus",
    vectors_config=models.VectorParams(
        size=768,  
        distance=models.Distance.COSINE
    )
)



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


texts = [chunk["text"] for chunk in processed_chunks]
vectors = embeddings.embed_documents(texts)


points = []
for i, (chunk, vector) in enumerate(zip(processed_chunks, vectors)):
    points.append(
        models.PointStruct(
            id=chunk["id"],
            vector=vector,
            payload={
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "title": chunk["metadata"]["title"],
                "chunk_id": chunk["metadata"]["chunk_id"]
            }
        )
    )


BATCH_SIZE = 100
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i+BATCH_SIZE]
    client.upsert(
        collection_name="advanced_corpus",
        points=batch
    )
    print(f"Inserted batch {i//BATCH_SIZE + 1}/{(len(points)//BATCH_SIZE) + 1}")

print(f"Successfully ingested {len(points)} chunks into advanced_corpus collection!")