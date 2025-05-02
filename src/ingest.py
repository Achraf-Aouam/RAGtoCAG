

import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


loader = DirectoryLoader("data/raw", glob="**/*.txt")
documents = loader.load()


text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
chunks = text_splitter.split_documents(documents)


client = QdrantClient(host="localhost", port=6333)


client.recreate_collection(
    collection_name="naive_corpus",
    vectors_config=VectorParams(
        size=768,  
        distance=Distance.COSINE
    )
)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  
texts = [chunk.page_content for chunk in chunks]
vectors = embeddings.embed_documents(texts)  


client.upsert(
    collection_name="naive_corpus",
    points=[
        {
            "id": idx,
            "vector": vector,
            "payload": {"text": text}
        }
        for idx, (text, vector) in enumerate(zip(texts, vectors))
    ]
)
print("Data ingested!")