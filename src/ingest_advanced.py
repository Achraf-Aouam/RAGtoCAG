from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rank_bm25 import BM25Okapi
import pickle

# Load documents
loader = DirectoryLoader("data/raw", glob="**/*.txt")
documents = loader.load()

# Recursive chunking
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

texts = [chunk.page_content for chunk in chunks]

# Tokenize documents for BM25 (simple whitespace split)
tokenized_docs = [doc.split() for doc in texts]  # Split by spaces

# Build BM25 model
bm25 = BM25Okapi(tokenized_docs)

# Save BM25 model to disk
with open("bm25_model.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("BM25 model saved!")


# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# Clean up existing collection
try:
    client.delete_collection("advanced_corpus")
except Exception as e:
    print(f"Collection deletion warning: {e}")

# Create new collection with hybrid config
# client.create_collection(
#     collection_name="advanced_corpus",
#     vectors_config=models.VectorParams(
#         size=768,
#         distance=models.Distance.COSINE,
#     ),
#     sparse_vectors_config={
#         "text": models.SparseVectorParams(
#             index=models.SparseIndexParams(
#                 on_disk=False,
#                 full_scan_threshold=10000
#             )
#         )
#     }
# )
client.create_collection(
    collection_name="advanced_corpus",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
    ),
    sparse_vectors_config={
        "bm25": models.SparseVectorParams()  # Name your sparse vector
    }
)

# Embed and upload
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
texts = [chunk.page_content for chunk in chunks]
vectors = embeddings.embed_documents(texts)

client.upsert(
    collection_name="advanced_corpus",
    points=[
        models.PointStruct(
            id=idx,
            vector=vector,
            payload={"text": text, "source": "wikipedia"}
        )
        for idx, (text, vector) in enumerate(zip(texts, vectors))
    ]
)

print("Advanced corpus created successfully!")