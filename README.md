# RAG Lab: Naive vs Advanced RAG Comparison

This repository contains the implementation of two RAG (Retrieval-Augmented Generation) pipelines:
1. **Naive RAG**: A minimal implementation with basic chunking, embeddings, and generation.
2. **Advanced RAG**: Enhanced with better chunking, query rewriting, reranking, and source citation.

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker Desktop
- Ubuntu 22.04 LTS (WSL2 or VM)

### Step 1: Environment Setup

Clone this repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-lab.git
cd rag-lab

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Start Qdrant Vector Database

Option 1: Using Docker command:
```bash
docker pull qdrant/qdrant:latest
docker run -d --name qdrant -p 6333:6333 -v ${PWD}/qdrant_data:/qdrant/storage qdrant/qdrant:latest
```

Option 2: Using Docker Compose:
```bash
docker-compose up -d
```

### Step 4: Set Up API Keys

Create a `.env` file in the root directory with your API keys:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Load environment variables:
```bash
export $(cat .env | xargs)
```

## Running the Pipelines

### Step 1: Data Ingestion

For Naive RAG:
```bash
python src/ingest.py
```

For Advanced RAG:
```bash
python src/ingest_advanced.py
```

### Step 2: Query the Systems

For Naive RAG:
```bash
python src/query.py
```

For Advanced RAG:
```bash
python src/query_advanced.py
```

### Step 3: Evaluate the Results

After generating results with both pipelines:
```bash
python src/evaluate.py
```

This will:
1. Calculate Ragas metrics for both pipelines
2. Generate a comparison chart
3. Save results to `evaluation.csv`

## Advanced RAG Improvements

This implementation includes several improvements:

1. **Recursive Chunking**: Using 200-token chunks with 50-token overlap
2. **Metadata Filtering**: Storing and filtering by document source
3. **Query Rewriting**: Using LLM to reformulate queries for better retrieval
4. **Reranking**: Applying cross-encoder reranking to improve result relevance
5. **Source Citation**: Adding source references to improve traceability

## Directory Structure

```
rag-lab/
├── data/
│   └── raw/         # Your corpus documents go here
├── notebooks/
│   ├── naive_rag.ipynb
│   └── advanced_rag.ipynb
├── src/
│   ├── ingest.py
│   ├── ingest_advanced.py
│   ├── query.py
│   ├── query_advanced.py
│   └── evaluate.py
├── docker-compose.yml
├── evaluation.csv
├── rag_comparison.png
├── requirements.txt
└── README.md
```

## Maintaining Qdrant Data Volumes

The Docker setup mounts the Qdrant data directory to `./qdrant_data` on your host machine. This ensures data persistence between Docker restarts. If you need to rebuild Qdrant:

```bash
# Stop and remove the container
docker stop qdrant
docker rm qdrant

# Start it again
docker run -d --name qdrant -p 6333:6333 -v ${PWD}/qdrant_data:/qdrant/storage qdrant/qdrant:latest
```

Alternatively, using Docker Compose:
```bash
docker-compose down
docker-compose up -d
```

## Troubleshooting

- **Qdrant Connection Issues**: Ensure Docker is running and the Qdrant container is up (`docker ps`)
- **Missing Dependencies**: Check requirements.txt and install any missing packages
- **API Rate Limits**: If using free tiers of Gemini API, you might hit rate limits