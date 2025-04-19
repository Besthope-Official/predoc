# Predoc

Preprocess document service for RAG(Retriveal Augumented Generation)

## Usage

```bash
cd docker && docker build -t rag:latest .
docker run rag:latest
```

1. Use API as a tool

2. Use RabbitMQ as a task consumer

## Supported features

- YOLO image recognition for PDF parsing
- LLM for text chunking, rule-based chunking(semantic)
- use `paraphrase-multilingual-mpnet-base-v2` as embedding model
