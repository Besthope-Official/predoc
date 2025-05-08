from prep.processor import PDFProcessor
from task.milvus import search_embedding

query = "Do men exhibit more competitive or more cooperative behavior in male groups? Why is this the case?"

processor = PDFProcessor(
    parse_method="auto",
    chunk_strategy="semantic_ollama",
)

embd = processor.embedding(query)
results = search_embedding(embd)
for result in results:
    print(result)