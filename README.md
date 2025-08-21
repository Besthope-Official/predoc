# Predoc

Preprocess document service for RAG (Retriveal Augumented Generation)

## Usage

### Dev Env setup

We recommend you use `conda` to isolate RAG environment, then install dependencies via `pip`:

```bash
conda create -y --name RAG python=3.10
```

Ensure that you are in RAG env, then install `pytorch` and required dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

You may need install `tesseract` for OCR capability, see [doc](https://github.com/UB-Mannheim/tesseract/wiki) for installation guide:

```bash
# for ubuntu/debian
apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim libgl1-mesa-glx \
```

For LLMChunker, you can use `ollama`(set it as default backend) to run the model locally, see [doc](https://ollama.com/docs/quickstart) for installation guide.

```python
from prep.chunker import LLMChunker

chunker = LLMChunker(
    backend='ollama',
    ollama_api_host='http://127.0.0.1:11434',
    model_name='qwen3:4b'
)
```

You can also use `api` backend to use any OpenAI-compatible API, e.g. you can use vLLM to deploy a model on your own server.

For contributor, install git hook before you commit:

```bash
pre-commit install
```

### Install from Docker

We provide CPU/GPU Docker images to suit different deployment needs. See [Docker Build Guide](docker/README.md) for detailed documentation.

We suggest you use CPU version as it not only provides smaller image size but also almost same performance (yolo-parsing and embedding will not be bottleneck).

## Getting started

1. Use API as a tool

check `/docs` for API documentation.

2. Use RabbitMQ as a task consumer

## Supported features

- YOLO image recognition for PDF parsing
- LLM for text chunking, rule-based chunking(semantic)
- use `paraphrase-multilingual-mpnet-base-v2` as embedding model

## TODOs

- paddleocr support
-
