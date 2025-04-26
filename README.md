# Predoc

Preprocess document service for RAG(Retriveal Augumented Generation)

## Usage

### Env setup

We recommend you use `conda` to isolate RAG environment, then install dependencies via `pip`:

```bash
conda create -y --name RAG python=3.10
```

Ensure that you are in RAG env, then install `pytorch` and required dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r /docker/requirements.txt
```

You may need install `tesseract` for OCR capability, see [doc](https://github.com/UB-Mannheim/tesseract/wiki) for installation guide:

```bash
# for ubuntu/debian
apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim libgl1-mesa-glx \
```

Currently, we use Ollama for local LLM inference, see [doc](https://ollama.com/download/windows) for installation guide:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
# we use gemma2:27b as default model
ollama run gemma2:27b
```

Run the service

```bash
python run.py
```

### Install from Docker

```bash
cd docker && docker build -t rag:latest .
docker run rag:latest
```

## Getting started

1. Use API as a tool

2. Use RabbitMQ as a task consumer

## Supported features

- YOLO image recognition for PDF parsing
- LLM for text chunking, rule-based chunking(semantic)
- use `paraphrase-multilingual-mpnet-base-v2` as embedding model
