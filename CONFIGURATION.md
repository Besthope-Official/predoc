# Configuration Guide

This document provides detailed configuration reference for the PreDoc service.

## Quick Start

1. Copy the example configuration file:
   ```bash
   cp config.yaml.example config.yaml
   ```

2. Edit required fields (marked as **Required** below)

3. Start the service:
   ```bash
   python run.py
   ```

## External Services

- Milvus(Required): vector db backend
- MinIO: external storage for parsing, can be set to localStorage
- RabbitMQ: message queue for async mode

## Configuration Priority

Configuration values are loaded in the following order (higher priority overrides lower):

1. **Environment variables** (highest priority)
2. **config.yaml** file
3. **Default values** (lowest priority)

## Configuration Sections

### 1. Application Settings (`app`)

Core application behavior configuration.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `env` | string | No | `prod` | Environment mode: `dev`, `test`, or `prod` |
| `enable_message_queue` | boolean | **Yes** | `false` | Operation mode:<br>• `false`: API server (sync)<br>• `true`: RabbitMQ consumer (async) |
| `enable_parallelism` | boolean | No | `true` | Enable parallel text processing |
| `parse_method` | string | No | `auto` | PDF parsing method:<br>• `auto`: Automatic selection<br>• `yolo`: Force YOLO parser |
| `chunk_strategy` | string | No | `semantic_api` | Text chunking strategy:<br>• `semantic_api`: LLM-based<br>• `sentence`: Sentence-based |
| `upload_to_oss` | boolean | No | `true` | Upload parsed results to object storage |

**Environment Variables:**
- `ENV`, `ENABLE_MASSAGE_QUEUE`, `PARSE_METHOD`, `CHUNK_STRATEGY`, `UPLOAD_TO_OSS`, `ENABLE_PARALLELISM`

---

### 2. Server Settings (`server`)

API server configuration (only for API mode).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `host` | string | No | `0.0.0.0` | Server listening address |
| `port` | integer | No | `8000` | Server listening port |
| `workers` | integer | No | `1` | Number of worker processes |
| `reload` | boolean | No | `false` | Auto-reload on code changes (dev only) |
| `backend_host` | string | No | `""` | Backend server URL (if applicable) |

**Environment Variables:**
- `SERVER_HOST`, `SERVER_PORT`, `SERVER_WORKERS`, `SERVER_RELOAD`

---

### 3. Message Queue (`rabbitmq`)

RabbitMQ configuration (only required when `enable_message_queue=true`).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `host` | string | **Yes*** | `127.0.0.1` | RabbitMQ server address |
| `port` | integer | No | `5672` | RabbitMQ server port |
| `user` | string | **Yes*** | `admin` | RabbitMQ username |
| `password` | string | **Yes*** | `admin` | RabbitMQ password |
| `task_queue` | string | No | `taskQueue` | Task queue name |
| `result_queue` | string | No | `respQueue` | Result queue name |
| `consumer_workers` | integer | No | `4` | Number of consumer worker threads |

**Required when `enable_message_queue=true`*

**Environment Variables:**
- `RABBITMQ_HOST`, `RABBITMQ_PORT`, `RABBITMQ_USER`, `RABBITMQ_PASSWORD`, `RABBITMQ_TASK_QUEUE`, `RABBITMQ_RESULT_QUEUE`, `RABBITMQ_CONSUMER_WORKERS`

---

### 4. Vector Database (`milvus`)

Milvus vector database configuration.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `host` | string | **Yes** | `127.0.0.1` | Milvus server address |
| `port` | integer | No | `19530` | Milvus server port |
| `user` | string | **Yes** | `root` | Milvus username |
| `password` | string | **Yes** | `Milvus` | Milvus password |
| `db_name` | string | No | `default` | Database name |
| `default_collection` | string | No | `default_collection` | Default collection name |
| `default_partition` | string | No | `default_partition` | Default partition name |

**Environment Variables:**
- `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_USER`, `MILVUS_PASSWORD`, `MILVUS_DB`, `MILVUS_DEFAULT_COLLECTION`, `MILVUS_DEFAULT_PARTITION`

---

### 5. Object Storage (`minio`)

MinIO object storage configuration (only required when `enable_message_queue=true`).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `endpoint` | string | **Yes*** | `127.0.0.1:9000` | MinIO server endpoint (with port) |
| `access_key` | string | **Yes*** | `minioadmin` | MinIO access key |
| `secret_key` | string | **Yes*** | `minioadmin` | MinIO secret key |
| `preprocessed_files_bucket` | string | No | `prep` | Bucket for parsed files |
| `pdf_bucket` | string | No | `mybucket` | Bucket for original PDFs |
| `secure` | boolean | No | `false` | Use HTTPS connection |

**Required when `enable_message_queue=true`*

**Environment Variables:**
- `MINIO_ENDPOINT`, `MINIO_ACCESS`, `MINIO_SECRET`, `PREPROCESSED_FILES_BUCKET`, `PDF_BUCKET`

---

### 6. AI Models (`models`)

AI model configuration for document processing.

#### 6.1 Chunking Model (`models.chunking`)

LLM API configuration for semantic chunking.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | string | **Yes*** | - | API key for LLM service |
| `api_url` | string | **Yes*** | `http://127.0.0.1:8000` | API endpoint URL |
| `model_name` | string | **Yes*** | `moonshot-v1-8k` | Model identifier |
| `max_qps` | integer | No | `10` | Max requests per second |

**Required when `chunk_strategy=semantic_api`*

**Supported Providers:**
- **OpenAI**: `https://api.openai.com/v1`
- **Moonshot AI**: `https://api.moonshot.cn/v1`
- **DeepSeek**: `https://api.deepseek.com/v1`
- **Ollama** (local): `http://localhost:11434/v1`

**Environment Variables:**
- `CHUNK_API_KEY`, `CHUNK_API_URL`, `CHUNK_MODEL_NAME`, `MAX_CHUNK_QPS`

#### 6.2 Embedding Model (`models.embedding`)

Sentence transformer model for text embeddings.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_name` | string | **Yes** | `paraphrase-multilingual-mpnet-base-v2` | HuggingFace model name |

**Popular Models:**
- `paraphrase-multilingual-mpnet-base-v2` (multilingual, 768-dim)
- `all-MiniLM-L6-v2` (English, 384-dim, fast)
- `all-mpnet-base-v2` (English, 768-dim, accurate)

**Environment Variables:**
- `EMBEDDING_MODEL_NAME`, `EMBEDDING_HF_REPO_ID`

#### 6.3 YOLO Model (`models.yolo`)

Document layout detection model.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_path` | string | No | `models/YOLOv10/doclayout_yolo_docstructbench_imgsz1024.pt` | Path to YOLO weights |
| `device` | string | No | `cpu` (auto-detect) | Inference device:<br>• `cpu`: CPU<br>• `cuda`: NVIDIA GPU<br>• `mps`: Apple Silicon |
| `image_size` | integer | No | `1024` | Input image size |
| `confidence` | float | No | `0.25` | Detection confidence threshold |

**Environment Variables:**
- `YOLO_MODEL_DIR`, `YOLO_MODEL_FILENAME`, `YOLO_HF_REPO_ID`

---

### 7. Text Processing (`text_processing`)

Text chunking and processing parameters.

| Field | Type | Required | Default | Validation | Description |
|-------|------|----------|---------|------------|-------------|
| `min_chunk_length` | integer | No | `100` | > 0 | Minimum chunk length (chars) |
| `max_chunk_length` | integer | No | `2048` | - | Maximum chunk length (chars) |
| `chunk_size` | integer | No | `1024` | 1-4096 | Target chunk size (chars) |
| `chunk_overlap` | integer | No | `128` | 0 to chunk_size | Overlap between chunks (chars) |

**Environment Variables:**
- `MIN_CHUNK_LENGTH`, `MAX_CHUNK_LENGTH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`

---

### 8. Logging (`logging`)

Application logging configuration.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `level` | string | No | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `format` | string | No | (loguru format) | Log message format |
| `file_enabled` | boolean | No | `false` | Enable logging to file |
| `file_path` | string | No | `logs/predoc.log` | Log file path |
| `rotation` | integer | No | `10` | Log rotation size (MB) |
| `retention` | integer | No | `30` | Log retention period (days) |

**Environment Variables:**
- `LOG_LEVEL`, `LOG_FILE_ENABLED`, `LOG_FILE_PATH`

---

## Configuration Examples

### API Server Mode (Minimal)

```yaml
app:
  enable_message_queue: false

milvus:
  host: localhost
  user: root
  password: Milvus

models:
  embedding:
    model_name: paraphrase-multilingual-mpnet-base-v2
```

### Task Consumer Mode (Production)

```yaml
app:
  env: prod
  enable_message_queue: true
  chunk_strategy: semantic_api

rabbitmq:
  host: rabbitmq.example.com
  user: admin
  password: secure_password

milvus:
  host: milvus.example.com
  user: root
  password: secure_password

minio:
  endpoint: https://minio.example.com:9000
  access_key: production_key
  secret_key: production_secret
  secure: true

models:
  chunking:
    api_key: sk-xxxxxxxxxxxxxxxx
    api_url: https://api.openai.com/v1
    model_name: gpt-3.5-turbo
  embedding:
    model_name: paraphrase-multilingual-mpnet-base-v2
```

### Development with Local Services

```yaml
app:
  env: dev
  enable_message_queue: false

server:
  reload: true

milvus:
  host: localhost

models:
  chunking:
    api_key: ollama
    api_url: http://localhost:11434/v1
    model_name: llama2
  yolo:
    device: cpu

logging:
  level: DEBUG
  file_enabled: true
```

---

## Environment Variable Override

You can override any configuration using environment variables:

```bash
# Override RabbitMQ host
export RABBITMQ_HOST=prod-rabbitmq.example.com

# Override chunk API key
export CHUNK_API_KEY=sk-production-key

# Override Milvus connection
export MILVUS_HOST=milvus-prod
export MILVUS_PASSWORD=secure_password

# Start service
python run.py
```

---

## Docker Deployment

Mount configuration file and override with environment variables:

```bash
docker run \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e MILVUS_HOST=milvus-service \
  -e MINIO_ENDPOINT=http://minio-service:9000 \
  -e CHUNK_API_KEY=sk-xxxxxxxx \
  -p 8000:8000 \
  predoc:latest
```

---

## Troubleshooting

### Configuration Loading Issues

If configuration fails to load:

1. Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`
2. Verify file permissions: `ls -la config.yaml`
3. Check logs for specific errors
4. Use environment variables to override problematic values

### Common Errors

**Error: `CHUNK_API_KEY` not set**
- Set the API key in `models.chunking.api_key` or via `CHUNK_API_KEY` environment variable

**Error: Cannot connect to Milvus**
- Verify `milvus.host` and `milvus.port` are correct
- Check network connectivity: `telnet <host> <port>`

**Error: MinIO bucket not found**
- Ensure buckets exist or service has permission to create them
- Check `minio.access_key` and `minio.secret_key`

---

## Migration from .env

If you're migrating from `.env` to `config.yaml`:

1. Map environment variables to YAML structure (see tables above)
2. Use nested structure: `MILVUS_HOST` → `milvus.host`
3. Keep sensitive values in environment variables
4. Remove `.env` file after migration

**Example:**
```bash
# Old .env
MILVUS_HOST=localhost
CHUNK_API_KEY=sk-xxx

# New config.yaml
milvus:
  host: localhost
models:
  chunking:
    api_key: ${CHUNK_API_KEY}  # Reference env var
```
