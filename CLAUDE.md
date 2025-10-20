# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predoc is a document preprocessing service for RAG (Retrieval Augmented Generation) systems. It provides:
- PDF parsing using YOLO-based layout recognition
- Text chunking with LLM and semantic strategies
- Vector embeddings using multilingual sentence transformers
- Two operational modes: FastAPI server OR RabbitMQ task consumer
- Storage integration with MinIO (object storage) and Milvus (vector database)

## Architecture

### Dual Operation Modes

The service operates in one of two modes controlled by `ENABLE_MASSAGE_QUEUE` environment variable:

1. **API Server Mode** (`ENABLE_MASSAGE_QUEUE=false`): Synchronous request/response via FastAPI endpoints
2. **Task Consumer Mode** (`ENABLE_MASSAGE_QUEUE=true`): Asynchronous processing via RabbitMQ task queue

### Core Components

- **`predoc/`**: Core document processing logic
  - `parser.py`: PDF parsing (YoloParser for layout detection)
  - `chunker.py`: Text segmentation (LLMChunker, SemanticChunker)
  - `embedding.py`: Vector embeddings generation
  - `processor.py`: Orchestrates parsing → chunking → embedding
  - `pipeline.py`: Task processing pipelines (e.g., DefaultPDFPipeline)
  - `storage.py`: Storage backend abstraction (MinioStorage, LocalStorage)

- **`backends/`**: Backend service clients (formerly part of `task/`)
  - `rabbitmq.py`: RabbitMQ base connection handling
  - `minio.py`: MinIO object storage operations
  - `milvus.py`: Milvus vector database operations

- **`messaging/`**: Message queue task management (formerly `task/`)
  - `producer.py`: TaskProducer publishes tasks to RabbitMQ
  - `consumer.py`: TaskConsumer processes tasks from queue

- **`api/`**: FastAPI endpoints
  - `api.py`: Main application with routes for `/preprocess`, `/parser`, `/chunker`, `/embedding`, `/retrieval`
  - `utils.py`: ModelLoader singleton for model caching
  - `search.py`: Document retrieval functionality (formerly `retrieve/search.py`)

- **`config/`**: Configuration management (Pydantic-based)
  - `base.py`: BaseConfig foundation with YAML/env support
  - `app.py`: Application settings (AppConfig)
  - `backend.py`: RabbitMQ, MinIO, Milvus connection configs
  - `model.py`: Model and processing parameters (ModelConfig)
  - `api.py`: API-specific configurations

- **`schemas/`**: Pydantic data models
  - `task.py`: Task and TaskStatus models
  - `document.py`: Document metadata schema

### Processing Pipeline Flow

In Task Consumer Mode:
1. TaskProducer publishes Task (containing Document) to RabbitMQ
2. TaskConsumer receives task and instantiates appropriate Pipeline (from `PIPELINE_REGISTRY`)
3. Pipeline processes document:
   - Check if parsed text exists in MinIO prep bucket; if yes, skip parsing
   - Otherwise, download PDF from MinIO, parse with YoloParser
   - Chunk text with configured chunker strategy
   - Generate embeddings with EmbeddingModel
4. Store embeddings + chunks + metadata to Milvus
5. Publish task status updates to result queue

## Common Development Commands

### Environment Setup

```bash
# Create conda environment
conda create -y --name RAG python=3.10
conda activate RAG

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install tesseract for OCR (Ubuntu/Debian)
apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim libgl1-mesa-glx

# Install pre-commit hooks
pre-commit install
```

### Running the Application

```bash
# Start API server (default mode)
python run.py

# Start with custom host/port
python run.py --host 0.0.0.0 --port 8000

# Enable auto-reload for development
python run.py --reload

# Start with multiple workers
python run.py --workers 4

# Task consumer mode: Set enable_message_queue=true in config.yaml, then:
python run.py
```

### Testing

```bash
# Run all tests
pytest

# Run tests excluding benchmarks
pytest -m "not benchmark"

# Run specific test file
pytest tests/test_chunker.py

# Run with verbose output
pytest -v

# Run single test function
pytest tests/test_chunker.py::test_semantic_chunker
```

### Code Quality

```bash
# Format code with black
black .

# Lint with ruff (auto-fix)
ruff check --fix .

# Run pre-commit hooks manually
pre-commit run --all-files
```

### Docker

```bash
# Build CPU version (recommended for most deployments)
DOCKER_BUILDKIT=1 docker build -t predoc:cpu -f docker/dockerfile-cpu .

# Build GPU version
DOCKER_BUILDKIT=1 docker build -t predoc:latest .

# Run with config file and model mounting
docker run -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  predoc:cpu

# Run GPU version
docker run -v $(pwd)/config.yaml:/app/config.yaml --gpus all \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  predoc:latest
```

## Key Design Patterns

### Configuration Management (Updated 2025-10)
- **Unified Pydantic BaseConfig**: All config classes inherit from `config/base.py::BaseConfig`
- **YAML + Environment Variable Hierarchy**:
  1. Load from `config.yaml` if present (via `from_yaml()`)
  2. Fall back to environment variables with `default_factory`
  3. Supports nested sections (e.g., `text_processing.device`)
- **Field Validation**: ModelConfig includes validators for batch_size, chunk_size, etc.
- **Alias Support**: Fields support both snake_case and legacy names (e.g., `access_key`/`access`)
- **Important**: Always use `Config.from_yaml()` for instantiation, not direct class access

**Migration Notes**:
- ❌ Old: `OSSConfig.pdf_bucket` (static access)
- ✅ New: `OSSConfig.from_yaml().pdf_bucket` (instance method)
- Config field names changed to snake_case: `DEVICE` → `device`, `BATCH_SIZE` → `batch_size`

### Storage Backend Abstraction (Added 2025-10)
- **StorageBackend Interface**: Unified abstraction for upload/download/exists operations
- **MinioStorage**: Production backend connecting to MinIO/OSS
  - Smart bucket selection: `"doc/text.txt"` → preprocessed bucket, `"doc.pdf"` → PDF bucket
  - Configuration loaded once via `OSSConfig.from_yaml()`
- **LocalStorage**: Development/testing backend using local filesystem
  - Simulates bucket structure: `base_dir/bucket_name/object_name`
  - No external dependencies, ideal for unit tests
- **Dependency Injection**: Storage backends are injected into Parser and Pipeline
  - `Parser(storage=MinioStorage())` for production
  - `Parser(storage=LocalStorage(Path("/tmp")))` for testing
  - `Parser(storage=None)` for local-only mode (no upload)

### Model Loading
- `ModelLoader` in `api/utils.py` uses singleton pattern with lazy-loading
- Models are cached in memory after first access
- **Parser Access**:
  - `loader.parser`: Property access (backward compatible, no storage)
  - `loader.get_parser(storage=...)`: Method for injecting storage backend
- Use `get_model_loader()` dependency injection in FastAPI routes

### Task Processing
- Pipeline registry pattern in `predoc/pipeline.py`: `PIPELINE_REGISTRY` maps task types to Pipeline classes
- `get_pipeline(task_type)` returns Pipeline class for instantiation
- Custom pipelines can be registered via `add_entry(key, PipelineClass)`

### Parser Classes
- **YoloParser**: The primary PDF parser using YOLO-based layout detection
- No backward compatibility aliases (PDFParser removed as of 2025-10)
- Use `YoloParser` directly for all PDF parsing needs

### Error Handling
- FastAPI uses HTTPException for expected errors with appropriate status codes
- General exceptions are caught by `general_exception_handler` and logged
- Task consumer publishes FAILED status to result queue on errors

### Async Patterns
- FastAPI routes use `async def` for I/O-bound operations
- Task consumer uses ThreadPoolExecutor to prevent blocking RabbitMQ heartbeat during long processing
- Connection callbacks use `add_callback_threadsafe()` for thread safety

## Configuration

The project uses YAML-based configuration with environment variable override support.

**Documentation**:
- **[CONFIGURATION.md](CONFIGURATION.md)**: Complete configuration reference with detailed tables for all sections
- **config.yaml.example**: Annotated example configuration file
- **config.yaml**: Clean production configuration (no comments)

**Quick Setup**:
1. Copy the example: `cp config.yaml.example config.yaml`
2. Edit required fields (see CONFIGURATION.md for details)
3. Environment variables override YAML values (higher priority)

**Configuration Sections**:
- `app`: Application mode and behavior
- `server`: API server settings (host, port, workers)
- `rabbitmq`: Message queue connection (required for consumer mode)
- `milvus`: Vector database connection
- `minio`: Object storage settings (required for consumer mode)
- `models`: AI models (chunking API, embedding, YOLO)
- `text_processing`: Text chunking parameters
- `logging`: Logging configuration

## FastAPI Guidelines (from .cursor/rules)

- Use functional programming style; avoid classes where possible
- Type hints required for all function signatures; prefer Pydantic models over raw dicts
- Early returns for error handling; place happy path last
- Use dependency injection via FastAPI's `Depends()`
- Minimize blocking I/O; use async for database/API calls
- Use lifespan context managers instead of `@app.on_event` decorators

## Docker Best Practices (from .cursor/rules)

- Multi-stage builds are used to minimize image size
- CPU dockerfile uses miniconda base for flexibility
- GPU dockerfile uses pytorch/pytorch base with pre-installed CUDA
- Models are mounted as volumes (`/app/models`) to persist downloads
- Health checks monitor `/health` endpoint every 30s
- Always use `DOCKER_BUILDKIT=1` for builds

---

## Recent Refactoring History

### 2025-10: Dead Code Removal - messaging/preprocess.py

**What Changed**:
- Deleted `messaging/preprocess.py` (38 lines) with zero usage across the codebase

**Rationale**:
The `preprocess()` function in this module was completely superseded by `TaskConsumer._process_task()` and had not been updated to follow recent architectural changes:

| Issue | preprocess.py | TaskConsumer._process_task() |
|-------|---------------|------------------------------|
| **Pipeline Selection** | ❌ Hardcoded `DefaultPDFPipeline` | ✅ Dynamic via `get_pipeline(task_type)` |
| **Storage Injection** | ❌ Missing (2025-10 refactor) | ✅ Full `storage` parameter support |
| **Thread Safety** | ❌ None | ✅ `add_callback_threadsafe()` |
| **Error Handling** | ❌ Simple exception propagation | ✅ Publishes FAILED status to result queue |
| **Flexibility** | ❌ Fixed parameters | ✅ Dynamic collection/partition selection |
| **Usage** | ❌ Zero imports found | ✅ Core consumer logic |

**Benefits**:
- ✅ **Eliminates Confusion**: Prevents accidental use of outdated interface
- ✅ **Reduces Maintenance**: No need to sync with future refactorings
- ✅ **Zero Breaking Changes**: No code references this module
- ✅ **Cleaner Architecture**: Single responsibility in TaskConsumer

**Migration Impact**: None (module had zero usage)

---

### 2025-10: Directory Structure Refactoring - Task & Retrieve Reorganization

**What Changed**:
1. **Directory Restructuring**: Reorganized `task/` and `retrieve/` directories for better clarity
   - **Renamed `task/` → Split into two directories**:
     - `backends/`: Backend service clients (RabbitMQ, MinIO, Milvus)
     - `messaging/`: Message queue task management (Producer, Consumer)
   - **Merged `retrieve/` → `api/`**:
     - `retrieve/search.py` → `api/search.py` (single-file module merged into API layer)

2. **New Directory Structure**:
   - **`backends/`**: Infrastructure clients for external services
     - `rabbitmq.py` (formerly `task/mq.py`)
     - `minio.py` (formerly `task/oss.py`)
     - `milvus.py` (formerly `task/milvus.py`)
   - **`messaging/`**: Task queue processing logic
     - `producer.py` (formerly `task/producer.py`)
     - `consumer.py` (formerly `task/consumer.py`)
     - ~~`preprocess.py` (formerly `task/preprocess.py`)~~ - *removed as dead code*
   - **`api/search.py`**: Document retrieval (formerly `retrieve/search.py`)

3. **Import Path Updates**: All imports updated throughout codebase
   - `from task.mq import` → `from backends.rabbitmq import`
   - `from task.oss import` → `from backends.minio import`
   - `from task.milvus import` → `from backends.milvus import`
   - `from task.producer import` → `from messaging.producer import`
   - `from task.consumer import` → `from messaging.consumer import`
   - `from retrieve.search import` → `from api.search import`

**Rationale**:
- ✅ **Clear Separation of Concerns**: `backends/` contains only client wrappers, `messaging/` contains business logic
- ✅ **Better Discoverability**: Names clearly indicate purpose (backend clients vs task queue)
- ✅ **Reduced Directory Clutter**: Single-file `retrieve/` module merged into existing `api/` directory
- ✅ **Scalability**: Easy to add new backend services (Redis, PostgreSQL) to `backends/`
- ✅ **Consistency**: Aligns with `predoc/storage.py` abstraction pattern

**Migration Guide**:
```python
# Old imports (no longer work)
from task.mq import RabbitMQBase
from task.oss import upload_file, download_file
from task.milvus import store_embedding_task
from task.producer import PDFTaskPublisher
from task.consumer import TaskConsumer
from retrieve.search import retrieve_documents

# New imports (correct)
from backends.rabbitmq import RabbitMQBase
from backends.minio import upload_file, download_file
from backends.milvus import store_embedding_task
from messaging.producer import PDFTaskPublisher
from messaging.consumer import TaskConsumer
from api.search import retrieve_documents
```

**Files Modified**: 15 files across codebase
- New directories: `backends/`, `messaging/`
- Removed directories: `task/`, `retrieve/`
- Updated imports: `predoc/pipeline.py`, `predoc/storage.py`, `api/api.py`, all test files
- Documentation: `CLAUDE.md` updated with new structure

---

### 2025-10: Configuration System Modernization & Backward Compatibility Cleanup

**What Changed**:
1. **Config System Migration**: Migrated all config classes from `@dataclass` to Pydantic `BaseConfig`
   - Supports YAML configuration files with environment variable fallback
   - Added field validation and type safety
   - Unified configuration loading pattern across the codebase

2. **Removed Backward Compatibility Code**:
   - Deleted `PDFParser` alias class (use `YoloParser` directly)
   - Removed `task/task.py` re-export module (use `task.producer`/`task.consumer` directly)
   - Removed deprecated `Task.to_metadata()` and `Task.validate()` methods

3. **Field Naming Standardization**:
   - Config fields renamed to snake_case: `DEVICE` → `device`, `BATCH_SIZE` → `batch_size`
   - OSSConfig: `access`/`secret` → `access_key`/`secret_key` (aliases supported)

**Breaking Changes**:
- Import paths: `from task.task import X` no longer works
- Config access: Must use `Config.from_yaml()` instead of static access
- Field names: Update all references to use snake_case names

**Migration Guide**:
```python
# Old imports (broken)
from task.task import TaskConsumer
from predoc.parser import PDFParser

# New imports (correct)
from messaging.consumer import TaskConsumer
from predoc.parser import YoloParser

# Old config access (broken)
device = CONFIG.DEVICE
bucket = OSSConfig.pdf_bucket

# New config access (correct)
device = CONFIG.device
bucket = OSSConfig.from_yaml().pdf_bucket
```

**Files Modified**: 18 files, +265/-203 lines
- Major: config/backend.py, config/model.py, config/app.py
- Cleanup: predoc/parser.py, schemas/task.py, task/task.py (deleted)

---

### 2025-10: Storage Backend Decoupling & Dependency Injection

**What Changed**:
1. **Storage Abstraction Layer**: Created `predoc/storage.py` with pluggable backends
   - `StorageBackend`: Abstract interface for upload/download/exists
   - `MinioStorage`: Production backend wrapping backends.minio functions
   - `LocalStorage`: Development/testing backend using local filesystem
   - Smart bucket routing: paths with `/` → preprocessed, others → PDF bucket

2. **Parser Refactoring**:
   - Removed direct `task.oss` imports and `upload_to_oss` boolean parameter
   - Added `storage` parameter to `__init__(storage=None)`
   - `_save_and_upload_file()` now conditionally uploads via `self.storage`
   - Removed `upload_to_oss` parameter from `parse()` method signature

3. **Processor Refactoring**:
   - Removed `upload_to_oss` parameter from `__init__()` and all methods
   - Simplified `parse()` call chain (no longer passes upload flag)

4. **Pipeline Refactoring**:
   - Added `storage` parameter to `BasePipeline` and all subclasses
   - Removed direct `OSSConfig.from_yaml()` calls (avoid config reload)
   - Added local file validation when `storage=None`
   - Smart bucket handling via storage backend

5. **ModelLoader Interface**:
   - Restored `parser` as `@property` for backward compatibility
   - Added `get_parser(storage=...)` method for dependency injection
   - Updated `preload_all(storage=...)` to use new method

6. **TaskConsumer Integration**:
   - Initializes `MinioStorage()` instance once
   - Injects storage into Pipeline during instantiation

**Benefits**:
- ✅ **Testability**: Can use `LocalStorage` or mocks in unit tests
- ✅ **Flexibility**: API mode can run without MinIO dependency
- ✅ **Performance**: Config loaded once, not per-operation
- ✅ **Maintainability**: Clear separation of storage concerns
- ✅ **Backward Compatible**: Existing code using `loader.parser` still works

**Migration Guide**:
```python
# Old pattern (tightly coupled to OSS)
parser = YoloParser()
text = parser.parse(pdf_path, output_dir, upload_to_oss=True)

# New pattern (dependency injection)
storage = MinioStorage()  # or LocalStorage(Path("/tmp"))
parser = YoloParser(storage=storage)
text = parser.parse(pdf_path, output_dir)  # auto-uploads if storage set

# Pipeline usage
pipeline = DefaultPDFPipeline(
    model_loader=loader,
    storage=MinioStorage(),  # production
    # storage=LocalStorage(Path("/tmp")),  # testing
    # storage=None,  # local-only mode
)

# ModelLoader (backward compatible)
loader = ModelLoader()
parser1 = loader.parser  # ✓ old code works
parser2 = loader.get_parser(storage=storage)  # ✓ new code with DI
```

**Files Modified**: 8 files, +280/-150 lines
- New: predoc/storage.py
- Major: predoc/parser.py, predoc/processor.py, predoc/pipeline.py
- Integration: messaging/consumer.py, api/utils.py
- Tests: tests/test_pipeline.py (updated for new architecture)

---

### 2025-10: Configuration Documentation Refactoring

**What Changed**:
1. **Documentation Structure**: Separated configuration documentation from config files
   - Created **CONFIGURATION.md**: Comprehensive 379-line configuration reference
   - Simplified **config.yaml**: Removed all comments, reduced from 214 to 79 lines (62% reduction)
   - Retained **config.yaml.example**: Detailed example with inline annotations (209 lines)

2. **Documentation Content**:
   - 8 detailed configuration section tables with field types, requirements, and defaults
   - Clear marking of required vs optional fields
   - Environment variable mapping for each configuration field
   - Multiple configuration examples (API mode, Consumer mode, Development)
   - Docker deployment configuration guide
   - Troubleshooting section with common errors
   - Migration guide from .env to config.yaml

3. **README Integration**:
   - Added "Configuration" section with links to CONFIGURATION.md
   - Added "Getting Started" guide with setup instructions
   - Added table of contents for better navigation

**Benefits**:
- ✅ **Cleaner Config Files**: Production config.yaml is now concise and easy to read
- ✅ **Better Documentation**: Comprehensive reference with searchable tables
- ✅ **Improved Onboarding**: Clear setup instructions for new users
- ✅ **Maintainability**: Single source of truth for configuration documentation
- ✅ **Developer Experience**: Easy to find specific configuration options

**Documentation Structure**:
- **CONFIGURATION.md**:
  - Quick Start guide
  - Configuration Priority explanation
  - 8 section tables (app, server, rabbitmq, milvus, minio, models, text_processing, logging)
  - 3 complete configuration examples
  - Environment variable override guide
  - Docker deployment examples
  - Troubleshooting and migration guides
- **config.yaml**: Clean production configuration (79 lines)
- **config.yaml.example**: Annotated example (209 lines)
- **README.md**: Quick setup with links to detailed docs

**Files Modified**: 4 files, +379 new/-135 removed lines
- New: CONFIGURATION.md (+379 lines)
- Updated: config.yaml (-135 lines, simplified)
- Updated: README.md (+50 lines, added configuration section)
- Updated: CLAUDE.md (this file, documentation)
