# Docker Image Build Guide

This directory contains Dockerfiles for building the predoc application with different configurations.

## Image Variants

### Default CUDA Version (dockerfile)

- Base image: `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime`
- Includes CUDA support for GPU acceleration
- Uses global pip installation
- Recommended for production deployment with GPU support

```bash
DOCKER_BUILDKIT=1 docker build -t predoc:latest .
```

### CPU Version (dockerfile-cpu)

- Base image: `continuumio/miniconda3:latest`
- Uses PyTorch CPU version from `https://download.pytorch.org/whl/cpu`
- Uses Miniconda for package management
- Suitable for development or deployment without GPU requirements

```bash
DOCKER_BUILDKIT=1 docker build -t predoc:cpu -f docker/dockerfile-cpu .
```

### CUDA Version (dockerfile-cuda)

- Same as default version but with explicit CUDA designation
- Base image: `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime`
- Uses global pip installation
- Recommended when you need to explicitly specify CUDA support

```bash
DOCKER_BUILDKIT=1 docker build -t predoc:cuda -f dockerfile-cuda .
```

## Package Management

### GPU Versions (default and cuda)

- Uses the Python environment from PyTorch base image
- Global pip installation for simplicity and smaller image size
- PyTorch and CUDA dependencies pre-installed in base image

### CPU Version

- Uses Miniconda for Python environment management
- Conda environment: `predoc`
- PyTorch CPU version installed via pip

## Usage

### Model Mounting

The application requires pre-trained models to function. These models can be mounted from the host machine:

```bash
models/
├── embedding/                # Sentence transformer models
│   └── paraphrase-multilingual-mpnet-base-v2/
└── YOLOv10/                  # YOLO models
    └── doclayout_yolo_docstructbench_imgsz1024.pt
```

Models will be automatically downloaded to these directories on first run if not present. The directories are mounted as Docker volumes to persist the models between container restarts.

### Running the Container

Note that you need to set the environment variables when running the container. `env-example` is provided.

```bash
# Run with GPU support and model mounting (default/cuda version)
docker run --env-file .env --gpus all \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  --name predoc \
  predoc:latest

# Run CPU version with model mounting
docker run --env-file .env \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  --name predoc \
  predoc:cpu

# Using docker-compose (recommended)
docker-compose up -d
```

### Environment Variables

The following environment variables control model paths:

- `EMBEDDING_MODEL_DIR`: Path to embedding models (default: `/app/models/embedding`)
- `YOLO_MODEL_DIR`: Path to YOLO models (default: `/app/models/YOLOv10`)
- `DEBIAN_FRONTEND=noninteractive`: Ensures non-interactive package installation
- `CONDA_DIR=/opt/conda`: Miniconda installation directory (CPU version only)

### Exposed Ports

- Port 8000: Main application port

### Health Check

All images include a health check that monitors the application:
- Interval: 30s
- Timeout: 30s
- Start period: 5s
- Retries: 3
- Endpoint: `/health`

## Development

### Building with Cache

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain .
```

### Cleaning Up

```bash
# Remove all unused images
docker image prune -a

# Remove specific image
docker rmi predoc:latest
```

## Troubleshooting

### Common Issues

1. GPU not detected:
   - Ensure NVIDIA drivers are installed
   - Verify docker-nvidia runtime is installed
   - Use `--gpus all` flag when running

2. Build failures:
   - Enable BuildKit: `export DOCKER_BUILDKIT=1`
   - Clear build cache: `docker builder prune`
   - Try using a mirror: Add `{"registry-mirrors": ["https://registry.cn-hangzhou.aliyuncs.com"]}` to `/etc/docker/daemon.json`

### Logs

- Container logs: `docker logs <container_id>`
- Build logs: Use `--progress=plain` during build
