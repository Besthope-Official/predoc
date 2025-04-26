# DOCKER_BUILDKIT=1 docker build -t rag:latest .

ARG CUDA_VERSION=11.8.0
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    sed -i 's/http:\/\/security.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git poppler-utils ghostscript tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -y --name rag-env python=3.10 && \
    conda clean -afy

SHELL ["conda", "run", "-n", "rag-env", "/bin/bash", "-c"]

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install Python dependencies with cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

WORKDIR /app
COPY . /app/

EXPOSE 8000
CMD ["conda", "run", "-n", "rag-env", "python", "run.py"]