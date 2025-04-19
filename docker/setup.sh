pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip install \
    faiss-cpu \
    pymupdf \
    tqdm \
    spacy \
    scikit-learn \
    numpy \
    pandas \
    python-docx \
    camelot-py[base] \
    pytesseract \
    pdf2image \
    fastapi \
    uvicorn \
    langchain \
    sentence-transformers \
    opencv-python \
    loguru \
    doclayout-yolo \
    minio \
    pika

python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm