import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict
from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


@dataclass
class ModelConfig:
    DEBUG: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    CHUNK_OUTPUT_DIR: str = os.getenv("CHUNK_OUTPUT_DIR", "./output/chunks")
    CHUNKS_FILE: str = field(default_factory=lambda: os.getenv("CHUNKS_FILE", ""))
    BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "4")))
    CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    CHUNK_OVERLAP: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "16"))
    )
    MIN_CHUNK_LENGTH: int = field(
        default_factory=lambda: int(os.getenv("MIN_CHUNK_LENGTH", "50"))
    )
    MAX_LENGTH: int = field(default_factory=lambda: int(os.getenv("MAX_LENGTH", "512")))
    ENCODING: str = os.getenv("ENCODING", "utf-8-sig")
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_MODEL_DIR: str = os.getenv("EMBEDDING_MODEL_DIR", "./models/embedding")
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2"
    )
    EMBEDDING_HF_REPO_ID: str = os.getenv(
        "EMBEDDING_HF_REPO_ID",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )

    YOLO_MODEL_DIR: str = os.getenv("YOLO_MODEL_DIR", "./models/YOLOv10")
    YOLO_MODEL_FILENAME: str = os.getenv(
        "YOLO_MODEL_FILENAME", "doclayout_yolo_docstructbench_imgsz1024.pt"
    )
    YOLO_HF_REPO_ID: str = os.getenv(
        "YOLO_HF_REPO_ID", "juliozhao/DocLayout-YOLO-DocStructBench"
    )

    def validate_path(self, path: str, needs_write: bool = True) -> str:
        dir_path = os.path.dirname(path) or path
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            if not os.access(dir_path, os.R_OK) or (
                needs_write and not os.access(dir_path, os.W_OK)
            ):
                raise PermissionError(f"目录 {dir_path} 无读写权限")
            if os.path.isfile(path) and needs_write and not os.access(path, os.W_OK):
                raise PermissionError(f"文件 {path} 无写权限")
            return path
        except OSError as e:
            logger.error(f"处理路径 {path} 失败: {e}")
            raise

    def __post_init__(self):
        try:
            if not 1 <= self.BATCH_SIZE <= 64:
                raise ValueError("BATCH_SIZE 应在 1-64 之间")
            if not 1 <= self.CHUNK_SIZE <= 4096:
                raise ValueError("CHUNK_SIZE 应在 1-4096 之间")
            if not 0 <= self.CHUNK_OVERLAP < self.CHUNK_SIZE:
                raise ValueError("CHUNK_OVERLAP 应在 0 到 CHUNK_SIZE 之间")
            if not self.MIN_CHUNK_LENGTH > 0:
                raise ValueError("MIN_CHUNK_LENGTH 应大于 0")
        except ValueError as e:
            logger.error(f"参数验证失败: {e}")
            raise

        logger.info(f"配置加载成功: EMBEDDING_MODEL={self.EMBEDDING_MODEL}")


try:
    CONFIG = ModelConfig()
except Exception as e:
    logger.error(f"配置初始化失败: {e}")
    raise
