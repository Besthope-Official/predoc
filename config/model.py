import os
import torch
from typing import ClassVar
from pydantic import Field, field_validator, model_validator
from loguru import logger
from .base import BaseConfig

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class ModelConfig(BaseConfig):
    """模型相关配置（解析、分块、嵌入）"""

    yaml_section: ClassVar[str] = "text_processing"

    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    chunk_output_dir: str = Field(
        default_factory=lambda: os.getenv("CHUNK_OUTPUT_DIR", "./output/chunks")
    )
    chunks_file: str = Field(default_factory=lambda: os.getenv("CHUNKS_FILE", ""))
    batch_size: int = Field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "4")))
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "16"))
    )
    min_chunk_length: int = Field(
        default_factory=lambda: int(os.getenv("MIN_CHUNK_LENGTH", "50"))
    )
    max_length: int = Field(default_factory=lambda: int(os.getenv("MAX_LENGTH", "512")))
    encoding: str = Field(default="utf-8-sig")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    embedding_model_dir: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_DIR", "./models/embedding")
    )
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2"
        )
    )
    embedding_hf_repo_id: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_HF_REPO_ID",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
    )
    yolo_model_dir: str = Field(
        default_factory=lambda: os.getenv("YOLO_MODEL_DIR", "./models/YOLOv10")
    )
    yolo_model_filename: str = Field(
        default_factory=lambda: os.getenv(
            "YOLO_MODEL_FILENAME", "doclayout_yolo_docstructbench_imgsz1024.pt"
        )
    )
    yolo_hf_repo_id: str = Field(
        default_factory=lambda: os.getenv(
            "YOLO_HF_REPO_ID", "juliozhao/DocLayout-YOLO-DocStructBench"
        )
    )

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if not 1 <= v <= 64:
            raise ValueError("batch_size 应在 1-64 之间")
        return v

    @field_validator("min_chunk_length")
    @classmethod
    def validate_min_chunk_length(cls, v: int) -> int:
        if not v > 0:
            raise ValueError("min_chunk_length 应大于 0")
        return v

    @model_validator(mode="after")
    def validate_chunk_config(self):
        if not 1 <= self.chunk_size <= 4096:
            raise ValueError("chunk_size 应在 1-4096 之间")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap 应在 0 到 chunk_size 之间")
        logger.info(f"配置加载成功: embedding_model={self.embedding_model}")
        return self

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


try:
    CONFIG = ModelConfig.from_yaml()
except Exception as e:
    logger.error(f"配置初始化失败: {e}")
    raise
