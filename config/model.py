import os
from dataclasses import dataclass, field
from typing import List
from loguru import logger


@dataclass
class ModelConfig:
    DATA_DIRS: List[str] = field(default_factory=lambda: [d.strip() for d in os.getenv(
        "DATA_DIRS", "./AI预实验文献/中文/,./AI预实验文献/英文/").split(",") if d.strip()])
    QUESTIONS_FILE: str = os.getenv("QUESTIONS_FILE", "./questions.json")
    INDEX_DIR: str = os.getenv("INDEX_DIR", "./data/indices")
    CHUNK_OUTPUT_DIR: str = os.getenv("CHUNK_OUTPUT_DIR", "./output/chunks")
    RESULT_OUTPUT_DIR: str = os.getenv("RESULT_OUTPUT_DIR", "./output/results")
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")
    CHUNKS_FILE: str = field(
        default_factory=lambda: os.getenv("CHUNKS_FILE", ""))
    CONFIG_FILE: str = field(
        default_factory=lambda: os.getenv("CONFIG_FILE", ""))
    BATCH_SIZE: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "4")))
    CHUNK_SIZE: int = field(default_factory=lambda: int(
        os.getenv("CHUNK_SIZE", "512")))
    CHUNK_OVERLAP: int = field(default_factory=lambda: int(
        os.getenv("CHUNK_OVERLAP", "16")))
    MIN_CHUNK_LENGTH: int = field(default_factory=lambda: int(
        os.getenv("MIN_CHUNK_LENGTH", "50")))
    ENCODING: str = os.getenv("ENCODING", "utf-8-sig")
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    SEMANTIC_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("SEMANTIC_THRESHOLD", "0.6")))
    SEMANTIC_BUFFER_SIZE: int = field(
        default_factory=lambda: int(os.getenv("SEMANTIC_BUFFER_SIZE", "3")))
    HNSW_EF_CONSTRUCTION: int = field(default_factory=lambda: int(
        os.getenv("HNSW_EF_CONSTRUCTION", "128")))
    HNSW_EF_SEARCH: int = field(default_factory=lambda: int(
        os.getenv("HNSW_EF_SEARCH", "64")))
    QUERY_STRATEGY: str = "combined"
    MAX_LENGTH: int = field(default_factory=lambda: int(
        os.getenv("MAX_LENGTH", "512")))

    def validate_path(self, path: str, needs_write: bool = True) -> str:
        dir_path = os.path.dirname(path) or path
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            if not os.access(dir_path, os.R_OK) or (needs_write and not os.access(dir_path, os.W_OK)):
                raise PermissionError(f"目录 {dir_path} 无读写权限")
            if os.path.isfile(path) and needs_write and not os.access(path, os.W_OK):
                raise PermissionError(f"文件 {path} 无写权限")
            return path
        except OSError as e:
            logger.error(f"处理路径 {path} 失败: {e}")
            raise

    def __post_init__(self):
        if not self.CHUNKS_FILE:
            self.CHUNKS_FILE = os.path.join(
                self.CHUNK_OUTPUT_DIR, "chunks_metadata.pkl")
        if not self.CONFIG_FILE:
            self.CONFIG_FILE = os.path.join(self.INDEX_DIR, "config.json")

        self.DATA_DIRS = [self.validate_path(
            d) for d in self.DATA_DIRS if os.path.isdir(d)]
        if not self.DATA_DIRS:
            logger.error("无有效 DATA_DIRS，程序无法继续")
            raise ValueError("无有效可读写的数据目录")
        self.QUESTIONS_FILE = self.validate_path(
            self.QUESTIONS_FILE, needs_write=False)
        self.INDEX_DIR = self.validate_path(self.INDEX_DIR)
        self.CHUNK_OUTPUT_DIR = self.validate_path(self.CHUNK_OUTPUT_DIR)
        self.RESULT_OUTPUT_DIR = self.validate_path(self.RESULT_OUTPUT_DIR)
        self.LOG_DIR = self.validate_path(self.LOG_DIR)
        self.CHUNKS_FILE = self.validate_path(self.CHUNKS_FILE)
        self.CONFIG_FILE = self.validate_path(self.CONFIG_FILE)

        try:
            if not 1 <= self.BATCH_SIZE <= 64:
                raise ValueError("BATCH_SIZE 应在 1-64 之间")
            if not 0 < self.SEMANTIC_THRESHOLD <= 1.0:
                raise ValueError("SEMANTIC_THRESHOLD 应在 0-1 之间")
            if not 1 <= self.CHUNK_SIZE <= 4096:
                raise ValueError("CHUNK_SIZE 应在 1-4096 之间")
            if not 0 <= self.CHUNK_OVERLAP < self.CHUNK_SIZE:
                raise ValueError("CHUNK_OVERLAP 应在 0 到 CHUNK_SIZE 之间")
            if not self.MIN_CHUNK_LENGTH > 0:
                raise ValueError("MIN_CHUNK_LENGTH 应大于 0")
            if not self.HNSW_EF_CONSTRUCTION > 0:
                raise ValueError("HNSW_EF_CONSTRUCTION 应大于 0")
            if not self.HNSW_EF_SEARCH > 0:
                raise ValueError("HNSW_EF_SEARCH 应大于 0")
            if not self.SEMANTIC_BUFFER_SIZE >= 0:
                raise ValueError("SEMANTIC_BUFFER_SIZE 应非负")
        except ValueError as e:
            logger.error(f"参数验证失败: {e}")
            raise

        logger.info(
            f"配置加载成功: DATA_DIRS={self.DATA_DIRS}, EMBEDDING_MODEL={self.EMBEDDING_MODEL}, QUERY_STRATEGY={self.QUERY_STRATEGY}")


try:
    CONFIG = ModelConfig()
except Exception as e:
    logger.error(f"配置初始化失败: {e}")
    raise
