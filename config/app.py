import os
from pydantic import Field
from .base import BaseConfig


class AppConfig(BaseConfig):
    """应用核心配置类"""

    env: str = Field(default_factory=lambda: os.getenv("ENV", "prod"))
    enable_message_queue: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_MASSAGE_QUEUE", "false").lower()
        == "true"
    )
    parse_method: str = Field(default_factory=lambda: os.getenv("PARSE_METHOD", "auto"))
    chunk_strategy: str = Field(
        default_factory=lambda: os.getenv("CHUNK_STRATEGY", "semantic_api")
    )
    upload_to_oss: bool = Field(
        default_factory=lambda: os.getenv("UPLOAD_TO_OSS", "true").lower() == "true"
    )
    enable_parallelism: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_PARALLELISM", "false").lower()
        == "true"
    )
