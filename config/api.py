"""LLM API相关配置"""

import os
from typing import ClassVar
from pydantic import Field
from .base import BaseConfig


class ChunkAPIConfig(BaseConfig):
    """分块用 LLM API 配置"""

    yaml_section: ClassVar[str] = "models.chunking"

    model_name: str = Field(
        default_factory=lambda: os.getenv("CHUNK_MODEL_NAME", "moonshot-v1-8k")
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("CHUNK_API_KEY", "your_openai_api_key")
    )
    api_url: str = Field(
        default_factory=lambda: os.getenv("CHUNK_API_URL", "http://127.0.0.1:8000")
    )
    max_qps: int = Field(default_factory=lambda: int(os.getenv("MAX_CHUNK_QPS", "50")))
