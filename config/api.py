"""LLM API相关配置"""

import os


class OpenAIConfig:
    """OpenAI API配置"""

    MODEL_NAME: str
    API_KEY: str
    API_URL: str
    # Suppose API has no QPS limit
    MAX_QPS: int = 50


class ChunkAPIConfig(OpenAIConfig):
    """分块用LLM API配置"""

    MODEL_NAME: str = os.getenv("CHUNK_MODEL_NAME", "moonshot-v1-8k")
    API_KEY: str = os.getenv("CHUNK_API_KEY", "your_openai_api_key")
    API_URL: str = os.getenv("CHUNK_API_URL", "http://127.0.0.1:8000")
    MAX_QPS: int = int(os.getenv("CHUNK_MAX_QPS", 50))
