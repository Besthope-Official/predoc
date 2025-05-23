import os

SUPPORTED_CHUNK_STRATEGIES = ['semantic', 'semantic_ollama', 'semantic_api']

class Config:
    """
    Base configuration class for predoc.
    """
    PARSE_METHOD: str = os.getenv("PARSE_METHOD", "auto")
    CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "semantic_api")
    UPLOAD_TO_OSS: bool = os.getenv("UPLOAD_TO_OSS", "true").lower() == "true"
    ENABLE_PARALLELISM: bool = os.getenv("ENABLE_PARALLELISM", "false").lower() == "true"
    
    def __post_init__(self):
        assert self.CHUNK_STRATEGY in SUPPORTED_CHUNK_STRATEGIES