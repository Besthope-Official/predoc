import os


class Config:
    """
    Base configuration class for predoc.
    """

    # set this false to use predoc as a API server
    ENABLE_MASSAGE_QUEUE: bool = (
        os.getenv("ENABLE_MASSAGE_QUEUE", "false").lower() == "true"
    )
    PARSE_METHOD: str = os.getenv("PARSE_METHOD", "auto")
    CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "semantic_api")
    UPLOAD_TO_OSS: bool = os.getenv("UPLOAD_TO_OSS", "true").lower() == "true"
    ENABLE_PARALLELISM: bool = (
        os.getenv("ENABLE_PARALLELISM", "false").lower() == "true"
    )
