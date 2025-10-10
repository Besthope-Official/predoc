"""后端服务配置（Milvus、MinIO、RabbitMQ）"""

import os
from typing import ClassVar
from pydantic import Field
from .base import BaseConfig


class MilvusConfig(BaseConfig):
    """Milvus 向量数据库配置"""

    host: str = Field(default_factory=lambda: os.getenv("MILVUS_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("MILVUS_PORT", "19530")))
    user: str = Field(default_factory=lambda: os.getenv("MILVUS_USER", "root"))
    password: str = Field(
        default_factory=lambda: os.getenv("MILVUS_PASSWORD", "Milvus")
    )
    db_name: str = Field(default_factory=lambda: os.getenv("MILVUS_DB", "default"))
    default_collection_name: str = Field(
        default_factory=lambda: os.getenv(
            "MILVUS_DEFAULT_COLLECTION", "default_collection"
        ),
        alias="default_collection",
    )
    default_partition_name: str = Field(
        default_factory=lambda: os.getenv(
            "MILVUS_DEFAULT_PARTITION", "default_partition"
        ),
        alias="default_partition",
    )


class OSSConfig(BaseConfig):
    """对象存储配置"""

    yaml_section: ClassVar[str] = "minio"

    endpoint: str = Field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
    )
    access_key: str = Field(
        default_factory=lambda: os.getenv("MINIO_ACCESS", "minioadmin"), alias="access"
    )
    secret_key: str = Field(
        default_factory=lambda: os.getenv("MINIO_SECRET", "minioadmin"), alias="secret"
    )
    preprocessed_files_bucket: str = Field(
        default_factory=lambda: os.getenv("PREPROCESSED_FILES_BUCKET", "prep")
    )
    pdf_bucket: str = Field(default_factory=lambda: os.getenv("PDF_BUCKET", "mybucket"))


class RabbitMQConfig(BaseConfig):
    """RabbitMQ 消息队列配置"""

    host: str = Field(default_factory=lambda: os.getenv("RABBITMQ_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("RABBITMQ_PORT", "5672")))
    user: str = Field(default_factory=lambda: os.getenv("RABBITMQ_USER", "admin"))
    password: str = Field(
        default_factory=lambda: os.getenv("RABBITMQ_PASSWORD", "admin")
    )
    task_queue: str = Field(
        default_factory=lambda: os.getenv("RABBITMQ_TASK_QUEUE", "taskQueue")
    )
    result_queue: str = Field(
        default_factory=lambda: os.getenv("RABBITMQ_RESULT_QUEUE", "respQueue")
    )
    consumer_workers: int = Field(
        default_factory=lambda: int(os.getenv("RABBITMQ_CONSUMER_WORKERS", "4"))
    )
