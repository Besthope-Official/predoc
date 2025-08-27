"""后端项目的配置文件"""

from dataclasses import dataclass
import os


@dataclass
class MilvusConfig:
    host: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    port: int = int(os.getenv("MILVUS_PORT", "19530"))
    user: str = os.getenv("MILVUS_USER", "root")
    password: str = os.getenv("MILVUS_PASSWORD", "Milvus")
    db_name: str = os.getenv("MILVUS_DB", "default")
    default_collection_name: str = os.getenv(
        "MILVUS_DEFAULT_COLLECTION", "default_collection"
    )
    default_partition_name: str = os.getenv(
        "MILVUS_DEFAULT_PARTITION", "default_partition"
    )


@dataclass
class OSSConfig:
    endpoint = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
    access = os.getenv("MINIO_ACCESS", "minioadmin")
    secret = os.getenv("MINIO_SECRET", "minioadmin")
    preprocessed_files_bucket = os.getenv("PREPROCESSED_FILES_BUCKET", "prep")
    pdf_bucket = os.getenv("PDF_BUCKET", "mybucket")


@dataclass
class RabbitMQConfig:
    host: str = os.getenv("RABBITMQ_HOST", "127.0.0.1")
    port: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    user: str = os.getenv("RABBITMQ_USER", "admin")
    password: str = os.getenv("RABBITMQ_PASSWORD", "admin")
    task_queue: str = os.getenv("RABBITMQ_TASK_QUEUE", "taskQueue")
    result_queue: str = os.getenv("RABBITMQ_RESULT_QUEUE", "respQueue")
    consumer_workers: int = int(os.getenv("RABBITMQ_CONSUMER_WORKERS", "4"))
