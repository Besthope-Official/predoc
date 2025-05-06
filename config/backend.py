'''后端项目的配置文件'''
from dataclasses import dataclass
import os


@dataclass
class MilvusConfig:
    host: str = os.getenv('MILVUS_HOST', '127.0.0.1')
    port: int = int(os.getenv('MILVUS_PORT', '19530'))
    user: str = os.getenv('MILVUS_USER', '')
    password: str = os.getenv('MILVUS_PASSWORD', '')
    db_name: str = os.getenv('MILVUS_DB', 'default')
    collection_name: str = os.getenv('MILVUS_COLLECTION', 'documents')


@dataclass
class OSSConfig:
    minio_endpoint = os.getenv('MINIO_ENDPOINT', '127.0.0.1:9000')
    minio_access = os.getenv('MINIO_ACCESS', 'minioadmin')
    minio_secret = os.getenv('MINIO_SECRET', 'minioadmin')
    minio_bucket = os.getenv('MINIO_BUCKET', 'prep') # Python 端的存储桶
    pdf_bucket = "mybucket"  # Java 端的存储桶


@dataclass
class RabbitMQConfig:
    host: str = os.getenv('RABBITMQ_HOST', '127.0.0.1')
    port: int = int(os.getenv('RABBITMQ_PORT', '5672'))
    user: str = os.getenv('RABBITMQ_USER', 'admin')
    password: str = os.getenv('RABBITMQ_PASSWORD', 'admin')
