'''后端项目的配置文件'''
from dataclasses import dataclass
import os
from . import env


@dataclass
class MilvusConfig:
    if env == 'dev':
        host: str = os.getenv('MILVUS_HOST', '127.0.0.1')
        port: int = int(os.getenv('MILVUS_PORT', '19530'))
        user: str = os.getenv('MILVUS_USER', 'root')
        password: str = os.getenv('MILVUS_PASSWORD', 'Milvus')
        db_name: str = os.getenv('MILVUS_DB', 'default')
        default_collection_name: str = os.getenv(
            'MILVUS_DEFAULT_COLLECTION', 'default_collection')
        default_partition_name: str = os.getenv(
            'MILVUS_DEFAULT_PARTITION', 'default_partition')
    elif env == 'test':
        host: str = os.getenv('MILVUS_HOST', '127.0.0.1')
        port: int = int(os.getenv('MILVUS_PORT', '19530'))
        user: str = os.getenv('MILVUS_USER', 'root')
        password: str = os.getenv('MILVUS_PASSWORD', 'Milvus')
        db_name: str = os.getenv('TEST_MILVUS_DB', 'test')
        default_collection_name: str = os.getenv(
            'TEST_MILVUS_DEFAULT_COLLECTION', 'test_collection')
        default_partition_name: str = os.getenv(
            'TEST_MILVUS_DEFAULT_PARTITION', 'test_partition')
    elif env == 'prod':
        host: str = os.getenv('MILVUS_HOST', '')
        port: int = int(os.getenv('MILVUS_PORT', '19530'))
        user: str = os.getenv('MILVUS_USER', 'root')
        password: str = os.getenv('MILVUS_PASSWORD', 'Milvus')
        db_name: str = os.getenv('MILVUS_DB', 'default')
        default_collection_name: str = os.getenv(
            'MILVUS_DEFAULT_COLLECTION', 'default_collection')
        default_partition_name: str = os.getenv(
            'MILVUS_DEFAULT_PARTITION', 'default_partition')
    else:
        raise ValueError(f"Unsupported environment: {env}")


@dataclass
class OSSConfig:
    if env == 'dev':
        minio_endpoint = os.getenv('MINIO_ENDPOINT', '127.0.0.1:9000')
        minio_access = os.getenv('MINIO_ACCESS', 'minioadmin')
        minio_secret = os.getenv('MINIO_SECRET', 'minioadmin')
        minio_bucket = os.getenv('MINIO_BUCKET', 'prep')
        pdf_bucket = os.getenv('PDF_BUCKET', 'mybucket')
    elif env == 'test':
        minio_endpoint = os.getenv('MINIO_ENDPOINT', '127.0.0.1:9000')
        minio_access = os.getenv('MINIO_ACCESS', 'minioadmin')
        minio_secret = os.getenv('MINIO_SECRET', 'minioadmin')
        minio_bucket = os.getenv('TEST_MINIO_BUCKET', 'test-prep')
        pdf_bucket = os.getenv('TEST_PDF_BUCKET', 'test-bucket')
    elif env == 'prod':
        minio_endpoint = os.getenv('MINIO_ENDPOINT', '')
        minio_access = os.getenv('MINIO_ACCESS', 'minioadmin')
        minio_secret = os.getenv('MINIO_SECRET', 'minioadmin')
        minio_bucket = os.getenv('MINIO_BUCKET', 'prep')
        pdf_bucket = os.getenv('PDF_BUCKET', 'mybucket')
    else:
        raise ValueError(f"Unsupported environment: {env}")


@dataclass
class RabbitMQConfig:
    if env == 'dev':
        host: str = os.getenv('RABBITMQ_HOST', '127.0.0.1')
        port: int = int(os.getenv('RABBITMQ_PORT', '5672'))
        user: str = os.getenv('RABBITMQ_USER', 'admin')
        password: str = os.getenv('RABBITMQ_PASSWORD', 'admin')
    elif env == 'test':
        host: str = os.getenv('RABBITMQ_HOST', '127.0.0.1')
        port: int = int(os.getenv('RABBITMQ_PORT', '5672'))
        user: str = os.getenv('RABBITMQ_USER', 'admin')
        password: str = os.getenv('RABBITMQ_PASSWORD', 'admin')
    elif env == 'prod':
        host: str = os.getenv('RABBITMQ_HOST', '')
        port: int = int(os.getenv('RABBITMQ_PORT', '5672'))
        user: str = os.getenv('RABBITMQ_USER', 'admin')
        password: str = os.getenv('RABBITMQ_PASSWORD', 'admin')
    else:
        raise ValueError(f"Unsupported environment: {env}")
