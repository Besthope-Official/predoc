"""
提供 OSS (基于 MinIO) 文件上传和下载功能
"""
import os
from typing import Optional, Union, IO
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from loguru import logger
from config.backend import OSSConfig

_minio_client = None


def get_minio_client() -> Minio:
    """获取 MinIO 客户端实例（单例模式）"""
    global _minio_client

    if _minio_client is not None:
        return _minio_client

    config = OSSConfig()
    secure = config.minio_endpoint.startswith("https://")

    endpoint = config.minio_endpoint
    if "://" in endpoint:
        endpoint = endpoint.split("://", 1)[1]

    _minio_client = Minio(
        endpoint,
        access_key=config.minio_access,
        secret_key=config.minio_secret,
        secure=secure
    )

    return _minio_client


def upload_file(
    file_path: Path,
    object_name: str,
    bucket_name: str
) -> str:
    """
    上传文件到 OSS

    Args:
        file_path: 本地文件路径
        object_name: 对象存储中的文件名，如果为 None 则使用原始文件名
        bucket_name: 存储桶名称，默认使用配置中的存储桶

    Returns:
        上传后的对象名称

    Raises:
        FileNotFoundError: 本地文件不存在
        S3Error: OSS操作错误
    """
    client = get_minio_client()

    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"创建存储桶 {bucket_name}")

        client.fput_object(
            bucket_name,
            object_name,
            str(file_path),
            content_type='application/octet-stream'
        )
        logger.info(f"文件 {file_path} 成功上传到 OSS 作为对象 {object_name}")
        return object_name
    except S3Error as e:
        logger.error(f"上传文件 {file_path} 到存储桶 {bucket_name} 失败: {e}")
        raise
    except Exception as e:
        logger.error(f"上传文件时发生未知错误: {e}")
        raise


def download_file(
    object_name: str,
    file_path: Union[str, Path],
    bucket_name: Optional[str] = OSSConfig.pdf_bucket
) -> Path:
    """
    从 OSS 下载文件

    Args:
        object_name: 对象存储中的文件名
        file_path: 本地保存路径
        bucket_name: 存储桶名称，默认使用配置中的存储桶

    Returns:
        下载后的本地文件路径对象

    Raises:
        S3Error: OSS操作错误或对象不存在
    """
    file_path = Path(file_path)

    os.makedirs(file_path.parent, exist_ok=True)

    client = get_minio_client()

    client.fget_object(
        bucket_name,
        object_name,
        str(file_path)
    )

    return file_path
