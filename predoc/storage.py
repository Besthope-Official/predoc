"""存储后端抽象层"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class StorageBackend(ABC):
    """存储后端抽象接口"""

    @abstractmethod
    def upload(
        self, local_path: Path, object_name: str, bucket: Optional[str] = None
    ) -> str:
        pass

    @abstractmethod
    def download(
        self, object_name: str, local_path: Path, bucket: Optional[str] = None
    ) -> Path:
        pass

    @abstractmethod
    def exists(self, object_name: str, bucket: Optional[str] = None) -> bool:
        pass


class MinioStorage(StorageBackend):
    """MinIO 存储后端

    默认 bucket 规则:
    - upload: preprocessed_files_bucket (解析后的文件)
    - download: 智能判断
      * "doc1/text.txt" -> preprocessed_files_bucket
      * "doc.pdf" -> pdf_bucket
      * "folder/doc.pdf" -> pdf_bucket
    - exists: preprocessed_files_bucket (检查缓存)
    """

    def __init__(self, oss_config=None):
        if oss_config is None:
            from config.backend import OSSConfig

            oss_config = OSSConfig.from_yaml()
        self.config = oss_config

    def upload(
        self, local_path: Path, object_name: str, bucket: Optional[str] = None
    ) -> str:
        from task.oss import upload_file

        target_bucket = bucket or self.config.preprocessed_files_bucket
        return upload_file(local_path, object_name, target_bucket)

    def download(
        self, object_name: str, local_path: Path, bucket: Optional[str] = None
    ) -> Path:
        from task.oss import download_file

        if bucket is None:
            # 智能判断规则:
            # 1. 路径包含 / 且扩展名不是 .pdf -> preprocessed (如 "doc1/text.txt")
            # 2. 其他情况 -> pdf_bucket (如 "doc.pdf" 或 "folder/doc.pdf")
            if "/" in object_name and not object_name.lower().endswith(".pdf"):
                bucket = self.config.preprocessed_files_bucket
            else:
                bucket = self.config.pdf_bucket

        return download_file(object_name, local_path, bucket)

    def exists(self, object_name: str, bucket: Optional[str] = None) -> bool:
        from task.oss import check_file_exists

        target_bucket = bucket or self.config.preprocessed_files_bucket
        return check_file_exists(object_name, target_bucket)


class LocalStorage(StorageBackend):
    """本地文件系统存储后端"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload(
        self, local_path: Path, object_name: str, bucket: Optional[str] = None
    ) -> str:
        import shutil

        dest = self.base_dir / (bucket or "default") / object_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists():
            raise FileNotFoundError(f"本地文件不存在: {local_path}")

        shutil.copy2(local_path, dest)
        return str(dest)

    def download(
        self, object_name: str, local_path: Path, bucket: Optional[str] = None
    ) -> Path:
        import shutil

        src = self.base_dir / (bucket or "default") / object_name
        if not src.exists():
            raise FileNotFoundError(f"本地存储中文件不存在: {src}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        return local_path

    def exists(self, object_name: str, bucket: Optional[str] = None) -> bool:
        path = self.base_dir / (bucket or "default") / object_name
        return path.exists()
