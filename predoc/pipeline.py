"""任务预处理 Pipeline 抽象与默认实现"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union, Type

from schemas.document import Document
from api.utils import ModelLoader
from predoc.processor import PDFProcessor
from predoc.chunker import LLMChunker
from predoc.parser import YoloParser
from predoc.embedding import EmbeddingModel


class BasePipeline(ABC):
    """
    基类：定义处理合同
    - process: 输入一个 Document，返回 (chunks, embeddings)
    子类可组合不同 parser/chunker/embedder，实现差异化流程。
    """

    def __init__(
        self,
        model_loader: Optional[ModelLoader] = None,
        storage=None,
        *,
        destination_collection: Optional[str] = None,
    ) -> None:
        self.model_loader = model_loader or ModelLoader()
        self.storage = storage
        self.destination_collection = destination_collection

    @abstractmethod
    def process(self, doc: Document) -> Tuple[List[str], List[List[float]]]:
        raise NotImplementedError

    def store_embedding(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        *,
        doc: Document,
        collection_name: Optional[str] = None,
        partition_name: Optional[str] = None,
    ) -> None:
        """默认实现：构建 metadata 并入库。子类可覆盖以自定义写入逻辑。"""
        from task.milvus import store_embedding_task

        metadata = doc.to_metadata()
        store_embedding_task(
            embeddings,
            chunks,
            metadata,
            collection_name=collection_name or self.destination_collection,
            partition_name=partition_name,
        )


class DefaultPDFPipeline(BasePipeline):
    """默认 PDF 处理流程：
    - 若存在已解析文本（prep bucket 下 <stem>/text.txt），则复用并直接 chunk+embed
    - 否则从 pdf bucket 下载 PDF，走完整 PDFProcessor 流程
    """

    def __init__(self, model_loader=None, storage=None, *, destination_collection=None):
        super().__init__(
            model_loader, storage, destination_collection=destination_collection
        )

    def process(self, doc: Document) -> Tuple[List[str], List[List[float]]]:
        from pathlib import Path
        import os
        import shutil

        file_name = doc.fileName
        file_path = Path(file_name)
        stem = file_path.stem
        parsed_text_obj = f"{stem}/text.txt"

        doc_bucket = getattr(doc, "bucket", None)

        # 检查缓存(仅当有 storage 时)
        if self.storage and self.storage.exists(parsed_text_obj):
            temp_dir = Path(os.environ.get("TEMP", "/tmp")) / f"pipeline_{stem}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            local_text_path = temp_dir / "text.txt"

            # storage 自己处理默认 bucket
            self.storage.download(parsed_text_obj, local_text_path, bucket=None)
            text = local_text_path.read_text(encoding="utf-8")

            chunker = (
                self.model_loader.get_chunker("semantic_api")
                if self.model_loader
                else LLMChunker()
            )
            embedder = (
                self.model_loader.embedder if self.model_loader else EmbeddingModel()
            )
            chunks = chunker.chunk(text)
            embeddings = embedder.generate_embeddings(chunks)
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                from loguru import logger

                logger.warning(
                    f"Failed to clean up temporary directory {temp_dir}: {e}"
                )
            return chunks, embeddings

        temp_dir = Path(os.environ.get("TEMP", "/tmp")) / f"pipeline_{stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_pdf = temp_dir / file_path.name

        # 下载 PDF (如果有 storage)
        if self.storage:
            self.storage.download(file_name, local_pdf, doc_bucket)
        else:
            # 本地文件,直接复制
            source_path = Path(file_name)
            if not source_path.exists():
                raise FileNotFoundError(f"本地文件不存在: {file_name}")
            shutil.copy2(source_path, local_pdf)

        # 创建 parser 并注入 storage
        if self.model_loader:
            parser = self.model_loader.get_parser(storage=self.storage)
        else:
            parser = YoloParser(storage=self.storage)

        processor = PDFProcessor(
            chunker=(
                self.model_loader.get_chunker("semantic_api")
                if self.model_loader
                else LLMChunker()
            ),
            parser=parser,
            embedder=(
                self.model_loader.embedder if self.model_loader else EmbeddingModel()
            ),
            output_dir=str(temp_dir),
        )
        try:
            chunks, embeddings = processor.preprocess(
                file_path=str(local_pdf), wrapper=False
            )
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                from loguru import logger

                logger.warning(
                    f"Failed to clean up temporary directory {temp_dir}: {e}"
                )
        return chunks, embeddings


# 注册表：taskType -> Pipeline
PIPELINE_REGISTRY = {
    "default": DefaultPDFPipeline,
}


class PrintFilenamePipeline(BasePipeline):
    """用于 Debug"""

    def __init__(self, model_loader=None, storage=None, *, destination_collection=None):
        super().__init__(
            model_loader, storage, destination_collection=destination_collection
        )

    def process(self, doc: Document) -> Tuple[List[str], List[List[float]]]:
        from loguru import logger

        logger.info(f"[PrintFilenamePipeline] {doc.fileName}")
        return [], []


PIPELINE_REGISTRY["print-filename"] = PrintFilenamePipeline


def add_entry(key: str, value: Union[Type[BasePipeline], BasePipeline]):
    """注册自定义 Pipeline。

    - 若传入实例，则登记其类，确保消费端可以使用类进行实例化；
    - 若传入类，直接登记。
    """
    cls: Type[BasePipeline]
    if isinstance(value, BasePipeline):
        cls = value.__class__
    else:
        cls = value  # type: ignore[assignment]
    PIPELINE_REGISTRY[key] = cls


def get_pipeline(task_type: str) -> type[BasePipeline]:
    return PIPELINE_REGISTRY.get(task_type, DefaultPDFPipeline)
