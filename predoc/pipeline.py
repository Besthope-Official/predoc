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
from task.oss import download_file, check_file_exists
from config.backend import OSSConfig

_oss_config = OSSConfig.from_yaml()


class BasePipeline(ABC):
    """
    基类：定义处理合同
    - process: 输入一个 Document，返回 (chunks, embeddings)
    子类可组合不同 parser/chunker/embedder，实现差异化流程。
    """

    def __init__(
        self,
        model_loader: Optional[ModelLoader] = None,
        *,
        destination_collection: Optional[str] = None,
    ) -> None:
        self.model_loader = model_loader or ModelLoader()
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

    def __init__(self, model_loader=None, *, destination_collection=None):
        super().__init__(model_loader, destination_collection=destination_collection)

    def process(self, doc: Document) -> Tuple[List[str], List[List[float]]]:
        file_name = doc.fileName
        stem = file_name.rsplit(".", 1)[0]
        parsed_text_obj = f"{stem}/text.txt"

        doc_bucket = getattr(doc, "bucket", None)

        if check_file_exists(parsed_text_obj):
            from pathlib import Path
            import os

            temp_dir = Path(os.environ.get("TEMP", "/tmp")) / f"pipeline_{stem}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            local_text_path = temp_dir / "text.txt"
            download_file(
                parsed_text_obj, local_text_path, _oss_config.preprocessed_files_bucket
            )
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
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            return chunks, embeddings

        from pathlib import Path
        import os
        import shutil

        temp_dir = Path(os.environ.get("TEMP", "/tmp")) / f"pipeline_{stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_pdf = temp_dir / (
            file_name if "/" not in file_name else file_name.split("/")[-1]
        )

        download_file(file_name, local_pdf, doc_bucket or _oss_config.pdf_bucket)

        processor = PDFProcessor(
            chunker=(
                self.model_loader.get_chunker("semantic_api")
                if self.model_loader
                else LLMChunker()
            ),
            parser=(self.model_loader.parser if self.model_loader else YoloParser()),
            embedder=(
                self.model_loader.embedder if self.model_loader else EmbeddingModel()
            ),
            output_dir=str(temp_dir),
            upload_to_oss=True,
        )
        try:
            chunks, embeddings = processor.preprocess(
                file_path=str(local_pdf), wrapper=False
            )
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        return chunks, embeddings


# 注册表：taskType -> Pipeline
PIPELINE_REGISTRY = {
    "default": DefaultPDFPipeline,
}


class PrintFilenamePipeline(BasePipeline):
    """用于 Debug"""

    def __init__(self, model_loader=None, *, destination_collection=None):
        super().__init__(model_loader, destination_collection=destination_collection)

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
