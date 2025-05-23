from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import numpy as np
from loguru import logger

from config.model import ModelConfig
from config.api import ChunkAPIConfig
from .chunker import Chunker, LLMChunker, SentenceChunker
from .parser import Parser, YoloParser
from .embedding import EmbeddingModel

from .error import ParseResultEmptyException


class Processor(ABC):
    BASE_EMBD_MODEL = ModelConfig.EMBEDDING_MODEL

    def __init__(self,
                 chunker: Chunker,
                 parser: Parser = None,
                 embedder: EmbeddingModel = None,
                 output_dir: str = ModelConfig.CHUNK_OUTPUT_DIR,
                 upload_to_oss: bool = True,
                 enable_parallel_processing: bool = False,
                 max_workers: Optional[int] = None):
        """
        初始化处理器

        Args:
            chunker: 分块器实例，如 LLMChunker() 或 SentenceChunker()
            parser: 解析器实例，默认使用YoloParser()
            embedder: 嵌入模型实例，默认使用EmbeddingModel()
            output_dir: 输出目录
            upload_to_oss: 是否上传到OSS
            enable_parallel_processing: 是否启用并行处理
            max_workers: 最大工作线程数，None时自动计算
        """
        self.chunker = chunker
        self.parser = parser if parser is not None else YoloParser()
        self.embedder = embedder if embedder is not None else EmbeddingModel()
        self.output_dir = output_dir
        self.upload_to_oss = upload_to_oss
        self.enable_parallel_processing = enable_parallel_processing
        self.max_workers = max_workers
        logger.info(
            f'分块器类型: {type(chunker).__name__}, 解析器类型: {type(self.parser).__name__}')

    def _validate_components(self):
        """验证组件类型是否正确"""
        if not isinstance(self.parser, Parser):
            raise TypeError(f"parser必须是Parser的实例，当前类型: {type(self.parser)}")
        if not isinstance(self.chunker, Chunker):
            raise TypeError(f"chunker必须是Chunker的实例，当前类型: {type(self.chunker)}")
        if not isinstance(self.embedder, EmbeddingModel):
            raise TypeError(
                f"embedder必须是EmbeddingModel的实例，当前类型: {type(self.embedder)}")

    def parse(self, file_path: str) -> str:
        """解析文档，提取文本，使用初始化时指定的parser"""
        text = self.parser.parse(
            file_path, self.output_dir, upload_to_oss=self.upload_to_oss)

        if not text.strip():
            raise ParseResultEmptyException("提取文本为空")
        return text

    def chunk(self, text: str) -> list[str]:
        """将文本分块，使用初始化时指定的chunker"""
        if not text.strip():
            return []
        return self.chunker.chunk(text)

    def embedding(self, text: str, embd_model: str = BASE_EMBD_MODEL) -> list[float]:
        """对单文本嵌入，使用初始化时指定的embedder"""
        embedding = self.embedder.generate_embedding(text)
        return embedding.tolist()

    def embeddings(self, chunks: list[str]) -> np.ndarray:
        """对多个文本块进行嵌入，使用初始化时指定的embedder"""
        embeddings = self.embedder.generate_embeddings(chunks)
        return embeddings

    @staticmethod
    def _wrapper(chunks: list[str], embeddings: np.ndarray) -> list[dict]:
        """包装分块和嵌入"""
        return [{"chunk": chunk, "embedding": embedding.tolist()}
                for chunk, embedding in zip(chunks, embeddings)]

    def preprocess(self, file_path: str, wrapper: bool = True):
        """预处理单个文档"""
        self._validate_components()

        try:
            text = self.parse(file_path)
        except Exception as e:
            raise Exception(f"解析文件 {file_path} 失败: {e}")

        try:
            chunks = self.chunk(text)
        except Exception as e:
            raise Exception(f"分块文本失败: {e}")

        try:
            embeddings = self.embeddings(chunks)
            if wrapper:
                return self._wrapper(chunks, embeddings)
            else:
                return (chunks, embeddings.tolist())
        except Exception as e:
            raise Exception(f"嵌入文本失败: {e}")


class PDFProcessor(Processor):
    """PDF文档处理类"""

    def __init__(self,
                 chunker: Chunker = None,
                 parser: Parser = None,
                 embedder: EmbeddingModel = None,
                 output_dir: str = ModelConfig.CHUNK_OUTPUT_DIR,
                 upload_to_oss: bool = True,
                 enable_parallel_processing: bool = False,
                 max_workers: Optional[int] = None):
        """
        初始化PDF处理器

        Args:
            chunker: 分块器实例，默认使用LLMChunker
            parser: 解析器实例，默认使用YoloParser()
            embedder: 嵌入模型实例，默认使用EmbeddingModel()
            output_dir: 输出目录
            upload_to_oss: 是否上传到OSS
            enable_parallel_processing: 是否启用并行处理
            max_workers: 最大工作线程数
        """
        if chunker is None:
            chunker = LLMChunker()
        if parser is None:
            parser = YoloParser()
        super().__init__(chunker, parser, embedder, output_dir,
                         upload_to_oss, enable_parallel_processing, max_workers)
