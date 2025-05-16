from abc import abstractmethod
from typing import Tuple
import numpy as np
from loguru import logger

from config.model import ModelConfig
from .model import generate_embeddings, init_model
from .chunker import Chunker
from .parser import Parser

from .error import ParseResultEmptyException

parser = Parser()
chunker = Chunker(model_name='qwen3:4b')


class Processor:
    """数据预处理接口，需实现 parse 和 chunk 方法"""
    BASE_EMBD_MODEL = ModelConfig.EMBEDDING_MODEL

    def __init__(self, output_dir=ModelConfig.CHUNK_OUTPUT_DIR,
                 parse_method="auto",
                 chunk_strategy="semantic_ollama"):
        self.output_dir = output_dir
        self.parse_method = parse_method.lower()
        self.chunk_strategy = chunk_strategy.lower()
        logger.info(f'解析方法: {self.parse_method} 分块策略为: {self.chunk_strategy}')

    @abstractmethod
    def parse(self, file_path: str) -> str:
        """解析PDF文档，提取文本"""
        raise NotImplementedError("parse method not implemented")

    @abstractmethod
    def chunk(self, text: str) -> list[dict]:
        """将文本分块"""
        raise NotImplementedError("chunk method not implemented")

    def embedding(self, text: str, embd_model: str = BASE_EMBD_MODEL) -> list[float]:
        """对单文本嵌入"""
        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer(embd_model)
        except Exception as e:
            logger.warning(f"指定模型 {embd_model} 不存在, 回退到默认模型: {str(e)}")
            model = SentenceTransformer(self.BASE_EMBD_MODEL)

        embedding = model.encode(text).tolist()
        return embedding

    def embeddings(self, chunks: list[str]) -> np.ndarray:
        embd_model_info = init_model()
        embeddings = generate_embeddings(embd_model_info, chunks)
        return embeddings

    @staticmethod
    def _wrapper(chunks: list[str], embeddings: np.ndarray) -> list[dict]:
        """包装分块和嵌入"""
        return [{"chunk": chunk, "embedding": embedding.tolist()}
                for chunk, embedding in zip(chunks, embeddings)]

    def preprocess(self, file_path: str, warpper=True):
        """预处理文档"""
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
            if warpper:
                return self._wrapper(chunks, embeddings)
            else:
                return (chunks, embeddings.tolist())
        except Exception as e:
            raise Exception(f"嵌入文本失败: {e}")


class PDFProcessor(Processor):
    """PDF文档处理类"""

    def __init__(self, output_dir=ModelConfig.CHUNK_OUTPUT_DIR, parse_method="auto", chunk_strategy="semantic_ollama"):
        super().__init__(output_dir, parse_method, chunk_strategy)

    def parse(self, file_path: str) -> str:
        """解析PDF文档，根据解析方法提取文本"""
        if self.parse_method == "auto" or self.parse_method == "yolo":
            text = parser.process_pdf(
                file_path, self.output_dir, upload_to_oss=True)

        if not text.strip():
            raise ParseResultEmptyException("提取文本为空")
        return text

    def chunk(self, text: str) -> list[dict]:
        """将文本分成语义块"""
        if not text.strip():
            return []
        doc_list = chunker.split_text(text, self.chunk_strategy)
        return [doc.page_content for doc in doc_list]
