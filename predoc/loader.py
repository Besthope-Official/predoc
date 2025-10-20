"""模型加载器 - 管理所有模型实例的初始化和访问"""

from functools import cached_property
from loguru import logger
import threading

from predoc.parser import Parser, YoloParser
from predoc.chunker import Chunker, LLMChunker, SentenceChunker
from predoc.embedding import EmbeddingModel


class ModelLoader:
    """模型加载器，使用懒加载管理模型实例

    使用 cached_property 实现自动懒加载，更简洁的 Python 风格
    """

    def __init__(self):
        """初始化模型加载器"""
        self._parser_storage = None
        self._lock = threading.Lock()
        logger.info("ModelLoader 已初始化")

    @property
    def parser(self) -> Parser:
        """获取解析器实例（懒加载，无 storage）"""
        return self.get_parser(storage=None)

    def get_parser(self, storage=None) -> Parser:
        """获取带 storage 的解析器实例

        Args:
            storage: StorageBackend 实例，None 表示不使用远程存储

        Returns:
            配置了 storage 的 Parser 实例
        """
        if not hasattr(self, "_parser_instance"):
            with self._lock:
                if not hasattr(self, "_parser_instance"):
                    self._parser_instance = YoloParser(storage=storage)
                    logger.info("解析器实例已加载")
        elif storage is not None:
            # 更新 storage（如果需要）
            self._parser_instance.storage = storage
        return self._parser_instance

    @cached_property
    def llm_chunker(self) -> LLMChunker:
        """获取 LLM 分块器实例（自动懒加载）"""
        logger.info("LLM分块器实例已加载")
        return LLMChunker()

    @cached_property
    def sentence_chunker(self) -> SentenceChunker:
        """获取句子分块器实例（自动懒加载）"""
        logger.info("句子分块器实例已加载")
        return SentenceChunker()

    @cached_property
    def embedder(self) -> EmbeddingModel:
        """获取嵌入模型实例（自动懒加载）"""
        logger.info("嵌入模型实例已加载")
        return EmbeddingModel()

    def get_chunker(self, strategy: str = "semantic") -> Chunker:
        """根据策略获取分块器"""
        if strategy == "semantic_api":
            return self.llm_chunker
        else:
            return self.sentence_chunker

    def preload_all(self, storage=None):
        """预加载所有模型

        Args:
            storage: StorageBackend 实例，传递给 parser
        """
        logger.info("开始预加载所有模型...")
        _ = self.get_parser(storage=storage)
        _ = self.llm_chunker
        _ = self.sentence_chunker
        _ = self.embedder
        logger.info("所有模型预加载完成")

    def clear_cache(self):
        """清理模型缓存"""
        logger.info("清理模型缓存...")
        # 删除 cached_property 创建的属性
        for attr in ["_parser_instance", "llm_chunker", "sentence_chunker", "embedder"]:
            if hasattr(self, attr):
                delattr(self, attr)
        logger.info("模型缓存已清理")
