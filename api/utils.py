"""API工具函数和模型加载器"""

from typing import Any, Dict
from typing import Optional
from loguru import logger

from prep.parser import Parser, YoloParser
from prep.chunker import Chunker, LLMChunker, SentenceChunker
from prep.embedding import EmbeddingModel


class ModelLoader:
    """模型加载器，管理所有模型实例的初始化和访问"""

    def __init__(self):
        """初始化模型加载器"""
        self._parser = None
        self._llm_chunker = None
        self._sentence_chunker = None
        self._embedder = None
        logger.info("ModelLoader 已初始化")

    @property
    def parser(self) -> Parser:
        """获取解析器实例（懒加载）"""
        if self._parser is None:
            self._parser = YoloParser()
            logger.info("解析器实例已加载")
        return self._parser

    @property
    def llm_chunker(self) -> LLMChunker:
        """获取LLM分块器实例（懒加载）"""
        if self._llm_chunker is None:
            self._llm_chunker = LLMChunker()
            logger.info("LLM分块器实例已加载")
        return self._llm_chunker

    @property
    def sentence_chunker(self) -> SentenceChunker:
        """获取句子分块器实例（懒加载）"""
        if self._sentence_chunker is None:
            self._sentence_chunker = SentenceChunker()
            logger.info("句子分块器实例已加载")
        return self._sentence_chunker

    @property
    def embedder(self) -> EmbeddingModel:
        """获取嵌入模型实例（懒加载）"""
        if self._embedder is None:
            self._embedder = EmbeddingModel()
            logger.info("嵌入模型实例已加载")
        return self._embedder

    def get_chunker(self, strategy: str = "semantic") -> Chunker:
        """根据策略获取分块器"""
        if strategy == "semantic_api":
            return self.llm_chunker
        else:
            return self.sentence_chunker

    def preload_all(self):
        """预加载所有模型"""
        logger.info("开始预加载所有模型...")
        _ = self.parser
        _ = self.llm_chunker
        _ = self.sentence_chunker
        _ = self.embedder
        logger.info("所有模型预加载完成")

    def clear_cache(self):
        """清理模型缓存"""
        logger.info("清理模型缓存...")
        self._parser = None
        self._llm_chunker = None
        self._sentence_chunker = None
        self._embedder = None
        logger.info("模型缓存已清理")


class ApiResponse:
    """API响应类型"""
    pass


def api_success(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """返回成功响应"""
    return {
        "success": True,
        "message": message,
        "data": data
    }


def api_fail(message: str = "Error", code: int = 400) -> Dict[str, Any]:
    """返回失败响应"""
    return {
        "success": False,
        "message": message,
        "code": code
    }
