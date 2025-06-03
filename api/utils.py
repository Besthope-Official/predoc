"""API工具函数和模型加载器"""

from typing_extensions import TypedDict
from typing import Any, Optional
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


class ApiResponse(TypedDict):
    """
    封装 API 的返回结果类型
    """
    success: bool
    data: Optional[Any]
    message: str


def api_response(success: bool, data: Any = None, message: str = "") -> ApiResponse:
    """
    封装 API 的返回结果

    :param success: 请求是否成功
    :param data: 返回的数据
    :param message: 错误或提示信息
    :return: 统一格式的字典
    """
    return {
        "success": success,
        "data": data,
        "message": message
    }


def api_success(data: Any = None, message: str = "Success") -> ApiResponse:
    """
    成功的 API 响应

    :param data: 返回的数据
    :param message: 提示信息
    :return: 成功响应格式的字典
    """
    return api_response(success=True, data=data, message=message)


def api_fail(message: str = "Fail", data: Any = None) -> ApiResponse:
    """
    失败的 API 响应

    :param message: 错误信息
    :param data: 相关的错误数据（如有）
    :return: 失败响应格式的字典
    """
    return api_response(success=False, data=data, message=message)
