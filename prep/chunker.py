"""分块器"""

import re
import requests
from typing import List
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from abc import ABC
from loguru import logger
from openai import OpenAI

from config.model import CONFIG
from config.api import ChunkAPIConfig
from config.app import Config
from .prompt import CHUNK_SYSTEM_PROMPT_TEMPLATE, CHUNK_PROMPT_TEMPLATE
from .utils import TextSplitter, extract_markers, reconstruct_chunks


class Chunker(ABC):
    """Base interface for chunker."""

    def __init__(self, enable_parallelism: bool):
        self.enable_parallelism = enable_parallelism
        if enable_parallelism:
            self.num_workers = self._get_optimal_worker_count()

    def split_text(self, text: str) -> List[str]:
        """Base method to split text into desired chunks. Text MAX_LENGTH is constrained by `CONFIG.CHUNK_SIZE`."""
        raise NotImplementedError("chunk method not implemented")

    @staticmethod
    def _get_optimal_worker_count() -> int:
        """Get the optimal number of workers for parallel processing."""
        try:
            cpu_cores = cpu_count()
            return min(8, max(1, cpu_cores * 3 // 4))
        except Exception as e:
            logger.warning(
                f"Proceeding with 1 worker. Error calculating optimal worker count: {e}"
            )
            return 1

    def _parallel_batch_processing(self, texts: List[str]) -> List[str]:
        logger.debug(f"Using {self.num_workers} workers for parallel processing.")
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            all_results = executor.map(self.split_text, texts)
            for result in all_results:
                results.extend(result)
        return results

    def chunk(self, text: str) -> List:
        """Chunk text.

        Args:
            text: The text to be chunked. Text will first be split into smaller sections, then call `split_text` to split each section into chunks.
        """
        if len(text) < CONFIG.MIN_CHUNK_LENGTH:
            return []

        markers, pages, clean_text = extract_markers(text)
        sections = TextSplitter.split_text_into_sections(clean_text)

        all_chunks = []
        if not self.enable_parallelism:
            for section in sections:
                all_chunks.extend(self.split_text(section))
        else:
            logger.info("使用多线程分块...")
            all_chunks = self._parallel_batch_processing(sections)
        text = reconstruct_chunks(all_chunks, markers, pages, len(clean_text))
        return text


class SentenceChunker(Chunker):
    """使用简单的基于句子的方法创建语义块"""

    def __init__(self, enable_parallelism: bool = Config.ENABLE_PARALLELISM):
        super().__init__(enable_parallelism=enable_parallelism)

    def split_text(self, text: str) -> List[str]:
        sentences = TextSplitter.split_into_sentences(text)

        if len(sentences) <= 3:
            logger.debug("句子数量不足，返回原始文本")
            return [text]

        chunks = []
        current_chunk = []
        sentences_per_chunk = min(10, max(7, len(sentences) // 2))

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            if (i + 1) % sentences_per_chunk == 0 or i == len(sentences) - 1:
                chunks.append("".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append("".join(current_chunk))
        return chunks


class LLMChunker(Chunker):
    """LLM-based Chunker. Similar to https://docs.chonkie.ai/python-sdk/chunkers/slumber-chunker,
    use a prompt template to chunk text into smaller pieces.
    You can change API backend to use Ollama or any OpenAI-compatible LLM API.
    """

    def __init__(
        self,
        backend="api",
        ollama_api_host="http://127.0.0.1:11434",
        model_name=ChunkAPIConfig.MODEL_NAME,
        api_base=ChunkAPIConfig.API_URL,
        api_key=ChunkAPIConfig.API_KEY,
        enable_parallelism=Config.ENABLE_PARALLELISM,
    ):
        """Initialize the LLMChunker with specified backend and API configurations.

        Args:
            backend: The API backend to use. Either 'api' for OpenAI-compatible
                APIs or 'ollama' for Ollama API. Defaults to 'api'.
            ollama_api_host: The host URL for the Ollama API.
                Defaults to `http://127.0.0.1:11434`.
            model_name: The name of the LLM model to use.
                Defaults to `ChunkAPIConfig.MODEL_NAME`.
            api_base: The base URL for the OpenAI-compatible API.
                Defaults to `ChunkAPIConfig.API_URL`.
            api_key: The API key for the OpenAI-compatible API.
                Defaults to `ChunkAPIConfig.API_KEY`.
            enable_parallelism: Use multi-threading for chunking. Defaults to `False`.
        """
        super().__init__(enable_parallelism=enable_parallelism)
        self.num_workers = min(self.num_workers, ChunkAPIConfig.MAX_QPS)
        self.model_name = model_name
        logger.debug(f"Using {self.model_name} as chunker LLM...")
        self.backend = backend
        self.ollama_api_host = ollama_api_host
        self.ollama_api_url = f"{ollama_api_host}/api/generate"
        self.api_url = api_base
        self.api_key = api_key
        self.client = OpenAI(base_url=api_base, api_key=api_key)

    @staticmethod
    def _remove_thinking(text: str) -> str:
        """
        对于推理系大模型, 输出如 `<think><Thinking Process></think>\n<Answer>`, 仅保留 `<Answer>` 部分
        """
        cleaned_text = re.sub(r"(?i)<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)
        cleaned_text = cleaned_text.lstrip()

        return cleaned_text

    @staticmethod
    def create_sentence_chunks(text: str) -> List[str]:
        """使用简单的基于句子的方法创建语义块"""
        logger.debug("使用备选分块方法...")

        chunker = SentenceChunker()
        chunks = chunker.split_text(text)

        logger.debug(f"备选方法生成了 {len(chunks)} 个语义块")
        return chunks

    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str:
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        if system_prompt:
            payload["system"] = system_prompt
        try:
            response = requests.post(self.ollama_api_url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return ""

    def _call_open_api(self, prompt: str, system_prompt: str = None) -> str:
        try:
            messages = []
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            messages.append({"role": "user", "content": full_prompt})
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, temperature=0.7, stream=False
            )
            content = response.choices[0].message.content.strip()
            if not content:
                raise ValueError(f"模型 {self.model_name} 返回空结果")
            return content
        except Exception as e:
            logger.error(f"API 调用失败: {e}, 请求: {prompt[:100]}...", exc_info=True)
            raise e

    def split_text(self, text: str) -> List[str]:
        if len(text) < 100:
            return [text]

        # Note, if one of the following conditions met:
        # 1. Ollama failed (e.g. ollama API request failed, specified model not found, etc.),
        # 2. result_chunks <= 1,
        # 3. LLM-based chunk performance is bad,
        # rule-based chunker (create_sentence_chunks) will be used.
        prompt = CHUNK_PROMPT_TEMPLATE.format(text=text, text_length=len(text))

        try:
            if self.backend == "ollama":
                response = self._call_ollama(prompt, CHUNK_SYSTEM_PROMPT_TEMPLATE)
            elif self.backend == "api":
                response = self._call_open_api(prompt, CHUNK_SYSTEM_PROMPT_TEMPLATE)

            # In case of using reasoning model
            # It's not recommended to use reasoning model(e.g. Deepseek-R1) for chunking,
            # as it requires more **TIME** and **TOKEN COST**
            cleaned_text = self._remove_thinking(response)

            result_chunks = [
                chunk.strip()
                for chunk in cleaned_text.split("[CHUNK_BREAK]")
                if chunk.strip()
            ]

            if len(result_chunks) <= 1:
                logger.warning("LLM只生成了一个块或返回空结果，使用备选分块方法")
                return self.create_sentence_chunks(text)

            # In case of LLM adds or removes some content unexpectedly
            reconstructed_text = "".join([chunk.strip() for chunk in result_chunks])
            original_text = text.strip()
            similarity_ratio = (
                len(reconstructed_text) / len(original_text) if original_text else 0
            )

            # ratio of 0.95-1.05 is considered acceptable
            # NO RETRY, use second choice for convenience
            if similarity_ratio < 0.95 or similarity_ratio > 1.05:
                logger.warning(
                    f"分块内容与原文不匹配 (原文长度：{len(original_text)} 分块后：{len(reconstructed_text)} 相似度比例: {similarity_ratio:.4f})"
                )
                return self.create_sentence_chunks(text)

            logger.info(f"LLM成功生成 {len(result_chunks)} 个语义块")
            return result_chunks

        except Exception as e:
            logger.error(f"LLM分块失败，使用备选方法: {e}")
            return self.create_sentence_chunks(text)
