'''分块器'''

import re
import requests
from typing import List
from loguru import logger
from langchain.docstore.document import Document as LangChainDocument
from concurrent.futures import ThreadPoolExecutor

from config.model import CONFIG
from .parser import Parser


class TwoStageSemanticChunker:
    def __init__(self, model_name="gemma2:27b", host="http://127.0.0.1:11434"):
        self.model_name = model_name
        self.host = host
        self.api_url = f"{host}/api/generate"
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return ""

    def split_text_into_sections(self, text: str, max_section_length=1500):
        """将文本分割成适合模型处理的大段"""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sections = []
        current_section = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)
            if para_length > max_section_length:
                if current_section:
                    sections.append("\n\n".join(current_section))
                    current_section = []
                    current_length = 0

                sentences = re.split(r'(?<=[。！？.!?])', para)
                sentences = [s.strip() for s in sentences if s.strip()]

                temp_section = []
                temp_length = 0

                for sentence in sentences:
                    sentence_length = len(sentence)
                    if temp_length + sentence_length > max_section_length and temp_section:
                        sections.append("".join(temp_section))
                        temp_section = [sentence]
                        temp_length = sentence_length
                    else:
                        temp_section.append(sentence)
                        temp_length += sentence_length

                if temp_section:
                    sections.append("".join(temp_section))
            elif current_length + para_length + 2 > max_section_length and current_section:
                sections.append("\n\n".join(current_section))
                current_section = [para]
                current_length = para_length
            else:
                current_section.append(para)
                current_length += para_length + 2

        if current_section:
            sections.append("\n\n".join(current_section))

        logger.info(
            f"文本已分割为 {len(sections)} 个大段，平均每段 {sum(len(s) for s in sections) / len(sections):.2f} 字符")
        return sections

    def split_into_sentences(self, text):
        """将文本分割成句子"""
        sentences = re.split(r'(?<=[。！？.!?])', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_semantic_chunks_simple(self, text):
        """使用简单的基于句子的方法创建语义块"""
        logger.info("使用备选分块方法...")
        sentences = self.split_into_sentences(text)

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

        logger.info(f"备选方法生成了 {len(chunks)} 个语义块")
        return chunks

    def _split_long_paragraph(self, para: str, max_length: int) -> List[str]:
        sentences = re.split(r'(?<=[。！？.!?])', para)
        sections = []
        current_section = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_length = len(sentence)
            if current_length + sent_length > max_length:
                if current_section:
                    sections.append("".join(current_section))
                current_section = [sentence]
                current_length = sent_length
            else:
                current_section.append(sentence)
                current_length += sent_length

        if current_section:
            sections.append("".join(current_section))
        return sections

    def create_semantic_chunks(self, text: str) -> List[str]:
        if len(text) < 100:
            return [text]

        system_prompt = """你是一个专门用于文本分块的AI助手。你的任务是分析给定文本，并识别语义边界，将文本分割成多个语义连贯的块。
        重要规则：
        1. 不要更改原文的任何内容，包括标点符号和空格
        2. 只在自然的语义边界处进行分块，通常是在一个主题结束、另一个主题开始的地方
        3. 分块后内容按顺序拼接应该与原文完全一致
        4. 如果文章内容较少，可以分成较少的块（2-3个）
        5. 如果文章内容较多，可以分成更多的块（4-6个）
        """

        prompt = f"""分析以下文本，将其分割成多个语义连贯的块。
        规则:
        - 不要删除、修改或概括任何原文
        - 在自然的语义边界处分块，如主题变化、段落转换等地方（必须在句子结束处分块，不能破坏句子的完整性）
        - 分块的数量根据文本长度适当调整，短文本2-3个块，长文本4-6个块
        - 返回格式：每个块之间使用 [CHUNK_BREAK] 分隔
        文本:
        {text}
        分块结果:"""

        response = self._call_ollama(prompt, system_prompt)
        result_chunks = [chunk.strip() for chunk in response.split(
            "[CHUNK_BREAK]") if chunk.strip()]
        if len(result_chunks) <= 1:
            logger.warning("LLM只生成了一个块或返回空结果，使用备选分块方法")
            return self.create_semantic_chunks_simple(text)

        reconstructed_text = "".join([chunk.strip()
                                     for chunk in result_chunks])
        original_text = text.strip()
        similarity_ratio = len(reconstructed_text) / \
            len(original_text) if original_text else 0

        if similarity_ratio < 0.95 or similarity_ratio > 1.05:
            logger.warning(
                f"分块内容与原文不匹配 (相似度比例: {similarity_ratio:.4f})，使用备选分块方法")
            return self.create_semantic_chunks_simple(text)
        return result_chunks


class Chunker:
    def __init__(self, model_name="gemma2:27b", host="http://127.0.0.1:11434"):
        self.chunker = TwoStageSemanticChunker(model_name, host)

    def split_text(self, text: str, strategy='semantic') -> List[LangChainDocument]:
        text = Parser.clean_text(text)
        if len(text) < CONFIG.MIN_CHUNK_LENGTH:
            return []
        markers, pages, clean_text = self._extract_markers(text)
        sections = self.chunker.split_text_into_sections(clean_text)

        all_chunks = []
        if strategy == 'semantic':
            chunk_method = self.chunker.create_semantic_chunks_simple
        # Note, if one of the following conditions met:
        # 1. Ollama failed (e.g. ollama API request failed, specified model not found, etc.),
        # 2. result_chunks <= 1,
        # 3. LLM-based chunk performance is bad,
        # rule-based chunker (create_semantic_chunks_simple) will be used.
        elif strategy == 'semantic_ollama':
            chunk_method = self.chunker.create_semantic_chunks

        for section in sections:
            all_chunks.extend(chunk_method(section))

        return self._reconstruct_chunks(all_chunks, markers, pages, len(clean_text))

    def _extract_markers(self, text: str) -> tuple[List, List, str]:
        markers = []
        pages = []
        clean_text = ""
        last_pos = 0

        for match in re.finditer(r'\[/(table|formula|figure)\]\[\d+\]\[/\1\]', text):
            start, end = match.span()
            markers.append((match.group(), start))
            clean_text += text[last_pos:start]
            last_pos = end

        remaining_text = text[last_pos:]
        last_pos = 0
        final_clean = ""

        for match in re.finditer(r'\[PAGE\]\[\d+\]\[PAGE\]', remaining_text):
            start, end = match.span()
            pages.append((match.group(), start))
            final_clean += remaining_text[last_pos:start]
            last_pos = end

        final_clean += remaining_text[last_pos:]
        return markers, pages, clean_text + final_clean

    def _reconstruct_chunks(self, chunks: List[str], markers: List, pages: List, orig_length: int) -> List[LangChainDocument]:
        reconstructed = []
        current_pos = 0

        for chunk in chunks:
            chunk_start = current_pos
            chunk_end = current_pos + len(chunk)
            new_chunk = chunk

            while markers and markers[0][1] < chunk_end:
                marker, pos = markers.pop(0)
                rel_pos = pos - chunk_start
                new_chunk = new_chunk[:rel_pos] + marker + new_chunk[rel_pos:]
                chunk_end += len(marker)

            while pages and pages[0][1] < chunk_end:
                page, pos = pages.pop(0)
                rel_pos = pos - chunk_start
                new_chunk = new_chunk[:rel_pos] + page + new_chunk[rel_pos:]
                chunk_end += len(page)

            reconstructed.append(new_chunk)
            current_pos += len(chunk)

        return [LangChainDocument(page_content=chunk) for chunk in reconstructed + [m[0] for m in markers] + [p[0] for p in pages]]
