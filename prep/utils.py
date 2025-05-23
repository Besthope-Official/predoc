import re
from loguru import logger
from typing import List


class TextSplitter:
    '''Utils for splitting text into smaller sections to chunk.'''
    @staticmethod
    def split_text_into_sections(text: str, max_section_length=1500, min_section_length=300):
        """将文本分割成适合模型处理的大段，增加最小长度控制"""
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

                sub_sections = TextSplitter._split_long_paragraph(
                    para, max_section_length)
                sections.extend(sub_sections)
            elif current_length + para_length + 2 > max_section_length and current_section:
                if current_length >= min_section_length:
                    sections.append("\n\n".join(current_section))
                    current_section = [para]
                    current_length = para_length
                else:
                    current_section.append(para)
                    current_length += para_length + 2
            else:
                current_section.append(para)
                current_length += para_length + 2

        if current_section:
            final_section = "\n\n".join(current_section)
            if len(final_section) < min_section_length and sections:
                sections[-1] = sections[-1] + "\n\n" + final_section
            else:
                sections.append(final_section)

        logger.info(
            f"文本已分割为 {len(sections)} 个大段，平均每段 {sum(len(s) for s in sections) / len(sections):.2f} 字符")
        return sections

    @staticmethod
    def split_into_sentences(text):
        """将文本分割成句子"""
        sentences = re.split(r'(?<=[。！？.!?])', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _split_long_paragraph(para: str, max_length: int) -> List[str]:
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


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\x00-\x1F\x7F-\x9F\u200b-\u200d\uFEFF]', '', text)
    return text.strip()


def extract_markers(text: str) -> tuple[List, List, str]:
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


def reconstruct_chunks(chunks: List[str], markers: List, pages: List, orig_length: int) -> List[str]:
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

    return [chunk for chunk in reconstructed + [m[0] for m in markers] + [p[0] for p in pages]]
