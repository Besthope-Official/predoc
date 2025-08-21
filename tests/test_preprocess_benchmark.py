"""Performance tests for document preprocessing pipeline.

use pytest-benchmark as benchmark plugin
for each run, 3 rounds of tests will be executed,
benchmark metrics mainly focus on execution time.

Usage:

```bash
export TEST_UPLOAD_TO_OSS=true
python -m pytest tests/test_preprocess_benchmark.py -v --benchmark-only -s
python -m pytest tests/test_preprocess_benchmark.py -v --benchmark-save=pdf_performance -s
```

"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from typing import List

from prep.processor import PDFProcessor
from prep.chunker import LLMChunker, SentenceChunker
from prep.parser import YoloParser


class TestPDFProcessing:
    """Performance tests for PDF preprocessing stages."""

    @pytest.fixture(scope="class")
    def pdf_files(self) -> List[Path]:
        pdf_dir = Path(__file__).parent.parent / "AI预实验文献/英文"
        files = list(pdf_dir.glob("*.pdf"))
        assert len(files) > 0, "No PDF files found for testing"
        return files

    @pytest.fixture(scope="class")
    def parsed_text(self) -> str:
        path = Path(__file__).parent / "examples/example_text.txt"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            assert len(text) > 0, "Parsed text should not be empty"
            return text

    @pytest.fixture(scope="class")
    def chunks(self) -> List[str]:
        path = Path(__file__).parent / "examples/chunks.json"
        with open(path, "r", encoding="utf-8") as f:
            chunks = list(json.load(f))
            assert len(chunks) > 0, "Chunks should not be empty"
            return chunks

    @pytest.fixture(scope="class")
    def processor(self, test_env, temp_test_dir, mock_oss_config) -> PDFProcessor:
        upload_to_oss = os.getenv("TEST_UPLOAD_TO_OSS", "false").lower() == "true"

        return PDFProcessor(
            parser=YoloParser(),
            chunker=SentenceChunker(),
            output_dir=str(temp_test_dir),
            upload_to_oss=upload_to_oss,
        )

    @pytest.fixture(scope="class", autouse=True)
    def setup_teardown(self, temp_test_dir):
        yield

    def test_parse_stage_benchmark(
        self, pdf_files: List[Path], processor: PDFProcessor, benchmark
    ):
        pdf_file = pdf_files[0]

        def parse_only():
            return processor.parse(str(pdf_file))

        text = benchmark.pedantic(parse_only, rounds=3)
        assert len(text) > 0, "解析的文本不能为空"

    def test_chunk_stage_benchmark(
        self, parsed_text: str, processor: PDFProcessor, benchmark
    ):
        def chunk_only():
            return processor.chunk(parsed_text)

        chunks = benchmark.pedantic(chunk_only, rounds=3)
        assert len(chunks) > 0, "分块数量必须大于0"

    def test_embedding_stage_benchmark(
        self, chunks, processor: PDFProcessor, benchmark
    ):
        def embedding_only():
            return processor.embeddings(chunks)

        embeddings = benchmark.pedantic(embedding_only, rounds=3)
        assert len(embeddings) > 0, "嵌入数量必须大于0"

    @pytest.mark.skip
    def test_single_pdf_parsing_benchmark(
        self, pdf_files: List[Path], processor: PDFProcessor, benchmark
    ):
        def pdf_processing():
            pdf_file = pdf_files[0]
            text = processor.parse(str(pdf_file))
            chunks = processor.chunk(text)
            embeddings = processor.embeddings(chunks)
            return {
                "text_length": len(text),
                "chunk_count": len(chunks),
                "embedding_dimensions": (
                    embeddings.shape[1] if len(embeddings) > 0 else 0
                ),
                "file_size": pdf_file.stat().st_size,
            }

        result = benchmark.pedantic(pdf_processing, rounds=1)

        assert result["text_length"] > 0, "提取的文本不能为空"
        assert result["chunk_count"] > 0, "分块数量必须大于0"

    @pytest.mark.skipif(
        not os.getenv("TEST_UPLOAD_TO_OSS", "false").lower() == "true",
        reason="OSS upload test skipped. Set TEST_UPLOAD_TO_OSS=true to enable.",
    )
    def test_preprocess_with_oss_benchmark(self, pdf_files: List[Path], benchmark):
        pdf_file = pdf_files[0]
        # validate if can connect to minio
        from task.oss import get_minio_client

        try:
            client = get_minio_client()
            if "test-prep-perf" not in [
                bucket.name for bucket in client.list_buckets()
            ]:
                pytest.fail("MinIO test bucket 'test-prep-perf' does not exist.")
        except Exception as e:
            pytest.skip(f"MinIO connection failed: {e}")

        def preprocess_with_oss():
            with tempfile.TemporaryDirectory(prefix="predoc_test_") as temp_dir:
                processor_with_oss = PDFProcessor(
                    chunker=LLMChunker(), output_dir=temp_dir, upload_to_oss=True
                )
                return processor_with_oss.preprocess(str(pdf_file), wrapper=True)

        result = benchmark.pedantic(preprocess_with_oss, rounds=1)

        assert len(result) > 0, "OSS上传预处理结果不能为空"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
