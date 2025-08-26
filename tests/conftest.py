"""测试配置文件"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path

os.environ["ENV"] = "test"


@pytest.fixture(scope="session")
def test_env():
    """确保测试运行在test环境中"""
    original_env = os.environ.get("ENV")
    os.environ["ENV"] = "test"
    yield
    if original_env:
        os.environ["ENV"] = original_env
    else:
        os.environ.pop("ENV", None)


@pytest.fixture(scope="class")
def temp_test_dir():
    """提供临时测试目录"""
    temp_dir = tempfile.mkdtemp(prefix="predoc_test_")
    yield Path(temp_dir)
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="class")
def mock_oss_config():
    """模拟OSS配置，避免真实上传"""
    original_endpoint = os.environ.get("MINIO_ENDPOINT")
    original_bucket = os.environ.get("TEST_MINIO_BUCKET")
    original_pdf_bucket = os.environ.get("TEST_PDF_BUCKET")

    os.environ["MINIO_ENDPOINT"] = "localhost:9000"
    os.environ["TEST_MINIO_BUCKET"] = "test-prep-perf"
    os.environ["TEST_PDF_BUCKET"] = "test-pdf-perf"

    yield

    if original_endpoint:
        os.environ["MINIO_ENDPOINT"] = original_endpoint
    if original_bucket:
        os.environ["TEST_MINIO_BUCKET"] = original_bucket
    if original_pdf_bucket:
        os.environ["TEST_PDF_BUCKET"] = original_pdf_bucket
