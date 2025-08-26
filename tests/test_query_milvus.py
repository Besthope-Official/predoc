from pathlib import Path
import json
import time
import pytest

from task.milvus import _get_milvus_client
from retrieve.search import retrieve_documents
from loguru import logger

QUESTION_JSON_DIR = Path(__file__).parent.parent / "AI预实验文献/questions.json"
K = 50
threshold = 0.3  # Minimum hit rate threshold for the test
zh_hits, zh_total = 0, 0
en_hits, en_total = 0, 0
query_times = []


def timed_retrieve_documents(query: str, k=K):
    start_time = time.time()
    result = retrieve_documents(query, k)
    end_time = time.time()

    elapsed_time = end_time - start_time
    query_times.append(elapsed_time)

    docs = result.get("docs", [])
    results = [{"title": doc["title"]} for doc in docs]

    return results


def hit_at_k(results, doc_title, k=K):
    """
    Validate query results title match corresponding doc title.
    >>> results = [
        {"title": "doc1", ...}
        {"title": "doc2", ...},
        {"title": "doc3", ...},
    ]
    >>> hit_at_k(results, "doc1", k=3)
    >>> True
    """
    return any(doc["title"] == doc_title for doc in results[:k])


@pytest.fixture(scope="session")
def qa_data():
    """Load Q&A data from JSON file."""
    with open(QUESTION_JSON_DIR, "r", encoding="utf-8") as f:
        return json.load(f)


def connect_milvus_available():
    """Check if Milvus connection is available."""
    try:
        client = _get_milvus_client()
        client.list_databases()
        return True
    except Exception as e:
        logger.error(f"Milvus connection failed: {e}")
        return False


class TestQueryMilvus:
    @pytest.mark.skipif(not connect_milvus_available(), reason="Milvus Unavailable")
    @pytest.mark.parametrize(
        "question_type,questions_key",
        [("chinese", "chinese_questions"), ("english", "english_questions")],
    )
    def test_direct_query(self, qa_data, question_type, questions_key):
        """Test questions retrieval for different languages."""
        global zh_hits, zh_total, en_hits, en_total

        questions = qa_data[questions_key]

        if question_type == "chinese":
            zh_total = len(questions)
        else:
            en_total = len(questions)

        for question in questions:
            results = timed_retrieve_documents(question["question"], k=K)
            answer_title = question["source"]

            if hit_at_k(results, answer_title):
                if question_type == "chinese":
                    zh_hits += 1
                else:
                    en_hits += 1

        if question_type == "chinese":
            hit_rate = zh_hits / zh_total if zh_total > 0 else 0
            assert (
                hit_rate >= threshold
            ), f"中文问题命中率 {hit_rate:.4f} 低于阈值 {threshold}"
        else:
            hit_rate = en_hits / en_total if en_total > 0 else 0
            assert (
                hit_rate >= threshold
            ), f"英文问题命中率 {hit_rate:.4f} 低于阈值 {threshold}"

    @pytest.fixture(scope="session", autouse=True)
    def print_final_stats(self):
        """Print final statistics after all tests complete."""
        yield

        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
            print("\n=== 测试统计结果 ===")
            print(f"总查询次数: {len(query_times)}")
            print(f"平均查询时间: {avg_query_time:.4f} 秒")

        zh_hit_rate = zh_hits / zh_total if zh_total > 0 else 0
        en_hit_rate = en_hits / en_total if en_total > 0 else 0

        print(f"中文问题 hit@{K}: {zh_hits}/{zh_total} = {zh_hit_rate:.4f}")
        print(f"英文问题 hit@{K}: {en_hits}/{en_total} = {en_hit_rate:.4f}")
        print(
            f"总体 hit@{K}: {(zh_hits + en_hits)}/{(zh_total + en_total)} = {(zh_hits + en_hits)/(zh_total + en_total):.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
