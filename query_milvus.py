import json
from prep.processor import PDFProcessor
from task.milvus import search_embedding

processor = PDFProcessor(
    parse_method="auto",
    chunk_strategy="semantic_ollama",
)

QUESTION_JSON_DIR = "AI预实验文献/questions.json"


def retrieve_documents(query, k=3, strategy="direct"):
    """Retrieve documents from Milvus using the query."""
    if strategy == "direct":
        embd = processor.embedding(query)
        results = search_embedding(embd, k)
    return results


def hit_at_k(results, doc_title, k=5):
    '''
        Validate query results title match corresponding doc title.
        >>> results = [
            {"title": "doc1", ...}
            {"title": "doc2", ...},
            {"title": "doc3", ...},
        ]
        >>> hit_at_k(results, "doc1", k=3)
        >>> True
    '''
    return any(
        doc["title"] == doc_title
        for doc in results[:k]
    )


if __name__ == "__main__":
    with open(QUESTION_JSON_DIR, "r", encoding="utf-8") as f:
        qa_json = json.load(f)
    zh_questions = qa_json["chinese_questions"]
    en_questions = qa_json["english_questions"]

    zh_n, en_n = len(zh_questions), len(en_questions)
    zh_hit_n, en_hit_n = 0, 0
    for question in zh_questions:
        raw_results = retrieve_documents(question["question"])[0]
        metadatas = [
            {
                "metadata": json.loads(raw_result['entity']['metadata'])
            } for raw_result in raw_results
        ]
        results = [{
            "title": metadata["metadata"]["title"]
        } for metadata in metadatas]
        answer_title = question["source"]
        if hit_at_k(results, answer_title):
            zh_hit_n += 1

    for question in en_questions:
        raw_results = retrieve_documents(question["question"])[0]
        metadatas = [
            {
                "metadata": json.loads(raw_result['entity']['metadata'])
            } for raw_result in raw_results
        ]
        results = [{
            "title": metadata["metadata"]["title"]
        } for metadata in metadatas]
        answer_title = question["source"]
        if hit_at_k(results, answer_title):
            en_hit_n += 1
    print(
        f"zh_hit_n: {zh_hit_n}, zh_n: {zh_n}, zh_hit_rate: {zh_hit_n/zh_n:.2f}")
    print(
        f"en_hit_n: {en_hit_n}, en_n: {en_n}, en_hit_rate: {en_hit_n/en_n:.2f}")
