'''知识库检索'''
import json
from typing import List, Dict, Any

from task.milvus import search_embedding
from api.utils import ModelLoader


def retrieve_documents(query: str, k: int = 50, strategy: str = "direct") -> List[Dict[str, Any]]:
    """
    从Milvus中检索文档

    Args:
        query: 查询字符串
        k: 返回结果数量
        strategy: 检索策略

    Returns:
        处理后的检索结果列表
    """
    if strategy == "direct":
        embedder = ModelLoader().embedder
        embd = embedder.generate_embedding(query)
        raw_results = search_embedding(embd, k)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    metadatas = [
        {
            "metadata": json.loads(raw_result['entity']['metadata'])
        } for raw_result in raw_results[0]
    ]
    return metadatas
