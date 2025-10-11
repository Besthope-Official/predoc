import json
from typing import List, Dict, Any

from loguru import logger
from urllib.parse import urljoin

from task.milvus import search_embedding
from api.utils import ModelLoader
from config.backend import OSSConfig

_oss_config = OSSConfig.from_yaml()


def generate_image_url(title: str, element_type: str, idx: int) -> str:
    """
    生成图片的公开访问 URL
    Args:
        title: 文档的标题
        element_type: 元素类型/(figure 或 table)
        idx: 元素的索引
    Returns:
        图片的公开访问 URL
    """
    object_name = f"{title}/{element_type}_{idx}.png"
    url = urljoin(
        f"http://{_oss_config.endpoint}",
        f"{_oss_config.preprocessed_files_bucket}/{object_name}",
    )
    return url


def retrieve_documents(
    query: str, k: int = 50, collection: str = "default_collection"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    从Milvus中检索文档

    Args:
        query: 查询字符串
        k: 返回结果数量
        collection: 查询的 collection 名称，默认全部 collection 在同一 Milvus 数据库下

    Returns:
        处理后的检索结果字典，包含去重后的文档和文本块列表
    """
    model_loader = ModelLoader()
    embedder = model_loader.embedder
    embd = embedder.generate_embedding(query)
    raw_results = search_embedding(embd, k, collection)

    docs = []
    chunks = []
    seen_titles = set()
    doc_id_map = {}

    for raw_result in raw_results[0]:
        try:
            metadata = json.loads(raw_result["entity"]["metadata"])
            title = metadata.get("title", "")

            if title not in seen_titles:
                seen_titles.add(title)
                doc_id = len(docs)
                doc_id_map[title] = doc_id

                doc = {
                    "idx": raw_result["id"],
                    "title": title,
                    "authors": metadata.get("authors", []),
                    "publicationDate": metadata.get("publicationDate", ""),
                    "language": metadata.get("language", ""),
                    "keywords": [item["name"] for item in metadata.get("keywords", [])],
                }
                docs.append(doc)
            images = []

            chunk = {
                "id": raw_result["id"],
                "doc_id": doc_id_map.get(title, -1),
                "page": raw_result["entity"]["page"],
                "text": raw_result["entity"]["chunk"],
                "images": images,
            }
            chunks.append(chunk)

        except (KeyError, json.JSONDecodeError) as e:
            logger.warning(f"解析Milvus结果失败: {e}, 跳过结果: {raw_result}")
            continue

    return {"docs": docs, "chunks": chunks}
