import json
from typing import List, Dict, Any

from task.milvus import search_embedding
from api.utils import ModelLoader
from loguru import logger
from config.backend import OSSConfig
from urllib.parse import urljoin

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
    url = urljoin(f"http://{OSSConfig.minio_endpoint}", f"{OSSConfig.minio_bucket}/{object_name}")
    return url
def retrieve_documents(query: str, k: int = 50, strategy: str = "direct") -> Dict[str, List[Dict[str, Any]]]:
    """
    从Milvus中检索文档

    Args:
        query: 查询字符串
        k: 返回结果数量
        strategy: 检索策略

    Returns:
        处理后的检索结果字典，包含去重后的文档和文本块列表
    """
    if strategy == "direct":
        model_loader = ModelLoader()
        embedder = model_loader.embedder
        embd = embedder.generate_embedding(query)
        raw_results = search_embedding(embd, k)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    docs = []
    chunks = []
    seen_titles = set()  # 用于去重的集合
    doc_id_map = {}      # 用于存储每篇文章的doc_id映射

    for raw_result in raw_results[0]:
        try:
            metadata = json.loads(raw_result['entity']['metadata'])
            title = metadata.get("title", "")

            # 如果标题未见过，则添加到docs列表，并记录doc_id
            if title not in seen_titles:
                seen_titles.add(title)
                doc_id = len(docs)  # 为新文档分配doc_id
                doc_id_map[title] = doc_id

                doc = {
                    "idx": raw_result['id'],
                    "title": title,
                    "authors": metadata.get("authors", []),
                    "publicationDate": metadata.get("publicationDate", ""),
                    "language": metadata.get("language", ""),
                    "keywords": [item["name"] for item in metadata.get("keywords", [])]
                }
                docs.append(doc)
            text = raw_result['entity']['chunk']
            images = []
            '''import re
            fig_matches = re.findall(r'\[/figure\]\[(\d+)\]\[/figure\]', text)
            table_matches = re.findall(r'\[/table\]\[(\d+)\]\[/table\]', text)
            for fig_id in fig_matches:
                fig_url = generate_image_url(title, "figure", int(fig_id))
                images.append({
                    "type": "figure",
                    "id": int(fig_id),
                    "url": fig_url
                })
                logger.debug(f"Generated figure URL: {fig_url}")

            for table_id in table_matches:
                table_url = generate_image_url(title, "table", int(table_id))
                images.append({
                    "type": "table",
                    "id": int(table_id),
                    "url": table_url
                })
                logger.debug(f"Generated table URL: {table_url}")'''
            chunk = {
                "id": raw_result['id'],
                "doc_id": doc_id_map.get(title, -1),
                "text": text,
                "images":images
            }
            chunks.append(chunk)

        except (KeyError, json.JSONDecodeError) as e:
            logger.warning(f"解析Milvus结果失败: {e}, 跳过结果: {raw_result}")
            continue

    return {"docs": docs, "chunks": chunks}
