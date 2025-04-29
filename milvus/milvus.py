"""
提供milvus向量数据库的插入和查询功能
"""
from pymilvus import MilvusClient, DataType
from config.backend import MilvusConfig
from loguru import logger
import json

_milvus_client = None
_default_collection_name = MilvusConfig.default_collection_name
_default_partition_name = MilvusConfig.default_partition_name

def _get_milvus_client() -> MilvusClient:
    """
    获取 Milvus 客户端实例
    """
    global _milvus_client

    if _milvus_client is None:
        config = MilvusConfig()
        
        uri = f"http://{config.host}:{config.port}"
        
        token = f"{config.user}:{config.password}" if config.user else None
        
        _milvus_client = MilvusClient(
            uri=uri,
            token=token,
            db_name=config.db_name
        )
    return _milvus_client

def _build_schema() -> None:
    """
    创建集合的schema
    """
    schema = MilvusClient.create_schema()
    
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=768
    )
    schema.add_field(
        field_name="chunk", 
        datatype=DataType.VARCHAR,
        max_length=2048
    )
    schema.add_field(
        field_name="metadata",
        datatype=DataType.JSON
    )
    return schema

def _build_index() -> None:
    """
    创建索引
    """
    client = _get_milvus_client()
    index_params = client.prepare_index_params()
    
    index_params.add_index(
        field_name="embedding",
        index_name="embedding_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"nlist": 128}
    )

    return index_params

def _create_collection(
        collection_name: str = _default_collection_name) -> None:
    """
    创建集合
    """
    client = _get_milvus_client()
    
    if client.has_collection(collection_name):
        raise ValueError(f"集合 {collection_name} 已存在")
    
    schema = _build_schema()
    index_params = _build_index()
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

def _create_partition(
    collection_name: str = _default_collection_name,
    partition_name: str = _default_partition_name
) -> None:
    """
    创建分区
    """
    client = _get_milvus_client()
    
    if not client.has_collection(collection_name):
        raise ValueError(f"集合 {collection_name} 不存在")
    
    if client.has_partition(collection_name, partition_name):
        raise ValueError(f"分区 {partition_name} 已存在")
    
    client.create_partition(
        collection_name=collection_name,
        partition_name=partition_name
    )

def _store_embedding(
    embedding: list,
    chunk_text: list,
    metadata: list,
    collection_name: str = _default_collection_name,
    partition_name: str = _default_partition_name
) -> None:
    try:
        client = _get_milvus_client()
        
        if not client.has_collection(collection_name):
            _create_collection(collection_name)

        if not client.has_partition(collection_name, partition_name):
            client.create_partition(
                collection_name=collection_name,
                partition_name=partition_name
            )

        data = []

        for i in range(len(embedding)):
            tmp = {
                "embedding": embedding[i],
                "chunk": chunk_text[i],
                "metadata": json.dumps(metadata[i])
            }
            data.append(tmp)

        
        client.insert(
            collection_name=collection_name,
            partition_name=partition_name,
            data=data
        )
    except Exception as e:
        raise RuntimeError(f"插入数据失败 of store_embedding: {e}")

def store_embedding_task(
        embedding: list,
        chunk_text: list,
        task: "Task") -> None:
    try:
        metadata_json = task.to_metadata()
        if metadata_json is None:
            raise ValueError("文档数据不能为空")
        _store_embedding(
            embedding,
            chunk_text,
            [metadata_json] * len(embedding),
            MilvusConfig.default_collection_name,
            MilvusConfig.default_partition_name
        )
        
    except Exception as e:
        raise RuntimeError(f"插入数据失败: {e}")

def search_embedding(
    query_embedding: list,
    top_k: int = 3,
    collection_name: str = _default_collection_name,
    partition_name: str = _default_partition_name,
) -> list:
    """
    查询集合中的向量数据
    """
    client = _get_milvus_client()
    
    if not client.has_collection(collection_name):
        raise ValueError(f"集合 {collection_name} 不存在")
    
    if not client.has_partition(collection_name, partition_name):
        raise ValueError(f"分区 {partition_name} 不存在")
    
    search_params = {
        "metric_type": "COSINE",
    }
    
    results = client.search(
        collection_name=collection_name,
        partition_names=[partition_name],
        data=[query_embedding],
        limit=top_k,
        search_params=search_params,
        output_fields=["chunk"]
    )
    
    return results