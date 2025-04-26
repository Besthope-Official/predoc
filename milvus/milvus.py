"""
提供milvus向量数据库的插入和查询功能
"""
from pymilvus import MilvusClient, DataType
from config.backend import MilvusConfig
import json
import os
from datetime import datetime
import random
from pydantic import BaseModel, Field
from peewee import PostgresqlDatabase, Model, CharField

_milvus_client = None

def get_milvus_client() -> MilvusClient:
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

def has_collection(collection_name: str) -> bool:
    """
    检查集合是否存在
    """
    client = get_milvus_client()
    return client.has_collection(collection_name)

def has_partition(
    collection_name: str,
    partition_name: str
) -> bool:
    """
    检查分区是否存在
    """
    client = get_milvus_client()
    
    if not client.has_collection(collection_name):
        raise ValueError(f"集合 {collection_name} 不存在")
    
    return client.has_partition(collection_name, partition_name)

def build_schema() -> None:
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

def build_index() -> None:
    """
    创建索引
    """
    client = get_milvus_client()
    index_params = client.prepare_index_params()
    
    index_params.add_index(
        field_name="embedding",
        index_name="embedding_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"nlist": 128}
    )

    return index_params

def create_collection(collection_name: str) -> None:
    """
    创建集合
    """
    client = get_milvus_client()
    
    if client.has_collection(collection_name):
        raise ValueError(f"集合 {collection_name} 已存在")
    
    schema = build_schema()
    index_params = build_index()
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

def create_partition(
    collection_name: str,
    partition_name: str
) -> None:
    """
    创建分区
    """
    client = get_milvus_client()
    
    if not client.has_collection(collection_name):
        raise ValueError(f"集合 {collection_name} 不存在")
    
    if client.has_partition(collection_name, partition_name):
        raise ValueError(f"分区 {partition_name} 已存在")
    
    client.create_partition(
        collection_name=collection_name,
        partition_name=partition_name
    )

def store_embedding(
    collection_name: str,
    embedding: list,
    chunk_text: list,
    metadata: list,
    partition_name: str = "Default partition"
) -> None:
    try:
        client = get_milvus_client()
        
        if not client.has_collection(collection_name):
            create_collection(collection_name)

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
                "metadata": metadata[i]
            }
            data.append(tmp)

        
        client.insert(
            collection_name=collection_name,
            partition_name=partition_name,
            data=data
        )
    except Exception as e:
        raise RuntimeError(f"插入数据失败: {e}")

def search_embedding(
    collection_name: str,
    query_embedding: list,
    top_k: int = 3,
    partition_name: str = "Default partition",
) -> list:
    """
    查询集合中的向量数据
    """
    client = get_milvus_client()
    
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

def drop_collection(collection_name: str) -> None:
    """
    删除集合
    """
    client = get_milvus_client()
    
    if not client.has_collection(collection_name):
        raise ValueError(f"集合 {collection_name} 不存在")
    
    client.drop_collection(collection_name)

def load_collection(collection_name: str) -> None:
    """
    加载集合
    """
    client = get_milvus_client()
    
    if not client.has_collection(collection_name):
        raise ValueError(f"集合 {collection_name} 不存在")
    
    client.load_collection(collection_name)

def release_collection(collection_name: str) -> None:
    """
    卸载集合
    """
    client = get_milvus_client()
    
    if not client.has_collection(collection_name):
        raise ValueError(f"集合 {collection_name} 不存在")
    
    client.release_collection(collection_name)