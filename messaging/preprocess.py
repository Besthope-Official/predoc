from loguru import logger

from schemas import Task
from typing import Optional
from api.utils import ModelLoader
from predoc.pipeline import DefaultPDFPipeline


def preprocess(
    task: Task,
    use_cached: bool = True,
    model_loader: Optional[ModelLoader] = None,
    collection_name: Optional[str] = None,
    partition_name: Optional[str] = None,
) -> None:
    """
    进行预处理任务，从 OSS 上获取文件，并进行预处理
    预处理后得到的图表等文件会上传到 OSS 上
    最后将处理结果存储到 Milvus 中

    Args:
        use_cached: 是否使用已解析的文本，跳过解析步骤
    """
    try:
        pipeline = DefaultPDFPipeline(model_loader=model_loader)
        chunks, embeddings = pipeline.process(task.document)
    except Exception as e:
        logger.error(f"预处理任务 {task.task_id} 出错: {e}")
        raise

    pipeline.store_embedding(
        chunks,
        embeddings,
        doc=task.document,
        collection_name=collection_name,
        partition_name=partition_name,
    )
    logger.info(f"任务 {task.task_id} 处理完成，结果已存储到 Milvus")
