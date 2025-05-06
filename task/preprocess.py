import os
import shutil
from pathlib import Path
from loguru import logger

from config.backend import OSSConfig
from prep.processor import PDFProcessor

from task.oss import download_file, upload_file
from milvus.milvus import store_embedding_task
from models import Task


def preprocess(task: Task):
    """
    进行预处理任务，从 OSS 上获取文件，并进行预处理
    预处理后得到的图表等文件会上传到 OSS 上
    """
    try:
        temp_base_dir = Path(os.environ.get('TEMP', '/tmp'))
        temp_dir = temp_base_dir / f"task_{task.task_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        file_name_without_extension = Path(task.document.fileName).stem
        local_file_path = temp_dir / task.document.fileName
        download_file(
            object_name=task.document.fileName,
            file_path=local_file_path
        )
        logger.info(
            f"文件 {task.document.fileName} 已从 OSS 下载到 {local_file_path}")

        save_path = temp_dir
        processor = PDFProcessor(
            output_dir=save_path,
            parse_method="auto",
            chunk_strategy="semantic_ollama",
        )

        chunks, embeddings = processor.preprocess(
            file_path=str(local_file_path),
            warpper=False
        )

        logger.info(
            f"文件 {task.document.fileName} 解析完成，解析结果已保存到 {save_path}")

        shutil.rmtree(temp_dir)
        logger.info(f"临时目录 {temp_dir} 已清理")

    except Exception as e:
        logger.error(f"预处理任务 {task.task_id} 上传文件出错: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"任务失败，临时目录 {temp_dir} 已清理")
        raise

    store_embedding_task(
        embedding=embeddings,
        chunk_text=chunks,
        task=task
    )
    logger.info(f"任务 {task.task_id} 处理完成，结果已存储到 Milvus")
