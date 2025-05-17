import os
import shutil
from pathlib import Path
from loguru import logger

from config.backend import OSSConfig
from prep.processor import PDFProcessor

from task.oss import download_file, upload_file, clear_directory, check_file_exists
from task.milvus import store_embedding_task
from models import Task


def preprocess(task: Task, use_cached: bool = True) -> None:
    """
    进行预处理任务，从 OSS 上获取文件，并进行预处理
    预处理后得到的图表等文件会上传到 OSS 上
    最后将处理结果存储到 Milvus 中

    Args:
        use_cached: 是否使用已解析的文本，跳过解析步骤
    """
    try:
        temp_base_dir = Path(os.environ.get('TEMP', '/tmp'))
        temp_dir = temp_base_dir / f"task_{task.task_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        file_name_without_extension = Path(task.document.fileName).stem
        local_file_path = temp_dir / task.document.fileName

        processor = PDFProcessor(
            output_dir=temp_dir,
            parse_method="auto",
            chunk_strategy="semantic",
            upload_to_oss=True
        )
        parsed_text = f"{file_name_without_extension}/text.txt"
        if use_cached and check_file_exists(parsed_text):
            local_text_path = temp_dir / "text.txt"

            download_file(
                object_name=parsed_text,
                file_path=local_text_path,
                bucket_name=OSSConfig.minio_bucket
            )
            logger.info(f"预解析文件 {parsed_text} 已从 OSS 下载到 {local_text_path}")
            with open(local_text_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = processor.chunk(text)
            embeddings = processor.embeddings(chunks)
        else:
            download_file(
                object_name=task.document.fileName,
                file_path=local_file_path
            )
            logger.info(
                f"文件 {task.document.fileName} 已从 OSS 下载到 {local_file_path}")

            chunks, embeddings = processor.preprocess(
                file_path=str(local_file_path),
                warpper=False
            )

            logger.info(
                f"文件 {task.document.fileName} 解析完成，解析结果已保存到 {temp_dir}")

            shutil.rmtree(temp_dir)
            logger.info(f"临时目录 {temp_dir} 已清理")

    except Exception as e:
        logger.error(f"预处理任务 {task.task_id} 出错: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            prefix = Path(task.document.fileName).stem
            clear_directory(
                prefix=prefix,
                bucket_name=OSSConfig.minio_bucket,
                recursive=True
            )
            logger.info(f"任务失败，临时目录 {temp_dir} 已清理")
        raise

    store_embedding_task(
        embedding=embeddings,
        chunk_text=chunks,
        task=task
    )
    logger.info(f"任务 {task.task_id} 处理完成，结果已存储到 Milvus")
