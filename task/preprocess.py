from task.task import Task
from oss import download_file, upload_file
from parser import Parser
from loguru import logger
import os
from pathlib import Path
import shutil
from datetime import datetime

def preprocess(task: Task):
    """
    进行预处理任务，从 OSS 上获取文件，并进行预处理
    预处理后得到的图表等文件会上传到 OSS 上
    """
    try:
        # 创建一个临时目录用于存储下载的文件和解析结果，TEMP为系统环境变量
        temp_dir = Path(os.environ.get('TEMP', '/tmp')) / f"task_{task.task_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 从 OSS 下载文件到本地临时目录
        local_file_path = temp_dir / task.document.fileName
        download_file(
            object_name=task.document.fileName,
            file_path=local_file_path,
            bucket_name=OSSConfig().minio_bucket
        )
        logger.info(f"文件 {task.document.fileName} 已从 OSS 下载到 {local_file_path}")

        # 初始化解析器
        parser = Parser()

        # 解析文件
        parser.process_pdf(
            pdf_path=str(local_file_path),
            output_dir=str(temp_dir)
        )
        logger.info(f"文件 {task.document.fileName} 解析完成，解析结果已保存到 {temp_dir}")

        # 将解析后的图表等文件上传到 OSS
        for file_type in ["figures", "formulas", "tables"]:
            type_dir = temp_dir / file_type
            if type_dir.exists():
                for file in type_dir.iterdir():
                    if file.is_file():
                        # 上传到 OSS
                        upload_file(
                            file_path=file,
                            object_name=str(file.relative_to(temp_dir)),
                            bucket_name=OSSConfig().minio_bucket
                        )
                        logger.info(f"文件 {file} 已上传到 OSS")

        # 清理临时文件
        shutil.rmtree(temp_dir)
        logger.info(f"临时目录 {temp_dir} 已清理")

        # 更新任务状态为成功
        task.status = TaskStatus.DONE
        task.finished_at = datetime.now()
        logger.info(f"任务 {task.task_id} 预处理完成")

    except Exception as e:
        logger.error(f"预处理任务 {task.task_id} 出错: {e}")
        task.status = TaskStatus.FAILED
        task.finished_at = datetime.now()
        # 确保在任务失败时也清理临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"任务失败，临时目录 {temp_dir} 已清理")
        raise

    pass
