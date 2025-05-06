from task.task import Task
from config.backend import OSSConfig
from task.oss import download_file, upload_file
from prep.parser import Parser
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
        # 创建一个临时目录用于存储下载的文件和解析结果
        temp_base_dir = Path(os.environ.get('TEMP', '/tmp'))
        temp_dir = temp_base_dir / f"task_{task.task_id}"  # 使用任务ID作为目录名
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 在临时目录下创建一个文件用于存储下载的PDF
        file_name_without_extension = Path(task.document.fileName).stem
        local_file_path = temp_dir / task.document.fileName
        download_file(
            object_name=task.document.fileName,
            file_path=local_file_path,
            bucket_name=OSSConfig().minio_bucket
        )
        logger.info(f"文件 {task.document.fileName} 已从 OSS 下载到 {local_file_path}")

        # 初始化解析器
        parser = Parser()
        parser.save_path = temp_dir  # 将解析结果保存到临时目录

        # 解析文件
        parser.process_pdf(
            pdf_path=str(local_file_path),
            output_dir=str(parser.save_path)  
        )
        logger.info(f"文件 {task.document.fileName} 解析完成，解析结果已保存到 {parser.save_path}")

        # 将解析后的图表等文件上传到 OSS
        uploaded_files = []
        for file_type in ["figures", "formulas", "tables"]:
            type_dir = parser.save_path / file_name_without_extension / file_type
            logger.info(f"检查目录 {type_dir} 是否存在以及是否有文件")
            if type_dir.exists():
                for file in type_dir.iterdir():
                    if file.is_file():
                        # 生成唯一的文件路径，包含文件名前缀和文件类型
                        unique_file_name = f"{file_name_without_extension}/{file_type}/{file.name}"
                        logger.info(f"尝试上传文件 {file} 到 MinIO，路径为 {unique_file_name}")
                        upload_result = upload_file(
                            file_path=file,
                            object_name=unique_file_name,
                            bucket_name=OSSConfig().minio_bucket
                        )
                        logger.info(f"文件 {file} 上传到 OSS 结果: {upload_result}")
                        uploaded_files.append(unique_file_name)
        if not uploaded_files:
            logger.info("没有找到要上传的文件")
        else:
            logger.info(f"上传到 MinIO 的文件列表: {uploaded_files}")

        # 清理临时文件
        shutil.rmtree(temp_dir)
        logger.info(f"临时目录 {temp_dir} 已清理")



    except Exception as e:
        logger.error(f"预处理任务 {task.task_id} 出错: {e}")
        task.status = TaskStatus.FAILED
        task.finished_at = datetime.now()
        # 确保在任务失败时也清理临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"任务失败，临时目录 {temp_dir} 已清理")
        raise  # 重新抛出异常，以便上层调用者可以处理
