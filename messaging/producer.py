"""任务生产者：基础 MQ 生产者 + 从本地/OSS 的 PDF 构建 Task 并推送到 MQ 的工具"""

from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Iterable, Optional
from loguru import logger

from config.backend import RabbitMQConfig, OSSConfig
from schemas import Task, TaskStatus
from schemas.document import Document
from backends.minio import upload_file
from backends.rabbitmq import RabbitMQBase
import pika

_oss_config = OSSConfig.from_yaml()


class TaskProducer(RabbitMQBase):
    """
    Producer for preprocess tasks in MQ.
    Publish tasks into a MQ for async processing.
    """

    def __init__(self, config: RabbitMQConfig) -> None:
        super().__init__(config)
        self._connect()

    def _connect(self) -> None:
        self._ensure_connection()
        # create or check (by default, taskQueue is created on published side)
        queue = getattr(self.config, "task_queue", "taskQueue")
        assert self.channel is not None
        self.channel.queue_declare(queue=queue, durable=True)
        logger.info(f"Connected! 任务队列: {queue}")

    def publish(self, task: Task) -> None:
        """发布任务到消息队列"""
        try:
            assert self.channel is not None
            queue = getattr(self.config, "task_queue", "taskQueue")
            self.channel.basic_publish(
                exchange="",
                routing_key=queue,
                body=task.to_json(),
                properties=pika.BasicProperties(delivery_mode=2),  # 使消息持久化
            )
            logger.info(f"任务 {task.task_id} 已发布")
        except Exception as e:
            logger.error(f"发布任务时出错: {e}")


def build_minimal_document_from_pdf(
    pdf_path: Path, *, object_name: Optional[str] = None, doc_type: str = "paper"
) -> Document:
    """
    基于 PDF 文件路径构建最小可用的 Document：
    - title: 去除扩展名的文件名
    - authors/keywords: 为空列表
    - fileName: 对象存储中的文件名（此处沿用本地文件名，生产环境建议带前缀）
    - doc_type: 由入参传入
    """
    title = pdf_path.stem
    return Document(
        title=title,
        authors=[],
        keywords=[],
        fileName=object_name or pdf_path.name,
        doc_type=doc_type,
    )


def build_task_for_document(
    doc: Document,
    *,
    task_type: str = "default",
    destination_collection: Optional[str] = None,
) -> Task:
    t = Task(
        taskId=uuid4(),
        status=TaskStatus.PENDING,
        document=doc,
        createdAt=datetime.utcnow(),
    )
    t.task_type = task_type
    if destination_collection:
        # attach destination collection hint
        t.destination_collection = destination_collection
    return t


class PDFTaskPublisher(TaskProducer):
    def publish_tasks_from_pdfs(
        self,
        pdf_paths: Iterable[Path],
        task_type: str = "default",
        *,
        upload_to_oss: bool = False,
        oss_bucket: Optional[str] = None,
        oss_object_prefix: str = "",
        destination_collection: Optional[str] = None,
    ) -> int:
        """遍历给定的文件/目录集合，查找所有 PDF 文件，构建并发布任务。
        - task_type: 任务类型（将写入 Task.task_type）
        - upload_to_oss: 若为 True，则先将本地 PDF 上传至 OSS 的 pdf_bucket
        - oss_bucket: 覆盖上传目标 bucket，默认为配置文件中的 pdf_bucket
        - oss_object_prefix: 上传到 OSS 的对象名前缀（如 "incoming/"）
        """

        def iter_all_pdfs(paths: Iterable[Path]):
            for p in paths:
                if p.is_dir():
                    for f in p.rglob("*"):
                        if f.is_file() and f.suffix.lower() == ".pdf":
                            yield f
                else:
                    if p.is_file() and p.suffix.lower() == ".pdf":
                        yield p

        published = 0
        bucket = oss_bucket or _oss_config.pdf_bucket
        prefix = (oss_object_prefix or "").strip("/")
        for pdf in iter_all_pdfs(pdf_paths):
            obj_name = f"{prefix}/{pdf.name}" if prefix else pdf.name

            if upload_to_oss:
                try:
                    upload_file(pdf, object_name=obj_name, bucket_name=bucket)
                except Exception as e:
                    logger.error(f"上传至 OSS 失败，跳过 {pdf}: {e}")
                    continue

            doc = build_minimal_document_from_pdf(pdf, object_name=obj_name)
            # 记录对象所在的 bucket，便于 consumer 侧准确读取
            doc.bucket = bucket if upload_to_oss or oss_bucket else None
            task = build_task_for_document(
                doc,
                task_type=task_type,
                destination_collection=destination_collection,
            )
            self.publish(task)
            published += 1

        logger.info(
            f"已发布 {published} 个任务到队列 {getattr(self.config, 'task_queue', 'taskQueue')}"
        )
        return published
