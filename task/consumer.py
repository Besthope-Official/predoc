"""任务消费者：消费 MQ 任务，运行 Pipeline 并入库，发布任务状态"""

from __future__ import annotations
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import pika

from config.backend import RabbitMQConfig, MilvusConfig
from schemas import TaskStatus, Task
from api.utils import ModelLoader
from predoc.pipeline import get_pipeline
from .mq import RabbitMQBase


class TaskConsumer(RabbitMQBase):
    """
    Consumer class for RabbitMQ document-preprocess tasks.
    Also producer of task result to provide task status for TaskProducer to persist (save to e.g. postgresql).
    """

    def __init__(
        self,
        config: RabbitMQConfig,
        queue_name: str | None = None,
        result_queue_name: str | None = None,
        collection_name: str | None = None,
        partition_name: str | None = None,
    ) -> None:
        super().__init__(config)
        self.queue_name = queue_name or getattr(self.config, "task_queue", "taskQueue")
        self.result_queue_name = result_queue_name or getattr(
            self.config, "result_queue", "respQueue"
        )
        self.collection_name = collection_name
        self.partition_name = partition_name
        # processing workers
        self.executor = ThreadPoolExecutor(
            max_workers=getattr(self.config, "consumer_workers", 4)
        )
        # shared model loader
        self.model_loader = ModelLoader()

        self._connect()

    def _connect(self) -> None:
        self._ensure_connection()
        assert self.channel is not None
        # create or check queues
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        self.channel.queue_declare(queue=self.result_queue_name, durable=True)
        logger.info(
            f"Connected! 任务队列: {self.queue_name}，结果队列: {self.result_queue_name}"
        )

    def callback(
        self,
        ch: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
    ) -> None:
        """处理接收到的消息"""
        try:
            task = Task.from_json(body.decode("utf-8"))
            logger.info(f"收到任务: {task.task_id}")
            self._publish_status(task, TaskStatus.PROCESSING, datetime.now())

            # In case of preprocess task blocking main thread KEEPING HEARTBEAT
            self.executor.submit(self._process_task, task, ch, method.delivery_tag)
        except Exception as e:
            logger.error(f"处理任务时出错: {e}")

    def _process_task(
        self, task: Task, ch: pika.channel.Channel, delivery_tag: int
    ) -> None:
        """在子线程中执行预处理任务"""

        def _on_task_done(task, ch, delivery_tag):
            try:
                self._publish_status(task, TaskStatus.DONE, datetime.now())
                ch.basic_ack(delivery_tag=delivery_tag)
                logger.info(f"任务 {task.task_id} 处理完成")
            except Exception as e:
                logger.error(f"发送ACK或状态更新失败: {e}")

        def _on_task_error(task, ch, delivery_tag, error):
            try:
                self._publish_status(task, TaskStatus.FAILED, datetime.now())
                ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
                logger.error(f"任务 {task.task_id} 处理失败: {error}")
            except Exception as e:
                logger.error(f"发送NACK或失败状态失败: {e}")

        try:
            default_collection = (
                getattr(task, "destination_collection", None)
                or self.collection_name
                or MilvusConfig().default_collection_name
            )
            logger.info(f"destination_collection: {default_collection}")
            # get_pipeline returns a Pipeline class; instantiate it here
            PipelineCls = get_pipeline(getattr(task, "task_type", "default"))
            pipeline = PipelineCls(
                model_loader=self.model_loader,
                destination_collection=default_collection,
            )
            chunks, embeddings = pipeline.process(task.document)

            # Store embeddings via pipeline's interface to reduce coupling
            pipeline.store_embedding(
                chunks,
                embeddings,
                doc=task.document,
                collection_name=default_collection,
                partition_name=self.partition_name,
            )
            self.connection.add_callback_threadsafe(
                lambda: _on_task_done(task, ch, delivery_tag)
            )
        except Exception as e:
            logger.error(f"任务处理失败: {e}")
            self.connection.add_callback_threadsafe(
                lambda e_ref=e: _on_task_error(task, ch, delivery_tag, e_ref)
            )

    def _publish_status(
        self, task: Task, status: TaskStatus, dateTime: datetime
    ) -> None:
        """更新 task 状态, 发布到结果队列"""
        if not self.connection or self.connection.is_closed:
            self._connect()
        assert self.channel is not None

        task.status = status
        if status == TaskStatus.PROCESSING:
            task.processed_at = dateTime
        elif status == TaskStatus.DONE or status == TaskStatus.FAILED:
            task.finished_at = dateTime

        self.channel.basic_publish(
            exchange="",
            routing_key=self.result_queue_name,
            body=task.to_resp_json(),
            properties=pika.BasicProperties(
                delivery_mode=2,
            ),
        )
        logger.info(
            f"任务 {task.task_id} 结果已发布到 {self.result_queue_name}，状态: {task.status}"
        )

    def start_consuming(self) -> None:
        """开始消费消息"""
        if not self.connection or self.connection.is_closed:
            self._connect()
        assert self.channel is not None

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name, on_message_callback=self.callback
        )

        logger.info("开始消费任务...")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
        finally:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
                logger.info("RabbitMQ连接已关闭")
