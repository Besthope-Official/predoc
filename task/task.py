"""消费文档预处理的任务"""

from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import pika
from loguru import logger

from config.backend import RabbitMQConfig
from task.preprocess import preprocess
from schemas import TaskStatus, Task


class TaskConsumer:
    """
    Consumer class for RabbitMQ document-preprocess tasks.
    Also producer of task result to provide task status for TaskProducer to persist (save to e.g. postgresql).
    """

    def __init__(
        self,
        config: RabbitMQConfig,
        queue_name: str = "taskQueue",
        result_queue_name: str = "respQueue",
    ) -> None:
        self.config = config
        self.credentials = pika.PlainCredentials(self.config.user, self.config.password)
        self.parameters = pika.ConnectionParameters(
            host=self.config.host,
            port=self.config.port,
            credentials=self.credentials,
            heartbeat=600,
        )

        self.queue_name = queue_name
        self.result_queue_name = result_queue_name
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
        # 4 threads for processing tasks
        self.executor = ThreadPoolExecutor(max_workers=4)

        self._connect()

    def _connect(self) -> None:
        if self.connection and not self.connection.is_closed:
            return
        logger.info(
            f"Connecting to RabbitMQ Server: {self.config.host}:{self.config.port}"
        )

        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()

        # create or check (by default, taskQueue is created on published side)
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
            preprocess(task)
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


if __name__ == "__main__":
    config = RabbitMQConfig()
    consumer = TaskConsumer(config)
    consumer.start_consuming()
