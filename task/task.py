'''消费文档预处理的任务'''
from datetime import datetime
from loguru import logger

import pika
from config.backend import RabbitMQConfig
from task.preprocess import preprocess
from models import TaskStatus, Task


class TaskConsumer:
    '''
        Consumer class for RabbitMQ document-preprocess tasks.
        Also producer of task result to provide task status for TaskProducer to persist (save to e.g. postgresql).
    '''

    def __init__(self,
                 config: RabbitMQConfig,
                 queue_name: str = "taskQueue",
                 result_queue_name: str = "respQueue",
                 ):
        self.config = config
        self.queue_name = queue_name
        self.result_queue_name = result_queue_name
        self.connection = None
        self.channel = None

        self._connect()

    def _connect(self):
        if self.connection and not self.connection.is_closed:
            return

        credentials = pika.PlainCredentials(
            self.config.user, self.config.password)
        parameters = pika.ConnectionParameters(
            host=self.config.host,
            port=self.config.port,
            credentials=credentials
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        # create or check (by default, taskQueue is created on published side)
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        self.channel.queue_declare(queue=self.result_queue_name, durable=True)
        logger.info(
            f"已连接到RabbitMQ，任务队列: {self.queue_name}，结果队列: {self.result_queue_name}")

    def callback(self, ch, method, properties, body):
        """处理接收到的消息"""
        try:
            task = Task.from_json(body.decode('utf-8'))
            logger.info(f"收到任务: {task.task_id}")
            self._publish_status(task, TaskStatus.PROCESSING, datetime.now())

            preprocess(task)

            self._publish_status(task, TaskStatus.DONE, datetime.now())
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"任务 {task.task_id} 处理完成")
        except Exception as e:
            logger.error(f"处理任务时出错: {e}")
            try:
                self._publish_status(task, TaskStatus.FAILED, datetime.now())
            except Exception as publish_error:
                logger.error(f"发布失败结果时出错: {publish_error}")

            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def _publish_status(self, task: Task, status: TaskStatus, dateTime: datetime):
        """更新 task 状态, 发布到结果队列"""
        if not self.connection or self.connection.is_closed:
            self._connect()

        task.status = status
        if status == TaskStatus.PROCESSING:
            task.processed_at = dateTime
        elif status == TaskStatus.DONE or status == TaskStatus.FAILED:
            task.finished_at = dateTime

        self.channel.basic_publish(
            exchange='',
            routing_key=self.result_queue_name,
            body=task.to_resp_json(),
            properties=pika.BasicProperties(
                delivery_mode=2,
            )
        )
        logger.info(
            f"任务 {task.task_id} 结果已发布到 {self.result_queue_name}，状态: {task.status}")

    def start_consuming(self):
        """开始消费消息"""
        if not self.connection or self.connection.is_closed:
            self._connect()

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.callback
        )

        logger.info('开始消费任务...')
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
        finally:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
                logger.info('RabbitMQ连接已关闭')


if __name__ == "__main__":
    config = RabbitMQConfig()
    consumer = TaskConsumer(config)
    consumer.start_consuming()
