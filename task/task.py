'''消费文档预处理的任务'''
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from uuid import UUID
from datetime import datetime
from loguru import logger

import pika
import json
from config.backend import RabbitMQConfig
from preprocess import preprocess


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, status_str):
        try:
            return cls(status_str.upper())
        except ValueError:
            return cls.PENDING


class Author(BaseModel):
    name: str
    institution: str


class Keyword(BaseModel):
    name: str


class Document(BaseModel):
    title: str
    authors: list[Author]
    keywords: list[Keyword]
    fileName: str
    docType: str = Field(alias='doc_type')
    publicationDate: datetime


class JournalArticle(Document):
    abstractText: str
    journal: str
    doi: str
    cited: str
    JEL: str


class Book(Document):
    publisher: str
    isbn: str


class Task(BaseModel):
    task_id: UUID = Field(alias='taskId')
    status: TaskStatus
    document: Document
    created_at: datetime = Field(alias='createdAt')
    finished_at: Optional[datetime] = Field(alias='finishedAt', default=None)
    
    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        logger.debug(f'Receiving JSON: {data}')
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = TaskStatus(data["status"])
            except ValueError:
                data["status"] = TaskStatus.PENDING
        return cls.model_validate(data)

    def to_json(self):
        return self.model_dump_json()


class TaskConsumer:
    """RabbitMQ任务消费者"""

    def __init__(self,
                 config: RabbitMQConfig,
                 queue_name: str = "taskQueue",
                 result_queue_name: str = "resultQueue",
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

            # 在这里处理任务
            # TODO: 实现具体的任务处理逻辑
            task.status = TaskStatus.PROCESSING
            
            preprocess(task)
            
            # 处理成功，更新任务状态为done
            task.status = TaskStatus.DONE
            task.finished_at = datetime.now()

            # 将结果发送到结果队列
            self._publish_result(task)

            # 确认消息已处理
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"任务 {task.task_id} 处理完成")
        except Exception as e:
            logger.error(f"处理任务时出错: {e}")

            try:
                # 更新任务状态为fail
                task.status = TaskStatus.FAILED
                task.finished_at = datetime.now()

                # 将失败结果发送到结果队列
                self._publish_result(task)
            except Exception as publish_error:
                logger.error(f"发布失败结果时出错: {publish_error}")

            # 如果处理失败，根据业务需求决定是否重新入队
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def _publish_result(self, task: Task):
        """将处理结果发布到结果队列"""
        if not self.connection or self.connection.is_closed:
            self.connect()

        self.channel.basic_publish(
            exchange='',
            routing_key=self.result_queue_name,
            body=task.to_json(),
            properties=pika.BasicProperties(
                delivery_mode=2,  # 持久化消息
            )
        )
        logger.info(
            f"任务 {task.task_id} 结果已发布到 {self.result_queue_name}，状态: {task.status}")

    def start_consuming(self):
        """开始消费消息"""
        if not self.connection or self.connection.is_closed:
            self.connect()

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
