"""RabbitMQ 基础封装，供 Producer/Consumer 复用"""

from __future__ import annotations
from typing import Optional
from loguru import logger
import pika

from config.backend import RabbitMQConfig


class RabbitMQBase:
    """封装 RabbitMQ 连接与通道初始化的共用逻辑"""

    def __init__(self, config: RabbitMQConfig) -> None:
        self.config = config
        self.credentials = pika.PlainCredentials(self.config.user, self.config.password)
        self.parameters = pika.ConnectionParameters(
            host=self.config.host,
            port=self.config.port,
            credentials=self.credentials,
            heartbeat=600,
        )
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None

    def _ensure_connection(self) -> None:
        if self.connection and not self.connection.is_closed:
            return
        logger.info(
            f"Connecting to RabbitMQ Server: {self.config.host}:{self.config.port}"
        )
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()  # type: ignore[assignment]
        logger.info("RabbitMQ 连接已建立")
