'''任务数据模型'''
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from uuid import UUID
from datetime import datetime
import json
from loguru import logger

from .document import Document


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


class Task(BaseModel):
    task_id: UUID = Field(alias='taskId')
    status: TaskStatus
    document: Document
    created_at: datetime = Field(alias='createdAt')
    processed_at: Optional[datetime] = Field(alias="processedAt", default=None)
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

    def to_resp_json(self):
        return self.to_json()

    def to_metadata(self):
        """将任务转换为元数据"""
        metadata = {
            "authors": [author.to_dict() for author in self.document.authors],
            "keywords": [keyword.to_dict() for keyword in self.document.keywords],
            "title": self.document.title,
            "publicationDate": str(self.document.publicationDate.isoformat()),
            "language": self.document.language
        }
        return metadata
