"""
数据模型
包含所有与数据库和API交互相关的数据模型定义
"""

from .document import Author, Keyword, Document, JournalArticle, Book
from .task import TaskStatus, Task

__all__ = [
    "Author",
    "Keyword",
    "Document",
    "JournalArticle",
    "Book",
    "TaskStatus",
    "Task",
]
