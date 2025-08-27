"""文档数据模型"""

from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List, Optional


class Author(BaseModel):
    name: str
    institution: str

    def to_dict(self):
        return {"name": self.name, "institution": self.institution}


class Keyword(BaseModel):
    name: str

    def to_dict(self):
        return {"name": self.name}


class Document(BaseModel):
    title: str
    authors: List[Author]
    keywords: List[Keyword]
    fileName: str
    docType: str = Field(alias="doc_type")
    bucket: Optional[str] = None  # will use default in config if none is provided
    publicationDate: Optional[datetime] = None
    language: Optional[str] = "unknown"

    # Allow population by field name or alias
    model_config = ConfigDict(populate_by_name=True)

    def to_metadata(self) -> dict:
        """构建用于入库/检索的元数据字典。"""
        return {
            "authors": [a.to_dict() for a in (self.authors or [])],
            "keywords": [k.to_dict() for k in (self.keywords or [])],
            "title": self.title,
            "publicationDate": (
                self.publicationDate.isoformat()
                if getattr(self, "publicationDate", None)
                else None
            ),
            "language": getattr(self, "language", None),
        }


class JournalArticle(Document):
    abstractText: str
    journal: str
    doi: str
    cited: str
    JEL: str


class Book(Document):
    publisher: str
    isbn: str
