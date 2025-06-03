'''文档数据模型'''
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


class Author(BaseModel):
    name: str
    institution: str

    def to_dict(self):
        return {
            "name": self.name,
            "institution": self.institution
        }


class Keyword(BaseModel):
    name: str

    def to_dict(self):
        return {
            "name": self.name
        }


class Document(BaseModel):
    title: str
    authors: List[Author]
    keywords: List[Keyword]
    fileName: str
    docType: str = Field(alias='doc_type')
    publicationDate: Optional[datetime] = None
    language: Optional[str] = "unknown"


class JournalArticle(Document):
    abstractText: str
    journal: str
    doi: str
    cited: str
    JEL: str


class Book(Document):
    publisher: str
    isbn: str
