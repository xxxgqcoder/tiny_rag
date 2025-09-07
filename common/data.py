import uuid
from enum import Enum
from sre_constants import SUCCESS
from typing import Any

from albucore import F
from pydantic import BaseModel, Field, model_validator
from strenum import StrEnum
from transformers.models.conditional_detr.modeling_conditional_detr import CONDITIONAL_DETR_INPUTS_DOCSTRING

from common.utils import hash64, now_in_utc


class SupportedFileType(StrEnum):
    PDF = "pdf"


class ContentType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TABLE = "table"


class Content(BaseModel):
    """
    Document content object.
    """

    content_type: ContentType = Field(ContentType.TEXT, description="content type")
    file_name: str = Field("", description="original file name")
    content: str = Field(
        "",
        description="the content, represented in string. If content type is not image / table, this field will be base64 encoded image content.",
    )
    extra_description: str = Field("", description="content extra description")
    content_url: str = Field(
        "",
        description="url to the content, set when content is not suitable for directly insert into db, for example image / audio data",
    )


class Chunk(Content):
    """
    Document chunk object. A chunk can be text paragraph, or a non-text asset, i.e., picture or audio.
    """

    uuid: str = Field("", description="unique id of the chunk")

    @model_validator(mode="after")
    def set_uuid(self):
        self.uuid = hash64((self.file_name + self.content + self.extra_description).encode("utf-8", errors="ignore"))
        return self


class RationalDBRecord(BaseModel):
    """
    Rational DB record.
    """

    file_name: str = Field(..., description="document name")
    # NOTE: recosinder this design.
    chunk_uuids: str = Field(..., description="id list of document's chunk, separated by '\x07'")
    created_date: str = Field(..., description="document created date")
    content_hash: str = Field(..., description="hash of document content")

    @model_validator(mode="after")
    def set_created_date(self):
        self.created_date = now_in_utc()
        return self


class VectorDBRecord(BaseModel):
    """
    Vector DB record.
    """

    uuid: str = Field(..., description="uuid of the chunk")
    file_name: str = Field(..., description="original file name")
    content_url: str = Field(..., description="url to the content")
    embedding: list[float] = Field(..., description="embedding vector")
    metadata: dict[str, Any] = Field(..., description="meta data of the chunk")


# -----------------------------------------------------------------------------------------------
# service request / response
class ServiceResponse(BaseModel):
    code: int = Field(0, description="0 for success")
    message: str = Field("", description="error message if any")
    data: dict[str, Any] = Field(default_factory=dict, description="data payload")


class NewDocumentRequest(BaseModel):
    file_name: str = Field(..., description="original file name")
    content_hash: str = Field(..., description="hash of document content")
    md_content: str = Field("", description="markdown content of the document")
    chunks: list[Chunk] = Field(..., description="list of chunks of the document")
    chunk_embedding: list[list[float]] = Field(..., description="list of chunk embeddings")


class GetDocumentRequest(BaseModel):
    file_name: str = Field(..., description="original file name")


class NewDocumentResponse(ServiceResponse):
    pass


class GetDocumentResponse(ServiceResponse):
    document: RationalDBRecord | None = Field(None, description="document record")
    md_content: str | None = Field("", description="markdown content of the document")
    chunks: list[Chunk] | None = Field([], description="list of chunks of the document")


class GetAllDocumentResponse(ServiceResponse):
    file_names: list[str] | None = Field(None, description="document name list")


class DeleteDocumentRequest(BaseModel):
    file_name: str = Field(..., description="original file name")


class DeleteDocumentResponse(ServiceResponse):
    pass


class SearchRequest(BaseModel):
    query: dict[str, list[float]] = Field(
        ..., description="search query, key is the vector db collection column name, value is the embedding vector."
    )
    query_params: dict[str, Any] = Field(default_factory=dict, description="additional search parameters")


class SearchResponse(ServiceResponse):
    chunks: list[Chunk] | None = Field([], description="list of chunks")
