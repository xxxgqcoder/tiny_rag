import uuid
from typing import Any

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
    content: bytes = Field(b"", description="the content, represented in bytes")
    extra_description: bytes = Field(b"", description="content extra description")
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
        if not self.uuid:
            self.uuid = hash64(self.file_name.encode("utf-8", errors="ignore") + self.content + self.extra_description)
        return self


# rational db record
class RationalDBRecord(BaseModel):
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
    uuid: str = Field(..., description="uuid of the chunk")
    file_name: str = Field(..., description="original file name")
    content_url: str = Field(..., description="url to the content")
    embedding: list[float] = Field(..., description="embedding vector")
    metadata: dict[str, Any] = Field(..., description="meta data of the chunk")
