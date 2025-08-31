from pydantic import BaseModel, Field, model_validator
from strenum import StrEnum
from common.utils import hash64

class SupportedFileType(StrEnum):
    PDF = "pdf"


class ChunkType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TABLE = "table"


class Chunk(BaseModel):
    """
    Document chunk object. A chunk can be text paragraph, or a non-text asset, i.e., picture or audio.
    """

    content_type: ChunkType = Field(ChunkType.TEXT, description="chunk content type")
    file_name: str = Field("", description="original file name")
    content: bytes = Field(b"", description="the content, represented in bytes")
    extra_description: bytes = Field(b"", description="content extra description")
    content_url: str = Field(
        "",
        description="url to the content, set when content is not suitable for directly insert into db, for example image / audio data",
    )
    uuid: str = Field("", description="unique id of the chunk")

    @model_validator(mode='after')
    def set_uuid(self):
        if not self.uuid:
            self.uuid = hash64(self.file_name.encode('utf-8', errors='ignore') + self.content + self.extra_description)
        return self