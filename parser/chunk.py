from strenum import StrEnum


class ChunkType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TABLE = "table"


class Chunk:
    """
    Document chunk object. A chunk can be text paragraph, or a non-text asset,
    i.e., picture or audio.
    """

    def __init__(
        self,
        content_type: ChunkType,
        content: bytes,
        extra_description: str,
    ):
        """
        Args:
        - content_type: chunk content type.
        - content: the content, represented in bytes.
        - extra_description: content extra description.
        """
        super().__init__()

        self.content_type = content_type
        self.content = content
        self.extra_description = extra_description
