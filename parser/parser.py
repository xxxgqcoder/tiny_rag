import xxhash
from strenum import StrEnum


class SupportedFileType(StrEnum):
    PDF = "pdf"


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
        extra_description: bytes,
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
        self.uuid = xxhash.xxh64(content + extra_description).hexdigest()

    def __str__(self, ):
        if self.content_type == ChunkType.TEXT:
            return self.content.decode('utf-8')
        elif self.content_type == ChunkType.IMAGE:
            return 'content is image, below is the image description:\n' \
                + self.extra_description.decode('utf-8')
        elif self.content_type == ChunkType.TABLE:
            return self.extra_description.decode('utf-8') \
                + self.content.decode('utf-8')
        else:
            return ""


class Parser():
    """
    Base parser class.
    """

    def __init__(self, ):
        """
        """
        super().__init__()

    def parse(
        self,
        file_path: str,
    ) -> list[Chunk]:
        """
        parse method.

        Args:
        - file_path: path to the file.

        Returns:
        - A list of parsed documents chunks.
        """
        return []
