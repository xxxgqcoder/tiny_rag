import xxhash
from strenum import StrEnum
from config import ChunkType
from abc import ABC, abstractmethod


class SupportedFileType(StrEnum):
    PDF = "pdf"


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
        content_url: str = '',
    ):
        """
        Args:
        - content_type: chunk content type.
        - content: the content, represented in bytes.
        - extra_description: content extra description.
        - content_url: url to the content, set when content is not suitable for
            directly insert into db, for example image / audio data.
        """
        super().__init__()

        self.content_type = content_type
        self.content = content
        self.extra_description = extra_description
        self.content_url = content_url
        self.uuid = xxhash.xxh64(content + extra_description).hexdigest()

    def __str__(self, ):
        if self.content_type == ChunkType.TEXT:
            return self.content.decode('utf-8')
        elif self.content_type == ChunkType.IMAGE:
            return 'content is image, below is the image description:\n' \
                + self.extra_description.decode('utf-8') \
                + f"content url: {self.content_url}"
        elif self.content_type == ChunkType.TABLE:
            return self.extra_description.decode('utf-8') \
                + self.content.decode('utf-8')
        else:
            return ""


class Parser(ABC):
    """
    Base parser class.
    """

    @abstractmethod
    def parse(
        self,
        file_path: str,
        asset_save_dir: str,
    ) -> list[Chunk]:
        """
        parse method.

        Args:
        - file_path: path to the file.
        - asset_save_dir: directory for saving parsed assets, for example images.

        Returns:
        - A list of parsed documents chunks.
        """
        raise NotImplementedError("Not implemented")
