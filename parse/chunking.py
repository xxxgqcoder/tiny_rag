import logging
from abc import ABC, abstractmethod

from common.cache import singleton
from common.config import TinyRAGConfig
from common.data import Chunk, Content, ContentType
from common.utils import safe_encode, safe_strip, Logger


class Chunking(ABC):
    """
    Document chunking strategy.
    """

    @abstractmethod
    def chunk(self, contents: list[Content]) -> list[Chunk]:
        """
        Chunk method.

        Args:
        - contents: original document contents to be chunked.

        Returns:
        - A list of chunks.
        """
        raise NotImplementedError("Not implemented")


def _filter_chunk(chunks: list[Chunk]) -> list[Chunk]:
    """
    Filter too short chunks
    """
    filtered_chunks = []
    for chunk in chunks:
        content = chunk.content
        if chunk.content_type != ContentType.TEXT:
            content = chunk.extra_description
        content = safe_strip(content)
        if len(content) < 1:
            Logger.info(f"{chunk.file_path}: remove chunk due to too short content: {str(chunk)}")
            continue

        filtered_chunks.append(chunk)

    return filtered_chunks


@singleton
class OverlapChunking(Chunking):
    def __init__(self) -> None:
        super().__init__()

        # used in chunking, number of consecutive block to be considered as one chunk.
        self.consecutive_block_num = TinyRAGConfig.chunking_config.consecutive_block_num  # type: ignore
        # used in chunking, number of overlapped block num between two consecutive chunks.
        self.block_overlap_num = TinyRAGConfig.chunking_config.block_overlap_num  # type: ignore

        assert self.block_overlap_num < self.consecutive_block_num, (
            f"block overlap num ({self.block_overlap_num}) be less than consecutive block num ({self.consecutive_block_num})"
        )

    def chunk(self, contents: list[Content]) -> list[Chunk]:
        """
        Chunk parsed pdf contents.

        Scan `self.consecutive_block_num` consecutive chunk and combine as one chunk.
        If image / table chunk is encountered within current consecutive chunks, then make the image / table chunk as independent chunk and continue scan until `self.consecutive_block_num` is met.

        Two consecutive merged chunks have `self.block_overlap_num` overlapped chunk to ensure semantic coherence.

        Returns:
        - List of chunks.
        """
        merged_chunks = []
        content_buffer = []
        i: int = 0
        # since we apply overlap,i can not exceed len(chunks) - self.block_overlap_num,
        # otherwise, infinite loop may happen.
        while i < len(contents) - self.block_overlap_num:
            # inner loop start from current chunk
            j: int = i
            while j < len(contents) and len(content_buffer) < self.consecutive_block_num:
                content = contents[j]

                # text chunk
                if content.content_type in [ContentType.TEXT]:
                    content_buffer.append(content)
                # image / table chunk
                elif content.content_type in [ContentType.IMAGE, ContentType.TABLE]:
                    merged_chunks.append(
                        Chunk(
                            content_type=content.content_type,
                            file_path=content.file_path,
                            content=content.content,
                            extra_description=content.extra_description,
                            content_url=content.content_url,
                            uuid="",
                        )
                    )
                else:
                    pass

                # move one step forward
                j += 1

            # inner loop ends when j == len(chunks) or len(block_buffer) == self.consecutive_block_num generate new chunk if buffer is not empty.
            if len(content_buffer) > 0:
                texts = "\n\n".join([chunk.content for chunk in content_buffer])
                merged_chunks.append(
                    Chunk(
                        content_type=content_buffer[0].content_type,
                        file_path=content_buffer[0].file_path,
                        content=safe_encode(texts),
                        extra_description="",
                        content_url="",
                        uuid="",
                    )
                )
                content_buffer.clear()

            # start next iteration
            i = j - self.block_overlap_num

        merged_chunks = _filter_chunk(merged_chunks)
        return merged_chunks


@singleton
class ByteOverlapChunking(Chunking):
    def __init__(self) -> None:
        super().__init__()

        self.consecutive_byte_num = TinyRAGConfig.chunking_config.consecutive_byte_num
        self.byte_overlap_num = TinyRAGConfig.chunking_config.byte_overlap_num

        assert self.byte_overlap_num < self.consecutive_byte_num, (
            f"block overlap num ({self.byte_overlap_num}) be less than consecutive block num ({self.consecutive_byte_num})"
        )

    def chunk(self, contents: list[Content]) -> list[Chunk]:
        if not contents:
            return []

        content = "\n\n\n\n".join([c.content for c in contents if c.content_type == ContentType.TEXT])

        i = 0
        chunks = []
        while i < len(content) - self.byte_overlap_num:
            chunk_content = content[i : i + self.consecutive_byte_num]
            chunk_content = safe_strip(chunk_content)
            if len(chunk_content) < 1:
                continue

            chunks.append(
                Chunk(
                    content_type=ContentType.TEXT,
                    file_path=contents[0].file_path,
                    content=safe_encode(chunk_content),
                    extra_description="",
                    content_url="",
                    uuid="",
                )
            )
            i = i + self.consecutive_byte_num - self.byte_overlap_num

        return _filter_chunk(chunks)


@singleton
class BypassChunking(Chunking):
    def chunk(self, contents: list[Content]) -> list[Chunk]:
        chunks = []
        for content in contents:
            chunks.append(
                Chunk(
                    content_type=content.content_type,
                    file_path=content.file_path,
                    content=content.content,
                    extra_description=content.extra_description,
                    content_url=content.content_url,
                    uuid="",
                )
            )
        return chunks


def get_chunking() -> Chunking:
    return OverlapChunking()


def get_chunking_by_file_type(file_type: str) -> Chunking:
    if file_type in ["pdf"]:
        return OverlapChunking()
    if file_type in ["txt", "md"]:
        return ByteOverlapChunking()
    if file_type in ["png", "jpg", "jpeg", "bmp", "gif"]:
        return BypassChunking()

    # default
    return ByteOverlapChunking()
