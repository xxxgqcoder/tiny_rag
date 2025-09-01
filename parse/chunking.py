import logging
from abc import ABC, abstractmethod

from common.config import TinyRAGConfig
from common.data import Chunk, Content, ContentType
from common.utils import safe_strip


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
                            file_name=content.file_name,
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
                texts = "\n\n".join([chunk.content.decode("utf-8") for chunk in content_buffer])
                merged_chunks.append(
                    Chunk(
                        content_type=content_buffer[0].content_type,
                        file_name=content_buffer[0].file_name,
                        content=texts.encode("utf-8", errors="ignore"),
                        extra_description="".encode("utf-8", errors="ignore"),
                        content_url="",
                        uuid="",
                    )
                )
                content_buffer.clear()

            # start next iteration
            i = j - self.block_overlap_num

        merged_chunks = self.filter_chunk(merged_chunks)
        return merged_chunks

    def filter_chunk(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Filter too short chunks
        """
        filtered_chunks = []
        for chunk in chunks:
            content = chunk.content
            if chunk.content_type != ContentType.TEXT:
                content = chunk.extra_description
            content = safe_strip(content.decode("utf-8"))
            if len(content) < 1:
                logging.info(f"{chunk.file_name}: remove chunk due to too short content: {str(chunk)}")
                continue

            filtered_chunks.append(chunk)

        return filtered_chunks
