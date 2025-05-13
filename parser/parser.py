from typing import List

from strenum import StrEnum

from .chunk import Chunk


class SupportedFileType(StrEnum):
    PDF = "pdf"


class Parser():
    """
    Base parser class.
    """

    def __init__(
        self,
        file_type: SupportedFileType = SupportedFileType.PDF,
        **kwargs,
    ):
        """
        Args:
        - file_type: file type, pdf, ppt, docx, etc.
        """
        super().__init__()
        self.file_type = file_type

    def parse(
        self,
        file_path: str,
    ) -> List[Chunk]:
        """
        parse method.

        Args:
        - file_path: path to the file.

        Returns:
        - A list of parsed documents chunks.
        """
        return []
