from strenum import StrEnum

from .chunk import Chunk


class SupportedFileType(StrEnum):
    PDF = "pdf"


class Parser():
    """
    Base parser class.
    """

    def __init__(self, ):
        """
        Args:
        - file_type: file type, pdf, ppt, docx, etc.
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
