import os
from abc import ABC, abstractmethod

from common.cache import singleton
from common.data import Content, ContentType


class Parser(ABC):
    """
    Base parser class.
    """

    @abstractmethod
    def parse(
        self,
        file_path: str,
    ) -> list[Content]:
        """
        parse method.

        Args:
        - file_path: path to the file.

        Returns:
        - A list of parsed documents chunks.
        """
        raise NotImplementedError("Not implemented")


@singleton
class TextParser(Parser):
    def parse(self, file_path: str) -> list[Content]:
        self.file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        content = Content(
            file_name=self.file_name,
            content_type=ContentType.TEXT,
            content=text,
            extra_description="",
            content_url="",
        )
        return [content]
