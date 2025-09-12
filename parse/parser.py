import os
from abc import ABC, abstractmethod

from common.cache import singleton
from common.data import Content, ContentType
from common.utils import load_base64_image


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
        self.file_path = file_path
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        content = Content(
            file_path=self.file_path,
            content_type=ContentType.TEXT,
            content=text,
            extra_description="",
            content_url="",
        )
        return [content]


@singleton
class ImageParser(Parser):
    def parse(self, file_path: str) -> list[Content]:
        self.file_path = file_path
        image_content = load_base64_image(file_path)
        content = Content(
            file_path=self.file_path,
            content_type=ContentType.IMAGE,
            content=image_content,
            extra_description="",
            content_url=file_path,
        )
        return [content]
