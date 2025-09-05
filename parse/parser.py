from abc import ABC, abstractmethod

from common.data import Content


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
