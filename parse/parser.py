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
        asset_save_dir: str,
    ) -> list[Content]:
        """
        parse method.

        Args:
        - file_path: path to the file.
        - asset_save_dir: directory for saving parsed assets, for example images.

        Returns:
        - A list of parsed documents chunks.
        """
        raise NotImplementedError("Not implemented")
