import unittest
import torch

from ..chunk import Chunk, ChunkType
from ..parser import Parser, SupportedFileType


class TestBaseClass(unittest.TestCase):

    def test_base_class(self):
        chunk = Chunk(ChunkType.TEXT, content='test', extra_description='')
        parser = Parser(file_type=SupportedFileType.PDF, )
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
