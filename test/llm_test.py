import unittest
import os
import json
import numpy as np

from chat import format_reference_info
from parse.parser import Chunk, ChunkType

import config


class TestLLM(unittest.TestCase):

    def test_base(self):
        from rag.llm import format_reference_info

        reference_meta = {
            '1': {
                'file_name': 'test file name',
                'content_url': 'test content url',
                'chunk_begin_digest': ['1', '2', '3'],
                'chunk_end_digest': ['1', '2', '3'],
            }
        }

        answer = "reference id: ##1@@"

        ret = format_reference_info(reference_meta, answer)
        print(ret)


if __name__ == '__main__':

    unittest.main()
