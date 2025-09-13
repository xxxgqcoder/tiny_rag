import os
import sys
import time
import unittest

from common.data import Content, ContentType
from parse.chunking import ByteOverlapChunking, OverlapChunking

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print(sys.path[-1])


from common.cache import get_cache_db
from common.config import CacheConfig, TinyRAGConfig


class TestChunk(unittest.TestCase):
    def test_base(self):
        content = Content(
            content_type=ContentType.TEXT,
            file_name="test_file_name",
            content="1234567",
            extra_description="",
            content_url="",
        )

        TinyRAGConfig.chunking_config.consecutive_byte_num = 4
        TinyRAGConfig.chunking_config.byte_overlap_num = 2

        chuking = ByteOverlapChunking()
        chunks = chuking.chunk([content])
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].content, "1234")
        self.assertEqual(chunks[1].content, "3456")
        self.assertEqual(chunks[2].content, "567")


if __name__ == "__main__":
    unittest.main()
