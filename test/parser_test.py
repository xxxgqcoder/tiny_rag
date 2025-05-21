import unittest
import os

from parse.parser import Parser, SupportedFileType, Chunk, ChunkType
from parse.pdf_parser import PDFParser

import config
from utils import get_project_base_directory


class TestPDFParser(unittest.TestCase):

    def test_filter_chunks(self):
        chunks = [
            # keep
            Chunk(content='text 1: should keep'.encode('utf-8'),
                  extra_description=''.encode('utf-8'),
                  content_type=config.ChunkType.TEXT),

            # rm
            Chunk(content='text 2'.encode('utf-8'),
                  extra_description=''.encode('utf-8'),
                  content_type=config.ChunkType.TEXT),

            # keep
            Chunk(content=''.encode('utf-8'),
                  extra_description='image 1: should keep'.encode('utf-8'),
                  content_type=config.ChunkType.IMAGE),

            # rm
            Chunk(content=''.encode('utf-8'),
                  extra_description='image 2'.encode('utf-8'),
                  content_type=config.ChunkType.TEXT),

            # keep
            Chunk(content=''.encode('utf-8'),
                  extra_description='table 1: should keep'.encode('utf-8'),
                  content_type=config.ChunkType.TABLE),

            # rm
            Chunk(content=''.encode('utf-8'),
                  extra_description='table 2'.encode('utf-8'),
                  content_type=config.ChunkType.TABLE),
        ]

        parser = PDFParser()
        ret = parser.filter_chunks(chunks)
        for chunk in ret:
            print(chunk)
            print('=' * 80)

        self.assertEqual(len(ret), 3)
        self.assertEqual(ret[0].content.decode('utf-8'), 'text 1: should keep')
        self.assertEqual(ret[1].extra_description.decode('utf-8'),
                         'image 1: should keep')
        self.assertEqual(ret[2].extra_description.decode('utf-8'),
                         'table 1: should keep')

    def test_filter_text_content(self, ):
        texts = ['', None, 'test ', 'block 1']
        parser = PDFParser()
        content = parser.filter_text_content(texts=texts)
        self.assertEqual(content, 'test\n\nblock 1')

    def test_parser_chunk(self):
        content_list = [
            {
                'type': 'text',
                'text_level': 1,
                'text': 'h1'
            },
            {
                'type': 'text',
                'text': 'p1'
            },
            {
                'type': 'text',
                'text': 'p2'
            },
            {
                'type': 'text',
                'text_level': 1,
                'text': 'h2'
            },
            {
                'type': 'table',
                'table_body': 't1',
                'table_caption': 't1 description',
                'table_footnote': 't1 footnote'
            },
        ]
        parser = PDFParser()
        chunks = parser.chunk(content_list=content_list,
                              temp_asset_dir='',
                              asset_save_dir='')

        # should have 3 chunks
        self.assertEqual(len(chunks), 3)

        # chunk 0
        self.assertTrue('h1' in str(chunks[0]) and 'p1' in str(chunks[0])
                        and 'p2' in str(chunks[0]))

        # chunk 1
        self.assertTrue('h2' in str(chunks[1]))


if __name__ == '__main__':

    unittest.main()
