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
                  file_name='/fake/path',
                  extra_description=''.encode('utf-8'),
                  content_type=config.ChunkType.TEXT),

            # rm
            Chunk(content='text 2'.encode('utf-8'),
                  file_name='/fake/path',
                  extra_description=''.encode('utf-8'),
                  content_type=config.ChunkType.TEXT),

            # keep
            Chunk(content=''.encode('utf-8'),
                  file_name='/fake/path',
                  extra_description='image 1: should keep'.encode('utf-8'),
                  content_type=config.ChunkType.IMAGE),

            # rm
            Chunk(content=''.encode('utf-8'),
                  file_name='/fake/path',
                  extra_description='image 2'.encode('utf-8'),
                  content_type=config.ChunkType.TEXT),

            # keep
            Chunk(content=''.encode('utf-8'),
                  file_name='/fake/path',
                  extra_description='table 1: should keep'.encode('utf-8'),
                  content_type=config.ChunkType.TABLE),

            # rm
            Chunk(content=''.encode('utf-8'),
                  file_name='/fake/path',
                  extra_description='table 2'.encode('utf-8'),
                  content_type=config.ChunkType.TABLE),
        ]

        parser = PDFParser()
        parser.file_name = '/fake/path'
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

    def test_strip_text_content(self, ):
        texts = ['', None, 'test ', 'block 1']
        parser = PDFParser()
        content = parser.strip_text_content(texts=texts)
        self.assertEqual(content, 'test\n\nblock 1')

    def test_parser_chunk(self):
        content_list = [
            {
                'type': 'text',
                'text': '1'
            },
            {
                'type': 'text',
                'text': '2'
            },
            {
                'type': 'table',
                'table_caption': 'table 1',
                'table_body': ''
            },
            {
                'type': 'text',
                'text': '3'
            },
            {
                'type': 'text',
                'text': '4'
            },
            {
                'type': 'text',
                'text': '5'
            },
            {
                'type': 'text',
                'text': '6'
            },
            {
                'type': 'table',
                'table_caption': 'table 2',
                'table_body': ''
            },
        ]
        parser = PDFParser()
        parser.consecutive_block_num = 4
        parser.block_overlap_num = 1
        parser.file_name = '/fake/path'

        chunks = parser.chunk(content_list=content_list,
                              temp_asset_dir='',
                              asset_save_dir='')

        # print('=' * 80)
        # for chunk in chunks:
        #     print(chunk)
        #     print('=' * 80)

        # should have 2 table chunk
        table_chunks = [
            chunk for chunk in chunks if chunk.content_type == 'table'
        ]
        self.assertEqual(len(table_chunks), 2)
        self.assertEqual(table_chunks[0].extra_description.decode('utf-8'),
                         'table 1')
        self.assertEqual(table_chunks[1].extra_description.decode('utf-8'),
                         'table 2')

        # should have 2 text chunk
        text_chunks = [
            chunk for chunk in chunks if chunk.content_type == 'text'
        ]
        self.assertEqual(len(text_chunks), 2)
        self.assertEqual(text_chunks[0].content.decode('utf-8'),
                         '\n\n'.join(['1', '2', '3', '4']))
        self.assertEqual(text_chunks[1].content.decode('utf-8'),
                         '\n\n'.join(['4', '5', '6']))

        # # chunk 0
        # self.assertTrue('h1' in str(chunks[0]) and 'p1' in str(chunks[0])
        #                 and 'p2' in str(chunks[0]))

        # # chunk 1
        # self.assertTrue('h2' in str(chunks[1]))

    def test_is_valid_block(self):
        parser = PDFParser()

        block = {}
        self.assertFalse(parser.is_valid_block(block))

        block = {'type': 'image', 'img_path': ''}
        self.assertFalse(parser.is_valid_block(block))

        block = {
            'type': 'table',
        }
        self.assertFalse(parser.is_valid_block(block))

        pass


if __name__ == '__main__':

    unittest.main()
