import unittest
import os

from parse.parser import Parser, SupportedFileType, Chunk, ChunkType
from parse.pdf_parser import PDFParser

import config
from utils import get_project_base_directory


class TestBaseClass(unittest.TestCase):

    def test_base_class(self):
        chunk = Chunk(ChunkType.TEXT, content='test', extra_description='')
        parser = Parser()
        chunks = parser.parse('fake_pdf_path.pdf', 'fake_path')
        self.assertEqual(1, 1)


class TestPDFParser(unittest.TestCase):

    def test_pdf_parser(self):
        parser = PDFParser()

        pdf_file_path = os.path.join(
            config.PROJECT_ASSET_FOLDER,
            'test/Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift.pdf'
        )

        chunks = parser.parse(pdf_file_path,
                              "/Users/xcoder/projects/tiny_rag/test_parse")
        # for i, chunk in enumerate(chunks):
        #     print(f'chunk {i}')
        #     print(chunk)
        #     print('=' * 120)
        self.assertEqual(1, 1)

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

        for i, chunk in enumerate(chunks):
            print(chunk)
            print('=' * 120)


if __name__ == '__main__':
    unittest.main()
