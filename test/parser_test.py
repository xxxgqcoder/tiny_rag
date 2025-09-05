import os
import sys
import time
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print(sys.path[-1])

from common.config import ParserConfig, TinyRAGConfig
from common.utils import get_project_base_directory
from parse.pdf_parser import PDFParser


class TestPDFParser(unittest.TestCase):
    def test_base(self):
        project_root_dir = get_project_base_directory()
        print("root dir", project_root_dir)

        pdf_file_path = os.path.join(project_root_dir, "test/test.pdf")

        TinyRAGConfig.parser_config = ParserConfig(  # type: ignore
            config_file_path=os.path.join(project_root_dir, "assets/MinerU/magic-pdf.json"),
            asset_save_dir=os.path.join(project_root_dir, "assets/parsed_assets"),
        )

        parser = PDFParser()

        # first round parse
        start_time = time.time()
        first_contents = parser.parse(file_path=pdf_file_path)
        end_time = time.time()

        first_time = end_time - start_time
        print(f"First round parsing took {first_time:.2f} seconds")

        # second round parse
        start_time = time.time()
        second_contents = parser.parse(file_path=pdf_file_path)
        end_time = time.time()

        second_time = end_time - start_time
        print(f"Second round parsing took {second_time:.2f} seconds")


if __name__ == "__main__":
    unittest.main()
