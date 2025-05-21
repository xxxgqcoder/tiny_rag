import os
import tempfile
import logging
import shutil
from typing import Tuple

import config
from .parser import Parser, Chunk, ChunkType
from config import MAGIC_PDF_CONFIG_PATH
from utils import safe_strip


class PDFParser(Parser):
    """
    PDF parser implementation, backed by [MinerU](https://github.com/opendatalab/MinerU).
    """

    def __init__(self, ):
        super().__init__()
        # set environment variable for magic_pdf to load config json file
        os.environ["MINERU_TOOLS_CONFIG_JSON"] = MAGIC_PDF_CONFIG_PATH

    def parse(
        self,
        file_path: str,
        asset_save_dir: str,
    ) -> list[Chunk]:
        logging.info(f'parsing file from {file_path}')

        os.makedirs(asset_save_dir, exist_ok=True)

        # get content list
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        logging.info(f'asset directory: {temp_dir.name}')

        content_list = self.parse_pdf_content(file_path=file_path,
                                              temp_asset_dir=temp_dir.name)
        self.content_list = content_list

        # get chunk list
        chunks = self.chunk(
            content_list=content_list,
            temp_asset_dir=temp_dir.name,
            asset_save_dir=asset_save_dir,
        )

        # filter chunks
        filtered_chunks = self.filter_chunks(chunks)

        temp_dir.cleanup()

        return filtered_chunks

    def parse_pdf_content(
        self,
        file_path: str,
        temp_asset_dir: str,
    ) -> list[dict]:
        """
        Parse PDF content and return content list. The result is a list of json 
        oject representing a pdf content block.
        
        Dict object key explanation:
            - `img_caption`: the image caption.
            - `img_footnote`:
            - `img_path`: path to parsed image.
            - `page_idx`: page index.
            - `table_body`: table content in html format.
            - `table_caption`: table caption.
            - `table_footnote`:
            - `text`: the block text content.
            - `text_format`: used in latex forumla block.
            - `text_level`: used in headline block.
            - `type`: block type, can be one of 'equation', 'image', 'table', 'text'.
        
        Typical paper parsed content is organized by list of content block. Headlines
        will stored in one separated block, with `text_level` = 1 while regular content
        block's `text_level` key is missing. Headline blocks are followed by regular
        content block, including `text`, `equation`, `table` and `image` (distinguished 
        by key `type`). All captions are stored in each block's caption key, for 
        example, caption of a parsed image is saved in `img_caption` key of the block.

        Refer [MinerU API demo](https://mineru.readthedocs.io/en/latest/user_guide/usage/api.html) 
        for more details.

        Returns:
        - A list of parsed content block dict.
        - A python TemporaryDirectory.
        """
        # NOTE: magic_pdf package uses singleton design and the model isntance is
        # initialized when the module is imported, so postpone the import statement
        # until parse method is called.
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod

        # prepare env
        name_without_suff = os.path.basename(file_path).split(".")[0]
        local_image_dir = os.path.join(temp_asset_dir, "images")
        local_md_dir = temp_asset_dir
        image_dir = os.path.basename(local_image_dir)
        os.makedirs(local_image_dir, exist_ok=True)

        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        # read bytes
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(file_path)
        logging.info(f"{file_path}: read bytes count: {len(pdf_bytes)}")

        # process
        ds = PymuDocDataset(pdf_bytes)

        # inference
        infer_result = ds.apply(doc_analyze,
                                ds.classify() == SupportedPdfParseMethod.OCR)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

        # draw model result on each page
        infer_result.draw_model(
            os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

        # get model inference result
        model_inference_result = infer_result.get_infer_res()

        # draw layout result on each page
        pipe_result.draw_layout(
            os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

        # draw spans result on each page
        pipe_result.draw_span(
            os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

        # get markdown content
        md_content = pipe_result.get_markdown(image_dir)

        # dump markdown
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

        # get content list content
        content_list = pipe_result.get_content_list(image_dir)

        # dump content list
        pipe_result.dump_content_list(
            md_writer, f"{name_without_suff}_content_list.json", image_dir)

        # get middle json
        middle_json_content = pipe_result.get_middle_json()

        # dump middle json
        pipe_result.dump_middle_json(md_writer,
                                     f'{name_without_suff}_middle.json')

        return content_list

    def chunk(
        self,
        content_list: list[dict],
        temp_asset_dir: str,
        asset_save_dir: str,
    ) -> Chunk:
        """
        Chunk parsed pdf contents.

        Rules:
        - Combine a headline block with following content blocks as one chunk. 
            Headline is merged into the chunk's content.
        - Filter table / image block from the chunk and make a separated chunk. 
            The chunk content will be table / image caption.

        Returns:
        - List of merged chunks.
        """
        chunks = []
        i = 0
        while i < len(content_list):
            # find first headline block
            j = i + 1
            while j < len(content_list) \
                    and 'text_level' not in content_list[j]:
                j += 1

            # find headline block or reach end of list
            blocks = content_list[i:j]
            all_types = sorted(list(set([block['type'] for block in blocks])))
            logging.info(f"all parsed block types: {all_types}")

            # filter and merge blocks
            text_blocks = [
                block for block in blocks
                if block['type'] in ['text', 'equation']
            ]
            if len(text_blocks) > 0:
                chunks.extend(
                    self.process_text_blocks(text_blocks, temp_asset_dir,
                                             asset_save_dir))

            # image blocks
            image_blocks = [
                block for block in blocks if block['type'] == 'image'
            ]
            if len(image_blocks) > 0:
                chunks.extend(
                    self.process_image_blocks(image_blocks, temp_asset_dir,
                                              asset_save_dir))

            # table blocks
            table_blocks = [
                block for block in blocks if block['type'] == 'table'
            ]
            if len(table_blocks) > 0:
                chunks.extend(
                    self.process_table_blocks(table_blocks, temp_asset_dir,
                                              asset_save_dir))

            # start next iteration
            i = j

        return chunks

    def process_text_blocks(
        self,
        text_blocks: list[dict],
        temp_asset_dir: str,
        asset_save_dir: str,
    ) -> list[Chunk]:
        texts = [block['text'] for block in text_blocks]
        content = self.filter_text_content(texts)
        return [
            Chunk(
                content_type=ChunkType.TEXT,
                content=content.encode('utf-8'),
                extra_description=''.encode('utf-8'),
            )
        ]

    def process_image_blocks(
        self,
        image_blocks: list[dict],
        temp_asset_dir: str,
        asset_save_dir: str,
    ) -> list[Chunk]:

        def _load_image(p: str) -> bytes:
            with open(p, 'rb') as f:
                image_bytes = f.read()
            return image_bytes

        def _save_image(src_path: str, dst_dir: str):
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            shutil.copyfile(src_path, dst_path)

        chunks = []
        for block in image_blocks:
            texts = [block['img_caption'], str(block['img_footnote'])]
            extra_description = self.filter_text_content(texts)
            if len(extra_description) == 0:
                extra_description = "no caption for this image"

            abs_img_path = os.path.join(temp_asset_dir, block['img_path'])
            _save_image(abs_img_path, asset_save_dir)

            chunk = Chunk(
                content_type=ChunkType.IMAGE,
                content=_load_image(abs_img_path),
                extra_description=(extra_description).encode('utf-8'),
                content_url=os.path.join(asset_save_dir,
                                         os.path.basename(block['img_path'])),
            )
            chunks.append(chunk)

        return chunks

    def process_table_blocks(
        self,
        table_blocks: list[dict],
        temp_asset_dir: str,
        asset_save_dir: str,
    ) -> list[Chunk]:
        chunks = []
        for block in table_blocks:
            texts = [block['table_caption'], str(block['table_footnote'])]
            extra_description = self.filter_text_content(texts)
            if len(extra_description) == 0:
                extra_description = "no caption for this table"

            chunk = Chunk(
                content_type=ChunkType.TABLE,
                content=block['table_body'].encode('utf-8'),
                extra_description=(extra_description).encode('utf-8'),
            )
            chunks.append(chunk)

        return chunks

    def filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Filter too short chunks
        """
        filtered_chunks = []
        for chunk in chunks:
            content = chunk.content
            if chunk.content_type != config.ChunkType.TEXT:
                content = chunk.extra_description
            content = safe_strip(content.decode('utf-8'))
            if len(content) < 8 or len(content.split()) < 3:
                logging.info(f'remove chunk due to too short content')
                logging.info('original content')
                logging.info(str(chunk))
                continue

            filtered_chunks.append(chunk)

        return filtered_chunks

    def filter_text_content(self, texts: list[str]) -> str:
        """
        Filter and merge text content
        """
        content = ""
        for text in texts:
            striped = safe_strip(text)
            if len(striped) == 0 or striped == '[]':
                continue
            content += striped
            content += "\n\n"
        return content.strip()
