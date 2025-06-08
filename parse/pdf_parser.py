import os
import tempfile
import logging
import shutil
import pickle
import json
from typing import Tuple, Dict, Any

import config
from utils import singleton, safe_strip, logging_exception
from parse.parser import Parser, Chunk, ChunkType

@singleton
class PDFParser(Parser):
    """
    PDF parser implementation, backed by [MinerU](https://github.com/opendatalab/MinerU).
    """

    def __init__(
        self,
    ):
        super().__init__()
        """
        Args:
        - consecutive_block_num: used in chunking, number of consecutive block to be considered as one chunk.
        - block_overlap_num: used in chunking, number of overlapped block num between two consecutive chunks.
        """
        with open(config.PDF_PARSER_CONFIG_PATH) as f:
            conf = json.load(f)

        self.consecutive_block_num = conf.get('consecutive_block_num', 8)
        self.block_overlap_num = conf.get('block_overlap_num', 3)

        logging.info(f"parsr config: {json.dumps(conf, indent=4)}")

        assert self.block_overlap_num < self.consecutive_block_num,\
            f"block overlap num ({self.block_overlap_num}) be less than consecutive block num ({self.consecutive_block_num})"

        # set environment variable for magic_pdf to load config json file
        os.environ["MINERU_TOOLS_CONFIG_JSON"] = config.PDF_PARSER_CONFIG_PATH

    def parse(
        self,
        file_path: str,
        asset_save_dir: str,
    ) -> list[Chunk]:
        os.makedirs(asset_save_dir, exist_ok=True)
        self.file_name = os.path.basename(file_path)

        # get content list
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        logging.info(f'asset directory: {temp_dir.name}')
        temp_asset_dir = temp_dir.name
        # temp_asset_dir = './parsed_assets'

        content_list = self.parse_pdf_content(
            file_path=file_path,
            temp_asset_dir=temp_asset_dir,
        )
        # with open(os.path.join(temp_asset_dir, 'content_list.pickle'),
        #           'wb') as f:
        #     pickle.dump(content_list, f)

        # with open(os.path.join(temp_asset_dir, 'content_list.pickle'),
        #           'rb') as f:
        #     print(f'loading content list from {temp_asset_dir}')
        #     content_list = pickle.load(f)

        filtered_content_list = []
        for block in content_list:
            if not self.is_valid_block(block):
                logging.info(
                    f'{self.file_name}: invalid block, ignore {block}')
                continue
            filtered_content_list.append(block)

        self.content_list = filtered_content_list
        all_types = sorted(
            list(set([block['type'] for block in self.content_list])))
        logging.info(f"all parsed block types: {all_types}")

        # get chunk list
        chunks = self.chunk(
            content_list=self.content_list,
            temp_asset_dir=temp_asset_dir,
            asset_save_dir=asset_save_dir,
        )

        # filter chunks
        filtered_chunks = self.filter_chunks(chunks)

        temp_dir.cleanup()

        logging.info(
            f'{self.file_name}: {len(filtered_chunks)} chunks after filtering')

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
        infer_result = ds.apply(doc_analyze)
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

        Scan `self.consecutive_block_num` consecutive blocks and combine as one
        chunk.
        If image / table block is encountered within current consecutive blocks,
        then make the image / table block as independent chunk and continue scan
        untile `self.consecutive_block_num` is met.

        Two consecutive chunks have `self.block_overlap_num` overlapped block to
        ensure semantic coherence.

        Returns:
        - List of chunks.
        """
        chunks = []
        block_buffer = []
        i = 0
        # since we apply overlap,i can not exceed len(content_list) - self.block_overlap_num,
        # otherwise, infinite loop may happen.
        while i < len(content_list) - self.block_overlap_num:
            print(f"i = {i}")

            # inner loop start from current block
            j = i
            while j < len(content_list) and len(
                    block_buffer) < self.consecutive_block_num:
                print(f"\tj = {j}")

                block = content_list[j]

                # text block
                if block['type'] in ['text', 'equation']:
                    block_buffer.append(block)

                # image / table block
                elif block['type'] in ['image', 'table']:
                    if block['type'] == 'table':
                        chunks.extend(
                            self.process_table_blocks(
                                table_blocks=content_list[j:j + 1],
                                temp_asset_dir=temp_asset_dir,
                                asset_save_dir=asset_save_dir,
                            ))
                    else:
                        chunks.extend(
                            self.process_image_blocks(
                                image_blocks=content_list[j:j + 1],
                                temp_asset_dir=temp_asset_dir,
                                asset_save_dir=asset_save_dir,
                            ))
                else:
                    pass

                # move one step forward
                j += 1

            # inner loop ends when j == len(content_list)
            # or len(block_buffer) == self.consecutive_block_num
            # generate new chunk if buffer is not empty.
            if len(block_buffer) > 0:
                chunks.extend(
                    self.process_text_blocks(
                        text_blocks=block_buffer,
                        temp_asset_dir=temp_asset_dir,
                        asset_save_dir=asset_save_dir,
                    ))
                block_buffer.clear()

            # start next iteration
            i = j - self.block_overlap_num

        return chunks

    def process_text_blocks(
        self,
        text_blocks: list[dict],
        temp_asset_dir: str,
        asset_save_dir: str,
    ) -> list[Chunk]:
        texts = [str(block['text']) for block in text_blocks]
        content = self.strip_text_content(texts)
        return [
            Chunk(
                content_type=ChunkType.TEXT,
                file_name=self.file_name,
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
            texts = [
                str(block.get('img_caption', '')),
                str(block.get('img_footnote', '')),
            ]
            extra_description = self.strip_text_content(texts)
            if len(extra_description) == 0:
                extra_description = "no caption for this image"

            abs_img_path = os.path.join(temp_asset_dir, block['img_path'])
            _save_image(abs_img_path, asset_save_dir)

            chunk = Chunk(
                content_type=ChunkType.IMAGE,
                file_name=self.file_name,
                content=_load_image(abs_img_path),
                extra_description=(extra_description).encode('utf-8'),
                content_url=os.path.join(asset_save_dir,
                                         os.path.basename(abs_img_path)),
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
            texts = [
                str(block.get('table_caption', '')),
                str(block.get('table_footnote', '')),
            ]
            extra_description = self.strip_text_content(texts)
            if len(extra_description) == 0:
                extra_description = "no caption for this table"

            chunk = Chunk(
                content_type=ChunkType.TABLE,
                file_name=self.file_name,
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
                logging.info(
                    f'{self.file_name}: remove chunk due to too short content: {str(chunk)}'
                )
                continue

            filtered_chunks.append(chunk)

        return filtered_chunks

    def strip_text_content(self, texts: list[str]) -> str:
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

    def is_valid_block(self, block: Dict[str, Any]) -> bool:
        """
        There are corner cases where returned blocks dont contain expected keys 
        or values are empty.

        Returns:
        - bool: true if block is valid.
        """
        # missing key
        if 'type' not in block:
            return False

        # text / equation
        if block['type'] in ['text', 'equation']:
            return 'text' in block

        # image
        if block['type'] == 'image':
            return 'img_path' in block and len(block['img_path']) > 0

        # table
        if block['type'] == 'table':
            return 'table_body' in block

        return True
