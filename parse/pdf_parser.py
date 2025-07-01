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

    def __init__(self, ):
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
                logging.info(f'{self.file_name}: invalid block, ignore {block}')
                continue
            filtered_content_list.append(block)

        self.content_list = filtered_content_list
        all_types = sorted(list(set([block['type'] for block in self.content_list])))
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

        logging.info(f'{self.file_name}: {len(filtered_chunks)} chunks after filtering')

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

        https://github.com/opendatalab/MinerU/blob/master/demo/demo.py for more details.

        Returns:
        - A list of parsed content block dict.
        """
        # NOTE: magic_pdf package uses singleton design and the model isntance is
        # initialized when the module is imported, so postpone the import statement
        # until parse method is called.

        import copy
        from pathlib import Path

        from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
        from mineru.utils.enum_class import MakeMode
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

        # prepare env
        try:
            shutil.rmtree(temp_asset_dir)
        except:
            pass
        os.makedirs(temp_asset_dir, exist_ok=True)

        lang = 'ch'
        start_page_id = 0
        end_page_id = None
        parse_method = 'auto'

        file_name = str(Path(file_path).stem)
        pdf_bytes = read_fn(file_path)

        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
            [new_pdf_bytes],
            [lang],
            parse_method=parse_method,
            formula_enable=True,
            table_enable=True,
        )

        model_list = infer_results[0]
        model_json = copy.deepcopy(model_list)
        local_image_dir, local_md_dir = prepare_env(temp_asset_dir, file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        middle_json = pipeline_result_to_middle_json(
            model_list,
            all_image_lists[0],
            all_pdf_docs[0],
            image_writer,
            lang,
            ocr_enabled_list[0],
            True,
        )

        pdf_info = middle_json["pdf_info"]

        # draw span and layout
        draw_layout_bbox(pdf_info, new_pdf_bytes, local_md_dir, f"{file_name}_layout.pdf")
        draw_span_bbox(pdf_info, new_pdf_bytes, local_md_dir, f"{file_name}_span.pdf")
        md_writer.write(f"{file_name}_origin.pdf", new_pdf_bytes)

        # dump md
        image_dir = str(os.path.basename(local_image_dir))
        md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        md_writer.write_string(f"{file_name}.md", md_content_str)

        # dump content list
        image_dir = str(os.path.basename(local_image_dir))
        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

        # dump middle json
        md_writer.write_string(
            f"{file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

        # dump model json
        md_writer.write_string(
            f"{file_name}_model.json",
            json.dumps(model_json, ensure_ascii=False, indent=4),
        )

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
            # inner loop start from current block
            j = i
            while j < len(content_list) and len(block_buffer) < self.consecutive_block_num:

                block = content_list[j]

                # text block
                if block['type'] in ['text', 'equation']:
                    block_buffer.append(block)

                # image / table block
                elif block['type'] in ['image', 'table']:
                    if block['type'] == 'table':
                        chunks.extend(self.process_table_blocks(
                            table_blocks=content_list[j:j + 1],
                            temp_asset_dir=temp_asset_dir,
                            asset_save_dir=asset_save_dir,
                        ))
                    else:
                        chunks.extend(self.process_image_blocks(
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
                chunks.extend(self.process_text_blocks(
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
        return [Chunk(
            content_type=ChunkType.TEXT,
            file_name=self.file_name,
            content=content.encode('utf-8'),
            extra_description=''.encode('utf-8'),
        )]

    def process_image_blocks(
        self,
        image_blocks: list[dict],
        temp_asset_dir: str,
        asset_save_dir: str,
    ) -> list[Chunk]:
        from pathlib import Path

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

            # NOTE: hard coded image path format
            abs_img_path = os.path.join(temp_asset_dir, str(Path(self.file_name).stem), 'auto', block['img_path'])
            _save_image(abs_img_path, asset_save_dir)

            chunk = Chunk(
                content_type=ChunkType.IMAGE,
                file_name=self.file_name,
                content=_load_image(abs_img_path),
                extra_description=(extra_description).encode('utf-8'),
                content_url=os.path.join(asset_save_dir, os.path.basename(abs_img_path)),
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
                logging.info(f'{self.file_name}: remove chunk due to too short content: {str(chunk)}')
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
