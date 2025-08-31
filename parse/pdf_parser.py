import json
import logging
import os
import shutil
import tempfile
from typing import Any

from common.config import TinyRAGConfig
from common.data import Chunk, ChunkType
from parse.parser import Parser
from utils import safe_strip, singleton


@singleton
class PDFParser(Parser):
    """
    PDF parser implementation, backed by [MinerU](https://github.com/opendatalab/MinerU).
    """

    def __init__(
        self,
    ):
        super().__init__()

        with open(file=TinyRAGConfig.parser_config.config_file_path) as f:  # type: ignore
            conf = json.load(f)
        # used in chunking, number of consecutive block to be considered as one chunk.
        self.consecutive_block_num = conf.get("consecutive_block_num", 8)
        # used in chunking, number of overlapped block num between two consecutive chunks.
        self.block_overlap_num = conf.get("block_overlap_num", 3)

        logging.info(f"Parsr config: {json.dumps(conf, indent=4)}")
        assert self.block_overlap_num < self.consecutive_block_num, (
            f"block overlap num ({self.block_overlap_num}) be less than consecutive block num ({self.consecutive_block_num})"
        )

        # set environment variable for magic_pdf to load config json file
        os.environ["MINERU_TOOLS_CONFIG_JSON"] = conf.get("mineru_tools_conf_json", "")
        os.environ["MINERU_MODEL_SOURCE"] = conf.get("mineru_model_source", "local")

    def parse(
        self,
        file_path: str,
        asset_save_dir: str,
    ) -> list[Chunk]:
        os.makedirs(asset_save_dir, exist_ok=True)
        self.file_name = os.path.basename(file_path)

        # get original chunk list
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        temp_asset_dir = temp_dir.name
        logging.info(f"temp asset directory: {temp_asset_dir}")

        chunks = self.parse_pdf_content(
            file_path=file_path,
            temp_asset_dir=temp_asset_dir,
            asset_save_dir=asset_save_dir,
        )
        logging.info(f"original chunk num: {len(chunks)}")

        # with open(os.path.join(temp_asset_dir, 'chunks.pickle'), 'wb') as f:
        #     pickle.dump(chunks, f)

        # with open(os.path.join(temp_asset_dir, 'chunks.pickle'), 'rb') as f:
        #     print(f'loading content list from {temp_asset_dir}')
        #     chunks = pickle.load(f)

        self.chunks = self.filter_chunks(chunks)
        logging.info(f"after filtering, chunk num: {len(self.chunks)}")

        all_types = sorted(list(set([str(chunk.content_type) for chunk in self.chunks])))
        logging.info(f"all parsed block types: {all_types}")

        # merge chunk list
        merged_chunks = self.merge_chunk(chunks=self.chunks)
        logging.info(f"{self.file_name}: total {len(merged_chunks)} chunks ")

        temp_dir.cleanup()
        return merged_chunks

    def parse_pdf_content(
        self,
        file_path: str,
        temp_asset_dir: str,
        asset_save_dir: str,
    ) -> list[Chunk]:
        """
        Parse PDF content and return content list.
        The result is a list of json oject representing a pdf content block.

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

        Typical parsed paper content is organized as list of content block.
        Headlines will stored in one separated block, with `text_level` = 1 while regular content block's `text_level` key is missing.
        Headline blocks are followed by regular content block, including `text`, `equation`, `table` and `image` (distinguished by key `type`).
        All captions are stored in each block's caption key, for example, caption of a parsed image is saved in `img_caption` key of the block.

        See https://github.com/opendatalab/MinerU/blob/master/demo/demo.py for more details.

        Returns:
        - A list of parsed chunk.
        """
        # NOTE: magic_pdf package uses singleton design and the model isntance is initialized when the module is imported,
        # so postpone the import statement until parse method is called.

        import copy
        from pathlib import Path

        from mineru.backend.pipeline.model_json_to_middle_json import (
            result_to_middle_json as pipeline_result_to_middle_json,
        )
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
        from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
        from mineru.utils.enum_class import MakeMode

        # prepare env
        try:
            shutil.rmtree(temp_asset_dir)
        except:
            pass
        os.makedirs(temp_asset_dir, exist_ok=True)

        lang = "ch"
        start_page_id = 0
        end_page_id = None
        parse_method = "auto"

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
        md_content_str: list[str] = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)  # type: ignore
        md_writer.write_string(f"{file_name}.md", str(md_content_str))

        # dump content list
        image_dir = str(os.path.basename(local_image_dir))
        content_list: list[dict[str, Any]] = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)  # type: ignore
        md_writer.write_string(f"{file_name}_content_list.json", json.dumps(content_list, ensure_ascii=False, indent=4))

        # dump middle json
        md_writer.write_string(f"{file_name}_middle.json", json.dumps(middle_json, ensure_ascii=False, indent=4))

        # dump model json
        md_writer.write_string(f"{file_name}_model.json", json.dumps(model_json, ensure_ascii=False, indent=4))

        # parse content list to chunks
        def _load_image(p: str) -> bytes:
            with open(p, "rb") as f:
                image_bytes = f.read()
            return image_bytes

        def _save_image(src_path: str, dst_dir: str) -> None:
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            shutil.copyfile(src_path, dst_path)

        def _is_valid_content(content: dict[str, Any]) -> bool:
            """
            There are corner cases where returned blocks dont contain expected keys or values are empty.

            Returns:
            - bool: true if block is valid.
            """
            # missing key
            if "type" not in content:
                return False
            # text / equation
            if content["type"] in ["text", "equation"]:
                return "text" in content
            # image
            if content["type"] == "image":
                return "img_path" in content and len(content["img_path"]) > 0
            # table
            if content["type"] == "table":
                return "table_body" in content
            return True

        def _format_caption(caption: Any) -> str:
            """
            Format caption as text.
            """
            if isinstance(caption, list):
                ret = "\n".join([str(e) for e in caption])
                return ret
            return str(caption)

        chunks = []
        for content in content_list:
            if not _is_valid_content(content):
                logging.info(f"invalid content: {json.dumps(content, indent=4)}")
                continue

            # text content
            if content["type"] in ["text", "equation"]:
                text = self.strip_text_content([content["text"]])
                chunks.append(
                    Chunk(
                        content_type=ChunkType.TEXT,
                        file_name=self.file_name,
                        content=text.encode("utf-8", errors="ignore"),
                        extra_description="".encode("utf-8", errors="ignore"),
                        content_url="",
                        uuid="",
                    )
                )

            # iamge content
            elif content["type"] in ["image"]:
                texts = [
                    _format_caption(content.get("img_caption", "")),
                    _format_caption(content.get("img_footnote", "")),
                ]
                extra_description = self.strip_text_content(texts)
                if len(extra_description) == 0:
                    extra_description = "no caption for this image"

                # NOTE: hard coded image path format
                abs_img_path = os.path.join(temp_asset_dir, str(Path(self.file_name).stem), "auto", content["img_path"])
                _save_image(abs_img_path, asset_save_dir)

                chunk = Chunk(
                    content_type=ChunkType.IMAGE,
                    file_name=self.file_name,
                    content=_load_image(abs_img_path),
                    extra_description=(extra_description).encode("utf-8", errors="ignore"),
                    content_url=os.path.join(asset_save_dir, os.path.basename(abs_img_path)),
                    uuid="",
                )
                chunks.append(chunk)

            # table content
            elif content["type"] in ["table"]:
                texts = [
                    _format_caption(content.get("table_caption", "")),
                    _format_caption(content.get("table_footnote", "")),
                ]
                extra_description = self.strip_text_content(texts)
                if len(extra_description) == 0:
                    extra_description = "no caption for this table"

                abs_img_path = os.path.join(temp_asset_dir, str(Path(self.file_name).stem), "auto", content["img_path"])
                if content["img_path"]:
                    _save_image(abs_img_path, asset_save_dir)

                chunk = Chunk(
                    content_type=ChunkType.TABLE,
                    file_name=self.file_name,
                    content=content["table_body"].encode("utf-8", errors="ignore"),
                    extra_description=(extra_description).encode("utf-8", errors="ignore"),
                    content_url=os.path.join(asset_save_dir, os.path.basename(abs_img_path))
                    if content["img_path"]
                    else "",
                    uuid="",
                )
                chunks.append(chunk)
            else:
                pass

        return chunks

    def merge_chunk(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Chunk parsed pdf contents.

        Scan `self.consecutive_block_num` consecutive chunk and combine as one chunk.
        If image / table chunk is encountered within current consecutive chunks, then make the image / table chunk as independent chunk and continue scan until `self.consecutive_block_num` is met.

        Two consecutive merged chunks have `self.block_overlap_num` overlapped chunk to ensure semantic coherence.

        Returns:
        - List of chunks.
        """
        merged_chunks = []
        chunk_buffer = []
        i: int = 0
        # since we apply overlap,i can not exceed len(chunks) - self.block_overlap_num,
        # otherwise, infinite loop may happen.
        while i < len(chunks) - self.block_overlap_num:
            # inner loop start from current chunk
            j: int = i
            while j < len(chunks) and len(chunk_buffer) < self.consecutive_block_num:
                chunk = chunks[j]

                # text chunk
                if chunk.content_type in [ChunkType.TEXT]:
                    chunk_buffer.append(chunk)
                # image / table chunk
                elif chunk.content_type in [ChunkType.IMAGE, ChunkType.TABLE]:
                    merged_chunks.append(chunk)
                else:
                    pass

                # move one step forward
                j += 1

            # inner loop ends when j == len(chunks) or len(block_buffer) == self.consecutive_block_num
            # generate new chunk if buffer is not empty.
            if len(chunk_buffer) > 0:
                texts = [chunk.content.decode("utf-8") for chunk in chunk_buffer]
                texts = "\n\n".join(texts)
                new_chunk = Chunk(
                    content_type=ChunkType.TEXT,
                    file_name=self.file_name,
                    content=texts.encode("utf-8", errors="ignore"),
                    extra_description="".encode("utf-8", errors="ignore"),
                    content_url="",
                    uuid="",
                )
                merged_chunks.append(new_chunk)
                chunk_buffer.clear()

            # start next iteration
            i: int = j - self.block_overlap_num

        return merged_chunks

    def filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Filter too short chunks
        """
        filtered_chunks = []
        for chunk in chunks:
            content = chunk.content
            if chunk.content_type != ChunkType.TEXT:
                content = chunk.extra_description
            content = safe_strip(content.decode("utf-8"))
            if len(content) < 1:
                logging.info(f"{self.file_name}: remove chunk due to too short content: {str(chunk)}")
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
            if len(striped) == 0 or striped == "[]":
                continue
            content += striped
            content += "\n\n"
        return content.strip()
