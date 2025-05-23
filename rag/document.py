import json
import sys
import argparse
import traceback
import logging
from typing import Union, Dict, List, Any

import config
from .nlp import EmbeddingModel
from parse.parser import Chunk


def make_record(chunk: Chunk, embed: EmbeddingModel) -> Dict[str, Any]:
    """
    Make a record from chunk.

    Args:
    - chunk: parsed chunk.
    - embed: embedding model.

    Returns:
    - A dict containing all columns of a record.
    """
    content = chunk.content
    if chunk.content_type != config.ChunkType.TEXT:
        content = chunk.extra_description
    content = content.decode('utf-8')

    meta = {}
    if chunk.content_type == config.ChunkType.IMAGE:
        meta['content_url'] = chunk.content_url
    if chunk.content_type == config.ChunkType.TABLE:
        meta['table_content'] = chunk.content.decode('utf-8')

    embeddings = embed.encode([content])

    return {
        'uuid': chunk.uuid,
        'content': content,
        'meta': json.dumps(meta, indent=4),
        'sparse_vector': embeddings['sparse'][[0]],
        'dense_vector': embeddings['dense'][0],
    }


def process_file(file_path: str) -> Dict[str, bool]:
    """
    Process a file, parse and save chunks into db.

    Args:
    - file_path: path to the file.

    Returns:
    - A list containing all successfuly inserted chunks' uuid, the order is aligned
        with the order of chunks in original file.
    """
    from parse.pdf_parser import PDFParser
    from rag.db import get_vector_db
    from rag.nlp import get_embed_model

    from config import PARSED_ASSET_DATA_DIR, MILVUS_COLLECTION_NAME, MILVUS_DB_NAME
    parser = PDFParser()

    vector_db = get_vector_db()
    embed_model = get_embed_model()

    # parse file
    chunks = parser.parse(file_path=file_path,
                          asset_save_dir=PARSED_ASSET_DATA_DIR)

    # insert into vector db
    records = [make_record(chunk, embed_model) for chunk in chunks]
    logging.info(f'total {len(records)} records')

    failed_records = []
    success_records = {}
    for record in records:
        try:
            vector_db.insert(record)
            success_records[record['uuid']] = True
        except Exception as e:
            logging.info(f"Exception: {type(e).__name__} - {e}")

            formatted_traceback = traceback.format_exc()
            logging.info(formatted_traceback)

            failed_records.append(record)

    # second try if any failure
    for record in failed_records:
        try:
            vector_db.insert(record)
            success_records[record['uuid']] = True
        except Exception as e:
            logging.info(f"Exception: {type(e).__name__} - {e}")

            formatted_traceback = traceback.format_exc()
            logging.info(formatted_traceback)

    logging.info(f'successfully insert {len(success_records)} records')

    # update file - chunking state
    # TODO

    return [
        record['uuid'] for record in records
        if record['uuid'] in success_records
    ]
