import json
import sys
import argparse
import traceback
import logging
import os
from typing import Union, Dict, List, Any

import config
from .nlp import EmbeddingModel
from parse.parser import Chunk
from utils import now_in_utc
from rag.db import get_vector_db, get_rational_db
from rag.nlp import get_embed_model


def make_record(
    file_path: str,
    chunk: Chunk,
    embed: EmbeddingModel,
) -> Dict[str, Any]:
    """
    Make a record from chunk.

    Args:
    - file_path: original file path.
    - chunk: parsed chunk.
    - embed: embedding model.

    Returns:
    - A dict containing all columns of a record.
    """
    content = chunk.content
    if chunk.content_type != config.ChunkType.TEXT:
        content = chunk.extra_description
    content = content.decode('utf-8')

    meta = {'file_path': file_path}
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


def process_new_file(file_path: str) -> Dict[str, bool]:
    """
    Process a file, parse and save chunks into db.

    Args:
    - file_path: path to the file.

    Returns:
    - A list containing all successfuly inserted chunks' uuid, the order is aligned
        with the order of chunks in original file.
    """
    from parse.pdf_parser import PDFParser
    from config import PARSED_ASSET_DATA_DIR

    parser = PDFParser()

    vector_db = get_vector_db()
    sql_db = get_rational_db()
    embed_model = get_embed_model()

    # clean up document
    clean_up_document(file_path=file_path)

    # parse file
    chunks = parser.parse(
        file_path=file_path,
        asset_save_dir=PARSED_ASSET_DATA_DIR,
    )

    # insert parsed chunks into vector db
    records = [make_record(chunk, embed_model) for chunk in chunks]
    logging.info(f'total {len(records)} records')

    failed_records = []
    success_records = {}
    for record in records:
        try:
            insert_cnt = vector_db.insert(record)
            if insert_cnt == 1:
                success_records[record['uuid']] = True
            else:
                failed_records.append(record)

        except Exception as e:
            logging.info(f"Exception: {type(e).__name__} - {e}")

            formatted_traceback = traceback.format_exc()
            logging.info(formatted_traceback)

            failed_records.append(record)

    for record in failed_records:
        try:
            insert_cnt = vector_db.insert(record)
            if insert_cnt == 1:
                success_records[record['uuid']] = True
        except Exception as e:
            logging.info(f"Exception: {type(e).__name__} - {e}")

            formatted_traceback = traceback.format_exc()
            logging.info(formatted_traceback)

    logging.info(
        f'successfully insert {len(success_records)} records into vector db')
    inserted_chunks = [
        record['uuid'] for record in records
        if record['uuid'] in success_records
    ]

    # update document record
    document_record = {
        'name': os.path.basename(file_path),
        'chunks': '\x07'.join(inserted_chunks),
        'created_date': now_in_utc(),
    }
    insert_cnt = sql_db.insert_document(document_record)
    if insert_cnt < 1:
        logging.info(f'fail to insert document: {file_path}, retrying...')
        sql_db.insert_document(document_record)

    return inserted_chunks


def clean_up_document(file_path: str):
    """
    Clean up document records.
    
    Args:
    - file_path: path to the file.
    """
    vector_db = get_vector_db()
    sql_db = get_rational_db()

    file_name = os.path.basename(file_path)

    # get document record
    document_record = sql_db.get_document(file_name)
    if document_record is None or len(document_record):
        logging.info(f'document record {file_name} not found, ignore')
        return

    # delete document record
    sql_db.delete_document(name=file_name)

    # delete chunks
    uuids = []
    if 'chunks' in document_record and len(document_record['chunks']) > 0:
        uuids = document_record['chunks'].split('\x07')
    logging.info(f'total {len(uuids)} chunks')

    total_delete_cnt = 0
    for uuid in uuids:
        delete_cnt = vector_db.delete(key=uuid)
        total_delete_cnt += delete_cnt
    logging.info(f'delete {total_delete_cnt} chunks from vector db')
