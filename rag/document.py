import json
import traceback
import logging
import os
from typing import Dict, Any

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
    DirCreatedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    DirDeletedEvent,
    FileClosedEvent,
    FileCreatedEvent,
    FileMovedEvent,
    FileModifiedEvent,
    FileClosedNoWriteEvent,
    FileDeletedEvent,
)
from watchdog.observers import Observer

import config
from .nlp import EmbeddingModel
from parse.parser import Chunk
from utils import now_in_utc, get_hash64, run_once
from .db import get_vector_db, get_rational_db
from .nlp import get_embed_model


def make_chunk_record(
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

    meta = {'file_path': os.path.basename(file_path)}
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
    Process new file, parse and save chunks into db.
    Steps:
    - check if file content is changed by content hash.
    - clean up previous document record if any once file content change detected.
    - run file content parse.
    - save chunks and document record

    Args:
    - file_path: path to the file.

    Returns:
    - A list containing all successfuly inserted chunks' uuid, the order is aligned
        with the chunks' original order in source file.
    """
    from parse.pdf_parser import PDFParser
    from config import PARSED_ASSET_DATA_DIR

    parser = PDFParser()

    vector_db = get_vector_db()
    sql_db = get_rational_db()
    embed_model = get_embed_model()

    # check if file content is changed
    file_name = os.path.basename(file_path)
    file_bytes = None
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
    except Exception as e:
        logging.info(
            f"""Exception when calculating contnet hash for file {file_path}:
                      {type(e).__name__} - {e}""")
        formatted_traceback = traceback.format_exc()
        logging.info(formatted_traceback)
        return
    file_content_hash = get_hash64(file_bytes)

    # get document record
    document_record = sql_db.get_document(file_name=file_name)
    stored_content_hash = None
    if document_record is not None:
        stored_content_hash = document_record['content_hash']
    if stored_content_hash == file_content_hash:
        logging.info(
            f'{file_path}: content hash ({file_content_hash}) unchanged, ignore'
        )
        return

    # delete document record if any
    process_delete_file(file_path=file_path)

    # parse file
    chunks = parser.parse(
        file_path=file_path,
        asset_save_dir=PARSED_ASSET_DATA_DIR,
    )

    # save parsed chunks into vector db
    records = [make_chunk_record(chunk, embed_model) for chunk in chunks]
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
    saved_chunks = [
        record['uuid'] for record in records
        if record['uuid'] in success_records
    ]

    # save document record
    document_record = {
        'name': os.path.basename(file_path),
        'chunks': '\x07'.join(saved_chunks),
        'created_date': now_in_utc(),
        'content_hash': get_hash64(file_bytes),
    }
    insert_cnt = sql_db.insert_document(document_record)
    if insert_cnt < 1:
        logging.info(f'fail to insert document: {file_path}, retrying...')
        sql_db.insert_document(document_record)

    return saved_chunks


def process_delete_file(file_path: str):
    """
    Delete document records.
    
    Args:
    - file_path: path to the file.
    """
    vector_db = get_vector_db()
    sql_db = get_rational_db()

    file_name = os.path.basename(file_path)

    # get document record
    document_record = sql_db.get_document(file_name)
    if document_record is None or len(document_record) == 0:
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


@run_once
def on_server_start_up():
    """
    Submit jobs for all files under root file directories.
    """

    pass


def ignore_file(file_path: str):
    """
    Rules on igore file.

    Args:
    - 
    """
    file_name = os.path.basename(file_path)
    # ignore hidden file
    if file_name.startswith('.'):
        return True

    # ignore non-supported file postfix
    postifx = file_name.split('.')[-1]
    if postifx not in ['pdf', 'docx', 'ppt']:
        return True

    return False


