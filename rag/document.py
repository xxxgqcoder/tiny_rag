import json
import traceback
import logging
import os
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
    DirCreatedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    DirDeletedEvent,
    FileCreatedEvent,
    FileMovedEvent,
    FileModifiedEvent,
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

    if ignore_file(file_path):
        logging.info(f'{file_path}: ignore')
        return

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
        logging.info(f"""{file_path} Exception: {type(e).__name__} - {e}""")
        formatted_traceback = traceback.format_exc()
        logging.info(formatted_traceback)
        return

    if len(file_bytes) == 0:
        logging.info(f'{file_path}: empty content, skip')
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
    if ignore_file(file_path):
        logging.info(f'{file_path}: ignore')
        return

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


def ignore_file(file_path: str):
    """
    Rules on igore file.

    Returns:
    - bool, true if file_path should be ignored.
    """
    file_name = os.path.basename(file_path)
    # ignore hidden file
    if file_name.startswith('.'):
        return True

    # ignore non-supported file postfix
    postifx = file_name.split('.')[-1]
    if postifx not in ['pdf', 'docx', 'ppt', 'md']:
        return True

    return False


_job_executor = None


def get_job_executor():
    global _job_executor
    if _job_executor is None:
        # NOTE: set only 1 thread to force sequencial job schedule.
        _job_executor = ThreadPoolExecutor(max_workers=1)

    return _job_executor


def test_process_new_file(file_path: str):
    if ignore_file(file_path):
        logging.info(f'{file_path}: ignore')
        return
    logging.info(f'{file_path}: on process new file')


def test_process_delete_file(file_path: str):
    if ignore_file(file_path):
        logging.info(f'{file_path}: ignore')
        return
    logging.info(f'{file_path}: on process delete file')


class FileHandler(FileSystemEventHandler):

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:

        job_executor = get_job_executor()
        src_path = event.src_path
        dest_path = event.dest_path

        if not os.path.isdir(src_path):
            job_executor.submit(process_delete_file, file_path=src_path)

        if not os.path.isdir(dest_path):
            job_executor.submit(process_new_file, file_path=dest_path)

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:

        job_executor = get_job_executor()
        src_path = event.src_path
        if os.path.isdir(src_path):
            return
        job_executor.submit(process_new_file, file_path=src_path)

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:

        job_executor = get_job_executor()
        src_path = event.src_path
        if os.path.isdir(src_path):
            return
        job_executor.submit(process_new_file, file_path=src_path)

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:

        job_executor = get_job_executor()
        src_path = event.src_path
        if os.path.isdir(src_path):
            return
        job_executor.submit(process_new_file, file_path=src_path)
