import traceback
import logging
import os
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

import watchdog.events as events
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from utils import now_in_utc, get_hash64, logging_exception, run_once
from .db import get_vector_db, get_rational_db


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
    from parse.parser import get_parser
    from config import PARSED_ASSET_DATA_DIR

    if ignore_file(file_path):
        logging.info(f'{file_path}: ignore')
        return

    logging.info(f'{file_path}: process new file')

    parser = get_parser()

    vector_db = get_vector_db()
    sql_db = get_rational_db()

    logging.info(f'{file_path}: begin processing')

    # check if file content is changed
    file_name = os.path.basename(file_path)
    file_bytes = None
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
    except Exception as e:
        logging_exception(e)
        return

    if len(file_bytes) == 0:
        logging.info(f'{file_path}: empty content, skip')
        return

    file_content_hash = get_hash64(file_bytes)
    logging.info(
        f'{file_path}: total {len(file_bytes)} bytes loaded, content hash: {file_content_hash}'
    )

    # get document record
    document_record = sql_db.get_document(name=file_name)
    stored_content_hash = None
    if document_record is not None:
        stored_content_hash = document_record['content_hash']
    if stored_content_hash == file_content_hash:
        logging.info(
            f'{file_path}: content hash ({file_content_hash}) unchanged, ignore'
        )
        return
    logging.info(f'{file_path}: file content chnaged or new file')

    # delete document record if any
    process_delete_file(file_path=file_path)

    # parse file
    chunks = parser.parse(
        file_path=file_path,
        asset_save_dir=PARSED_ASSET_DATA_DIR,
    )
    logging.info(f'{file_path}: total {len(chunks)} chunks')
    if len(chunks) == 0:
        return

    # save parsed chunks into vector db
    failed_chunks = []
    success_chunks = {}
    for chunk in chunks:
        try:
            insert_cnt = vector_db.insert(chunk)
            if insert_cnt == 1:
                success_chunks[chunk.uuid] = True
            else:
                failed_chunks.append(chunk)

        except Exception as e:
            logging_exception(e)
            failed_chunks.append(chunk)

    for chunk in failed_chunks:
        try:
            insert_cnt = vector_db.insert(chunk)
            if insert_cnt == 1:
                success_chunks[chunk.uuid] = True
        except Exception as e:
            logging_exception(e)

    logging.info(
        f'successfully insert {len(success_chunks)} records into vector db')
    saved_chunks = [
        chunk.uuid for chunk in chunks if chunk.uuid in success_chunks
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
        logging.info(f'{file_path}: fail to insert document, retrying...')
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
    logging.info(f'{file_path}: process delete file')

    vector_db = get_vector_db()
    sql_db = get_rational_db()

    file_name = os.path.basename(file_path)

    # get document record
    document_record = sql_db.get_document(name=file_name)
    if document_record is None:
        logging.info(f'{file_path}: document record not found, ignore')
        return
    logging.info(f'{file_path}: document record: {document_record}')

    # delete document record
    delete_cnt = sql_db.delete_document(name=file_name)
    logging.info(f'delete document record from db, delete cnt: {delete_cnt}')

    # delete chunks
    uuids = []
    if 'chunks' in document_record and len(document_record['chunks']) > 0:
        uuids = document_record['chunks'].split('\x07')
    logging.info(f'{file_path}: total {len(uuids)} chunks')

    delete_cnt = vector_db.delete(keys=uuids)
    logging.info(f'delete {delete_cnt} chunks from vector db')


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
    if postifx not in ['pdf', 'docx', 'ppt', 'md', 'txt']:
        return True

    return False


_job_executor = None


def get_job_executor():
    global _job_executor
    if _job_executor is None:
        # NOTE: set only 1 thread to force sequencial job schedule.
        _job_executor = ThreadPoolExecutor(max_workers=1)

    return _job_executor


def on_process_new_file(file_path: str):
    try:
        process_new_file(file_path=file_path)
    except Exception as e:
        logging_exception(e)


def on_process_delete_file(file_path: str):
    try:
        process_delete_file(file_path=file_path)
    except Exception as e:
        logging_exception(e)


class FileHandler(FileSystemEventHandler):

    def on_any_event(self, event: FileSystemEvent) -> None:

        job_executor = get_job_executor()
        src_path = event.src_path
        dest_path = event.dest_path

        if event.event_type == events.EVENT_TYPE_MOVED:
            if not os.path.isdir(src_path):
                job_executor.submit(on_process_delete_file, file_path=src_path)

            if not os.path.isdir(dest_path):
                job_executor.submit(on_process_new_file, file_path=src_path)

        elif event.event_type == events.EVENT_TYPE_DELETED:
            if not os.path.isdir(src_path):
                job_executor.submit(on_process_delete_file, file_path=src_path)

        elif event.event_type == events.EVENT_TYPE_CREATED:
            if not os.path.isdir(src_path):
                job_executor.submit(on_process_new_file, file_path=src_path)

        elif event.event_type == events.EVENT_TYPE_MODIFIED:
            if not os.path.isdir(src_path):
                job_executor.submit(on_process_new_file, file_path=src_path)

        else:
            pass


@run_once
def initial_file_process(file_dir: str):
    """
    Submit initial file content check.
    """
    job_executor = get_job_executor()
    sql_db = get_rational_db()

    # get all documents
    all_documents = sql_db.get_all_documents()
    file_names = os.listdir(file_dir)

    # delete documents that are not found in file_dir
    to_delete = list(set(all_documents) - set(file_names))
    logging.info(
        f"Below files are founded in db but not in file folder, delete: {to_delete}"
    )
    for file_name in to_delete:
        job_executor.submit(on_process_delete_file,
                            file_path=os.path.join(file_dir, file_name))

    for file_name in file_names:
        job_executor.submit(on_process_new_file,
                            file_path=os.path.join(file_dir, file_name))
