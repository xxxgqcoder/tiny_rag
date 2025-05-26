import logging
import sqlite3
import traceback
import os
import time

from watchdog.observers import Observer

import config
from utils import run_once
from rag.document import FileHandler
from rag.document import get_job_executor, process_new_file


@run_once
def create_milvus_collection(
    conn_url: str = config.MILVUS_DB_NAME,
    token: str = None,
    collection_name: str = config.MILVUS_COLLECTION_NAME,
    **kwargs,
) -> None:
    """
    Create milvus collection.

    Args:
    - conn_url: the milvus connection url, or db_name if deployed as lite.
    - token: connection token if any.
    - collection_name: the collection name.
    - kwargs: should contain at least `dense_embed_dim` representing the embedding dim.
    """
    from pymilvus import MilvusClient
    from pymilvus import DataType

    logging.info(f"initialize milvus db: {conn_url}, token: {token}")

    # NOTE: assume local file path
    os.makedirs(os.path.dirname(conn_url), exist_ok=True)

    client = MilvusClient(conn_url)

    if client.has_collection(collection_name=collection_name):
        logging.info(
            f'collection {collection_name} found in {conn_url}, skip collection creation'
        )
        logging.info('existing collection schema')
        client.describe_collection(collection_name=collection_name)
        return

    # data schema
    dense_embed_dim = kwargs['dense_embed_dim']
    schema = client.create_schema(enable_dynamic_field=True)

    schema.add_field(
        field_name="uuid",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=128,
    )
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=10240,
    )
    schema.add_field(
        field_name="meta",
        datatype=DataType.JSON,
        nullable=True,
    )
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=dense_embed_dim,
    )
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    # index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    # create collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        enable_dynamic_field=True,
    )

    logging.info(f'milvus collection created: {collection_name}')
    logging.info('collection schema')
    client.describe_collection(collection_name=collection_name)

    client.close()


@run_once
def create_sqlite_table(
    conn_url: str = config.SQLITE_DB_NAME,
    token: str = None,
    table_name: str = config.SQLITE_DOCUMENT_TABLE_NAME,
    **kwargs,
) -> None:
    """
    Create SQLite table.

    Args:
    - conn_url: sqlite connection url. Currently only support local file path.
    - token: not used.
    - table_name: document table name.
    """

    sql_create_table = """
    CREATE TABLE IF NOT EXISTS document (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        chunks TEXT NOT NULL,
        created_date TEXT NOT NULL,
        content_hash TEXT NOT NULL
    )
    """
    # NOTE: assume local file path
    os.makedirs(os.path.dirname(conn_url), exist_ok=True)

    with sqlite3.connect(conn_url) as conn:
        cur = conn.cursor()
        try:
            ret = cur.execute("SELECT name FROM sqlite_master WHERE name = ?",
                              (table_name, ))
            res = ret.fetchall()
            if len(res) > 0:
                logging.info(
                    f'table {table_name} found in {conn_url}, skip table creation'
                )
                return
            cur.execute(sql_create_table)
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logging.info(f"Exception: {type(e).__name__} - {e}")
            formatted_traceback = traceback.format_exc()
            logging.info(formatted_traceback)

    logging.info(f'table created {table_name}')


@run_once
def initial_file_process(file_dir: str):
    """
    Submit initial file content check.
    """
    files = os.listdir(file_dir)
    file_paths = [os.path.join(file_dir, f) for f in files]
    job_executor = get_job_executor()
    for file_path in file_paths:
        job_executor.submit(process_new_file, file_path=file_path)


if __name__ == '__main__':
    # set up db
    create_milvus_collection(
        conn_url=config.MILVUS_DB_NAME,
        collection_name=config.MILVUS_COLLECTION_NAME,
        dense_embed_dim=config.BGE_DENSE_EMBED_DIM,
    )
    create_sqlite_table(
        conn_url=config.SQLITE_DB_NAME,
        table_name=config.SQLITE_DOCUMENT_TABLE_NAME,
    )

    # start file monitor
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, config.RAG_FILE_DIR, recursive=False)
    observer.start()

    # initial file direcory process
    initial_file_process(config.RAG_FILE_DIR)

    # event loop
    try:
        print('start file monitor')
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()
