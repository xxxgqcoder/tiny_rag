import logging
import sqlite3
import traceback

import config
from utils import run_once


@run_once
def create_milvus_collection(
        conn_url: str = config.MILVUS_DB_NAME,
        token: str = None,
        collection_name: str = config.MILVUS_COLLECTION_NAME,
        **kwargs):
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
    conn_url: str,
    token: str,
    table_name: str,
    **kwargs,
):
    sql_create_table = """
    CREATE TABLE IF NOT EXISTS document (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        chunks TEXT NOT NULL,
        created_date TEXT NOT NULL
    )
    """

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
