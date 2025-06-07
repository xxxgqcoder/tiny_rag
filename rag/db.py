import logging
import traceback
import json
import sqlite3
import os

from typing import Union, Dict, List, Any
from abc import ABC, abstractmethod
from strenum import StrEnum

import config
from utils import singleton, run_once
from . import nlp
from parse.parser import Chunk


class VectorDB(ABC):
    """
    Abstract class for vector db.
    """

    def __init__(self, conn_url: str, token: str = None, **kwargs):
        """
        Args:
        - conn_url: db connection url.
        - token: db connection token.
        """
        super().__init__()

        self.conn_url = conn_url
        self.token = token
        for k, v in kwargs.items():
            setattr(self, k, v)

    # CRUD
    @abstractmethod
    def insert(self, data: Chunk) -> int:
        """
        Insert or update records.
        Returns:
        - An int of how many records are successfully insert.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete(self, keys: list[str]) -> int:
        """
        Delete records.

        Returns:
        - An int of how many records are successfully deleted.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get(self, keys: list[str]) -> list[Any]:
        """
        Get records.

        Returns:
        - A list of records
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search(self, query: str, params: Dict[str,
                                              Any]) -> list[Dict[str, Any]]:
        raise NotImplementedError("Not implemented")


@singleton
class MilvusLiteDB(VectorDB):

    def __init__(self, conn_url: str, token: str = None, **kwargs):
        super().__init__(conn_url=conn_url, token=token, **kwargs)
        from pymilvus import MilvusClient
        self.client = MilvusClient(conn_url)

    def insert(self, data: Chunk) -> int:
        # embed chunks
        embed_model = nlp.get_embed_model()
        content = data.content

        if data.content_type != config.ChunkType.TEXT:
            content = data.extra_description
        content = content.decode('utf-8')

        meta = {'file_name': data.file_name}
        if data.content_type == config.ChunkType.IMAGE:
            meta['content_url'] = data.content_url
        if data.content_type == config.ChunkType.TABLE:
            meta['table_content'] = data.content.decode('utf-8')

        embeddings = embed_model.encode([content])
        record = {
            'uuid': data.uuid,
            'content': content,
            'meta': json.dumps(meta, indent=4),
            'sparse_vector': embeddings['sparse'][[0]],
            'dense_vector': embeddings['dense'][0],
        }

        stats = self.client.upsert(self.collection_name, record)
        logging.info(f'insert stats: {stats}')
        return stats['upsert_count']

    def delete(self, keys: list[str]) -> Any:
        stats = self.client.delete(
            collection_name=self.collection_name,
            ids=keys,
        )
        logging.info(f'delete stats: {stats}')
        return len(stats)

    def search(self, query: str, params: Dict[str,
                                              Any]) -> list[Dict[str, Any]]:
        """
        Run hybird search by default

        Args:
        - query: user query, natural language.
        - params: the query params.

        Returns:
        - Query result
        """
        from pymilvus import AnnSearchRequest, WeightedRanker

        output_fields = ['content', 'meta', 'uuid']
        limit = params.get('limit', 10)
        sparse_weight = params.get('sparse_weight', 0.7)
        dense_weight = params.get('dense_weight', 1.0)

        # embed query
        embed_model = nlp.get_embed_model()
        embed = embed_model.encode([query])
        query_embed = {
            'sparse': embed['sparse'][[0]],
            'dense': embed['dense'][0]
        }

        query_dense_embedding = query_embed['dense']
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest([query_dense_embedding],
                                     "dense_vector",
                                     dense_search_params,
                                     limit=limit)

        query_sparse_embedding = query_embed['sparse']
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest([query_sparse_embedding],
                                      "sparse_vector",
                                      sparse_search_params,
                                      limit=limit)

        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[sparse_req, dense_req],
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
        )
        if len(res) == 0:
            return []

        ret = []
        for hit in res[0]:
            entity = hit['entity']
            meta = entity['meta']
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}

            file_name = meta.get('file_name', '')
            content = entity.get('content', '')
            uuid = entity.get('uuid', '')
            ret.append({
                'file_name': file_name,
                'content': content,
                'uuid': uuid,
            })

        return ret

    def get(self, keys: list[str]) -> list[Any]:
        res = self.client.get(
            collection_name=self.collection_name,
            ids=keys,
            output_fields=['uuid', 'content', 'meta'],
        )
        return res


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
        max_length=65535,
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


def get_vector_db():
    return MilvusLiteDB(
        conn_url=config.MILVUS_DB_NAME,
        collection_name=config.MILVUS_COLLECTION_NAME,
    )


class RationalDB(ABC):
    """
    Abstract class of rational db.
    """

    @abstractmethod
    def insert_document(self, data: Dict[str, Any]) -> int:
        """
        Insert document.

        Returns:
        - An int counting how many records are inserted.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_document(self, name: str):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete_document(self, name: str):
        """
        Delete document.

        Returns:
        - An int counting how many records are deleted.
        """
        raise NotImplementedError("Not implemented")


@singleton
class SQLiteDB(RationalDB):

    def __init__(self, conn_url: str, token: str = None, **kwargs):
        """
        SQLite DB:
        Args:
        - kwargs: should contain `document_table`.
        """
        super().__init__()
        import sqlite3
        self.conn = sqlite3.connect(conn_url, check_same_thread=False)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def insert_document(self, data: Dict[str, Any]) -> int:
        import sqlite3
        cur = self.conn.cursor()
        key_col = 'name'

        cur.execute(f"SELECT id FROM {self.document_table} WHERE name = ?",
                    (data[key_col], ))
        record_exists = cur.fetchone() is not None

        try:
            if record_exists:
                # update
                update_query = f"UPDATE {self.document_table} SET "
                update_query_values = []
                for column, value in data.items():
                    update_query += f"{column} = ?, "
                    update_query_values.append(value)
                update_query = update_query.rstrip(', ')
                update_query += f" WHERE {key_col} = ?"
                update_query_values.append(data[key_col])
                cur.execute(update_query, update_query_values)
            else:
                # insert
                columns = []
                values = []
                for column, value in data.items():
                    columns.append(column)
                    values.append(value)

                columns = ', '.join(columns)
                placeholders = ', '.join(['?'] * len(data))
                insert_query = f"INSERT INTO {self.document_table} ({columns}) VALUES ({placeholders})"
                cur.execute(insert_query, tuple(values))
            self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()

            logging.info(f"Exception: {type(e).__name__} - {e}")

            formatted_traceback = traceback.format_exc()
            logging.info(formatted_traceback)
            return 0
        finally:
            return 1

    def get_document(self, name: str):
        cur = self.conn.cursor()
        query = f"SELECT * FROM {self.document_table} WHERE name = ?"

        ret = cur.execute(query, (name, ))
        res = ret.fetchall()
        if len(res) < 1:
            return None
        res = res[0]
        return {
            'name': res[1],
            'chunks': res[2],
            'created_date': res[3],
            'content_hash': res[4],
        }

    def delete_document(self, name: str) -> int:
        import sqlite3

        cur = self.conn.cursor()
        query = f"DELETE FROM {self.document_table} WHERE name = ?"
        logging.info(f'delete document: {name}')

        try:
            res = cur.execute(query, (name, ))
            self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()

            logging.info(
                f"Initial delete fail, exception: {type(e).__name__} - {e}")

            formatted_traceback = traceback.format_exc()
            logging.info(formatted_traceback)

            return 0

        finally:
            return 1

    def get_all_documents(self, ) -> list[str]:
        query = f"SELECT name FROM {self.document_table}"
        cur = self.conn.cursor()

        ret = cur.execute(query, ())
        res = ret.fetchall()
        if len(res) < 1:
            return []

        names = [r[0] for r in res]
        return names


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
    sql_create_table = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        chunks TEXT NOT NULL,
        created_date TEXT NOT NULL,
        content_hash TEXT NOT NULL
    )
    """
    sql_create_index = f"CREATE INDEX idx_name ON {table_name} (name)"
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
            cur.execute(sql_create_index)
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logging.info(f"Exception: {type(e).__name__} - {e}")
            formatted_traceback = traceback.format_exc()
            logging.info(formatted_traceback)

    logging.info(f'table created {table_name}')


def get_rational_db():
    return SQLiteDB(
        conn_url=config.SQLITE_DB_NAME,
        document_table=config.SQLITE_DOCUMENT_TABLE_NAME,
    )
