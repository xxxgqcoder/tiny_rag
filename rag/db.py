import logging
import traceback

from typing import Union, Dict, List, Any
from abc import ABC, abstractmethod
from strenum import StrEnum

import config
from utils import singleton


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
    def insert(self, data: Union[Dict, List[Dict]]) -> int:
        """
        Insert or update records.
        Returns:
        - An int of how many records are successfully insert.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete(self, key: str) -> int:
        """
        Delete record.

        Returns:
        - An int of how many records are successfully deleted.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search(self, query: Dict[str, Any], params: Dict[str, Any]) -> Any:
        raise NotImplementedError("Not implemented")


@singleton
class MilvusLiteDB(VectorDB):

    def __init__(self, conn_url: str, token: str = None, **kwargs):
        super().__init__(conn_url=conn_url, token=token, **kwargs)
        from pymilvus import MilvusClient
        self.client = MilvusClient(conn_url)

    def insert(self, data: Union[Dict, List[Dict]]) -> int:
        stats = self.client.upsert(self.collection_name, data)
        logging.info(f'insert stats: {stats}')
        return stats['upsert_count']

    def delete(self, keys: list[str]) -> Any:
        stats = self.client.delete(
            collection_name=self.collection_name,
            ids=keys,
        )
        logging.info(f'delete stats: {stats}')
        return len(stats)

    def search(self, query: Dict[str, Any], params: Dict[str, Any]) -> Any:
        """
        Run hybird search by default

        Args:
        - query: the query dict, should contain both dense and sparse embeddings
        - params: the query params.

        Returns:
        - Query result
        """
        from pymilvus import AnnSearchRequest, WeightedRanker

        output_fields = params.get('output_fields', ['content'])
        limit = params.get('limit', 10)
        sparse_weight = params.get('sparse_weight', 0.7)
        dense_weight = params.get('dense_weight', 1.0)

        query_dense_embedding = query['dense']
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest([query_dense_embedding],
                                     "dense_vector",
                                     dense_search_params,
                                     limit=limit)

        query_sparse_embedding = query['sparse']
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

        return res[0]


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
        self.conn = sqlite3.connect(conn_url)
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

        cur = self.conn
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


def get_rational_db():
    return SQLiteDB(
        conn_url=config.SQLITE_DB_NAME,
        document_table=config.SQLITE_DOCUMENT_TABLE_NAME,
    )
