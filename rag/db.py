import json
import logging
import os
import sqlite3
import traceback
from abc import ABC, abstractmethod
from typing import Any

from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker

from common.config import TinyRAGConfig
from common.data import RationalDBRecord, VectorDBRecord
from common.utils import logging_exception, run_once, singleton, time_it


class VectorDB(ABC):
    """
    Abstract class for vector db.
    """

    def __init__(self, conn_url: str, token: str = "", collection_name: str = "", **kwargs):
        """
        Args:
        - conn_url: db connection url.
        - token: db connection token.
        """
        super().__init__()

        self.conn_url = conn_url
        self.token = token
        self.collection_name = collection_name
        for k, v in kwargs.items():
            setattr(self, k, v)

    # CRUD
    @abstractmethod
    def insert(self, data: VectorDBRecord) -> int:
        """
        Insert or update records.

        Args:
        - data: data to insert.

        Returns:
        - An int of how many records are successfully insert.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete(self, keys: list[str]) -> int:
        """
        Delete records.

        Args:
        - keys: list of record keys to delete.

        Returns:
        - An int of how many records are successfully deleted.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get(self, keys: list[str]) -> list[VectorDBRecord]:
        """
        Get records.

        Args:
        - keys: list of record keys to get.

        Returns:
        - A list of records
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search(self, query: dict[str, list[float]], params: dict[str, Any]) -> list[VectorDBRecord]:
        """
        Search records.

        Args:
        - query: query vector dict, key is the vector column, value the query vector.
        - params: the query params.

        Returns:
        - list of quert results.
        """
        raise NotImplementedError("Not implemented")


class RationalDB(ABC):
    """
    Abstract class of rational db.
    """

    @abstractmethod
    def insert_document(self, data: RationalDBRecord) -> int:
        """
        Insert a document.

        Args:
        - data: data to insert.

        Returns:
        - An int indicating how many records are inserted.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_document(self, name: str) -> RationalDBRecord:
        """
        Get a document.

        Args:
        - name: document name.

        Returns:
        - A document record.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete_document(self, name: str) -> int:
        """
        Delete a document.

        Args:
        - name: document name.

        Returns:
        - An int indicating how many records are deleted.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_all_documents(self) -> list[str]:
        """
        Get all documnets.

        Returns:
        - A list of document names.
        """
        raise NotImplementedError("Not implemented")


@singleton
class MilvusLiteDB(VectorDB):
    def __init__(self, conn_url: str, token: str = "", **kwargs):
        super().__init__(conn_url=conn_url, token=token, **kwargs)

        self.client = MilvusClient(conn_url)

    @time_it
    def insert(self, data: VectorDBRecord) -> int:
        record = data.model_dump()
        stats = self.client.upsert(self.collection_name, record)
        logging.info(f"Upsert stats: {stats}")
        return stats["upsert_count"]

    @time_it
    def delete(self, keys: list[str]) -> Any:
        stats = self.client.delete(
            collection_name=self.collection_name,
            ids=keys,
        )
        logging.info(f"Delete stats: {stats}")
        return len(stats)

    @time_it
    def search(
        self,
        query: dict[str, list[float]],
        params: dict[str, Any],
    ) -> list[VectorDBRecord]:
        output_fields = ["uuid", "file_name", "content_url", "metadata"]
        limit = params.get("limit", 100)
        ranker_weights = []

        search_reqs = []
        for vector_col, vector in query.items():
            query_dense_embedding = [vector]
            dense_search_params = {"metric_type": "IP", "params": {}}
            dense_req = AnnSearchRequest([query_dense_embedding], vector_col, dense_search_params, limit=limit)
            search_reqs.append(dense_req)
            ranker_weights.append(params.get(f"{vector_col}_weight", 1.0))

        rerank = WeightedRanker(*ranker_weights)
        res = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=search_reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
        )
        if len(res) == 0:
            return []

        ret = []
        for hit in res[0]:
            entity = hit["entity"]
            record = VectorDBRecord(
                uuid=entity.get("uuid", ""),
                file_name=entity.get("file_name", ""),
                content_url=entity.get("content_url", ""),
                metadata=entity.get(["metadata"], {}),
                embedding=[],  # do not return embedding
            )
            ret.append(record)

        return ret

    @time_it
    def get(self, keys: list[str]) -> list[Any]:
        res = self.client.get(
            collection_name=self.collection_name,
            ids=keys,
            output_fields=["uuid", "file_name", "content_url", "metadata"],
        )

        all_records = {}
        for ret in res:
            record = VectorDBRecord(
                uuid=ret.get("uuid", ""),
                file_name=ret.get("file_name", ""),
                content_url=ret.get("content_url", ""),
                metadata=ret.get("metadata", {}),
                embedding=[],  # do not return embedding
            )
            all_records[record.uuid] = record

        ret_records = [all_records[uuid] for uuid in keys if uuid in all_records]
        return ret_records


@run_once
def create_vector_db_collection(
    conn_url: str = "",
    token: str = "",
    collection_name: str = "",
    **kwargs,
) -> None:
    """
    Create milvus collection.

    Args:
    - conn_url: the milvus connection url, or db_name if deployed as lite.
    - token: connection token if any.
    - collection_name: the collection name.
    - kwargs: should contain `dense_embed_dim`.
    """

    logging.info(f"initialize milvus db: {conn_url}, token: {token}")

    # NOTE: assume local file path
    os.makedirs(os.path.dirname(conn_url), exist_ok=True)

    client = MilvusClient(conn_url)

    if client.has_collection(collection_name=collection_name):
        logging.info(f"collection {collection_name} found in {conn_url}, skip collection creation")
        return

    # data schema
    dense_embed_dim = kwargs["embedding_dim"]
    schema = client.create_schema(enable_dynamic_field=True)

    schema.add_field(
        field_name="uuid",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=128,
    )
    schema.add_field(
        field_name="file_name",
        datatype=DataType.VARCHAR,
        is_primary=False,
        auto_id=False,
        max_length=1024,
    )
    schema.add_field(
        field_name="content_url",
        datatype=DataType.VARCHAR,
        is_primary=False,
        auto_id=False,
        max_length=1024,
    )
    schema.add_field(
        field_name="metadata",
        datatype=DataType.JSON,
        nullable=True,
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=dense_embed_dim,
    )
    # index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="IP",
    )

    # create collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        enable_dynamic_field=True,
    )

    logging.info(f"milvus collection created: {collection_name}")
    client.close()


def get_vector_db() -> VectorDB:
    return MilvusLiteDB(
        conn_url=TinyRAGConfig.vector_db_config.db_name,  # type: ignore
        collection_name=TinyRAGConfig.vector_db_config.collection_name,  # type: ignore
    )


@singleton
class SQLiteDB(RationalDB):
    def __init__(self, conn_url: str, token: str = "", document_table_name: str = "", **kwargs):
        """
        SQLite DB.

        Args:
        - conn_url: connection url.
        - token:
        - document_table_name: document table name.
        - kwargs: should contain `document_table`.
        """
        super().__init__()

        self.conn = sqlite3.connect(conn_url, check_same_thread=False)
        self.token = token
        self.document_table_name = document_table_name
        for k, v in kwargs.items():
            setattr(self, k, v)

    @time_it
    def insert_document(self, data: RationalDBRecord) -> int:
        cur = self.conn.cursor()
        key_col = "name"
        data_dict: dict[str, Any] = data.model_dump()
        cur.execute(f"SELECT id FROM {self.document_table_name} WHERE name = ?", (data_dict[key_col],))
        record_exists = cur.fetchone() is not None

        try:
            if record_exists:
                # update
                update_query = f"UPDATE {self.document_table_name} SET "
                update_query_values = []
                for column, value in data_dict.items():
                    update_query += f"{column} = ?, "
                    update_query_values.append(value)
                update_query = update_query.rstrip(", ")
                update_query += f" WHERE {key_col} = ?"
                update_query_values.append(data_dict[key_col])
                cur.execute(update_query, update_query_values)
            else:
                # insert
                columns = []
                values = []
                for column, value in data_dict.items():
                    columns.append(column)
                    values.append(value)

                columns = ", ".join(columns)
                placeholders = ", ".join(["?"] * len(data_dict))
                insert_query = f"INSERT INTO {self.document_table_name} ({columns}) VALUES ({placeholders})"
                cur.execute(insert_query, tuple(values))
            self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()

            logging_exception(e)
            return 0
        finally:
            return 1

    @time_it
    def get_document(self, name: str) -> RationalDBRecord:
        cur = self.conn.cursor()
        query = f"SELECT * FROM {self.document_table_name} WHERE name = ?"

        ret = cur.execute(query, (name,))
        res = ret.fetchall()
        if len(res) < 1:
            return None  # type: ignore
        res = res[0]
        return RationalDBRecord.model_validate(
            {
                "name": res[1],
                "chunk_uuids": res[2].split("\x07"),
                "created_date": res[3],
                "content_hash": res[4],
            }
        )

    @time_it
    def delete_document(self, name: str) -> int:
        import sqlite3

        cur = self.conn.cursor()
        query = f"DELETE FROM {self.document_table_name} WHERE name = ?"
        logging.info(f"delete document: {name}")

        try:
            res = cur.execute(query, (name,))
            self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()
            logging_exception(e)
            return 0

        finally:
            return 1

    def get_all_documents(self) -> list[str]:
        query = f"SELECT name FROM {self.document_table_name}"
        cur = self.conn.cursor()

        ret = cur.execute(query, ())
        res = ret.fetchall()
        if len(res) < 1:
            return []

        names = [r[0] for r in res]
        return names


@run_once
def create_rational_db_table(
    conn_url: str = "",
    token: str = "",
    table_name: str = "",
    **kwargs,
) -> None:
    """
    Create rational db table.

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
            ret = cur.execute("SELECT name FROM sqlite_master WHERE name = ?", (table_name,))
            res = ret.fetchall()
            if len(res) > 0:
                logging.info(f"table {table_name} found in {conn_url}, skip table creation")
                return
            cur.execute(sql_create_table)
            cur.execute(sql_create_index)
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logging_exception(e)

    logging.info(f"table created {table_name}")


def get_rational_db():
    return SQLiteDB(
        conn_url=TinyRAGConfig.rational_db_config.db_name,  # type: ignore
        document_table=TinyRAGConfig.rational_db_config.document_table_name,  # type: ignore
    )
