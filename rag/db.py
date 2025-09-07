import io
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any

from minio import Minio
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
    def upsert(self, data: VectorDBRecord) -> int:
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
    def upsert_document(self, record: RationalDBRecord) -> int:
        """
        Insert a document.

        Args:
        - data: data to insert.

        Returns:
        - An int indicating how many records are inserted.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_document(self, file_name: str) -> RationalDBRecord:
        """
        Get a document.

        Args:
        - file_name: document file name.

        Returns:
        - A document record.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete_document(self, file_name: str) -> int:
        """
        Delete a document.

        Args:
        - file_name: document file name.

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


class ObjectStore(ABC):
    """
    Abstract class of object store.
    """

    @abstractmethod
    def get(self, key: str) -> bytes:
        """
        Get an object.

        Args:
        - key: object key.

        Returns:
        - The object bytes.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def put(self, key: str, obj: bytes) -> int:
        """
        Put an object.

        Args:
        - key: object key.
        - obj: object bytes.

        Returns:
        - An int indicating how many objects are put.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete(self, key: str) -> int:
        """
        Delete an object.

        Args:
        - key: object key.

        Returns:
        - An int indicating how many objects are deleted.
        """
        raise NotImplementedError("Not implemented")


# implementations


# vector db
@singleton
class MilvusLiteDB(VectorDB):
    def __init__(self, conn_url: str, token: str = "", collection_name: str = "", **kwargs):
        self.collection_name = collection_name
        self.client = MilvusClient(conn_url)

    @time_it(prefix="vector db")
    def upsert(self, data: VectorDBRecord) -> int:
        record = data.model_dump()
        stats = self.client.upsert(self.collection_name, record)
        logging.info(f"Upsert stats: {stats}")
        return stats["upsert_count"]

    @time_it(prefix="vector db")
    def delete(self, keys: list[str]) -> int:
        stats = self.client.delete(
            collection_name=self.collection_name,
            ids=keys,
        )
        logging.info(f"Delete stats: {stats}")
        return len(keys)

    @time_it(prefix="vector db")
    def search(
        self,
        query: dict[str, list[float]],
        params: dict[str, Any],
    ) -> list[VectorDBRecord]:
        output_fields = ["uuid", "file_name", "content_url", "metadata"]
        limit = params.get("limit", 100)

        if len(query) == 1:
            vector_col = list(query.keys())[0]
            vector = query[vector_col]
            query_dense_embedding = [vector]
            dense_search_params = {"metric_type": "IP"}
            res = self.client.search(
                collection_name=self.collection_name,
                data=query_dense_embedding,
                anns_field=vector_col,
                search_params=dense_search_params,
                limit=limit,
                output_fields=output_fields,
            )
        else:
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
                metadata=entity.get("metadata", {}),
                embedding=[],  # do not return embedding
            )
            ret.append(record)

        return ret

    @time_it(prefix="vector db")
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
def create_vector_db_collection() -> None:
    """
    Create milvus collection.

    Args:
    - conn_url: the milvus connection url, or db_name if deployed as lite.
    - token: connection token if any.
    - kwargs: should contain `dense_embed_dim`.
    """
    conn_url = TinyRAGConfig.vector_db_config.db_name  # type: ignore
    logging.info(f"initialize milvus db: {conn_url}")

    # NOTE: assume local file path
    os.makedirs(os.path.dirname(conn_url), exist_ok=True)
    client = MilvusClient(conn_url)
    collection_name = StorageManager.vector_db_collection_name()
    logging.info(f"milvus collection name: {collection_name}")
    if client.has_collection(collection_name=collection_name):
        logging.info(f"collection {collection_name} found in {conn_url}, skip collection creation")
        return

    # data schema
    embedding_dim = TinyRAGConfig.embedding_config.embedding_dim  # type: ignore
    logging.info(f"Embedding dim: {embedding_dim}")
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
        field_name=TinyRAGConfig.vector_db_config.embedding_column_name,  # type: ignore
        datatype=DataType.FLOAT_VECTOR,
        dim=embedding_dim,
    )
    # index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=TinyRAGConfig.vector_db_config.embedding_column_name,  # type: ignore
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
        collection_name=StorageManager.vector_db_collection_name(),  # type: ignore
    )


# rational db
@singleton
class SQLiteDB(RationalDB):
    def __init__(
        self,
        conn_url: str,
        token: str = "",
        document_table_name: str = "",
        **kwargs,
    ):
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

    @time_it(prefix="rational db")
    def upsert_document(self, record: RationalDBRecord) -> int:
        cur = self.conn.cursor()
        key_col = "file_name"
        data_dict: dict[str, Any] = record.model_dump()
        cur.execute(f"SELECT id FROM {self.document_table_name} WHERE file_name = ?", (data_dict[key_col],))
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
                ret = cur.execute(insert_query, tuple(values))

            self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()

            logging_exception(e)
            return 0

        return 1

    @time_it(prefix="rational db")
    def get_document(self, file_name: str) -> RationalDBRecord:
        cur = self.conn.cursor()
        query = f"SELECT * FROM {self.document_table_name} WHERE file_name = ?"

        ret = cur.execute(query, (file_name,))
        res = ret.fetchall()
        if len(res) < 1:
            return None  # type: ignore
        res = res[0]
        return RationalDBRecord.model_validate(
            {
                "file_name": res[1],
                "chunk_uuids": res[2],
                "created_date": res[3],
                "content_hash": res[4],
            }
        )

    @time_it(prefix="rational db")
    def delete_document(self, file_name: str) -> int:
        import sqlite3

        cur = self.conn.cursor()
        query = f"DELETE FROM {self.document_table_name} WHERE file_name = ?"
        logging.info(f"delete document: {file_name}")

        try:
            res = cur.execute(query, (file_name,))
            self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()
            logging_exception(e)
            return 0

        return 1

    @time_it(prefix="rational db")
    def get_all_documents(self) -> list[str]:
        query = f"SELECT file_name FROM {self.document_table_name}"
        cur = self.conn.cursor()

        ret = cur.execute(query, ())
        res = ret.fetchall()
        if len(res) < 1:
            return []

        names = [r[0] for r in res]
        return names


@run_once
def create_rational_db_table() -> None:
    """
    Create rational db table.

    Args:
    - conn_url: sqlite connection url. Currently only support local file path.
    - token: not used.
    - table_name: document table name.
    """
    conn_url = TinyRAGConfig.rational_db_config.db_name  # type: ignore
    document_table_name = StorageManager.rational_db_document_table_name()
    logging.info(f"Rational db: {conn_url}, document table name: {document_table_name}")

    sql_create_table = f"""
    CREATE TABLE IF NOT EXISTS {document_table_name} (
        id INTEGER PRIMARY KEY,
        file_name TEXT NOT NULL,
        chunk_uuids TEXT NOT NULL,
        created_date TEXT NOT NULL,
        content_hash TEXT NOT NULL
    )
    """
    sql_create_index = f"CREATE INDEX idx_name ON {document_table_name} (file_name)"
    # NOTE: assume local file path
    os.makedirs(os.path.dirname(conn_url), exist_ok=True)

    with sqlite3.connect(conn_url) as conn:
        cur = conn.cursor()
        try:
            ret = cur.execute("SELECT name FROM sqlite_master WHERE name = ?", (document_table_name,))
            res = ret.fetchall()
            if len(res) > 0:
                logging.info(f"Table {document_table_name} found in {conn_url}, skip table creation")
                return
            cur.execute(sql_create_table)
            cur.execute(sql_create_index)
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logging_exception(e)

    logging.info(f"table created {document_table_name}")


def get_rational_db() -> RationalDB:
    return SQLiteDB(
        conn_url=TinyRAGConfig.rational_db_config.db_name,  # type: ignore
        document_table_name=StorageManager.rational_db_document_table_name(),  # type: ignore
    )


# object storage
@singleton
class MinioStore(ObjectStore):
    def __init__(
        self,
        conn_url: str,
        user: str = "",
        token: str = "",
        bucket_name: str = "",
        **kwargs,
    ):
        """
        Minio store.

        Args:
        - conn_url: connection url.
        - user: connection user.
        - token: connection token.
        - bucket_name: bucket name.
        - kwargs: not used.
        """
        super().__init__()

        self.client = Minio(
            endpoint=conn_url,
            access_key=user,
            secret_key=token,
            secure=False,
        )

        self.bucket_name = bucket_name

    @time_it(prefix="object store")
    def get(self, key: str) -> bytes:
        try:
            response = self.client.get_object(self.bucket_name, key)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            logging_exception(e)
            return b""

        return b""

    @time_it(prefix="object store")
    def put(self, key: str, obj: bytes) -> int:
        binary_io = io.BytesIO(obj)
        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=binary_io,
                length=len(obj),
            )
        except Exception as e:
            logging_exception(e)
            return 0

        return len(obj)

    @time_it(prefix="object store")
    def delete(self, key: str) -> int:
        try:
            self.client.remove_object(self.bucket_name, key)
        except Exception as e:
            logging_exception(e)
            return 0

        return 1


@run_once
def create_object_store_bucket() -> None:
    """
    Create obejct storage bucket.
    """
    conn_url = TinyRAGConfig.object_store_config.conn_url  # type: ignore
    user = TinyRAGConfig.object_store_config.user  # type: ignore
    token = TinyRAGConfig.object_store_config.token  # type: ignore
    logging.info(f"initialize minio object store: {conn_url}, user: {user}")
    client = Minio(
        endpoint=conn_url,
        access_key=user,
        secret_key=token,
        secure=False,
    )
    bucket_name = StorageManager.object_store_bucket_name()
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        logging.info(f"Created bucket {bucket_name}")
    else:
        logging.info(f"Bucket {bucket_name} already exists")


def get_object_store() -> ObjectStore:
    return MinioStore(
        conn_url=TinyRAGConfig.object_store_config.conn_url,  # type: ignore
        user=TinyRAGConfig.object_store_config.user,  # type: ignore
        token=TinyRAGConfig.object_store_config.token,  # type: ignore
        bucket_name=StorageManager.object_store_bucket_name(),  # type: ignore
    )


@singleton
class _StorageManager:
    def document_content_key(self, content_hash: str) -> str:
        return f"document_content:{content_hash}"

    def chunk_content_key(self, uuid: str) -> str:
        return f"chunk_content:{uuid}"

    def vector_db_collection_name(self) -> str:
        return f"document_chunks_{TinyRAGConfig.embedding_config.embedding_model_name}_{TinyRAGConfig.embedding_config.embedding_dim}"  # type: ignore

    def rational_db_document_table_name(self) -> str:
        return "documents"

    def object_store_bucket_name(self) -> str:
        return "minio-tiny-rag"


StorageManager = _StorageManager()
