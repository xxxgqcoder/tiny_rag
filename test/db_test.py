import os
import sys
import time
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print(sys.path[-1])


import numpy as np

from common.config import CacheConfig, ObjectStoreConfig, RationalDBConfig, TinyRAGConfig, VectorDBConfig
from common.data import Chunk, ContentType, VectorDBRecord
from rag.db import (
    RationalDBRecord,
    create_object_store_bucket,
    create_rational_db_table,
    create_vector_db_collection,
    get_cache_db,
    get_object_store,
    get_rational_db,
    get_vector_db,
)


class TestMilvusDB(unittest.TestCase):
    def test_base(self):
        embedding_dim = 10
        collection_name = "test_milvus_collection"
        vector_db_name = "test/test_milvius.db"
        try:
            os.remove(vector_db_name)
        except:
            pass
        embeding_vector = np.random.rand(embedding_dim).astype("float32").tolist()

        create_vector_db_collection(
            conn_url=vector_db_name,
            collection_name=collection_name,
            embedding_dim=embedding_dim,
        )

        TinyRAGConfig.vector_db_config = VectorDBConfig(  # type: ignore
            db_name=vector_db_name,
            db_root_data_dir="",
            collection_name=collection_name,
        )
        db = get_vector_db()
        self.assertEqual(db.collection_name, collection_name)

        # insert record
        record1 = VectorDBRecord(
            uuid="uuid1",
            file_name="fake_file_name",
            content_url="fake_content_url",
            embedding=embeding_vector,
            metadata={"key": "value"},
        )

        # insert
        insert_cnt = db.insert(record1)
        self.assertEqual(insert_cnt, 1)

        # get record by key
        ret = db.get(keys=[record1.uuid])
        self.assertEqual(ret[0].uuid, record1.uuid)

        # search
        ret = db.search(query={"embedding": embeding_vector}, params={})
        self.assertEqual(ret[0].uuid, record1.uuid)

        # test delete
        delete_cnt = db.delete(
            keys=[record1.uuid],
        )
        self.assertEqual(delete_cnt, 1)


class TestSQLiteDB(unittest.TestCase):
    def test_base(
        self,
    ):
        db_name = "test/test_sql_lite.db"
        document_table = "document"
        try:
            os.remove(db_name)
        except:
            pass

        TinyRAGConfig.rational_db_config = RationalDBConfig(  # type: ignore
            db_name=db_name,
            document_table_name=document_table,
        )
        db = get_rational_db()

        file_path = "/var/share/tiny_rag_files/test_file.pdf"
        file_name = os.path.basename(file_path)
        chunk_uuids = ["4e03170d52fd201a", "57e68f3d1e1ebcfb"]
        record = RationalDBRecord(
            file_name=file_name,
            chunk_uuids="\x07".join(chunk_uuids),
            created_date="1",
            content_hash="1",
        )
        # create table
        create_rational_db_table(conn_url=db_name, table_name=document_table)

        # insert
        insert_cnt = db.insert_document(record=record)
        self.assertEqual(insert_cnt, 1)

        # duplicate insert
        insert_cnt = db.insert_document(record=record)
        self.assertEqual(insert_cnt, 1)

        # get
        ret = db.get_document(file_name=file_name)
        self.assertEqual(ret.file_name, file_name)

        # get total
        ret = db.get_all_documents()
        self.assertEqual(len(ret), 1)

        # delete
        delete_cnt = db.delete_document(file_name=file_name)
        self.assertEqual(delete_cnt, 1)

        # get total
        ret = db.get_all_documents()
        self.assertEqual(len(ret), 0)


class TestMinio(unittest.TestCase):
    def test_base(self):
        conn_url = "localhost:9000"
        bucket_name = "test-bucket"
        user = "minioadmin"
        token = "minioadmin"

        TinyRAGConfig.object_store_config = ObjectStoreConfig(  # type: ignore
            conn_url=conn_url,
            user=user,
            token=token,
            bucket_name=bucket_name,
        )

        # create bucket
        create_object_store_bucket(user=user, conn_url=conn_url, token=token, bucket_name=bucket_name)

        # get object store
        store = get_object_store()
        self.assertIsNotNone(store)

        # put
        key = "test_key"
        data = b"test data"
        ret = store.put(key, data)
        self.assertEqual(ret, len(data))

        # get
        ret = store.get(key)
        self.assertEqual(ret, data)

        # delete
        _ = store.delete(key)
        ret = store.get(key)
        self.assertEqual(len(ret), 0)


class TestCache(unittest.TestCase):
    def test_base(self):
        conn_url = "redis://localhost:6379/0"
        token = ""
        key_ttl = 5

        key = "test_key"
        value = b"test_value"

        TinyRAGConfig.cache_config = CacheConfig(  # type: ignore
            conn_url=conn_url,
            token=token,
            key_ttl_seconds=key_ttl,
        )

        # get cache
        cache = get_cache_db()

        # put key
        ret = cache.put(key, value)
        self.assertEqual(ret, len(value))

        # get key
        ret = cache.get(key)
        self.assertEqual(ret, value)

        # wait for key expire
        time.sleep(key_ttl + 1)
        ret = cache.get(key)
        self.assertEqual(len(ret), 0)


if __name__ == "__main__":
    unittest.main()
