import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print(sys.path[-1])


import numpy as np

from common.config import TinyRAGConfig, VectorDBConfig
from common.data import Chunk, ContentType, VectorDBRecord
from rag.db import create_vector_db_collection, get_vector_db


class TestMilvusDB(unittest.TestCase):
    def test_base(self):
        embedding_dim = 10
        collection_name = "test_milvus_collection"
        vector_db_name = "test/test_milvius.db"
        os.remove(vector_db_name)
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


# class TestSQLiteDB(unittest.TestCase):
#     def test_base(
#         self,
#     ):
#         from rag.db import get_rational_db
#         from start_server import create_sqlite_table
#         from utils import get_hash64, now_in_utc

#         # create table
#         db_name = "./test_sql_lite.db"
#         document_table = "document"
#         config.SQLITE_DB_NAME = db_name
#         config.SQLITE_DOCUMENT_TABLE_NAME = document_table
#         create_sqlite_table(
#             conn_url=config.SQLITE_DB_NAME,
#             table_name=config.SQLITE_DOCUMENT_TABLE_NAME,
#         )

#         db = get_rational_db()

#         # insert
#         file_path = "/var/share/tiny_rag_files/test_file.pdf"
#         file_name = os.path.basename(file_path)

#         chunks = """
#         4e03170d52fd201a
#         57e68f3d1e1ebcfb
#         """.strip().split()

#         data = {
#             "name": file_name,
#             "chunks": chunks,
#             "created_date": now_in_utc(),
#             "content_hash": get_hash64(b"test"),
#         }

#         insert_cnt = db.insert_document(data=data)
#         self.assertEqual(insert_cnt, 1)

#         # get
#         ret = db.get_document(name=file_name)
#         print(ret)
#         self.assertEqual(ret["chunks"], chunks)
#         self.assertEqual(ret["content_hash"], get_hash64(b"test"))

#         # delete
#         delete_cnt = db.delete_document(name=file_name)
#         self.assertEqual(delete_cnt, 1)
#         ret = db.get_document(name=file_name)
#         self.assertTrue(ret is None)


if __name__ == "__main__":
    unittest.main()
