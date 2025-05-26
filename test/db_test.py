import unittest
import os
import json
import numpy as np

from parse.parser import Parser, SupportedFileType, Chunk, ChunkType
from parse.pdf_parser import PDFParser

import config
from utils import get_project_base_directory, get_hash64
from start_server import create_milvus_collection, create_sqlite_table


class TestMilvusDB(unittest.TestCase):

    def test_base(self):
        from scipy.sparse import csr_array
        from pymilvus import DataType
        from rag.db import MilvusLiteDB, get_vector_db

        dense_embed_dim = 10
        collection_name = 'test_milvus_collection'

        # create collection
        config.MILVUS_DB_NAME = './test_milvus.db'
        config.MILVUS_COLLECTION_NAME = collection_name
        create_milvus_collection(
            conn_url=config.MILVUS_DB_NAME,
            collection_name=config.MILVUS_COLLECTION_NAME,
            dense_embed_dim=dense_embed_dim,
        )

        db = get_vector_db()
        self.assertEqual(db.collection_name, collection_name)

        self.assertEqual(db.collection_name, 'test_milvus_collection')
        self.assertTrue(db.client.has_collection('test_milvus_collection'))

        # test insert
        row = np.array([0, 1, 2, 0])
        col = np.array([0, 1, 1, 0])
        data = np.array([1, 2, 4, 8])
        ret = csr_array((data, (row, col)), shape=(3, 3))

        record = {
            'uuid': '123456',
            'content': 'test content',
            'meta': json.dumps({'key': 'value'}),
            'dense_vector': np.random.uniform(low=0.0, high=1.0, size=10),
            'sparse_vector': ret[[0]],
        }

        insert_cnt = db.insert(record)
        self.assertEqual(insert_cnt, 1)

        record['uuid'] = '654321'
        insert_cnt = db.insert(record)
        self.assertEqual(insert_cnt, 1)

        ret = db.client.get(collection_name='test_milvus_collection',
                            ids=['123456'])
        self.assertTrue(ret is not None)

        # test delete
        delete_cnt = db.delete(keys=['123456', '654321'], )
        self.assertEqual(delete_cnt, 2)
        ret = db.client.get(
            collection_name='test_milvus_collection',
            ids=['123456'],
        )
        self.assertTrue(len(ret) == 0)


class TestSQLiteDB(unittest.TestCase):

    def test_base(self, ):
        import os

        from rag.db import get_rational_db
        from utils import now_in_utc

        # create table
        db_name = './test_sql_lite.db'
        document_table = 'document'
        config.SQLITE_DB_NAME = db_name
        config.SQLITE_DOCUMENT_TABLE_NAME = document_table
        create_sqlite_table(
            conn_url=config.SQLITE_DB_NAME,
            table_name=config.SQLITE_DOCUMENT_TABLE_NAME,
        )

        db = get_rational_db()

        # insert
        file_path = '/var/share/tiny_rag_files/test_file.pdf'
        file_name = os.path.basename(file_path)

        chunks = """
        4e03170d52fd201a
        57e68f3d1e1ebcfb
        """.strip().split()
        chunks = "\x07".join(chunks)

        data = {
            'name': file_name,
            'chunks': chunks,
            'created_date': now_in_utc(),
            'content_hash': get_hash64('test'.encode('utf-8')),
        }

        insert_cnt = db.insert_document(data=data)
        self.assertEqual(insert_cnt, 1)

        # get
        ret = db.get_document(name=file_name)
        print(ret)
        self.assertEqual(ret['chunks'], chunks)
        self.assertEqual(ret['content_hash'],
                         get_hash64('test'.encode('utf-8')))

        # delete
        delete_cnt = db.delete_document(name=file_name)
        self.assertEqual(delete_cnt, 1)
        ret = db.get_document(name=file_name)
        self.assertTrue(len(ret) == 0)


if __name__ == '__main__':

    unittest.main()
