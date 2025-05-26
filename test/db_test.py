import unittest
import os
import json
import numpy as np

from parse.parser import Parser, SupportedFileType, Chunk, ChunkType
from parse.pdf_parser import PDFParser

import config
from utils import get_project_base_directory


class TestMilvusDB(unittest.TestCase):

    def test_base(self):
        from scipy.sparse import csr_array
        from pymilvus import DataType
        from rag.db import MilvusLiteDB

        collection_name = 'test_milvus_collection'
        db = MilvusLiteDB(conn_url='./test_milvus.db',
                          collection_name=collection_name)
        self.assertEqual(db.collection_name, collection_name)

        if db.client.has_collection(collection_name):
            db.client.drop_collection(collection_name)

        # create collection
        # data schema
        schema = db.client.create_schema(enable_dynamic_field=True)
        dense_embed_dim = 10

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
        index_params = db.client.prepare_index_params()
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
        db.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True,
        )

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

        from rag.db import SQLiteDB
        from utils import now_in_utc

        db_name = 'test_sql_lite.db'
        document_table = 'document'

        db = SQLiteDB(conn_url=db_name, document_table=document_table)
        # create table
        cur = db.conn.cursor()
        sql_drop_table = """
        DROP TABLE IF EXISTS document
        """
        sql_create_table = """
        CREATE TABLE IF NOT EXISTS document (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            chunks TEXT NOT NULL,
            created_date TEXT NOT NULL
        )
        """
        cur.execute(sql_drop_table)
        cur.execute(sql_create_table)
        db.conn.commit()

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
        }

        insert_cnt = db.insert_document(data=data)
        self.assertEqual(insert_cnt, 1)

        # get
        ret = db.get_document(name=file_name)
        self.assertEqual(ret['chunks'], chunks)

        # delete
        delete_cnt = db.delete_document(name=file_name)
        self.assertEqual(delete_cnt, 1)
        ret = db.get_document(name=file_name)
        self.assertTrue(len(ret) == 0)

        pass


if __name__ == '__main__':

    unittest.main()
