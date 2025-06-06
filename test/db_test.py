import unittest
import os
import json
import numpy as np

from parse.parser import Chunk

import config


class TestMilvusDB(unittest.TestCase):

    def test_base(self):
        from typing import Dict, Any
        from scipy.sparse import csr_array

        from rag.db import get_vector_db
        from start_server import create_milvus_collection

        dense_embed_dim = 10
        collection_name = 'test_milvus_collection'

        # mock embed func
        from rag import nlp

        class MockEmbedingModel(nlp.EmbeddingModel):

            def __init__(self):
                print(f'call mock embedding model')
                pass

            def encode(self, texts: list[str]) -> Dict[str, Any]:
                dense_vector = np.random.uniform(low=0.0,
                                                 high=1.0,
                                                 size=dense_embed_dim),

                row = np.array([0, 1, 2, 0])
                col = np.array([0, 1, 1, 0])
                data = np.array([1, 2, 4, 8])
                sparse_vector = csr_array((data, (row, col)), shape=(3, 3))

                return {
                    'dense': dense_vector,
                    'sparse': sparse_vector,
                }

            def dense_embed_dim(self):
                return dense_embed_dim

        def mock_embed_model() -> nlp.EmbeddingModel:
            return MockEmbedingModel()

        nlp.get_embed_model = mock_embed_model
        embed_model = nlp.get_embed_model()

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

        # insert chunk1
        chunk1 = Chunk(
            content_type=config.ChunkType.TEXT,
            file_name='fake_file_name',
            content='chunk 1'.encode('utf-8'),
            extra_description=''.encode('utf-8'),
        )
        uuid1 = chunk1.uuid

        insert_cnt = db.insert(chunk1)
        self.assertEqual(insert_cnt, 1)

        # insert chunk2
        chunk2 = Chunk(
            content_type=config.ChunkType.TEXT,
            file_name='fake_file_name',
            content='chunk 2'.encode('utf-8'),
            extra_description=''.encode('utf-8'),
        )
        uuid2 = chunk2.uuid
        insert_cnt = db.insert(chunk2)
        self.assertEqual(insert_cnt, 1)

        ret = db.client.get(collection_name='test_milvus_collection',
                            ids=[uuid1])
        self.assertTrue(ret is not None)

        # test delete
        delete_cnt = db.delete(keys=[uuid1, uuid2], )
        self.assertEqual(delete_cnt, 2)
        ret = db.client.get(
            collection_name='test_milvus_collection',
            ids=[uuid1],
        )
        self.assertTrue(len(ret) == 0)


class TestSQLiteDB(unittest.TestCase):

    def test_base(self, ):
        import os

        from rag.db import get_rational_db
        from utils import now_in_utc
        from start_server import create_sqlite_table
        from utils import get_hash64

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
        self.assertTrue(ret is None)


if __name__ == '__main__':

    unittest.main()
