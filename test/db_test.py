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
        from rag.db import MilvusLiteDB

        db = MilvusLiteDB(conn_url='./test_milvus.db',
                          collection_name='test_milvus_collection')
        self.assertEqual(db.collection_name, 'test_milvus_collection')

        if db.client.has_collection('test_milvus_collection'):
            db.client.drop_collection('test_milvus_collection')
        db.create_table(table_name='test_milvus_collection', dense_embed_dim=10)
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
            'dense_vector':  np.random.uniform(low=0.0, high=1.0, size=10),
            'sparse_vector': ret[[0]],
        }

        db.insert(record)

        ret = db.client.get(collection_name='test_milvus_collection', ids=['123456'])
        self.assertTrue(ret is not None)

        # test delete
        _ = db.delete(key='123456')
        ret = db.client.get(collection_name='test_milvus_collection', ids=['123456'])
        self.assertTrue(len(ret) == 0)



if __name__ == '__main__':

    unittest.main()
