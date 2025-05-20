import logging

from typing import Union, Dict, List, Any
from abc import ABC, abstractmethod
from strenum import StrEnum

from utils import singleton


class VectorDB(ABC):
    """
    Abstract class for vector db.
    """

    def __init__(self, conn_url: str, token: str = None):
        """
        Args:
        - conn_url: db connection url.
        - token: db connection token.
        """
        super().__init__()

        self.conn_url = conn_url
        self.token = token

    # CRUD
    @abstractmethod
    def insert(self, data: Union[Dict, List[Dict]]):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete(self, ):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search(self, query: Dict[str, Any], params: Dict[str, Any]):
        raise NotImplementedError("Not implemented")


@singleton
class MilvusLiteDB(VectorDB):

    def __init__(self, conn_url: str, token: str = None):
        logging.info(f"initialize milvus db: {conn_url}, token: {token}")
        super().__init__(conn_url=conn_url, token=token)
        from pymilvus import MilvusClient
        self.client = MilvusClient(conn_url)

    def create_collection(self, collection_name: str, dense_embed_dim) -> None:
        from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

        if self.client.has_collection(collection_name=collection_name):
            logging.info('collection found in db, skip creation')
            return

        # data schema
        schema = self.client.create_schema(enable_dynamic_field=True)

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
        index_params = self.client.prepare_index_params()
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
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True,
        )

        self.collection_name = collection_name
        logging.info(f'milvus collection created: {collection_name}')

    def use_collection(self, collection_name: str):
        if not self.client.has_collection(collection_name=collection_name):
            raise Exception(
                f'collection: {collection_name} not found in current db, please check'
            )
        self.collection_name = collection_name

    def insert(self, data: Union[Dict, List[Dict]]):
        stats = self.client.upsert(self.collection_name, data)
        logging.info(f'insert stats: {stats}')

    def delete(self, ):
        pass

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
