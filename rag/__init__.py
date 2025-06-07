import numpy as np
from typing import Dict, Any

from .nlp import BGEM3EmbeddingModel, EmbeddingModel


class MockEmbedingModel(EmbeddingModel):

    def __init__(self, dense_embed_dim=10):
        print(f'call mock embedding model')
        self.dense_embed_dim = dense_embed_dim

    def encode(self, texts: list[str]) -> Dict[str, Any]:
        import numpy as np
        from scipy.sparse import csr_array

        dense_vector = np.random.uniform(low=0.0,
                                         high=1.0,
                                         size=self.dense_embed_dim),

        row = np.array([0, 1, 2, 0])
        col = np.array([0, 1, 1, 0])
        data = np.array([1, 2, 4, 8])
        sparse_vector = csr_array((data, (row, col)), shape=(3, 3))

        return {
            'dense': dense_vector,
            'sparse': sparse_vector,
        }

    def dense_embed_dim(self):
        return self.dense_embed_dim


Embeddings = {
    'bge-m3': BGEM3EmbeddingModel,
    'mock_for_test': MockEmbedingModel,
}


def get_embed_model(name: str = "bge-m3") -> EmbeddingModel:
    if name not in Embeddings:
        msg = f"unknown parser: {name}" + "\n" \
            f"supported parsers are {[k for k in Embeddings]}"
        raise Exception(msg)
    p = Embeddings[name]
    return p()
