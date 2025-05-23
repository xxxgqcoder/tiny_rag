import json
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Union, Dict, Any
from scipy.sparse import csr_array, vstack

import config
from utils import singleton


class EmbeddingModel(ABC):

    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        """
        Encode text as vector. Some model is versatile and can return both dense 
            and sparse vector.

        Args:
        - texts: the texts to encode.

        Returns:
        - Encoded vector, represented by a dict. Key is the encoded vector
            type, i.e., `dense`, `sparse`. Value is the encoded vector value.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def dense_embed_dim(self, ) -> int:
        raise NotImplementedError("Not implemented")


@singleton
class BGEM3EmbeddingModel(EmbeddingModel):

    def __init__(self, name='default'):
        super().__init__(name=name)

        from FlagEmbedding import BGEM3FlagModel

        with open(config.BGE_MODEL_CONFIG_PATH) as f:
            model_config = json.load(f)

        extra_model_config = {
            'devices': 'cpu',
            'normalize_embeddings': True,
            'use_fp16': False,
        }
        model_config.update(extra_model_config)
        self._model_config = model_config
        self.model = BGEM3FlagModel(**model_config)

        logging.info('BGE-M3 model config')
        logging.info(json.dumps(model_config, indent=4))

        encode_config = {
            "batch_size": 16,
            "return_dense": True,
            "return_sparse": True,
            "return_colbert_vecs": False,
        }
        self._encode_config = encode_config

    def encode(self, texts: list[str]) -> Dict[str, Any]:
        output = self.model.encode(sentences=texts, **self._encode_config)
        results = {}
        if self._encode_config["return_dense"]:
            results["dense"] = list(output["dense_vecs"])

        if self._encode_config["return_sparse"]:
            sparse_dim = self.dim["sparse"]
            results["sparse"] = []
            for sparse_vec in output["lexical_weights"]:
                indices = [int(k) for k in sparse_vec]
                values = np.array(list(sparse_vec.values()), dtype=np.float64)
                row_indices = [0] * len(indices)
                csr = csr_array((values, (row_indices, indices)),
                                shape=(1, sparse_dim))
                results["sparse"].append(csr)
            results["sparse"] = self.stack_sparse_embeddings(
                results["sparse"]).tocsr()

        if self._encode_config["return_colbert_vecs"]:
            results["colbert_vecs"] = output["colbert_vecs"]
        return results

    def stack_sparse_embeddings(self, sparse_embs: csr_array):
        """
        Vertical stack sparse vectors
        """
        return vstack(
            [sparse_emb.reshape((1, -1)) for sparse_emb in sparse_embs])

    def dense_embed_dim(self):
        return self.dim["dense"]

    @property
    def dim(self) -> Dict:
        return {
            "dense": self.model.model.model.config.hidden_size,
            "colbert_vecs": self.model.model.colbert_linear.out_features,
            "sparse": len(self.model.tokenizer),
        }


def get_embed_model():
    return BGEM3EmbeddingModel()
