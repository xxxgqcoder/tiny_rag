from abc import ABC, abstractmethod
from typing import Union, Dict, Any

from utils import singleton


class EmbeddingModel(ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name
        
        
    @abstractmethod
    def encode(self, text: str) -> Union[list[float], Dict[str, Any]]:
        """
        Encode text as vector. Some model is versatile and can return both dense and sparse vector.

        Args:
        - text: the text to encode

        Returns:
        - dense vector, represented by list of floats, or a dict containing both dense and sparse vector
        """
        raise NotImplementedError("Not implemented")

    def dense_embed_dim(self, ) -> int:
        raise NotImplementedError("Not implemented")


@singleton
class BGEM3EmbeddingModel(EmbeddingModel):
    def __init__(self, name='default'):
        super().__init__(name=name)

        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        self.ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        
    def encode(self, text: str) -> Union[list[float], Dict[str, Any]]:
        doc_embed = self.ef([text])
        return {
            'dense': doc_embed['dense'][0],
            'sparse': doc_embed['sparse'],
        }


    def dense_embed_dim(self):
        return self.ef.dim["dense"]
