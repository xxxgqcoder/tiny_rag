from abc import ABC, abstractmethod
from typing import Any

from sentence_transformers import SentenceTransformer

from common.config import TinyRAGConfig
from common.utils import singleton, time_it


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: list[str], **kwargs) -> dict[str, Any]:
        """
        Encode text as vector. Some model is versatile and can return both dense and sparse vector.

        Args:
        - texts: the texts to encode.

        Returns:
        - Encoded vector, represented by a dict. Key is the encoded vector type, i.e., `dense`, `sparse`. Value is the encoded vector value.
        """
        raise NotImplementedError("Not implemented")


@singleton
class Qwen3Embedding(EmbeddingModel):
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = SentenceTransformer(model_dir)

    @time_it(prefix="Qwen3 ebmedding")
    def encode(self, texts: list[str], **kwargs) -> dict[str, Any]:
        prompt_name = kwargs.get("prompt_name", None)
        embeddings = self.model.encode(texts, prompt_name=prompt_name)
        return {"dense": embeddings}


def get_embedding_model() -> EmbeddingModel:
    return Qwen3Embedding(model_dir=TinyRAGConfig.embedding_config.embedding_model_dir)  # type: ignore


class RankerModel(ABC):
    @abstractmethod
    def rank(
        self,
    ):
        """
        Ranking function.
        """
        raise NotImplementedError("Not implemented")


class Qwen3Ranker(RankerModel):
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    @time_it(prefix="Qwen3 ranker")
    def rank(
        self,
    ):
        pass


def get_ranker() -> RankerModel:
    return Qwen3Ranker(model_dir=TinyRAGConfig.ranker_config.ranker_model_dir)  # type: ignore
