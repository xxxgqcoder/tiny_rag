import random
from abc import ABC, abstractmethod
from typing import Any

from ollama import Client as OllamaClient
from ollama import EmbedResponse
from pandas.tests.test_algos import test_infinity_against_nan
from sentence_transformers import SentenceTransformer

from common.cache import cache_it
from common.config import TinyRAGConfig
from common.utils import hash64, singleton, time_it


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: list[str], **kwargs) -> dict[str, list[float]]:
        """
        Encode text as vector. Some model is versatile and can return both dense and sparse vector.

        Args:
        - texts: List of texts to be encoded.

        Returns:
        - Encoded vector, represented by a dict. Key is the encoded vector type, i.e., `dense`, `sparse`. Value is the encoded vector value.
        """
        raise NotImplementedError("Not implemented")


@singleton
class Qwen3Embedding(EmbeddingModel):
    def __init__(self, model_dir: str, model_name: str):
        self.model_dir = model_dir
        self.model_name = model_name
        self.client = OllamaClient(
            host=TinyRAGConfig.ollama_host,
            timeout=5,
        )

    def key_generator(self, texts: list[str], **kwargs) -> str:
        prompt_name = kwargs.get("prompt_name", "")
        content = ",".join(texts) + self.model_name + prompt_name
        return "embedding::text_hash::" + hash64(content.encode("utf-8", errors="ignore"))

    @time_it(prefix="Qwen3 ebmedding")
    @cache_it(key_generator=key_generator)
    def encode(self, texts: list[str], **kwargs) -> dict[str, list[float]]:
        prompt_name = kwargs.get("prompt_name", None)
        if prompt_name:
            for i, text in enumerate(texts):
                texts[i] = f"{prompt_name} {text}"
        response: EmbedResponse = self.client.embed(model=self.model_name, input=texts)
        return {
            TinyRAGConfig.vector_db_config.embedding_column_name: response.embeddings,  # type: ignore
        }


def get_embedding_model() -> EmbeddingModel:
    return Qwen3Embedding(
        model_dir=TinyRAGConfig.embedding_config.embedding_model_dir,  # type: ignore
        model_name=TinyRAGConfig.embedding_config.embedding_model_name,  # type: ignore
    )


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
