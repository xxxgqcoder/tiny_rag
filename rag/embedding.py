from abc import ABC, abstractmethod

from ollama import Client as OllamaClient
from ollama import EmbedResponse

from common.cache import cache_it
from common.config import TinyRAGConfig
from common.utils import hash64, singleton, time_it


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        Encode texts as vector.

        Args:
        - texts: List of texts to be encoded.

        Returns:
        - Encoded vector.
        """
        raise NotImplementedError("Not implemented")


@singleton
class Qwen3Embedding(EmbeddingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OllamaClient(
            host=TinyRAGConfig.ollama_host,
            timeout=10 * 60,
        )

    def key_generator(self, texts: list[str], **kwargs) -> str:
        prompt_name = kwargs.get("prompt_name", "")
        content = ",".join(texts) + self.model_name + prompt_name
        return "embedding::text_hash::" + hash64(content.encode("utf-8", errors="ignore"))

    @time_it(prefix="Qwen3 ebmedding")
    @cache_it(key_generator=key_generator, key_ttl_seconds=60 * 60 * 24 * 30)
    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        prompt_name = kwargs.get("prompt_name")
        if prompt_name:
            for i, text in enumerate(texts):
                texts[i] = f"{prompt_name} {text}"
        response: EmbedResponse = self.client.embed(model=self.model_name, input=texts)
        return [list(embedding) for embedding in response.embeddings]


def get_embedding_model() -> EmbeddingModel:
    return Qwen3Embedding(
        model_name=TinyRAGConfig.embedding_config.embedding_model_name,
    )
