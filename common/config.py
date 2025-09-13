import os

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from common.utils import get_project_base_directory, init_root_logger


class RationalDBConfig(BaseSettings):
    db_name: str = Field("", description="Name of the SQL database.")


class VectorDBConfig(BaseSettings):
    db_name: str = Field("", description="Name of the vector database.")
    embedding_column_name: str = Field("embedding", description="Name of the embedding column.")


class EmbeddingConfig(BaseSettings):
    embedding_model_name: str = Field("", description="Name of the embedding model.")
    embedding_dim: int = Field(1024, description="Dimension of the embedding vectors.")
    embedding_support_sparse_vector: bool = Field(True, description="Whether to use sparse vectors for embeddings.")
    embedding_model_dir: str = Field("", description="Directory of the embedding model.")


class ParserConfig(BaseSettings):
    config_file_path: str = Field("", description="Path to the configuration file.")
    asset_save_dir: str = Field("parsed_assets", description="Directory for saving parsed assets.")


class ChunkingConfig(BaseSettings):
    consecutive_block_num: int = Field(8, description="Number of consecutive blocks to form a chunk.")
    block_overlap_num: int = Field(3, description="Number of overlapping blocks between consecutive chunks.")

    consecutive_byte_num: int = Field(1000, description="Number of consecutive bytes to form a chunk.")
    byte_overlap_num: int = Field(500, description="Number of overlapping bytes between consecutive chunks.")


class ObjectStoreConfig(BaseSettings):
    conn_url: str = Field("", description="Connection URL for the object store.")
    user: str = Field("", description="Username for the object store.")
    token: str = Field("", description="Access token for the object store.")


class CacheConfig(BaseSettings):
    conn_url: str = Field("", description="Connection URL for the object store.")
    token: str = Field("", description="Access token for the object store.")
    key_ttl_seconds: int = Field(12 * 60 * 60, description="Access token for the object store.")


class GenerationConf(BaseSettings):
    temperature: float = Field(0.7, description="Temperature for text generation.")
    top_p: float = Field(0.3, description=" Top-p (nucleus) sampling parameter.")
    repeat_penalty: float = Field(1.1, description=" Repetition penalty for text generation.")
    num_ctx: int = Field(64000, description=" Maximum context length for the model.")


class Config(BaseSettings):
    """Centralized configuration class for the entire Tiny RAG project."""

    ollama_host: str = Field("", description="Host for the Ollama server.")
    ollama_model: str = Field("", description="Model name for the Ollama server.")
    vision_model: str = Field("", description="Vision model name for the Ollama server.")
    search_service_url: str = Field("", description="URL for the search service.")
    search_service_port: int = Field(8080, description="Port for the search service.")
    root_data_dir: str = Field("data", description="Root directory for all data storage.")
    max_context_token_num: int = Field(64000, description="Maximum number of tokens in the context.")
    host_file_dir: str = Field("", description="Directory for host files.")
    ignore_path_pattern: list[str] = Field(..., description="List of path patterns to ignore.")

    parser_config: ParserConfig = Field(..., description="Parser configuration.")
    rational_db_config: RationalDBConfig = Field(..., description="Rational database configuration.")
    vector_db_config: VectorDBConfig =Field(..., description="Vector database configuration.")
    embedding_config: EmbeddingConfig = Field(..., description="Embedding configuration.")
    chunking_config: ChunkingConfig = Field(..., description="Embedding configuration.")
    object_store_config: ObjectStoreConfig = Field(..., description="Object store configuration.")
    cache_config: CacheConfig = Field(..., description="Cache configuration.")
    gen_conf: GenerationConf = Field(..., description="Generation configuration.")

    model_config = SettingsConfigDict(
        yaml_file=os.path.join(get_project_base_directory(), "config.yaml"),
        env_prefix="TINY_RAG@@",
        env_nested_delimiter="@@",
        nested_model_default_partial_update=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (env_settings, YamlConfigSettingsSource(settings_cls),)


TinyRAGConfig = Config()  # type: ignore

init_root_logger("tiny_rag")
