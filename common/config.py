import json
import os

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict
from pydantic_settings_yaml import YamlBaseSettings

from common.utils import get_project_base_directory, init_root_logger


class RationalDBConfig(BaseModel):
    db_name: str = Field("", description="Name of the SQL database.")
    document_table_name: str = Field("", description="Name of the document table.")
    db_data_dir: str = Field("", description="Directory for SQL database data storage.")


class VectorDBConfig(BaseModel):
    db_name: str = Field("", description="Name of the vector database.")
    collection_name: str = Field("", description="Name of the collection in the vector database.")


class EmbeddingConfig(BaseModel):
    embedding_model_name: str = Field("", description="Name of the embedding model.")
    embedding_dim: int = Field(1024, description="Dimension of the embedding vectors.")
    embedding_support_sparse_vector: bool = Field(True, description="Whether to use sparse vectors for embeddings.")
    embedding_model_dir: str = Field("", description="Directory of the embedding model.")


class ParserConfig(BaseModel):
    config_file_path: str = Field("", description="Path to the configuration file.")
    asset_save_dir: str = Field("parsed_assets", description="Directory for saving parsed assets.")


class ChunkingConfig(BaseModel):
    consecutive_block_num: int = Field(8, description="Number of consecutive blocks to form a chunk.")
    block_overlap_num: int = Field(3, description="Number of overlapping blocks between consecutive chunks.")


class ObjectStoreConfig(BaseModel):
    conn_url: str = Field("", description="Connection URL for the object store.")
    user: str = Field("", description="Username for the object store.")
    token: str = Field("", description="Access token for the object store.")
    bucket_name: str = Field("", description="Name of the bucket in the object store.")


class CacheConfig(BaseModel):
    conn_url: str = Field("", description="Connection URL for the object store.")
    token: str = Field("", description="Access token for the object store.")
    key_ttl_seconds: int = Field(12 * 60 * 60, description="Access token for the object store.")


class RankerConfig(BaseModel):
    ranker_mode_dir: str = Field("", description="Directory of the ranker model.")


class Config(YamlBaseSettings):
    """Centralized configuration class for the entire Tiny RAG project."""

    search_service_url: str = Field("", description="URL for the search service.")
    search_service_port: int = Field(8080, description="Port for the search service.")
    root_data_dir: str = Field("data", description="Root directory for all data storage.")

    parser_config: ParserConfig | None = Field(None, description="Parser configuration.")
    rational_db_config: RationalDBConfig | None = Field(None, description="Rational database configuration.")
    vector_db_config: VectorDBConfig | None = Field(None, description="Vector database configuration.")
    embedding_config: EmbeddingConfig | None = Field(None, description="Embedding configuration.")
    chunking_config: ChunkingConfig | None = Field(None, description="Embedding configuration.")
    object_store_config: ObjectStoreConfig | None = Field(None, description="Object store configuration.")
    cache_config: CacheConfig | None = Field(None, description="Cache configuration.")
    ranker_config: RankerConfig | None = Field(None, description="Ranker configuration.")

    model_config = SettingsConfigDict(yaml_file=(os.path.join(get_project_base_directory(), "config.yaml")))


TinyRAGConfig = Config()  # type: ignore

init_root_logger("tiny_rag")
