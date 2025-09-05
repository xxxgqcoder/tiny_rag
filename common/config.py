import json

from pydantic import BaseModel, Field

from common.utils import run_once


class RationalDBConfig(BaseModel):
    db_name: str = Field("", description="Name of the SQL database.")
    document_table_name: str = Field("", description="Name of the document table.")
    db_data_dir: str = Field("", description="Directory for SQL database data storage.")


class VectorDBConfig(BaseModel):
    db_name: str = Field("", description="Name of the vector database.")
    collection_name: str = Field("", description="Name of the collection in the vector database.")
    db_root_data_dir: str = Field("", description="Root directory for vector database data storage.")


class EmbedConfig(BaseModel):
    embed_model_name: str = Field("", description="Name of the embedding model.")
    embed_dim: int = Field(1024, description="Dimension of the embedding vectors.")
    embed_support_sparse_vector: bool = Field(True, description="Whether to use sparse vectors for embeddings.")


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


class Config(BaseModel):
    """Centralized configuration class for the entire Tiny RAG project."""

    parser_config: ParserConfig | None = Field(None, description="Parser configuration.")
    rational_db_config: RationalDBConfig | None = Field(None, description="Rational database configuration.")
    vector_db_config: VectorDBConfig | None = Field(None, description="Vector database configuration.")
    embed_config: EmbedConfig | None = Field(None, description="Embedding configuration.")
    chunking_config: ChunkingConfig | None = Field(None, description="Embedding configuration.")
    object_store_config: ObjectStoreConfig | None = Field(None, description="Object store configuration.")
    cache_config: CacheConfig | None = Field(None, description="Cache configuration.")


@run_once
def from_config_file(file_path: str) -> Config:
    with open(file_path, encoding="utf-8") as f:
        config_data = json.load(f)
        return Config.model_validate(config_data)
    return None


TinyRAGConfig: Config | None = from_config_file("config.json")
