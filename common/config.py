import json

from pydantic import BaseModel, Field

from common.utils import run_once


class SQLDBConfig(BaseModel):
    sql_db_name: str = Field("", description="Name of the SQL database.")
    sql_db_data_dir: str = Field("", description="Directory for SQL database data storage.")


class VectorDBConfig(BaseModel):
    vector_db_name: str = Field("", description="Name of the vector database.")
    vector_db_root_data_dir: str = Field("", description="Root directory for vector database data storage.")


class EmbedConfig(BaseModel):
    embed_model_name: str = Field("", description="Name of the embedding model.")
    embed_dim: int = Field(1024, description="Dimension of the embedding vectors.")
    embed_support_sparse_vector: bool = Field(True, description="Whether to use sparse vectors for embeddings.")


class ParserConfig(BaseModel):
    config_file_path: str = Field("", description="Path to the configuration file.")


class Config(BaseModel):
    """Centralized configuration class for the entire Tiny RAG project."""

    parser_config: ParserConfig = Field(None, description="Parser configuration.")  # type: ignore
    sql_db_config: SQLDBConfig = Field(None, description="SQL database configuration.")  # type: ignore
    vector_db_config: VectorDBConfig = Field(None, description="Vector database configuration.")  # type: ignore
    embed_config: EmbedConfig = Field(None, description="Embedding configuration.")  # type: ignore


@run_once
def from_config_file(file_path: str) -> Config | None:
    with open(file_path, encoding="utf-8") as f:
        config_data = json.load(f)
        return Config.model_validate(config_data)
    return None


TinyRAGConfig: Config | None = from_config_file("config.json")
