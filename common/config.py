import logging
import os
import json
import re
from typing import Dict, Any, Optional

from common.utils import singleton
from pydantic import BaseModel, Field


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


@singleton
class TinyRAGConfig(BaseModel):
    """Centralized configuration class for the entire Tiny RAG project."""

    sql_db_config: SQLDBConfig = Field(None, description="SQL database configuration.")  # type: ignore
    vector_db_config: VectorDBConfig = Field(None, description="Vector database configuration.")  # type: ignore
    embed_config: EmbedConfig = Field(None, description="Embedding configuration.")  # type: ignore
