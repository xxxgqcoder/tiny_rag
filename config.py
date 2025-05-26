# top level config
import logging
import os
from strenum import StrEnum

from utils import get_project_base_directory, init_root_logger

# ============================================================================ #
# init project level resouces
init_root_logger("tiny_rag")

PROJECT_ASSET_FOLDER = os.path.join(get_project_base_directory(), 'assets')
logging.info(f'project assets folder: {PROJECT_ASSET_FOLDER}')

# directory for saving vector db, parsed assets, etc.
RAG_DATA_DIR = os.environ.get('RAG_DATA_DIR', '/var/share/tiny_rag_data')
logging.info(f'root storage directory: {RAG_DATA_DIR}')

# directory for saving knowledge files, i.e., pdf papers.
RAG_FILE_DIR = os.environ.get('RAG_FILE_DIR', '/var/share/tiny_rag_files')
logging.info(f'files storage folder: {RAG_FILE_DIR}')

# parsed resouces
PARSED_ASSET_DATA_DIR = os.path.join(RAG_DATA_DIR, 'tiny_rag_parsed_assets')
logging.info(f"parsed asset data directory: {PARSED_ASSET_DATA_DIR}")

# ============================================================================ #
# parser config
## MinerU
MAGIC_PDF_CONFIG_PATH = os.path.join(PROJECT_ASSET_FOLDER,
                                     'MinerU/magic-pdf.json')
logging.info(f'magic pdf config file path: {MAGIC_PDF_CONFIG_PATH}')

# ============================================================================ #
# embedding model
BGE_MODEL_CONFIG_PATH = os.path.join(PROJECT_ASSET_FOLDER,
                                     'bge-m3/bge-m3.json')
logging.info(f'bge-m3 model config file path: {BGE_MODEL_CONFIG_PATH}')

# ============================================================================ #
# db config
## milvus config
MILVUS_ROOT_DATA_DIR = os.path.join(RAG_DATA_DIR, 'milvus_data')
MILVUS_DB_NAME = os.path.join(MILVUS_ROOT_DATA_DIR, 'tiny_rag.db')
MILVUS_COLLECTION_NAME = 'knowledge_collection'
logging.info(f"""milvus root data directory: {MILVUS_ROOT_DATA_DIR}
milvus db name: {MILVUS_DB_NAME}
milvus collection name: {MILVUS_COLLECTION_NAME}
""")

# SQLite config
SQLITE_ROOT_DATA_DIR = os.path.join(RAG_DATA_DIR, 'sqlite_data')
SQLITE_DB_NAME = os.path.join(SQLITE_ROOT_DATA_DIR, 'tiny_rag_documents.db')
SQLITE_DOCUMENT_TABLE_NAME = 'document'
logging.info(f"""sqlite root data directory: {SQLITE_ROOT_DATA_DIR}
sqlite db name: {SQLITE_DB_NAME}
sqlite table name: {SQLITE_DOCUMENT_TABLE_NAME}
""")


# ============================================================================ #
class ChunkType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TABLE = "table"
