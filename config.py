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

ROOT_STORAGE_DIR = os.environ.get('TINY_RAG_DATA_DIR',
                                  '/var/share/tiny_rag_data')
logging.info(f'root storage directory: {ROOT_STORAGE_DIR}')

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
MILVUS_ROOT_DATA_DIR = os.path.join(ROOT_STORAGE_DIR, 'milvus_data')
logging.info(f"milvus root data directory: {MILVUS_ROOT_DATA_DIR}")

# ============================================================================ #
# parsed resouces
PARSED_ASSET_DATA_DIR = os.path.join(ROOT_STORAGE_DIR,
                                     'tiny_rag_parsed_assets')
logging.info(f"parsed asset data directory: {PARSED_ASSET_DATA_DIR}")


# ============================================================================ #
class ChunkType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TABLE = "table"
