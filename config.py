# top level config

import pathlib
import logging
import os

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
# milvus config
