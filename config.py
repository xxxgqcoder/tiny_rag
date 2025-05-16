# top level config

import pathlib
import logging
import os

from utils import get_project_base_directory, init_root_logger

# ============================================================================ #
# init project level resouces
init_root_logger("tiny_rag")


project_asset_folder = os.path.join(get_project_base_directory(), 'assets')
logging.info(f'project assets folder: {project_asset_folder}')

# root_storage_dir = os.path.join()

# ============================================================================ #
# parser config
## MinerU
magic_pdf_config_path = os.path.join(project_asset_folder, 'MinerU/magic-pdf.json')
logging.info(f'magic pdf config file path: {magic_pdf_config_path}')


# ============================================================================ #
# milvus config

