# top level config

import pathlib
import logging
import os

from utils import get_project_base_directory, init_root_logger

# initialize root logger
init_root_logger("tiny_rag")

asset_folder = os.path.join(get_project_base_directory(), 'assets')
logging.info(f'project assets folder: {asset_folder}')

# ============================================================================ #
# parser config
## MinerU
magic_pdf_config_path = os.path.join(asset_folder, 'MinerU/magic-pdf.json')
logging.info(f'magic pdf config file path: {magic_pdf_config_path}')
