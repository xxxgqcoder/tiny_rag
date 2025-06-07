# top level config
import logging
import os
import json
import re
from strenum import StrEnum

from utils import get_project_base_directory, init_root_logger, run_once


# ============================================================================ #
class ChunkType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TABLE = "table"


@run_once
def init_root_config():
    # logger
    init_root_logger("tiny_rag")

    # parse environment file
    global HOST_RAG_FILE_DIR
    HOST_RAG_FILE_DIR = os.environ.get('HOST_RAG_FILE_DIR', '')
    logging.info(f"host rag file dir: {HOST_RAG_FILE_DIR}")

    global PROJECT_ASSET_DIR
    PROJECT_ASSET_DIR = os.path.join(get_project_base_directory(), 'assets')
    logging.info(f'project assets dir: {PROJECT_ASSET_DIR}')

    # directory for saving vector db, parsed assets, etc.
    global RAG_DATA_DIR
    RAG_DATA_DIR = os.environ.get('RAG_DATA_DIR', '/var/share/tiny_rag_data')
    logging.info(f'root storage dir: {RAG_DATA_DIR}')

    # directory for saving knowledge files, i.e., pdf papers.
    global RAG_FILE_DIR
    RAG_FILE_DIR = os.environ.get('RAG_FILE_DIR', '/var/share/tiny_rag_files')
    logging.info(f'files storage dir: {RAG_FILE_DIR}')

    # parsed resouces
    global PARSED_ASSET_DATA_DIR
    PARSED_ASSET_DATA_DIR = os.path.join(RAG_DATA_DIR,
                                         'tiny_rag_parsed_assets')
    logging.info(f"parsed asset data dir: {PARSED_ASSET_DATA_DIR}")

    # ======================================================================== #
    # PDF parser
    global PDF_PARSER_NAME, PDF_PARSER_CONFIG_PATH
    PDF_PARSER_NAME = os.environ.get('PDF_PARSER_NAME', 'MinerU')
    PDF_PARSER_CONFIG_PATH = os.path.join(PROJECT_ASSET_DIR,
                                          'MinerU/magic-pdf.json')
    logging.info(f'pdf parser name: {PDF_PARSER_NAME}')
    logging.info(f'pdf parser config path: {PDF_PARSER_CONFIG_PATH}')

    # ======================================================================== #
    # embedding model
    global EMBED_MODEL_CONFIG_PATH, EMBED_DENSE_DIM, EMBED_MODEL_NAME
    EMBED_MODEL_NAME = 'bge-m3'
    EMBED_MODEL_CONFIG_PATH = os.path.join(PROJECT_ASSET_DIR,
                                           'bge-m3/bge-m3.json')
    EMBED_DENSE_DIM = 1024
    logging.info(f'embed model name: {EMBED_MODEL_NAME}')
    logging.info(f'embed model config file path: {EMBED_MODEL_CONFIG_PATH}')
    logging.info(f'embed model dense embed dim: {EMBED_DENSE_DIM}')

    # ============================================================================ #
    # vector db config
    global MILVUS_ROOT_DATA_DIR, MILVUS_DB_NAME, MILVUS_COLLECTION_NAME
    MILVUS_ROOT_DATA_DIR = os.path.join(RAG_DATA_DIR, 'milvus_data')
    MILVUS_DB_NAME = os.path.join(MILVUS_ROOT_DATA_DIR, 'tiny_rag.db')
    # NOTE: collection name subject to embedding model name and dense dimension,
    # Thus changing embedding model may cause tables re-creation and re-parse
    # existing pdf files.
    MILVUS_COLLECTION_NAME = f'knowledge_collection_{EMBED_MODEL_NAME}_{EMBED_DENSE_DIM}'
    MILVUS_COLLECTION_NAME = re.sub(r"[^a-zA-Z0-9_]", "_",
                                    MILVUS_COLLECTION_NAME)

    logging.info(f"milvus root data directory: {MILVUS_ROOT_DATA_DIR}")
    logging.info(f"milvus db name: {MILVUS_DB_NAME}")
    logging.info(f"milvus collection name: {MILVUS_COLLECTION_NAME}")

    # ============================================================================ #
    # sqlite db config
    global SQLITE_ROOT_DATA_DIR, SQLITE_DB_NAME, SQLITE_DOCUMENT_TABLE_NAME
    SQLITE_ROOT_DATA_DIR = os.path.join(RAG_DATA_DIR, 'sqlite_data')
    SQLITE_DB_NAME = os.path.join(SQLITE_ROOT_DATA_DIR,
                                  'tiny_rag_documents.db')
    # NOTE: document table subject to embedding model and dense dimension, thus
    # changing embedding model may cause table re-creation and re-parse existing
    # pdf files.
    SQLITE_DOCUMENT_TABLE_NAME = f'document_{EMBED_MODEL_NAME}_{EMBED_DENSE_DIM}'
    SQLITE_DOCUMENT_TABLE_NAME = re.sub(r"[^a-zA-Z0-9_]", "_",
                                        SQLITE_DOCUMENT_TABLE_NAME)

    logging.info(f"sqlite root data directory: {SQLITE_ROOT_DATA_DIR}")
    logging.info(f"sqlite db name: {SQLITE_DB_NAME}")
    logging.info(f"sqlite table name: {SQLITE_DOCUMENT_TABLE_NAME}")

    # ============================================================================ #
    # chat server
    global CHAT_MODEL_URL, CHAT_MODEL_NAME, CHAT_GEN_CONF, CONVERSATION_SAVE_PATH

    CHAT_MODEL_URL = os.environ.get('CHAT_MODEL_URL',
                                    'http://host.docker.internal:11434')
    CHAT_MODEL_NAME = os.environ.get('CHAT_MODEL_NAME', 'qwen3:30b-a3b')
    CHAT_GEN_CONF = {
        'temperature': 0.1,
        'top_p': 0.3,
        'presence_penalty': 0.4,
        'frequency_penalty': 0.7,
    }
    # where to save conversation data.
    CONVERSATION_SAVE_PATH = os.path.join(RAG_FILE_DIR,
                                          'conversation/conversation.json')

    logging.info(f'chat model url: {CHAT_MODEL_URL}')
    logging.info(f'chat model name: {CHAT_MODEL_NAME}')
    logging.info(f"chat model gen conf: {json.dumps(CHAT_GEN_CONF, indent=4)}")


init_root_config()
