import logging
import traceback
import os
import time

from flask import Flask
from watchdog.observers import Observer

import config
from rag.document import FileHandler, initial_file_process
from rag.db import create_milvus_collection, create_sqlite_table

if __name__ == '__main__':
    # set up db
    create_milvus_collection(
        conn_url=config.MILVUS_DB_NAME,
        collection_name=config.MILVUS_COLLECTION_NAME,
        dense_embed_dim=config.EMBED_DENSE_DIM,
    )
    create_sqlite_table(
        conn_url=config.SQLITE_DB_NAME,
        table_name=config.SQLITE_DOCUMENT_TABLE_NAME,
    )

    # initial file direcory process
    initial_file_process(config.RAG_FILE_DIR)

    # start file monitor
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, config.RAG_FILE_DIR, recursive=False)
    observer.start()

    # http server
    # NOTE: debug=True cause milvus start failure, no idea why.
    app = Flask(__name__)
    from rag import rag_server
    app.register_blueprint(rag_server.bp)
    app.run(
        debug=False,
        host='0.0.0.0',
        port=4567,
    )

    logging.info('server shutdown')
