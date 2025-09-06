import logging

import uvicorn

from common.config import TinyRAGConfig
from rag.service import app

if __name__ == "__main__":
    # set up service
    logging.info(f"Server started on port {TinyRAGConfig.search_service_port}")

    uvicorn.run(app, host="0.0.0.0", port=TinyRAGConfig.search_service_port)

    logging.info(f"Server shut down")
