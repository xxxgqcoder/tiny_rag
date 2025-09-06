import asyncio
import logging

from watchdog.observers import Observer

from common.config import TinyRAGConfig
from rag.document import FileHandler, initial_file_process


def main():
    # start file monitor
    initial_file_process()

    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, TinyRAGConfig.host_file_dir, recursive=False)
    observer.start()

    observer.join()

    logging.info(f"shutdown")


if __name__ == "__main__":
    main()
