import logging
import os
from concurrent.futures import ThreadPoolExecutor

import watchdog.events as events
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from common.config import TinyRAGConfig
from common.data import Content, ContentType, GetDocumentResponse
from common.utils import ensure_max_token, hash64, logging_exception, run_once, time_it
from parse import get_parser_by_file_type
from parse.chunking import get_chunking_by_file_type
from parse.parser import Parser
from rag.embedding import EmbeddingModel, get_embedding_model
from rag.functions import delete_document, get_all_document, get_document, upsert_document
from rag.llm import ChatModel, get_chat_model, get_vision_model


def _format_md_content(content_list: list[Content]) -> str:
    ret = ""
    for content in content_list:
        if content.content_type in ContentType.TEXT:
            ret += content.content + "\n\n"
        elif content.content_type in [ContentType.IMAGE, ContentType.TABLE]:
            ret += content.content_url + "\n\n"
            ret += content.extra_description + "\n\n"
        else:
            pass

    return ret


def process_new_file(file_path: str):
    if _ignore_file(file_path):
        logging.info(f"{file_path}: ignore")
        return

    chat_model: ChatModel = get_chat_model()
    embedding_model: EmbeddingModel = get_embedding_model()

    # get file content hash
    file_type = file_path.rsplit(".", 1)[-1]
    logging.info(f"{file_path}: type {file_type}, begin processing")
    file_bytes = ""
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        logging_exception(e)
        return

    if len(file_bytes) == 0:
        logging.info(f"{file_path}: empty content, skip")
        return

    file_content_hash = hash64(file_bytes)
    logging.info(f"{file_path}: total {len(file_bytes)} bytes loaded, content hash: {file_content_hash}")

    # get document record
    document_record: GetDocumentResponse = get_document(file_path=file_path)
    stored_content_hash = ""
    try:
        stored_content_hash = document_record.document.content_hash  # type: ignore
    except:
        pass
    if stored_content_hash == file_content_hash:
        logging.info(f"{file_path}: content hash ({file_content_hash}) unchanged, ignore")
        return
    logging.info(f"{file_path}: file content changed or new file")

    # delete document record if any
    delete_document(file_path=file_path)

    # parse file
    parser: Parser = get_parser_by_file_type(file_type=file_type)
    content_list: list[Content] = parser.parse(file_path=file_path)
    logging.info(f"{file_path}: total {len(content_list)} content")
    if len(content_list) == 0:
        return

    # chunking
    chunker = get_chunking_by_file_type(file_type=file_type)
    chunks = chunker.chunk(contents=content_list)
    logging.info(f"{file_path}: total {len(chunks)} chunks")

    # NOTE: add title, authors and keywords to each chunk cause search to focus on meta info of each chunk.
    # # add document meta data to chunk
    # md_content = _format_md_content(content_list=content_list)
    # content = _ensure_max_token(md_content, 2000)
    # document_meta = chat_model.instant_chat(prompt=PROMPT_DOCUMENT_META.format(content=content))
    # logging.info(f"{file_name}: document meta:\n{document_meta}")

    # for chunk in chunks:
    #     if chunk.content_type == ContentType.TEXT:
    #         chunk.content = chunk.content + "\n\n\n\n" + f"<document_meta>\n{document_meta}\n</document_meta>"
    #     else:
    #         chunk.extra_description = (
    #             chunk.extra_description + "\n\n\n\n" + f"<document_meta>\n{document_meta}\n</document_meta>"
    #         )

    # process image chunks
    vision_model = get_vision_model()
    for chunk in chunks:
        if chunk.content_type not in [ContentType.IMAGE, ContentType.TABLE]:
            continue
        if not chunk.content or not chunk.content_url:
            continue
        img_description = vision_model.image_chat(
            prompt="summarize what you see in the picture",
            image_content=chunk.content,
        )
        chunk.extra_description = chunk.extra_description + "\n\n\n\n" + img_description

    # embedding chunks
    embedding_max_token_num = 16 * 1024
    chunk_embedding = []
    embedding_batch_size = 64
    for i in range(0, len(chunks), embedding_batch_size):
        chunk_batch = chunks[i : i + embedding_batch_size]
        chunk_content = []
        for chunk in chunk_batch:
            text = chunk.content if chunk.content_type == ContentType.TEXT else chunk.extra_description
            chunk_content.append(ensure_max_token(text, embedding_max_token_num))
        embeddings = embedding_model.encode(texts=chunk_content)
        chunk_embedding.extend(embeddings)
    logging.info(f"{file_path}: chunk embedding done")

    # save to db
    upsert_document(
        file_path=file_path,
        content_hash=file_content_hash,
        md_content=_format_md_content(content_list=content_list),
        chunks=chunks,
        chunk_embedding=chunk_embedding,
    )

    logging.info(f"{file_path}: finish processing")


def process_delete_file(file_path: str):
    if _ignore_file(file_path):
        logging.info(f"{file_path}: ignore")
        return
    logging.info(f"{file_path}: process delete file")

    delete_document(file_path=file_path)


def _ignore_file(file_path: str) -> bool:
    """
    Rules on igore file.

    Returns:
    - bool, true if file_path should be ignored.
    """
    for pattern in TinyRAGConfig.ignore_path_pattern:
        if pattern in file_path:
            return True
    file_name = os.path.basename(file_path)
    # ignore hidden file
    if file_name.startswith("."):
        return True

    # ignore non-supported file postfix
    postifx = file_name.rsplit(".", 1)[-1]
    if postifx in ["pdf", "docx", "ppt", "md", "txt"]:
        return False
    if postifx in ["png", "jpg", "jpeg", "bmp", "gif"]:
        return False

    return True


# --------------------------------------------------------------------------------------------------------
# job executor

_job_executor = None


def get_job_executor() -> ThreadPoolExecutor:
    global _job_executor
    if _job_executor is None:
        # NOTE: set only 1 thread to force sequencial job schedule.
        _job_executor = ThreadPoolExecutor(max_workers=1)

    return _job_executor


@time_it(prefix="process new file")
def on_process_new_file(file_path: str) -> None:
    try:
        process_new_file(file_path=file_path)
    except Exception as e:
        logging_exception(e)


@time_it(prefix="process delete file")
def on_process_delete_file(file_path: str) -> None:
    try:
        process_delete_file(file_path=file_path)
    except Exception as e:
        logging_exception(e)


class FileHandler(FileSystemEventHandler):
    def on_any_event(self, event: FileSystemEvent) -> None:
        job_executor = get_job_executor()
        src_path = event.src_path
        dest_path = event.dest_path

        if event.event_type == events.EVENT_TYPE_MOVED:
            if not os.path.isdir(src_path):
                job_executor.submit(on_process_delete_file, file_path=src_path)

            if not os.path.isdir(dest_path):
                job_executor.submit(on_process_new_file, file_path=dest_path)

        elif event.event_type == events.EVENT_TYPE_DELETED:
            if not os.path.isdir(src_path):
                job_executor.submit(on_process_delete_file, file_path=src_path)

        elif event.event_type == events.EVENT_TYPE_CREATED:
            if not os.path.isdir(src_path):
                job_executor.submit(on_process_new_file, file_path=src_path)

        elif event.event_type == events.EVENT_TYPE_MODIFIED:
            if not os.path.isdir(src_path):
                job_executor.submit(on_process_new_file, file_path=src_path)

        else:
            pass


@run_once
def initial_file_process() -> None:
    job_executor = get_job_executor()

    # get all documents
    all_document = get_all_document()
    remote_file_paths = all_document.file_paths if all_document.file_paths else []
    logging.info(f"Total {len(remote_file_paths)} files in db")
    full_file_paths = [
        os.path.join(root, file) for root, _, files in os.walk(TinyRAGConfig.host_file_dir) for file in files
    ]

    filered_file_paths = [file_path for file_path in full_file_paths if not _ignore_file(file_path)]
    logging.info(f"Total {len(filered_file_paths)} files to process")

    # delete documents that are not found in file_dir
    to_delete = list(set(remote_file_paths) - set(filered_file_paths))
    logging.info(f"Below files are founded in db but not in file folder, delete: {to_delete}")
    for file_path in to_delete:
        job_executor.submit(on_process_delete_file, file_path=file_path)

    for file_path in filered_file_paths:
        job_executor.submit(on_process_new_file, file_path=os.path.join(file_path))


def main():
    # start file monitor
    initial_file_process()

    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, TinyRAGConfig.host_file_dir, recursive=True)
    observer.start()

    observer.join()

    logging.info(f"shutdown")


if __name__ == "__main__":
    main()
