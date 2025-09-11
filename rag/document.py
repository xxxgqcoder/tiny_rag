import logging
import os
from concurrent.futures import ThreadPoolExecutor

import watchdog.events as events
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from common.config import TinyRAGConfig
from common.data import Content, ContentType, GetDocumentResponse
from common.utils import estimate_token_num, hash64, logging_exception, run_once, time_it
from parse import get_parser
from parse.chunking import get_chunking
from parse.parser import Parser
from rag.embedding import EmbeddingModel, get_embedding_model
from rag.functions import delete_document, get_all_document, get_document, upsert_document
from rag.llm import ChatModel, get_chat_model
from rag.prompt import PROMPT_DOCUMENT_META


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


def _ensure_max_token(content: str, max_token_num: int) -> str:
    token_num = estimate_token_num(content)[0]
    if token_num > max_token_num:
        truncate_ratio = float(max_token_num / token_num)
        content = content[: int(len(content) * truncate_ratio)]
        logging.info(
            f"Truncate text due to token num exceed max token num {max_token_num}, estimate token num: {token_num}, new byte len: {len(content)}, truncate ratio: {truncate_ratio}"
        )

    return content


def process_new_file(file_path: str):
    if ignore_file(file_path):
        logging.info(f"{file_path}: ignore")
        return

    parser: Parser = get_parser()
    chat_model: ChatModel = get_chat_model()
    embedding_model: EmbeddingModel = get_embedding_model()

    # get file content hash
    file_name = os.path.basename(file_path)
    logging.info(f"{file_name}: begin processing")
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
    logging.info(f"{file_name}: total {len(file_bytes)} bytes loaded, content hash: {file_content_hash}")

    # get document record
    document_record: GetDocumentResponse = get_document(file_name=file_name)
    stored_content_hash = ""
    try:
        stored_content_hash = document_record.document.content_hash  # type: ignore
    except:
        pass
    if stored_content_hash == file_content_hash:
        logging.info(f"{file_name}: content hash ({file_content_hash}) unchanged, ignore")
        return
    logging.info(f"{file_name}: file content changed or new file")

    # delete document record if any
    delete_document(file_name=file_name)

    # parse file
    content_list: list[Content] = parser.parse(file_path=file_path)
    logging.info(f"{file_name}: total {len(content_list)} content")
    if len(content_list) == 0:
        return

    # chunking
    chunker = get_chunking()
    chunks = chunker.chunk(contents=content_list)
    logging.info(f"{file_name}: total {len(chunks)} chunks")

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

    # embedding chunks
    embedding_max_token_num = 16 * 1024
    chunk_embedding = []
    embedding_batch_size = 64
    for i in range(0, len(chunks), embedding_batch_size):
        chunk_batch = chunks[i: i + embedding_batch_size]
        chunk_content = []
        for chunk in chunk_batch:
            text = chunk.content if chunk.content_type == ContentType.TEXT else chunk.extra_description
            chunk_content.append(_ensure_max_token(text, embedding_max_token_num))
        embeddings = embedding_model.encode(texts=chunk_content)
        chunk_embedding.extend(embeddings)
    logging.info(f"{file_name}: chunk embedding done")

    # save to db
    upsert_document(
        file_name=file_name,
        content_hash=file_content_hash,
        md_content=_format_md_content(content_list=content_list),
        chunks=chunks,
        chunk_embedding=chunk_embedding,
    )

    logging.info(f"{file_name}: finish processing")


def process_delete_file(file_path: str):
    if ignore_file(file_path):
        logging.info(f"{file_path}: ignore")
        return
    logging.info(f"{file_path}: process delete file")

    file_name = os.path.basename(file_path)
    delete_document(file_name=file_name)


def ignore_file(file_path: str) -> bool:
    """
    Rules on igore file.

    Returns:
    - bool, true if file_path should be ignored.
    """
    file_name = os.path.basename(file_path)
    # ignore hidden file
    if file_name.startswith("."):
        return True

    # ignore non-supported file postfix
    postifx = file_name.rsplit(".", 1)[-1]
    if postifx not in ["pdf", "docx", "ppt", "md", "txt"]:
        return True

    return False


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
                job_executor.submit(on_process_new_file, file_path=src_path)

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
    remote_file_names = all_document.file_names if all_document.file_names else []
    file_names = os.listdir(TinyRAGConfig.host_file_dir)

    # delete documents that are not found in file_dir
    to_delete = list(set(remote_file_names) - set(file_names))
    logging.info(f"Below files are founded in db but not in file folder, delete: {to_delete}")
    for file_name in to_delete:
        job_executor.submit(on_process_delete_file, file_path=os.path.join(TinyRAGConfig.host_file_dir, file_name))

    for file_name in file_names:
        job_executor.submit(on_process_new_file, file_path=os.path.join(TinyRAGConfig.host_file_dir, file_name))


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
