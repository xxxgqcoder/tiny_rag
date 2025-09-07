import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import watchdog.events as events
from watchdog.events import FileSystemEvent, FileSystemEventHandler

from common.config import TinyRAGConfig
from common.data import Content, ContentType, GetDocumentResponse
from common.utils import estimate_token_num, hash64, logging_exception, run_once, time_it
from parse import get_parser
from parse.chunking import get_chunking
from rag.embedding import get_embedding_model
from rag.functions import delete_document, get_all_document, get_document, upsert_document
from rag.llm import ChatModel, get_chat_model

_prompt_text_summary = """"
summarize below content, use no more than {max_token_num} words.

below is the content
----

{content}

----

now let's tink step by step and give a concise summary.
"""

_prompt_image_summary = """
"""


def format_md_content(content_list: list[Content]) -> str:
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
    if ignore_file(file_path):
        logging.info(f"{file_path}: ignore")
        return

    parser = get_parser()

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

    # add llm summary to chunk
    chat_model: ChatModel = get_chat_model()
    for chunk in chunks:
        if chunk.content_type != ContentType.TEXT:
            continue

        content = chunk.content
        estimated_token_num = estimate_token_num(content)[0]
        if estimated_token_num > TinyRAGConfig.max_context_token_num:
            truncate_ratio = float(TinyRAGConfig.max_context_token_num / estimated_token_num)
            logging.info(
                f"estimated token num ({estimated_token_num}) exceed max token num ({TinyRAGConfig.max_context_token_num}), prompt byte num: {len(content)}, truncated by ratio: {truncate_ratio}"
            )
            content = content[: int(len(content) * truncate_ratio)]
            logging.info(f"truncated byte num: {len(content)}")

        prompt = _prompt_text_summary.format(
            content=content,
            max_token_num=int(estimated_token_num * 0.1),
        )
        summary = chat_model.instant_chat(
            prompt=prompt,
            gen_conf=TinyRAGConfig.gen_conf.model_dump(),  # type: ignore
        )
        # need to regenerate uuid
        chunk.content += f"\n\n\n\n<llm_content><summary>{summary}</summary></llm_content>"
        logging.info(f"{file_name}: Finish adding llm summary to chunk, summary:\n{summary}")

    # embedding chunks
    embedding_model = get_embedding_model()
    embedding_max_token_num = 8 * 1024
    chunk_embedding = []
    for i, chunk in enumerate(chunks):
        text: str = chunk.content if chunk.content_type == ContentType.TEXT else chunk.extra_description
        token_num = estimate_token_num(text)[0]
        logging.info(
            f"{file_name}: chunk {i}, uuid: {chunk.uuid}, byte len: {len(text)}, estimated token num: {token_num}"
        )
        if token_num > embedding_max_token_num:
            truncate_ratio = float(embedding_max_token_num / token_num)
            text = text[: int(len(text) * truncate_ratio)]
            logging.info(
                f"Truncate text due to token num exceed embedding model limit, new byte len: {len(text)}, truncate ratio: {truncate_ratio}"
            )
        embedding: dict[str, Any] = embedding_model.encode(texts=[text])
        chunk_embedding.append(embedding["dense"][0])
        logging.info(f"{file_name}: finish embedding for chunk: {i}")

    # save to db
    upsert_document(
        file_name=file_name,
        content_hash=file_content_hash,
        md_content=format_md_content(content_list=content_list),
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
    file_names = os.listdir(TinyRAGConfig.host_file_dir)

    # delete documents that are not found in file_dir
    to_delete = list(set(all_document.file_names) - set(file_names))  # type: ignore
    logging.info(f"Below files are founded in db but not in file folder, delete: {to_delete}")
    for file_name in to_delete:
        job_executor.submit(on_process_delete_file, file_path=os.path.join(TinyRAGConfig.host_file_dir, file_name))

    for file_name in file_names:
        job_executor.submit(on_process_new_file, file_path=os.path.join(TinyRAGConfig.host_file_dir, file_name))
