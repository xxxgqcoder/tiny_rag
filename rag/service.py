import json
import logging
from typing import Any

from fastapi import FastAPI

from common.config import TinyRAGConfig
from common.data import (
    Chunk,
    DeleteDocumentResponse,
    GetAllDocumentResponse,
    GetDocumentResponse,
    NewDocumentResponse,
    RationalDBRecord,
    SearchResponse,
    VectorDBRecord,
)
from common.utils import logging_exception
from rag.db import (
    ObjectStore,
    RationalDB,
    StorageManager,
    VectorDB,
    create_object_store_bucket,
    create_rational_db_table,
    create_vector_db_collection,
    get_object_store,
    get_rational_db,
    get_vector_db,
)

create_rational_db_table(
    conn_url=TinyRAGConfig.rational_db_config.db_name,  # type: ignore
    table_name=TinyRAGConfig.rational_db_config.document_table_name,  # type: ignore
)

create_vector_db_collection(
    conn_url=TinyRAGConfig.vector_db_config.db_name,  # type: ignore
    collection_name=TinyRAGConfig.vector_db_config.collection_name,  # type: ignore
    embedding_dim=TinyRAGConfig.embeding_config.embeding_dim,  # type: ignore
)

create_object_store_bucket(
    conn_url=TinyRAGConfig.object_store_config.conn_url,  # type: ignore
    user=TinyRAGConfig.object_store_config.user,  # type: ignore
    token=TinyRAGConfig.object_store_config.token,  # type: ignore
    bucket_name=TinyRAGConfig.object_store_config.bucket_name,  # type: ignore
)


app = FastAPI()


rational_db: RationalDB = get_rational_db()
vector_db: VectorDB = get_vector_db()
object_store: ObjectStore = get_object_store()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Service is running"}


@app.post("/upsert_document", response_model=NewDocumentResponse)
async def upsert_document(
    file_name: str,
    content_hash: str,
    md_content: str,
    chunks: list[Chunk],
    chunk_embedding: list[list[float]],
) -> NewDocumentResponse:
    """
    Add new document.
    """

    # Save to rational db
    try:
        insert_cnt = rational_db.upsert_document(
            RationalDBRecord(
                file_name=file_name,
                chunk_uuids="\x07".join([chunk.uuid for chunk in chunks]),
                created_date="",
                content_hash=content_hash,
            )
        )
        if insert_cnt != 1:
            raise Exception(f"Failed to insert document: expected 1 record inserted, got {insert_cnt}")
    except Exception as e:
        logging_exception(e)
        return NewDocumentResponse(code=1, message="error when saving document to rational db, error:\n" + str(e))

    # Save to vector db
    try:
        for i, chunk in enumerate(chunks):
            insert_cnt = vector_db.upsert(
                VectorDBRecord(
                    uuid=chunk.uuid,
                    file_name=chunk.file_name,
                    content_url=chunk.content_url,
                    embedding=chunk_embedding[i],
                    metadata={
                        "extra_description": chunk.extra_description,
                        "content_type": str(chunk.content_type),
                    },
                )
            )
            if insert_cnt != 1:
                raise Exception(f"Failed to insert chunk: expected 1 record inserted, got {insert_cnt}")
            logging.info(f"Inserted chunk to vector db: {chunk.uuid}, file_name: {file_name}")

    except Exception as e:
        logging_exception(e)
        return NewDocumentResponse(code=1, message="error when saving chunks to vector db, error:\n" + str(e))

    # Save to object store
    try:
        # save document content
        if md_content:
            content_bytes = md_content.encode("utf-8", errors="ignore")
            insert_byte_cnt = object_store.put(
                key=StorageManager.document_content_key(content_hash),
                obj=content_bytes,
            )
            if insert_byte_cnt != len(content_bytes):
                raise Exception(f"Failed to insert md_content: expected {len(md_content)} bytes, got {insert_byte_cnt}")
            logging.info(f"Inserted md_content: {file_name}, size: {insert_byte_cnt} bytes")

        # save document chunks
        for chunk in chunks:
            data_dict = chunk.model_dump()
            json_data = json.dumps(data_dict, ensure_ascii=False)
            json_bytes = json_data.encode("utf-8", errors="ignore")
            if chunk.content_url and chunk.content:
                insert_byte_cnt = object_store.put(
                    key=StorageManager.chunk_content_key(chunk.uuid),
                    obj=json_bytes,
                )
                if insert_byte_cnt != len(json_bytes):
                    raise Exception(
                        f"Failed to insert chunk content: expected {len(json_data)} bytes, got {insert_byte_cnt}"
                    )
                logging.info(f"Inserted chunk content: {chunk.uuid}, size: {insert_byte_cnt} bytes")

    except Exception as e:
        logging_exception(e)
        return NewDocumentResponse(code=1, message="error when saving to object store, error:\n" + str(e))

    return NewDocumentResponse(
        code=0, message="success", data={"file_name": file_name, "chunks": [c.uuid for c in chunks]}
    )


@app.post("/get_document", response_model=GetDocumentResponse)
async def get_document(
    file_name: str,
) -> GetDocumentResponse:
    if not file_name:
        return GetDocumentResponse(
            code=1,
            message="file_name is required",
            document=None,
            md_content="",
            chunks=[],
        )

    record: RationalDBRecord = rational_db.get_document(file_name=file_name)
    if not record:
        return GetDocumentResponse(
            code=1,
            message=f"document not found: {file_name}",
        )

    # md content
    md_content: str = ""
    content_bytes: bytes = object_store.get(StorageManager.document_content_key(record.content_hash))
    if content_bytes:
        md_content = content_bytes.decode("utf-8")

    # chunks
    chunks: list[Chunk] = []
    for uuid in record.chunk_uuids.split("\x07"):
        chunk_data: bytes = object_store.get(StorageManager.chunk_content_key(uuid))
        if not chunk_data:
            continue
        chunk: Chunk = Chunk.model_validate(json.loads(chunk_data.decode(encoding="utf-8")))
        chunks.append(chunk)

    return GetDocumentResponse(
        code=0,
        message="success",
        document=record,
        md_content=md_content,
        chunks=chunks,
    )


@app.post("/get_all_document", response_model=GetAllDocumentResponse)
async def get_all_document() -> GetAllDocumentResponse:
    file_names: list[str] = rational_db.get_all_documents()
    return GetAllDocumentResponse(
        code=0,
        message="success",
        file_names=file_names,
    )


@app.post("/delete_document", response_model=DeleteDocumentResponse)
async def delete_document(file_name: str) -> DeleteDocumentResponse:
    # get record from rational db
    record = rational_db.get_document(file_name=file_name)
    if not record:
        return DeleteDocumentResponse(code=1, message=f"document not found: {file_name}")

    # delete from rational db
    delete_cnt = rational_db.delete_document(file_name=file_name)
    logging.info(f"Deleted {delete_cnt} records from rational db for document: {file_name}")

    # delete from vector db
    uuids = record.chunk_uuids.split("\x07")
    delete_cnt = vector_db.delete(keys=uuids)
    logging.info(f"Deleted {delete_cnt} records from vector db for chunk: {file_name}")

    # delete from object store
    try:
        object_store.delete(StorageManager.document_content_key(record.content_hash))
        for uuid in uuids:
            object_store.delete(StorageManager.chunk_content_key(uuid))
    except Exception as e:
        logging_exception(e)

    return DeleteDocumentResponse(
        code=0,
        message="success",
    )


@app.post("/search", response_model=SearchResponse)
async def search(
    query: dict[str, Any],
    query_params: dict[str, Any],
) -> SearchResponse:
    return SearchResponse(
        code=1,
        message="not implemented",
        data={},
        chunks=[],
    )
