import json

import uvicorn
from fastapi import FastAPI

from common.config import TinyRAGConfig
from common.data import (
    Chunk,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    GetAllDocumentResponse,
    GetDocumentRequest,
    GetDocumentResponse,
    NewDocumentRequest,
    NewDocumentResponse,
    RationalDBRecord,
    SearchRequest,
    SearchResponse,
    VectorDBRecord,
)
from common.logger import Logger
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

create_rational_db_table()
create_vector_db_collection()
create_object_store_bucket()


app = FastAPI()


rational_db: RationalDB = get_rational_db()
vector_db: VectorDB = get_vector_db()
object_store: ObjectStore = get_object_store()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Service is running"}


@app.post("/upsert_document", response_model=NewDocumentResponse)
async def upsert_document(request: NewDocumentRequest) -> NewDocumentResponse:
    """
    Add new document.
    """

    # Save to rational db
    try:
        upsert_cnt = rational_db.upsert_document(
            RationalDBRecord(
                file_path=request.file_path,
                chunk_uuids="\x07".join([chunk.uuid for chunk in request.chunks]),
                created_date="",
                content_hash=request.content_hash,
            )
        )
        if upsert_cnt != 1:
            raise Exception(f"Failed to insert document: expected 1 record inserted, got {upsert_cnt}")
    except Exception as e:
        logging_exception(e)
        return NewDocumentResponse(code=1, message="error when saving document to rational db, error:\n" + str(e))

    # Save to vector db
    try:
        for i, chunk in enumerate(request.chunks):
            upsert_cnt = vector_db.upsert(
                VectorDBRecord(
                    uuid=chunk.uuid,
                    file_path=chunk.file_path,
                    content_url=chunk.content_url,
                    embedding=request.chunk_embedding[i],
                    metadata={
                        "extra_description": chunk.extra_description,
                        "content_type": str(chunk.content_type),
                    },
                )
            )
            if upsert_cnt != 1:
                raise Exception(f"Failed to insert chunk: expected 1 record inserted, got {upsert_cnt}")

    except Exception as e:
        logging_exception(e)
        return NewDocumentResponse(code=1, message="error when saving chunks to vector db, error:\n" + str(e))

    # Save to object store
    try:
        # save document content
        if request.md_content:
            content_bytes = request.md_content.encode("utf-8", errors="ignore")
            insert_byte_cnt = object_store.put(
                key=StorageManager.document_content_key(request.content_hash),
                obj=content_bytes,
            )
            if insert_byte_cnt != len(content_bytes):
                raise Exception(
                    f"Failed to insert md_content: expected {len(request.md_content)} bytes, got {insert_byte_cnt}"
                )

        # save document chunks
        for i, chunk in enumerate(request.chunks):
            data_dict = chunk.model_dump()
            json_data = json.dumps(data_dict, ensure_ascii=False)
            json_bytes = json_data.encode("utf-8", errors="ignore")
            insert_byte_cnt = object_store.put(
                key=StorageManager.chunk_content_key(chunk.uuid),
                obj=json_bytes,
            )
            if insert_byte_cnt != len(json_bytes):
                raise Exception(
                    f"Failed to insert chunk content: expected {len(json_data)} bytes, got {insert_byte_cnt}"
                )

    except Exception as e:
        logging_exception(e)
        return NewDocumentResponse(code=1, message="error when saving to object store, error:\n" + str(e))

    return NewDocumentResponse(
        code=0, message="success", data={"file_path": request.file_path, "chunks": [c.uuid for c in request.chunks]}
    )


@app.post("/get_document", response_model=GetDocumentResponse)
async def get_document(
    request: GetDocumentRequest,
) -> GetDocumentResponse:
    file_path = request.file_path
    if not file_path:
        return GetDocumentResponse(
            code=1,
            message="file_path is required",
            document=None,
            md_content="",
            chunks=[],
        )

    record: RationalDBRecord = rational_db.get_document(file_path=file_path)
    if not record:
        return GetDocumentResponse(
            code=1,
            message=f"document not found: {file_path}",
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
    file_paths: list[str] = rational_db.get_all_documents()
    return GetAllDocumentResponse(
        code=0,
        message="success",
        file_paths=file_paths,
    )


@app.post("/delete_document", response_model=DeleteDocumentResponse)
async def delete_document(request: DeleteDocumentRequest) -> DeleteDocumentResponse:
    file_path = request.file_path
    # get record from rational db
    record = rational_db.get_document(file_path=file_path)
    if not record:
        return DeleteDocumentResponse(code=1, message=f"document not found: {file_path}")

    # delete from rational db
    delete_cnt = rational_db.delete_document(file_path=file_path)
    Logger.info(f"Deleted {delete_cnt} records from rational db for document: {file_path}")

    # delete from vector db
    uuids = record.chunk_uuids.split("\x07")
    delete_cnt = vector_db.delete(keys=uuids)
    Logger.info(f"Deleted {delete_cnt} records from vector db for chunk: {file_path}")

    # delete from object store
    try:
        object_store.delete(StorageManager.document_content_key(record.content_hash))
    except Exception as e:
        logging_exception(e)

    for uuid in uuids:
        try:
            object_store.delete(StorageManager.chunk_content_key(uuid))
        except Exception as e:
            logging_exception(e)

    return DeleteDocumentResponse(
        code=0,
        message="success",
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    query = request.query
    query_params = request.query_params

    records = vector_db.search(query=query, params=query_params)
    chunks = []
    for record in records:
        uuid = record.uuid
        try:
            chunk_bytes = object_store.get(key=StorageManager.chunk_content_key(uuid))
            if not chunk_bytes:
                continue
        except Exception as e:
            logging_exception(e)
            continue
        chunk = Chunk.model_validate_json(chunk_bytes.decode(encoding="utf-8", errors="ignore"))
        chunks.append(chunk)

    return SearchResponse(
        code=0,
        message="",
        data={},
        chunks=chunks,
    )


if __name__ == "__main__":
    # set up service
    Logger.info(f"Server started on port {TinyRAGConfig.search_service_port}")
    
    uvicorn.run(app, host="0.0.0.0", port=TinyRAGConfig.search_service_port)

    Logger.info("Server shutting down")
