from typing import Any

import requests

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
    SearchRequest,
    SearchResponse,
)
from common.logger import Logger
from rag.embedding import get_embedding_model

_service_url = TinyRAGConfig.search_service_url


def http_call(url, request: Any, out_cls) -> Any:
    try:
        response = requests.post(
            url=url, json=request.model_dump() if request else "", headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            response_data = response.json()
            return out_cls.model_validate(response_data)
        else:
            Logger.error(f"Http call fail with status {response.status_code}")
            raise ValueError(f"Http call fail with status {response.status_code}")
    except Exception as e:
        Logger.error(f"Unexpected error: {e}")
        raise


def upsert_document(
    file_path: str,
    content_hash: str,
    md_content: str,
    chunks: list[Chunk],
    chunk_embedding: list[list[float]],
) -> NewDocumentResponse:
    """
    Args:
    - file_path: original file path.
    - content_hash: hash of document content.
    - md_content: markdown content of the document.
    - chunks: list of chunks of the document.
    - chunk_embedding: list of chunk embeddings.
    """
    request = NewDocumentRequest(
        file_path=file_path,
        content_hash=content_hash,
        md_content=md_content,
        chunks=chunks,
        chunk_embedding=chunk_embedding,
    )
    try:
        ret = http_call(url=_service_url + "/upsert_document", request=request, out_cls=NewDocumentResponse)
        return ret
    except Exception as e:
        Logger.error(e)
        return NewDocumentResponse.model_validate({})


def get_document(
    file_path: str,
) -> GetDocumentResponse:
    """
    Args:
    - file_path: original file path.

    Returns:
    - GetDocumentResponse: document content and metadata.
    """
    request = GetDocumentRequest(
        file_path=file_path,
    )
    try:
        ret = http_call(url=_service_url + "/get_document", request=request, out_cls=GetDocumentResponse)
        return ret
    except Exception as e:
        Logger.error(e)
        return GetDocumentResponse.model_validate({})


def get_all_document() -> GetAllDocumentResponse:
    """
    Get all documents name in db.

    Returns:
    - GetAllDocumentResponse: list of document names.
    """
    try:
        ret = http_call(url=_service_url + "/get_all_document", request={}, out_cls=GetAllDocumentResponse)
        return ret
    except Exception as e:
        Logger.error(e)
        return GetAllDocumentResponse.model_validate({})


def delete_document(file_path: str) -> DeleteDocumentResponse:
    """
    Delete document from db.
    """
    request = DeleteDocumentRequest(
        file_path=file_path,
    )

    try:
        ret = http_call(url=_service_url + "/delete_document", request=request, out_cls=DeleteDocumentResponse)
        return ret
    except Exception as e:
        Logger.error(e)
        return DeleteDocumentResponse.model_validate({})


def search(
    query: str,
    limit: int = 10,
    prompt_name: str = "",
) -> SearchResponse:
    """
    Search documents in db using semantic search.

    Args:
    - query: query requirement in natural language.
    - limit: number of results to return.

    Returns:
    - SearchResponse: list of search results.
    """
    embedding_model = get_embedding_model()
    embedding_vector = embedding_model.encode([query], prompt_name=prompt_name)

    request = SearchRequest(
        query={
            TinyRAGConfig.vector_db_config.embedding_column_name: embedding_vector[0],
        },
        query_params={
            "limit": limit,
        },
    )

    try:
        ret = http_call(url=_service_url + "/search", request=request, out_cls=SearchResponse)
        return ret
    except Exception as e:
        Logger.error(e)
        return SearchResponse.model_validate({})
