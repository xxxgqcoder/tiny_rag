import asyncio
import logging
from typing import Any

import aiohttp
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
    RationalDBRecord,
    SearchRequest,
    SearchResponse,
    VectorDBRecord,
)

_service_url = TinyRAGConfig.search_service_url


async def http_call(url, request: Any, out_cls) -> Any:
    _session = aiohttp.ClientSession()
    try:
        response = await _session.post(
            url=url, json=request.model_dump() if request else "", headers={"Content-Type": "application/json"}
        )

        if response.status == 200:
            response_data = await response.json()
            return out_cls.model_validate(response_data)
        else:
            error_text = await response.text()
            logging.error(f"Http call fail with status {response.status}: {error_text}")
            raise ValueError(f"Http call fail with status {response.status}: {error_text}")

    except aiohttp.ClientError as e:
        logging.error(f"Network error during document upsert: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during document upsert: {e}")
        raise


async def upsert_document(
    file_name: str,
    content_hash: str,
    md_content: str,
    chunks: list[Chunk],
    chunk_embedding: list[list[float]],
) -> NewDocumentResponse:
    """
    Args:
    - file_name: original file name.
    - content_hash: hash of document content.
    - md_content: markdown content of the document.
    - chunks: list of chunks of the document.
    - chunk_embedding: list of chunk embeddings.
    """
    request = NewDocumentRequest(
        file_name=file_name,
        content_hash=content_hash,
        md_content=md_content,
        chunks=chunks,
        chunk_embedding=chunk_embedding,
    )
    try:
        ret = await http_call(url=_service_url + "/upsert_document", request=request, out_cls=NewDocumentResponse)
        return ret
    except Exception as e:
        logging.error(e)
        return NewDocumentResponse.model_validate({})


async def get_document(
    file_name: str,
) -> GetDocumentResponse:
    """
    Args:
    - file_name: original file name.
    """
    request = GetDocumentRequest(
        file_name=file_name,
    )
    try:
        ret = await http_call(url=_service_url + "/get_document", request=request, out_cls=GetDocumentResponse)
        return ret
    except Exception as e:
        logging.error(e)
        return GetDocumentResponse.model_validate({})


async def get_all_document() -> GetAllDocumentResponse:
    """
    Get all documents name in db.
    """
    try:
        ret = await http_call(url=_service_url + "/get_all_document", request={}, out_cls=GetAllDocumentResponse)
        return ret
    except Exception as e:
        logging.error(e)
        return GetAllDocumentResponse.model_validate({})


async def delete_document(file_name: str) -> DeleteDocumentResponse:
    """
    Delete document from db.
    """
    request = DeleteDocumentRequest(
        file_name=file_name,
    )

    try:
        ret = await http_call(url=_service_url + "/delete_document", request=request, out_cls=DeleteDocumentResponse)
        return ret
    except Exception as e:
        logging.error(e)
        return DeleteDocumentResponse.model_validate({})


async def search(query: dict[str, Any], query_params: dict[str, Any]) -> SearchResponse:
    request = SearchRequest(
        query=query,
        query_params=query_params,
    )

    try:
        ret = await http_call(url=_service_url + "/search", request=request, out_cls=SearchResponse)
        return ret
    except Exception as e:
        logging.error(e)
        return SearchResponse.model_validate({})
    pass
