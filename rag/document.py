import json
from typing import Union, Dict, List, Any

import logging
import config
from parse.parser import Chunk
from rag.nlp import EmbeddingModel


def make_record(chunk: Chunk, embed: EmbeddingModel) -> Dict[str, Any]:
    """
    Make a record from chunk.

    Args:
    - chunk: parsed chunk.
    - embed: embedding model.

    Returns:
    - A dict containing all columns of a record.
    """
    content = chunk.content
    if chunk.content_type != config.ChunkType.TEXT:
        content = chunk.extra_description
    content = content.decode('utf-8')

    meta = {}
    if chunk.content_type == config.ChunkType.IMAGE:
        meta['content_url'] = chunk.content_url
    if chunk.content_type == config.ChunkType.TABLE:
        meta['table_content'] = chunk.content.decode('utf-8')

    embeddings = embed.encode([content])

    return {
        'uuid': chunk.uuid,
        'content': content,
        'meta': json.dumps(meta, indent=4),
        'sparse_vector': embeddings['sparse'][0],
        'dense_vector': embeddings['dense'][[0]],
    }
