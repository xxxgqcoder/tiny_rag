import logging
import traceback
import re
import os

from typing import Union, Dict, List, Any, Generator, Tuple
from abc import ABC, abstractmethod

from ollama import Client as OllamaClient

import config
from utils import singleton, estimate_token_num
from parse.parser import Chunk, ChunkType


class ChatModel(ABC):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def chat(
        self,
        history: list[Dict[str, Any]],
        gen_conf: Dict[str, Any],
    ) -> Generator[Union[str, int], Any, Any]:
        """
        Chat API.

        Args:
        - history: a list of json objects representing coversation history.
        - gen_conf: dict containing LLM generation configuration.

        Returns:
        - A generator of token sequence, last element will be total generated 
            token num.
        """
        raise NotImplementedError("Not implemented")

    def _calculate_dynamic_ctx(self, history: list[Dict[str, Any]]) -> int:
        """
        Calculate dynamic context window size

        Args:
        - history: conversation history, a list of json objects.

        """

        def count_tokens(text):
            # Simple calculation: 1 token per ASCII character
            # 2 tokens for non-ASCII characters (Chinese, Japanese, Korean, etc.)
            total = 0
            for char in text:
                if ord(char) < 128:  # ASCII characters
                    total += 1
                else:  # Non-ASCII characters (Chinese, Japanese, Korean, etc.)
                    total += 2
            return total

        # Calculate total tokens for all messages
        total_tokens = 0
        for message in history:
            content = message.get("content", "")
            # Calculate content tokens
            content_tokens = count_tokens(content)
            # Add role marker token overhead
            role_tokens = 4
            total_tokens += content_tokens + role_tokens

        # Apply 1.2x buffer ratio
        total_tokens_with_buffer = int(total_tokens * 1.2)

        if total_tokens_with_buffer <= 8192:
            ctx_size = 8192
        else:
            ctx_multiplier = (total_tokens_with_buffer // 8192) + 1
            ctx_size = ctx_multiplier * 8192

        return ctx_size

    @abstractmethod
    def instant_chat(
        self,
        prompt: str,
        gen_conf: Dict[str, Any],
    ) -> str:
        """
        Instant chat.
        Args:
        - prompt: prompt text.
        - gen_conf: dict containing LLM generation configuration.
        """
        raise NotImplementedError("Not implemented")


@singleton
class OllamaChat(ChatModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OllamaClient(
            host=config.CHAT_MODEL_URL if 'ollama_host' not in kwargs else kwargs['ollama_host'],
            timeout=15 * 60,  # time out 15 min
        )
        self.model_name = config.CHAT_MODEL_NAME if 'ollama_model_name' not in kwargs else kwargs['ollama_model_name']

    def chat(
        self,
        history: list[Dict[str, Any]],
        gen_conf: Dict[str, Any],
    ) -> Generator[Union[str, int], Any, Any]:
        ctx_size = self._calculate_dynamic_ctx(history)
        if "max_tokens" in gen_conf:
            del gen_conf["max_tokens"]

        options = {"num_ctx": ctx_size}
        if "temperature" in gen_conf:
            options["temperature"] = gen_conf["temperature"]
        if "max_tokens" in gen_conf:
            options["num_predict"] = gen_conf["max_tokens"]
        if "top_p" in gen_conf:
            options["top_p"] = gen_conf["top_p"]
        if "presence_penalty" in gen_conf:
            options["presence_penalty"] = gen_conf["presence_penalty"]
        if "frequency_penalty" in gen_conf:
            options["frequency_penalty"] = gen_conf["frequency_penalty"]

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                stream=True,
                options=options,
                keep_alive=10,
            )
            for resp in response:
                # ollama generates one token per response
                if resp["done"]:
                    token_count = resp.get("prompt_eval_count", 0) + resp.get("eval_count", 0)
                    yield token_count
                yield resp["message"]["content"]
        except Exception as e:
            yield "\n**ERROR**: " + str(e)
        yield 0

    def instant_chat(
        self,
        prompt: str,
        gen_conf: Dict[str, Any],
    ) -> str:
        est_token_num = estimate_token_num(prompt)[0]
        if est_token_num > config.MAX_TOKEN_NUM:
            truncate_ratio = float(config.MAX_TOKEN_NUM / est_token_num)
            logging.info(f'estimated token num exceed max token num, prompt byte num: {len(prompt)}, truncated by ratio: {truncate_ratio}')
            prompt = prompt[:int(len(prompt) * truncate_ratio)]
            logging.info(f'truncated byte num: {len(prompt)}')

        history = [{'role': 'user', 'content': prompt}]
        ctx_size = self._calculate_dynamic_ctx(history)
        if "max_tokens" in gen_conf:
            del gen_conf["max_tokens"]

        options = {"num_ctx": ctx_size}
        if "temperature" in gen_conf:
            options["temperature"] = gen_conf["temperature"]
        if "max_tokens" in gen_conf:
            options["num_predict"] = gen_conf["max_tokens"]
        if "top_p" in gen_conf:
            options["top_p"] = gen_conf["top_p"]
        if "presence_penalty" in gen_conf:
            options["presence_penalty"] = gen_conf["presence_penalty"]
        if "frequency_penalty" in gen_conf:
            options["frequency_penalty"] = gen_conf["frequency_penalty"]

        try:
            response = self.client.chat(model=self.model_name, messages=history, options=options, keep_alive=10)
        except Exception as e:
            return f"Exception: {e}"

        ans = response["message"]["content"].strip()
        if '</think>' in ans:
            ans = ans.split('</think>')[-1]
        return ans.strip()


def get_chat_model(name: str = 'Ollama') -> ChatModel:
    return OllamaChat(
        ollama_host=config.CHAT_MODEL_URL,
        ollama_model_name=config.CHAT_MODEL_NAME,
    )


def format_host_url(content_url: str) -> str:
    if content_url is None or len(content_url) == 0:
        return None
    if content_url.startswith(config.HOST_RAG_FILE_DIR):
        return content_url
    file_name = os.path.basename(content_url)
    ret = os.path.join(config.HOST_RAG_FILE_DIR, 'tiny_rag_parsed_assets', file_name)
    return ret


def max_token_truncate(history: list[str], max_token_num: int) -> int:
    """
    Truncate by max token num. Return index such that token num of history[:index + 1] <= max_token_num

    Args:
    - history: a list of string.
    - max_token_num: maximum token num.
    """
    total_token_num = 0
    index = 0
    for i, msg in enumerate(history):
        token_num, _ = estimate_token_num(msg)
        if total_token_num + token_num >= max_token_num:
            break
        else:
            index = i
    return index


def assemble_knowledge_base(chunks: list[Chunk]) -> Tuple[str, Dict[str, Any]]:
    """
    Assemble knowledge in chunk and return formatted knowledge base.

    Args:
    - chunks: chunks used to format knowledge base.

    Returns:
    - knowledge base: formated knowledge base.
    - A dict of reference id to chunk meta, key is reference id within current knowledge base.
    """
    _content_divider = '\n\n'
    # dedup chunks
    deduped_chunk = {}
    for chunk in chunks:
        if chunk.uuid not in deduped_chunk:
            deduped_chunk[chunk.uuid] = chunk

    # document name to chunks mapping
    document2chunks = {}
    for _, chunk in deduped_chunk.items():
        if chunk.file_name not in document2chunks:
            document2chunks[chunk.file_name] = []
        document2chunks[chunk.file_name].append(chunk)

    knowledge_base = []
    chunk_idx = 0  # reference index within current knowledge base
    refid2meta = {}  # reference index to chunk meta info
    for file_name, chunks in document2chunks.items():
        knowledge_base.append(f"Document: {file_name}{_content_divider}")
        knowledge_base.append(f'Relevant fragments as following:{_content_divider}')
        for chunk in chunks:
            content = ""
            if chunk.content_type in [ChunkType.TEXT]:
                content = chunk.content.decode('utf-8')
            else:
                content = chunk.extra_description.decode('utf-8')

            # trim llm summary if any
            content = re.sub(r"<summary>[.\s\S]*</summary>", "", content)
            knowledge_base.append(f"ID:{chunk_idx}\n{content}")

            tokens = estimate_token_num(content)[-1]

            refid2meta[chunk_idx] = {
                'uuid': chunk.uuid,
                'file_name': chunk.file_name,
                'content_type': chunk.content_type,
                'content_url': format_host_url(chunk.content_url),
                'chunk_begin_digest': tokens[:6],
                'chunk_end_digest': tokens[-6:],
            }

            chunk_idx += 1

    knowledge_base = _content_divider.join(knowledge_base)

    return knowledge_base, refid2meta
