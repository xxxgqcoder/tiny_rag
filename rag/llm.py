import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Generator, Union

from numpy import single
from ollama import Client as OllamaClient

from common.cache import cache_it, logging_exception
from common.config import TinyRAGConfig
from common.utils import hash64, singleton, time_it


class ChatModel(ABC):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def chat(
        self,
        history: list[dict[str, Any]],
        gen_conf: dict[str, Any] = {},
    ) -> Generator[Union[str, int], Any, Any]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def instant_chat(
        self,
        prompt: str,
        gen_conf: dict[str, Any] = {},
    ) -> str:
        raise NotImplementedError("Not implemented")


class VisionModel(ABC):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def image_chat(
        self,
        prompt: str,
        image_content: str,
        gen_conf: dict[str, Any] = {},
    ) -> str:
        """
        Args:
        - prompt: prompt to guide image to text generation.
        - image_content: base64 encoded image content.
        - gen_conf: generation configuration.

        Returns:
        - generated text.
        """
        raise NotImplementedError("Not implemented")


def _ollama_options(gen_conf: dict[str, Any]) -> dict[str, Any]:
    if not gen_conf:
        gen_conf = TinyRAGConfig.gen_conf.model_dump()

    if "max_tokens" in gen_conf:
        del gen_conf["max_tokens"]

    options = {}
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
    if "repeat_penalty" in gen_conf:
        options["repeat_penalty"] = gen_conf["repeat_penalty"]
    if "num_ctx" in gen_conf:
        options["num_ctx"] = gen_conf["num_ctx"]

    return options


# implementation
@singleton
class OllamaChat(ChatModel):
    def __init__(self, ollama_host: str, ollama_model: str):
        self.client = OllamaClient(
            host=ollama_host,
            timeout=15 * 60,  # time out 15 min
        )
        self.model_name = ollama_model

    def chat(
        self,
        history: list[dict[str, Any]],
        gen_conf: dict[str, Any] = {},
    ) -> Generator[Union[str, int], Any, Any]:
        options = _ollama_options(gen_conf)
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

    def key_generator(self, prompt: str) -> str:
        return f"ollama_chat::prompt_hash::{hash64(prompt.encode('utf-8', errors='ignore'))}"

    @time_it(prefix="llm instant chat")
    @cache_it(key_generator=key_generator)
    def instant_chat(
        self,
        prompt: str,
        gen_conf: dict[str, Any] = {},
    ) -> str:
        options = _ollama_options(gen_conf)

        history = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat(model=self.model_name, messages=history, options=options, keep_alive=10)
        except Exception as e:
            return f"Exception: {e}"

        ans = response["message"]["content"].strip()
        if "</think>" in ans:
            ans = ans.split("</think>")[-1]

        if "</thinking>" in ans:
            ans = ans.split("</thinking>")[-1]
        return ans.strip()


def get_chat_model() -> ChatModel:
    return OllamaChat(
        ollama_host=TinyRAGConfig.ollama_host,
        ollama_model=TinyRAGConfig.ollama_model,
    )


@singleton
class OllamaVisionModel(VisionModel):
    def __init__(self, ollama_host: str, ollama_model: str):
        self.client = OllamaClient(
            host=ollama_host,
            timeout=15 * 60,  # time out 15 min
        )
        self.model_name = ollama_model

    def key_generator(self, prompt: str, image_content: str) -> str:
        content = prompt + image_content
        return f"ollama_vision::prompt_hash::{hash64(content.encode('utf-8', errors='ignore'))}"

    @time_it(prefix="llm image chat")
    @cache_it(key_generator=key_generator, key_ttl_seconds=25 * 60 * 60 * 100)
    def image_chat(
        self,
        prompt: str,
        image_content: str,
        gen_conf: dict[str, Any] = {},
    ) -> str:
        options = _ollama_options(gen_conf)
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt, "images": [image_content]}],
            options=options,
        )
        try:
            return response["message"]["content"]
        except Exception as e:
            logging_exception(e)

        return ""


def get_vision_model() -> VisionModel:
    return OllamaVisionModel(
        ollama_host=TinyRAGConfig.ollama_host,
        ollama_model=TinyRAGConfig.vision_medel,
    )


# def assemble_knowledge_base(chunks: list[Chunk]) -> Tuple[str, Dict[str, Any]]:
#     """
#     Assemble knowledge in chunk and return formatted knowledge base.

#     Args:
#     - chunks: chunks used to format knowledge base.

#     Returns:
#     - knowledge base: formated knowledge base.
#     - A dict of reference id to chunk meta, key is reference id within current knowledge base.
#     """
#     _content_divider = "\n\n"
#     # dedup chunks
#     deduped_chunk = {}
#     for chunk in chunks:
#         if chunk.uuid not in deduped_chunk:
#             deduped_chunk[chunk.uuid] = chunk

#     # document name to chunks mapping
#     document2chunks = {}
#     for _, chunk in deduped_chunk.items():
#         if chunk.file_name not in document2chunks:
#             document2chunks[chunk.file_name] = []
#         document2chunks[chunk.file_name].append(chunk)

#     knowledge_base = []
#     chunk_idx = 0  # reference index within current knowledge base
#     refid2meta = {}  # reference index to chunk meta info
#     for file_name, chunks in document2chunks.items():
#         knowledge_base.append(f"Document: {file_name}{_content_divider}")
#         knowledge_base.append(f"Relevant fragments as following:{_content_divider}")
#         for chunk in chunks:
#             content = ""
#             if chunk.content_type in [ChunkType.TEXT]:
#                 content = chunk.content.decode("utf-8")
#             else:
#                 content = chunk.extra_description.decode("utf-8")

#             # trim llm summary if any
#             content = re.sub(r"<summary>[.\s\S]*</summary>", "", content)
#             knowledge_base.append(f"ID:{chunk_idx}\n{content}")

#             tokens = estimate_token_num(content)[-1]

#             refid2meta[str(chunk_idx)] = {
#                 "uuid": chunk.uuid,
#                 "file_name": chunk.file_name,
#                 "content_type": chunk.content_type,
#                 "content_url": format_host_url(chunk.content_url),
#                 "chunk_begin_digest": tokens[:6],
#                 "chunk_end_digest": tokens[-6:],
#             }

#             chunk_idx += 1

#     knowledge_base = _content_divider.join(knowledge_base)

#     return knowledge_base, refid2meta


# def format_reference_info(reference_meta: Dict[str, str], answer: str) -> str:
#     """
#     Fomat reference data in `answer`.

#     Args:
#     - reference_meta: reference meta info, key is reference id.
#     - answer: generated answer containing reference id.

#     Returns:
#     - formatted reference info.
#     """
#     formatted_reference_info = "\n\n"

#     answer = re.sub(r"<think>[.\S\s]*</think>", "", answer)

#     reference = re.findall(r"##[0-9]+@@", answer)
#     ref_ids = {}
#     for ref in reference:
#         ref_id = ref.strip("##").strip("@@")
#         ref_ids[ref_id] = True

#     for ref_id in sorted([ref_id for ref_id in ref_ids]):
#         ref_info = ""
#         meta = reference_meta[ref_id]
#         ref_info += f"<reference ID={ref_id}>,"
#         ref_info += "file=" + meta["file_name"] + ","
#         if meta["content_url"]:
#             ref_info += "url=" + meta["content_url"] + ","

#         chunk_begin_digest = " ".join(meta["chunk_begin_digest"])
#         chunk_end_digest = " ".join(meta["chunk_end_digest"])
#         ref_info += "ref content=" + f"{chunk_begin_digest} ... {chunk_end_digest}"
#         formatted_reference_info += ref_info + "\n\n"

#     return formatted_reference_info
