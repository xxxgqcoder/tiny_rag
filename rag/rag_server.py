import time
import logging
import json
import re
from typing import Tuple, Dict, Any

from flask import (
    request,
    Blueprint,
    Response,
)

import config
from .llm import get_chat_model
from .db import get_vector_db
from parse.parser import Chunk
from utils import estimate_token_num

bp = Blueprint('rag', __name__, url_prefix='/')

_prompt_system = """
you are a knowledge assistance, please use below knowledge to answer user questions.
If user questions are not included in knowledge, you must reply with "not found in knowledgebase".


Below is the knowledge base


{knowledge_base}


Above is the knowledge base
"""

_promot_citation = """
# Citation requirements:
- Inserts CITATIONS in format '##i@@ ##j@@' where i,j are the ID of the content you are citing and encapsulated with '##' and '@@'.
- Inserts the CITATION symbols at the end of a sentence, AND NO MORE than 4 citations.
- DO NOT insert CITATION in the answer if the content is not from retrieved chunks.
- DO NOT use standalone Document IDs (e.g., '#ID#').
- Under NO circumstances any other citation styles or formats (e.g., '~~i==', '[i]', '(i)', etc.) be used.
- Citations ALWAYS the '##i@@' format.
- Any failure to adhere to the above rules, including but not limited to incorrect formatting, use of prohibited styles, or unsupported citations, will be considered a error, should skip adding Citation for this sentence.

--- Example START ---
<SYSTEM>: Here is the knowledge base:

Document: Elon Musk Breaks Silence on Crypto, Warns Against Dogecoin ...
URL: https://blockworks.co/news/elon-musk-crypto-dogecoin
ID: 0
The Tesla co-founder advised against going all-in on dogecoin, but Elon Musk said it’s still his favorite crypto...

Document: Elon Musk's Dogecoin tweet sparks social media frenzy
ID: 1
Musk said he is 'willing to serve' D.O.G.E. – shorthand for Dogecoin.

Document: Causal effect of Elon Musk tweets on Dogecoin price
ID: 2
If you think of Dogecoin — the cryptocurrency based on a meme — you can’t help but also think of Elon Musk...

Document: Elon Musk's Tweet Ignites Dogecoin's Future In Public Services
ID: 3
The market is heating up after Elon Musk's announcement about Dogecoin. Is this a new era for crypto?...

      The above is the knowledge base.

<USER>: What's the Elon's view on dogecoin?

<ASSISTANT>: Musk has consistently expressed his fondness for Dogecoin, often citing its humor and the inclusion of dogs in its branding. He has referred to it as his favorite cryptocurrency ##0@@ ##1@@.
Recently, Musk has hinted at potential future roles for Dogecoin. His tweets have sparked speculation about Dogecoin's potential integration into public services ##3@@.
Overall, while Musk enjoys Dogecoin and often promotes it, he also warns against over-investing in it, reflecting both his personal amusement and caution regarding its speculative nature.

--- Example END ---
"""

_content_divider = "\n\n"


def assemble_knowledge_base(chunks: list[Chunk]) -> Tuple[str, Dict[str, Any]]:
    """
    Assemble knowledge in chunk and return formatted knowledge base.
    """
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
        knowledge_base.append(
            f'Relevant fragments as following:{_content_divider}')
        for chunk in chunks:
            if chunk.content_type in [config.ChunkType.TEXT]:
                knowledge_base.append(
                    f"ID:{chunk_idx}\n{chunk.content.decode('utf-8')}")
            else:
                knowledge_base.append(
                    f"ID:{chunk_idx}\n{chunk.extra_description.decode('utf-8')}"
                )

            refid2meta[chunk_idx] = {
                'uuid': chunk.uuid,
                'file_name': chunk.file_name,
                'content_type': chunk.content_type,
                'content_url': chunk.content_url,
            }

            chunk_idx += 1

    knowledge_base = _content_divider.join(knowledge_base)

    return knowledge_base, refid2meta


@bp.route('/chat_completion', methods=['POST'])
def chat_completion():
    """
    Input json:
    - `history`: chat history json ojbect:
        - `role`: one of `user` / `assistant` / `system`. `user` represents user input,
            `assistant` represents LLM response, `system` represents system context
            setting.
        - `content`: the chat message.


    Output json:
    - `code`: 0 for success.
    - `message`: error message if any.
    - `data`: data load. Empty data payload indicates end of generation.
        - `answer`: str, LLM generated answer.
        - `prompt`: str, prompt used to generate the answer.
        - `reference_meta`: dict, reference id to meta info.
        
    Return object is generated in incremental way, each returned object has
        newly generated token appended to previous returned answer.
    """
    logging.info(f'**DEBUG** chat_completion: request={request}')
    logging.info(
        f'**DEBUG** chat_completion: request.json={json.dumps(request.json, indent=4, ensure_ascii=False)}'
    )

    req = request.json
    history = req["history"]
    logging.info(
        f"**DEBUG** chat_completion: history={json.dumps(history, indent=4, ensure_ascii=False)}"
    )

    model = get_chat_model()
    vector_db = get_vector_db()

    message = [{
        'role': m['role'],
        'content': m['content']
    } for m in history if m['role'] != 'system']

    user_questions = [m['content'] for m in message
                      if m['role'] == 'user'][-3:]
    chunks = []
    for question in user_questions:
        ret = vector_db.search(query=question, params={'limit': 4})
        chunks.extend(ret)

    knowledge_base, refid2meta = assemble_knowledge_base(chunks)

    logging.info(
        f'**DEBUG** chat_completion, knowledge_base = \n{knowledge_base}')
    logging.info('=' * 120)

    prompt = _prompt_system.format(knowledge_base=knowledge_base) \
                + f'---- {_content_divider} ' \
                + _promot_citation

    message.insert(0, {'role': 'system', 'content': prompt})

    final_ans = ''

    def stream():
        nonlocal model, final_ans
        try:
            for ans in model.chat(history=message,
                                  gen_conf=config.CHAT_GEN_CONF):
                if isinstance(ans, int):
                    break
                # append to previous ans
                final_ans += ans

                yield json.dumps(
                    {
                        "code": 0,
                        "message": "",
                        "data": {
                            "answer": final_ans,
                            "reference_meta": refid2meta,
                            'prompt': prompt,
                            'prompt_token_num': estimate_token_num(prompt)[0],
                            'answer_token_num': estimate_token_num(final_ans)[0],
                        }
                    },
                    ensure_ascii=False) + _content_divider

        except Exception as e:
            yield json.dumps(
                {
                    "code": 500,
                    "message": str(e),
                    "data": {
                        "answer": "**ERROR**: " + str(e),
                        "reference_meta": [],
                        "prompt": prompt,
                    }
                },
                ensure_ascii=False) + _content_divider
        yield json.dumps({
            "code": 0,
            "message": "",
            "data": {}
        },
                         ensure_ascii=False) + _content_divider

    resp = Response(stream(), mimetype="text/event-stream")
    resp.headers.add_header("Cache-control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")
    resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
    return resp
