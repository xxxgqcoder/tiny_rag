import time
import logging
import json

from flask import (
    jsonify,
    request,
    Blueprint,
    Response,
)

import config
from rag.llm import get_chat_model

bp = Blueprint('rag', __name__, url_prefix='/')


@bp.route('/chat_completion', methods=['POST'])
def chat_completion():
    """
    Return json object:
    - `code`: 0 for success.
    - `message`: error message if any.
    - `data`: data load. Empty data payload indicates end of generation.
        - `answer`: str, LLM generated answer.
        - `reference`: list, reference used for generating this answer.
    """
    logging.info(f'chat_completion: request={request}')
    logging.info(f'chat_completion: request.json={request.json}')

    req = request.json

    history = req["history"]
    logging.info(f"request {history}")

    model = get_chat_model()
    final_ans = ''

    def stream():
        nonlocal model, final_ans
        try:
            for ans in model.chat(
                    history=history,
                    gen_conf=config.OLLAMA_GEN_CONF,
            ):
                logging.info(f'chat_completion: ans = {ans}')
                if isinstance(ans, int):
                    break
                final_ans += ans

                yield json.dumps(
                    {
                        "code": 0,
                        "message": "",
                        "data": {
                            "answer": final_ans,
                            "reference": [],
                        }
                    },
                    ensure_ascii=False) + "\n\n"

        except Exception as e:
            yield json.dumps(
                {
                    "code": 500,
                    "message": str(e),
                    "data": {
                        "answer": "**ERROR**: " + str(e),
                        "reference": [],
                    },
                },
                ensure_ascii=False) + "\n\n"
        yield json.dumps({
            "code": 0,
            "message": "",
            "data": {},
        },
                         ensure_ascii=False) + "\n\n"

    resp = Response(stream(), mimetype="text/event-stream")
    resp.headers.add_header("Cache-control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")
    resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
    return resp
