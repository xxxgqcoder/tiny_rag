import sys

import config
from rag.llm import get_chat_model
from utils import logging_exception


def generate_response(user_input: str) -> str:
    model = get_chat_model()

    history = []
    history.append({
        'role': 'user',
        'content': user_input,
    })

    ret, tok_cnt = model.chat(
        history=history,
        gen_conf=config.OLLAMA_GEN_CONF,
    )
    return ret


def run_chat():
    while True:
        try:
            user_input = input(">>>")
            response = generate_response(user_input)
            print(response)
        except Exception as e:
            logging_exception(e)
            break


if __name__ == '__main__':
    try:
        run_chat()
    except Exception as e:
        sys.exit(0)
