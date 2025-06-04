import sys
import json
import time

import requests
from prompt_toolkit import prompt
from concurrent.futures import ThreadPoolExecutor

import config
from utils import logging_exception

# current ongoing conversation
conversation = []

chat_server_url = 'http://127.0.0.1:4567/chat_completion'
input_placeholder = '<type <esc> then <enter> to finish input>'

job_executor = None
is_generating = False


def get_job_executor():
    global job_executor
    if job_executor is None:
        # NOTE: set only 1 thread to force sequencial job schedule.
        job_executor = ThreadPoolExecutor(max_workers=1)

    return job_executor


def generate_response(user_input: str) -> requests.models.Response:
    user_input = user_input.strip()
    history = []
    history.append({
        'role': 'user',
        'content': user_input,
    })

    response = requests.post(
        url=chat_server_url,
        json={'history': history},
        stream=True,
    )
    return response


def print_response(response: requests.models.Response) -> None:
    last_ans = ""
    json_buffer = ""
    for chunk in response.iter_content(
            chunk_size=8192,
            decode_unicode=True,
    ):
        if not chunk:
            continue
        json_buffer += chunk
        try:
            ret = json.loads(json_buffer)
            cur_ans = ret['data']
            if not isinstance(ret['data'], str):
                break

            print(
                cur_ans[len(last_ans):],
                end='',
                flush=True,
            )

            json_buffer = ""
            last_ans = cur_ans
        except json.JSONDecodeError:
            continue
    print()




def print_loading_mark():
    global is_generating

    loading_mark = ['-', '\\', '|', '/']
    idx = 0
    while True:
        if is_generating:
            ch = loading_mark[idx % len(loading_mark)]
            idx = (idx + 1) % len(loading_mark)
            print(ch, end='\r', flush=True)

        time.sleep(0.05)


def run_chat():
    global is_generating

    while True:
        try:
            is_generating = False
            user_input = prompt(
                ">>",
                multiline=True,
                placeholder=input_placeholder,
            )
            user_input = user_input.strip()
            if len(user_input) == 0:
                continue

            is_generating = True
            response = generate_response(user_input=user_input)
            response.raise_for_status()

            is_generating = False
            print('', end='\r', flush=True)

            print_response(response=response)

        except Exception as e:
            logging_exception(e)
            pass


if __name__ == '__main__':
    is_generating = False

    job_executor = get_job_executor()
    job_executor.submit(print_loading_mark)

    try:
        run_chat()
    except Exception as e:
        sys.exit(0)
