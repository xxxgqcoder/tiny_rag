import sys
import json
import time
import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from prompt_toolkit import prompt

import config
from utils import logging_exception

chat_server_url = 'http://127.0.0.1:4567/search'
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
    global is_generating
    if user_input.startswith('uuid:'):
        uuids = user_input.split('uuid:')[-1].split()
        query = {
            'uuid': uuids,
        }
    else:
        query = {
            'question': user_input,
        }

    is_generating = True
    response = requests.post(
        url=chat_server_url,
        json={'query': query},
        stream=False,
    )
    is_generating = False

    response.raise_for_status()

    # print response
    print('', end='\r', flush=True)

    print(json.dumps(json.loads(response.text), indent=4))


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


def parse_user_instruct(user_input: str):
    user_input = user_input.strip()
    if len(user_input) == 0:
        return

    if user_input in ['?', '/help']:
        print("""Help info:
/help: show help info.
/exit: exit and save conversation as json.
""")
    elif user_input == '/exit':
        print('byte:)')
        os._exit(0)

    else:
        # talk to LLM
        generate_response(user_input)


def run_chat():
    global is_generating

    while True:
        try:
            # get user input
            is_generating = False
            user_input = prompt(">>", multiline=True, placeholder=input_placeholder)
            parse_user_instruct(user_input=user_input)

        except KeyboardInterrupt:
            os._exit(0)
        except EOFError:
            os._exit(0)


if __name__ == '__main__':
    is_generating = False

    job_executor = get_job_executor()
    job_executor.submit(print_loading_mark)

    try:
        run_chat()
    except Exception as e:
        logging_exception(e)
        os._exit(0)
