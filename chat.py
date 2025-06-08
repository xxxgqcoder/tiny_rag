import sys
import json
import time
import datetime
import os
import re
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

import requests
from prompt_toolkit import prompt
from strenum import StrEnum

import config
from utils import logging_exception
from utils import singleton


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# current ongoing conversation
conversation = {}

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


def generate_response() -> requests.models.Response:
    global conversation, is_generating

    reduced_conv = {}
    reduced_conv['history'] = [{
        'role': history['role'],
        'content': history['content'],
    } for history in conversation['history']]

    is_generating = True
    response = requests.post(
        url=chat_server_url,
        json=reduced_conv,
        stream=True,
    )
    is_generating = False

    response.raise_for_status()

    # print response
    print('', end='\r', flush=True)

    last_ans = ""
    json_buffer = ""
    context_prompt = ''
    reference_meta = None
    prompt_token_num = 0
    answer_token_num = 0
    try:
        for chunk in response.iter_content(
                chunk_size=8192,
                decode_unicode=True,
        ):
            if not chunk:
                continue
            json_buffer += chunk
            try:
                ret = json.loads(json_buffer)
                data = ret.get('data', {})
                if len(data) == 0:
                    break
                if not context_prompt:
                    context_prompt = data['prompt']
                if not reference_meta:
                    reference_meta = data['reference_meta']
                    
                answer_token_num = data['answer_token_num']
                prompt_token_num = data['prompt_token_num']

                print(data['answer'][len(last_ans):], end='', flush=True)

                json_buffer = ""
                last_ans = data['answer']
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logging_exception(e)
        return

    conversation['history'].append({
        'role': 'assistant',
        'content': last_ans,
        'prompt': context_prompt,
        'reference_meta': reference_meta,
        'answer_token_num': answer_token_num,
        'prompt_token_num': prompt_token_num,
    })
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
        os.makedirs(os.path.dirname(config.CONVERSATION_SAVE_PATH),
                    exist_ok=True)

        if len(conversation['history']) >= 2:
            with open(config.CONVERSATION_SAVE_PATH, 'w+') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=4)
                print(f'save conversation to {config.CONVERSATION_SAVE_PATH}')

        print('byte:)')
        os._exit(0)

    else:
        # talk to LLM
        conversation['history'].append({'role': 'user', 'content': user_input})

        generate_response()


def run_chat():
    global is_generating

    while True:
        try:
            # get user input
            is_generating = False
            user_input = prompt(">>",
                                multiline=True,
                                placeholder=input_placeholder)
            parse_user_instruct(user_input=user_input)

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    is_generating = False

    job_executor = get_job_executor()
    job_executor.submit(print_loading_mark)

    if 'history' not in conversation:
        conversation['history'] = []

    try:
        run_chat()
    except Exception as e:
        logging_exception(e)
        os._exit(0)
