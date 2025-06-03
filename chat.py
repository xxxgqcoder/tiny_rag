import sys
import json
import logging
import time
from datetime import datetime

import aiohttp
import asyncio
import requests
from prompt_toolkit import prompt

import config
from utils import logging_exception

# current ongoing conversation
conversation = []

chat_server_url = 'http://127.0.0.1:4567/chat_completion'
prompt_placeholder = '<input your question. type <esc> then <enter> to finish>'


def generate_response(user_input: str) -> str:
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


# async def get_response(user_input: str) -> None:
#     async with aiohttp.ClientSession() as session:
#         history = [{
#             'role': 'user',
#                     'content': user_input,
#                 }]
#         task = asyncio.create_task(session.post(
#                         url=chat_server_url,
#                         json={'history': history},
#                     ))

#         while not task.done():
#             print("*", end='r')
#             time.sleep(0.1)

#         response = await task
#         try:
#             for chunk in response.iter_content(chunk_size=8192):
#                 # response.iter_content return bytes
#                 chunk = json.loads(chunk)
#                 if not isinstance(chunk['data'], str):
#                     break

#                 print(chunk['data'].encode('utf-8'), end='\r')

#         except Exception as e:
#             print(f"Error reading response: {e}")


def run_chat():
    while True:
        try:
            user_input = prompt(
                ">>",
                multiline=True,
                placeholder=prompt_placeholder,
                mouse_support=True,
            )
            user_input = user_input.strip()

            begin = datetime.now()
            response = generate_response(user_input=user_input)
            response.raise_for_status()
            end = datetime.now()

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
            print(f'total genertion time: {end - begin}')

        except Exception as e:
            logging_exception(e)
            break


if __name__ == '__main__':
    try:
        run_chat()
    except Exception as e:
        sys.exit(0)
