import logging
import os
import traceback
from typing import Any, Tuple
from logging.handlers import RotatingFileHandler

import xxhash

initialized_root_logger = False


def get_project_base_directory():
    project_base = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    return project_base


def init_root_logger(
    logfile_basename: str,
    log_format: str = "%(asctime)-15s %(levelname)-8s %(process)d %(message)s",
):
    global initialized_root_logger
    if initialized_root_logger:
        return
    initialized_root_logger = True

    logger = logging.getLogger()
    logger.handlers.clear()
    log_path = os.path.abspath(
        os.path.join(get_project_base_directory(), "logs",
                     f"{logfile_basename}.log"))

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    formatter = logging.Formatter(log_format)

    handler1 = RotatingFileHandler(log_path,
                                   maxBytes=10 * 1024 * 1024,
                                   backupCount=5)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)

    logging.captureWarnings(True)

    LOG_LEVELS = os.environ.get("LOG_LEVELS", "")
    pkg_levels = {}
    for pkg_name_level in LOG_LEVELS.split(","):
        terms = pkg_name_level.split("=")
        if len(terms) != 2:
            continue
        pkg_name, pkg_level = terms[0], terms[1]
        pkg_name = pkg_name.strip()
        pkg_level = logging.getLevelName(pkg_level.strip().upper())
        if not isinstance(pkg_level, int):
            pkg_level = logging.INFO
        pkg_levels[pkg_name] = logging.getLevelName(pkg_level)

    for pkg_name in ['peewee', 'pdfminer']:
        if pkg_name not in pkg_levels:
            pkg_levels[pkg_name] = logging.getLevelName(logging.WARNING)
    if 'root' not in pkg_levels:
        pkg_levels['root'] = logging.getLevelName(logging.INFO)

    for pkg_name, pkg_level in pkg_levels.items():
        pkg_logger = logging.getLogger(pkg_name)
        pkg_logger.setLevel(pkg_level)

    msg = f"{logfile_basename} log path: {log_path}, log levels: {pkg_levels}"
    logger.info(msg)


def safe_strip(d: Any) -> str:
    """
    Safely strip d.
    """
    if d is None:
        return ''
    if isinstance(d, str):
        return d.strip()
    return str(d).strip()


def singleton(cls):
    instances = {}

    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return getinstance


def run_once(func):
    has_run = False
    ret = None

    def wrapper(*args, **kwargs):
        nonlocal has_run, ret
        if not has_run:
            has_run = True
            ret = func(*args, **kwargs)
        return ret

    return wrapper


def now_in_utc():
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc)
    return now_utc.strftime('%Y-%m-%d %H:%M:%S.%f')


def get_hash64(content: bytes) -> str:
    return xxhash.xxh64(content).hexdigest()


def logging_exception(e: Exception):
    logging.info(f"Exception: {type(e).__name__} - {e}")
    formatted_traceback = traceback.format_exc()
    logging.info(formatted_traceback)


from typing import Tuple


def estimate_token_num(text: str) -> Tuple[int, list[str]]:
    """
    Estimate tokens in text. Combine consecutive ascii character as one token,
    treat each non-ascii character as one token. Each ascii token accounts for 2.3
    token, each non-ascii token accounts for 1.2 token.

    Args:
    - text: the string to parse.

    Return:
    - int, estimated token num.
    - list of string, estimated tokens.
    """

    def is_space(ch: str) -> bool:
        if ord(ch) >= 128:
            return False
        if ch.strip() == '':
            return True
        return False

    def find_token_bound(text: str, i: int, j: int) -> bool:
        if ord(text[i]) < 127:
            # space met or non-ascii character met
            return (is_space(text[j]) or ord(text[j]) > 127)
        else:
            # count one non-ascii character as one token
            return j > i

    token_buffer = []
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and not find_token_bound(text, i, j):
            j += 1

        token = text[i:j]
        token_buffer.append(token)

        i = j
        while i < len(text) and is_space(text[i]):
            i += 1

    token_num = 0
    for token in token_buffer:
        if ord(token[0]) < 128:
            token_num += 2.3
        else:
            token_num += 1.2

    return int(token_num), token_buffer


if __name__ == '__main__':
    print(get_project_base_directory())
