import logging
import os
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

import xxhash


def get_project_base_directory() -> str:
    project_base = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    return project_base


initialized_root_logger = False


def init_root_logger(
    logfile_basename: str,
    log_format: str = "%(asctime)-15s %(levelname)-4s %(filename)s:%(lineno)d: %(message)s",
    need_stream: bool = True,
) -> None:
    global initialized_root_logger
    if initialized_root_logger:
        return
    initialized_root_logger = True

    logger = logging.getLogger()
    logger.handlers.clear()
    log_path = os.path.abspath(os.path.join(get_project_base_directory(), "logs", f"{logfile_basename}.log"))

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    formatter = logging.Formatter(log_format)

    handler1 = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    if need_stream:
        handler2 = logging.StreamHandler()
        handler2.setFormatter(formatter)
        logger.addHandler(handler2)

    logger.setLevel(level=logging.INFO)
    logging.captureWarnings(True)


def safe_strip(d: Any) -> str:
    """
    Safely strip d.
    """
    if d is None:
        return ""
    if isinstance(d, str):
        return d.strip()
    return str(d).strip()


_instances = {}


def singleton(cls) -> Callable[..., Any]:
    def getinstance(*args, **kwargs) -> Any:
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    return getinstance


def run_once(func) -> Callable[..., Any | None]:
    has_run = False
    ret = None

    def wrapper(*args, **kwargs) -> Any | None:
        nonlocal has_run, ret
        if not has_run:
            has_run = True
            ret = func(*args, **kwargs)
        return ret

    return wrapper


def now_in_utc() -> str:
    now_utc = datetime.now(timezone.utc)
    return now_utc.strftime("%Y-%m-%d %H:%M:%S.%f")


def hash64(content: bytes) -> str:
    return xxhash.xxh64(content).hexdigest()


def logging_exception(e: Exception) -> None:
    logging.info(f"Exception: {type(e).__name__} - {e}")
    formatted_traceback = traceback.format_exc()
    logging.info(formatted_traceback)


def estimate_token_num(text: str) -> tuple[int, list[str]]:
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
    if text is None or len(text.strip()) == 0:
        return 0, []

    text = text.strip()

    def is_space(ch: str) -> bool:
        if ord(ch) >= 128:
            return False
        if ch.strip() == "":
            return True
        return False

    def token_bound_found(text: str, i: int, j: int) -> bool:
        if ord(text[i]) < 127:
            # space met or non-ascii character met
            return is_space(text[j]) or ord(text[j]) > 127
        else:
            # count one non-ascii character as one token
            return j > i

    token_buffer = []
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and not token_bound_found(text, i, j):
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


def time_it(func):
    def wrapper(*kargs, **kwargs):
        begin = time.time_ns()
        ret = func(*kargs, **kwargs)
        elapse = (time.time_ns() - begin) // 1000000
        logging.info(
            f"func {func.__name__} took {elapse // 60000}min {(elapse % 60000) // 1000}sec {elapse % 60000 % 1000}ms to finish"
        )

        return ret

    return wrapper
