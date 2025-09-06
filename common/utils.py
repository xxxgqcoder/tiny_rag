import asyncio
import functools
import inspect
import logging
import os
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, TypeVar

import xxhash


def get_project_base_directory() -> str:
    project_base = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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

    handler1 = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=1)
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


T = TypeVar("T")


def _sync_run_once(func: Callable[..., T]) -> Callable[..., T]:
    """Thread-safe run_once for synchronous functions."""
    import threading

    _called = False
    _result = None
    _lock = threading.Lock()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal _called, _result

        if _called:
            return _result

        with _lock:
            if not _called:
                _result = func(*args, **kwargs)
                _called = True

        return _result

    return wrapper  # type: ignore


def _async_run_once(func: Callable[..., T]) -> Callable[..., T]:
    """Async-safe run_once for coroutine functions."""
    _called = False
    _result = None
    _lock = None

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        nonlocal _called, _result, _lock

        if _called:
            return _result

        # Initialize lock lazily to avoid event loop issues
        if _lock is None:
            _lock = asyncio.Lock()

        async with _lock:
            if not _called:
                _result = await func(*args, **kwargs)  # type: ignore
                _called = True

        return _result

    return wrapper  # type: ignore


def run_once(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that ensures a function runs only once, supporting both sync and async functions.

    For sync functions: Uses threading.Lock for thread safety
    For async functions: Uses asyncio.Lock for async safety

    Args:
        func: Function to wrap (sync or async)

    Returns:
        Wrapped function that executes only once
    """
    if inspect.iscoroutinefunction(func):
        return _async_run_once(func)  # type: ignore
    else:
        return _sync_run_once(func)


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


def time_it(prefix: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*kargs, **kwargs):
            begin = time.time_ns()
            ret = func(*kargs, **kwargs)
            elapse = (time.time_ns() - begin) // 1000000

            func_name = f"{prefix} {func.__name__}" if prefix else func.__name__
            logging.info(
                f"{func_name} took {elapse // 60000}min {(elapse % 60000) // 1000}sec {elapse % 60000 % 1000}ms to finish"
            )

            return ret

        return wrapper

    return decorator


def safe_encode(text: str) -> str:
    try:
        return text.encode(encoding="utf-8", errors="ignore").decode(encoding="utf-8", errors="ignore")
    except Exception as e:
        logging_exception(e)
        return ""
