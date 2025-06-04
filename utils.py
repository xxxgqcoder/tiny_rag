import logging
import os
import traceback
from typing import Any
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


if __name__ == '__main__':
    print(get_project_base_directory())
