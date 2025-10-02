import logging
import os
from logging.handlers import RotatingFileHandler

_loggers = {}


def get_logger(
    log_module_name: str = "",
    log_format: str = "%(asctime)-15s %(levelname)-4s %(filename)s:%(lineno)d: %(message)s",
    need_stream: bool = True,
):
    """
    Get logger for a specific module.

    Args:
    - log_module_name: Name of the module for which to get the logger. If empty.

    Returns:
    - Logger instance.
    """
    logger_key = f"{log_module_name}_{need_stream}"
    if not logger_key:
        logger_key = "default"
    if logger_key in _loggers:
        return _loggers[logger_key]

    project_root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    logger = logging.getLogger(name=logger_key)
    logger.handlers.clear()
    log_path = os.path.abspath(os.path.join(project_root_dir, "logs", f"{log_module_name}.log"))

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    formatter = logging.Formatter(log_format)

    handler1 = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=1)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    if need_stream:
        handler2 = logging.StreamHandler()
        handler2.setFormatter(formatter)
        logger.addHandler(handler2)
    else:
        logger.propagate = False

    logger.setLevel(level=logging.INFO)
    logging.captureWarnings(True)

    _loggers[logger_key] = logger
    return logger


Logger = get_logger()
