import functools
import io
import logging
import os
import pickle
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

import redis

from common.config import TinyRAGConfig
from common.utils import logging_exception, singleton, time_it


class CacheDB(ABC):
    @abstractmethod
    def get(self, key: str) -> bytes:
        """
        Get an cached bytes.

        Returns:
        - The object bytes.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def put(self, key: str, obj: bytes) -> int:
        """
        Put an cache bytes.

        Returns:
        - An int indicating how many bytes are put.
        """
        raise NotImplementedError("Not implemented")


@singleton
class RedisCache(CacheDB):
    def __init__(self, conn_url: str, token: str = "", key_ttl_seconds: int = 12 * 60 * 60, **kwargs):
        """
        Redis cache.

        Args:
        - conn_url: connection url.
        - token: connection token.
        - kwargs: not used.
        """
        super().__init__()
        self.key_ttl_seconds = key_ttl_seconds
        self.conn_url = conn_url
        self.token = token
        self.client = redis.Redis.from_url(conn_url, password=token)  # type: ignore
        try:
            self.client.ping()
        except redis.ConnectionError as e:
            logging_exception(e)
            raise e

    @time_it(prefix="redis")
    def get(self, key: str) -> bytes:
        ret = self.client.get(key)
        if ret is None:
            return b""
        return ret  # type: ignore

    @time_it(prefix="redis")
    def put(self, key: str, obj: bytes) -> int:
        ret = self.client.setex(
            name=key,
            time=self.key_ttl_seconds,
            value=obj,
        )
        if not ret:
            return 0
        return len(obj)


def get_cache_db() -> CacheDB:
    return RedisCache(
        conn_url=TinyRAGConfig.cache_config.conn_url,  # type: ignore
        token=TinyRAGConfig.cache_config.token,  # type: ignore
        key_ttl_seconds=TinyRAGConfig.cache_config.key_ttl_seconds,  # type: ignore
    )


T = TypeVar("T")


def cache_it(key_generator: Callable[..., str]) -> Callable[..., Callable[..., T]]:
    """
    Redis cache decorator with customized key generator.

    Args:
        key_generator: Function that takes the same args as decorated function and returns cache key
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = key_generator(*args, **kwargs)
            redis_cache_db = get_cache_db()
            cached_data = redis_cache_db.get(cache_key)
            if cached_data:
                try:
                    return pickle.loads(cached_data)
                except:
                    pass

            result = func(*args, **kwargs)
            try:
                serialized_result = pickle.dumps(result)
                redis_cache_db.put(cache_key, serialized_result)
            except:
                pass
            return result

        return wrapper

    return decorator
