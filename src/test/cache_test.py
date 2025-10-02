import os
import sys
import time
import pytest
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.cache import get_cache_db, cache_it, RedisCache
from common.config import CacheConfig, TinyRAGConfig


@pytest.fixture(scope="function")
def setup_cache():
    """Setup cache configuration before each test"""
    conn_url = "redis://localhost:23456/0"
    token = ""
    key_ttl = 5

    TinyRAGConfig.cache_config = CacheConfig(
        conn_url=conn_url,
        token=token,
        key_ttl_seconds=key_ttl,
    )
    
    cache = get_cache_db()
    yield cache
    
    # Cleanup: flush test keys
    try:
        cache.client.flushdb() # type: ignore
    except:
        pass


class TestCacheBasic:
    def test_put_and_get(self, setup_cache):
        """Test basic put and get operations"""
        cache = setup_cache
        key = "test_key"
        value = b"test_value"

        ret = cache.put(key, value, -1)
        assert ret == len(value)

        ret = cache.get(key)
        assert ret == value

    def test_key_expiration(self, setup_cache):
        """Test key expiration with TTL"""
        cache = setup_cache
        key = "test_expire_key"
        value = b"test_expire_value"
        key_ttl = 2

        cache.put(key, value, key_ttl)
        
        # Key should exist immediately
        ret = cache.get(key)
        assert ret == value

        # Wait for key to expire
        time.sleep(key_ttl + 1)
        ret = cache.get(key)
        assert len(ret) == 0

    def test_get_nonexistent_key(self, setup_cache):
        """Test getting a key that doesn't exist"""
        cache = setup_cache
        ret = cache.get("nonexistent_key")
        assert ret == b""

    def test_overwrite_existing_key(self, setup_cache):
        """Test overwriting an existing key"""
        cache = setup_cache
        key = "test_overwrite"
        value1 = b"first_value"
        value2 = b"second_value"

        cache.put(key, value1, -1)
        assert cache.get(key) == value1

        cache.put(key, value2, -1)
        assert cache.get(key) == value2

    def test_multiple_keys(self, setup_cache):
        """Test storing and retrieving multiple keys"""
        cache = setup_cache
        keys_values = {
            "key1": b"value1",
            "key2": b"value2",
            "key3": b"value3",
        }

        for key, value in keys_values.items():
            cache.put(key, value, -1)

        for key, expected_value in keys_values.items():
            assert cache.get(key) == expected_value

    def test_empty_value(self, setup_cache):
        """Test storing empty bytes"""
        cache = setup_cache
        key = "empty_key"
        value = b""

        ret = cache.put(key, value, -1)
        assert ret == 0

        ret = cache.get(key)
        assert ret == value

    def test_large_value(self, setup_cache):
        """Test storing large binary data"""
        cache = setup_cache
        key = "large_key"
        value = b"x" * 10000  # 10KB

        ret = cache.put(key, value, -1)
        assert ret == len(value)

        ret = cache.get(key)
        assert ret == value
        assert len(ret) == 10000


class TestCacheDecorator:
    def test_cache_it_decorator_basic(self, setup_cache):
        """Test cache_it decorator caches function results"""
        call_count = 0

        def key_gen(*args, **kwargs):
            return "test_decorator_key"

        @cache_it(key_generator=key_gen, key_ttl_seconds=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

    def test_cache_it_decorator_with_different_args(self, setup_cache):
        """Test cache_it decorator with different arguments"""
        call_count = 0

        def key_gen(*args, **kwargs):
            return f"test_key_{args[0]}"

        @cache_it(key_generator=key_gen, key_ttl_seconds=60)
        def add_ten(x):
            nonlocal call_count
            call_count += 1
            return x + 10

        result1 = add_ten(5)
        assert result1 == 15
        assert call_count == 1

        result2 = add_ten(10)
        assert result2 == 20
        assert call_count == 2

        # Cached result for 5
        result3 = add_ten(5)
        assert result3 == 15
        assert call_count == 2

    def test_cache_it_decorator_expiration(self, setup_cache):
        """Test cache_it decorator respects TTL"""
        call_count = 0

        def key_gen(*args, **kwargs):
            return "test_ttl_key"

        @cache_it(key_generator=key_gen, key_ttl_seconds=2)
        def get_value():
            nonlocal call_count
            call_count += 1
            return "computed_value"

        result1 = get_value()
        assert result1 == "computed_value"
        assert call_count == 1

        # Should use cache
        result2 = get_value()
        assert result2 == "computed_value"
        assert call_count == 1

        # Wait for expiration
        time.sleep(3)
        result3 = get_value()
        assert result3 == "computed_value"
        assert call_count == 2

    def test_cache_it_decorator_with_complex_return(self, setup_cache):
        """Test cache_it decorator with complex return types"""
        call_count = 0

        def key_gen(*args, **kwargs):
            return "complex_return_key"

        @cache_it(key_generator=key_gen, key_ttl_seconds=60)
        def return_dict():
            nonlocal call_count
            call_count += 1
            return {"name": "test", "value": 42, "items": [1, 2, 3]}

        result1 = return_dict()
        assert result1 == {"name": "test", "value": 42, "items": [1, 2, 3]}
        assert call_count == 1

        result2 = return_dict()
        assert result2 == {"name": "test", "value": 42, "items": [1, 2, 3]}
        assert call_count == 1


class TestRedisCacheSingleton:
    def test_singleton_pattern(self, setup_cache):
        """Test RedisCache follows singleton pattern"""
        cache1 = get_cache_db()
        cache2 = get_cache_db()
        assert cache1 is cache2

    def test_redis_connection(self, setup_cache):
        """Test Redis connection is established"""
        cache = setup_cache
        assert cache.client is not None
        assert cache.client.ping() is True
