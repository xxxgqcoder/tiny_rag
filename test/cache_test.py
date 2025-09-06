import os
import sys
import time
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print(sys.path[-1])


from common.cache import get_cache_db
from common.config import CacheConfig, TinyRAGConfig


class TestCache(unittest.TestCase):
    def test_base(self):
        conn_url = "redis://localhost:6379/0"
        token = ""
        key_ttl = 5

        key = "test_key"
        value = b"test_value"

        TinyRAGConfig.cache_config = CacheConfig(  # type: ignore
            conn_url=conn_url,
            token=token,
            key_ttl_seconds=key_ttl,
        )

        # get cache
        cache = get_cache_db()

        # put key
        ret = cache.put(key, value)
        self.assertEqual(ret, len(value))

        # get key
        ret = cache.get(key)
        self.assertEqual(ret, value)

        # wait for key expire
        time.sleep(key_ttl + 1)
        ret = cache.get(key)
        self.assertEqual(len(ret), 0)


if __name__ == "__main__":
    unittest.main()
