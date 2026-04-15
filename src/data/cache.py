import time

class SimpleCache:
    """A very simple in-memory cache with TTL."""
    def __init__(self):
        self._cache = {}

    def get(self, key):
        if key in self._cache:
            item = self._cache[key]
            if time.time() < item['expires_at']:
                return item['value']
            else:
                del self._cache[key]
        return None

    def set(self, key, value, ttl_seconds):
        self._cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl_seconds
        }

    def delete(self, key):
        self._cache.pop(key, None)

    def clear(self):
        self._cache = {}

cache = SimpleCache()
