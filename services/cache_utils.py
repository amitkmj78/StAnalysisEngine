"""
Framework-agnostic TTL cache — replaces st.cache_data / st.cache_resource so
the decorated functions work identically whether called from a Streamlit
rerun or a FastAPI request. Semantics match Streamlit's: cache keyed on
call args, per-process, entries expire after ttl_seconds.
"""
import threading

from cachetools import TTLCache, cached


def ttl_cache(maxsize: int, ttl_seconds: int):
    return cached(cache=TTLCache(maxsize=maxsize, ttl=ttl_seconds), lock=threading.Lock())
