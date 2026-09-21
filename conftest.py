"""Shared pytest configuration.

Must be imported before `torch`, so we set memory-allocator related env vars
here. `expandable_segments:True` greatly reduces GPU memory fragmentation in
tests that allocate / free many large tensors
"""

import os

# Set the HIP/CUDA caching-allocator config _before_ torch is imported anywhere.
_alloc_conf_keys = ("PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
_default_conf = "expandable_segments:True"
for _key in _alloc_conf_keys:
    existing = os.environ.get(_key, "")
    if "expandable_segments" not in existing:
        os.environ[_key] = (existing + "," + _default_conf).strip(",") if existing else _default_conf
