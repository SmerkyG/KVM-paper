"""Benchmark-only worker hook, enabled by this directory on PYTHONPATH.

This hook is loaded in vLLM spawned workers as well as the API process. The
public LoD configuration stays fixed; only single-token recursive decode is
overridden for this ablation. It is never on the normal package import path.
"""

import os
import sys
import importlib.util

pages = int(os.environ.get("RULER_PAGES_PER_SLOT", "1"))
snapshot = os.environ.get("PAGE_KERNEL_SNAPSHOT")
if "RULER_PAGES_PER_SLOT" in os.environ or snapshot:
    if pages not in (1, 2, 4, 8):
        raise ValueError("RULER_PAGES_PER_SLOT must be 1, 2, 4, or 8")
    from lod_attention.kernels import paged_prefill

    original = paged_prefill.query_major_residual_page_attention
    if snapshot:
        spec = importlib.util.spec_from_file_location(
            "lod_attention.kernels._page_opening_snapshot", snapshot
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        original = module.query_major_residual_page_attention

    def attention(query, *args, **kwargs):
        if query.size(-2) == 1:
            kwargs["pages_per_slot"] = pages
        return original(query, *args, **kwargs)

    paged_prefill.query_major_residual_page_attention = attention
    print(f"RULER decode ablation: {pages} pages per routed centroid", file=sys.stderr, flush=True)
