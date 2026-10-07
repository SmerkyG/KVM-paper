"""Isolated, in-memory backport of AITER 8cfa0902 (PR #5909).

No installed file is edited. Clone the two affected Gluon JIT functions,
apply only the unconditional loop-tail carry, and give each clone a fresh
device cache. The kernel's source hash separates the old/new disk compiles.
This is a conventional paged-attention experiment, NOT a K3 MLA improvement.
"""

import ast
import copy
from collections import defaultdict
from contextlib import contextmanager


def unconditional_prefetch_carry(source):
    """Remove exactly the unsafe conditional; reject a changed source contract."""
    predicate = ast.parse("sequence_partition_idx + CONTEXT_PARTITION_SIZE_PER_BLOCK < sequence_partition_end_idx", mode="eval").body
    matches = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.If)
               and ast.dump(n.test) == ast.dump(predicate)]
    if len(matches) != 1:
        raise ValueError(f"expected one conditional prefetch carry, found {len(matches)}")
    node = matches[0]
    expected = {"kv_block_numbers": "kv_block_numbers2", "key_tensor": "key_tensor2",
                "kv_block_start_idx": "kv_block_start_idx2"}
    if len(node.body) == 4:
        expected["page_offset"] = "page_offset2"
    if node.orelse or len(node.body) != len(expected):
        raise ValueError("unexpected prefetch carry body")
    for n in node.body:
        if (not isinstance(n, ast.Assign) or len(n.targets) != 1 or not isinstance(n.targets[0], ast.Name)
                or not isinstance(n.value, ast.Name) or expected.get(n.targets[0].id) != n.value.id):
            raise ValueError("unexpected prefetch carry assignment")
    lines = source.splitlines(keepends=True)
    body = lines[node.body[0].lineno - 1:node.end_lineno]
    lines[node.lineno - 1:node.end_lineno] = [line[4:] for line in body]
    result = "".join(lines)
    ast.parse(result)
    return result


@contextmanager
def patched_prefetch(module):
    originals, candidates = {}, {}
    for name in ("paged_attention_decode_sliding_window_head_1", "paged_attention_decode_sliding_window"):
        original = getattr(module, name)
        patched = copy.copy(original)
        patched._unsafe_update_src(unconditional_prefetch_carry(original.src))
        patched.device_caches = defaultdict(patched.create_binder)
        patched.used_global_vals = {}
        originals[name], candidates[name] = original, patched
    try:
        for name, kernel in candidates.items():
            setattr(module, name, kernel)
        yield candidates
    finally:
        for name, kernel in originals.items():
            setattr(module, name, kernel)
