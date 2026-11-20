"""Make a separate copy of Squeezed Attention's Triton kernels with GQA addressing.

The attention math and per-query-head selection are unchanged. The lookup
and sparse kernels map their input reads to the shared physical KV head.
"""
import argparse
import difflib
from pathlib import Path

def patch_kernel(source, name, next_kernel, offsets, *, index_group=False, kv_group=True, fixed_blocks=False):
    start = source.index(f'def {name}(')
    end = source.index(f'@triton.jit\ndef {next_kernel}(', start)
    part = source[start:end]
    signature_start = part.index('BLOCK_DMODEL: tl.constexpr')
    signature_end = part.index('\n', signature_start)
    indent = part[part.rindex('\n', 0, signature_start) + 1:signature_start]
    params = ['BLOCK_DMODEL: tl.constexpr']
    if kv_group:
        params.append('GQA_GROUP: tl.constexpr = 1')
    if index_group:
        params.append('GQA_INDEX_GROUP: tl.constexpr = 1')
    if fixed_blocks:
        params.append('FIXED_BLOCKS_PER_HEAD: tl.constexpr = 0')
    part = part[:signature_start] + (',\n' + indent).join(params) + part[signature_end:]
    for old, new, expected in offsets:
        if part.count(old) != expected:
            raise RuntimeError(f'{name}: expected {expected} occurrences of {old!r}, got {part.count(old)}')
        part = part.replace(old, new)
    return source[:start] + part + source[end:]


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--patch', type=Path, required=True)
a = p.parse_args()
before = a.source.read_text()
after = patch_kernel(before, '_fwd_centroid_simple_kernel_qk', '_fwd_kernel_qk', [
    ('offs_k = head_id * skh +', 'offs_k = (head_id // GQA_GROUP) * skh +', 2),
    ('offs_nk = head_id * N_CENTROIDS +', 'offs_nk = (head_id // GQA_GROUP) * N_CENTROIDS +', 1),
])
after = patch_kernel(after, '_fwd_kernel_qk', '_reduce_kernel_qk', [
    ('offs_k_base = head_id * skh +', 'offs_k_base = (head_id // GQA_GROUP) * skh +', 1),
    ('offs_v_base = head_id * svh +', 'offs_v_base = (head_id // GQA_GROUP) * svh +', 1),
    ('offs_kidx = head_id * skih +', 'offs_kidx = (head_id // GQA_INDEX_GROUP) * skih +', 1),
    ('head_kv_len + head_id)', 'head_kv_len + (head_id // GQA_INDEX_GROUP))', 1),
    ('''    head_id = -1
    for i in range(NUMHEADS):
        head_start = tl.load(head_start_block + i).to(tl.int32)
        is_in_head = kv_block_id2 >= head_start
        head_id += is_in_head.to(tl.int32)
''', '''    if FIXED_BLOCKS_PER_HEAD:
        head_id = kv_block_id2 // FIXED_BLOCKS_PER_HEAD
    else:
        head_id = -1
        for i in range(NUMHEADS):
            head_start = tl.load(head_start_block + i).to(tl.int32)
            is_in_head = kv_block_id2 >= head_start
            head_id += is_in_head.to(tl.int32)
''', 1),
], index_group=True, fixed_blocks=True)
after = patch_kernel(after, '_reduce_kernel_qk', '_fwd_kernel_causal', [
    ('head_kv_len + head_id)', 'head_kv_len + (head_id // GQA_INDEX_GROUP))', 1),
], index_group=True, kv_group=False)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(after)
a.patch.parent.mkdir(parents=True, exist_ok=True)
a.patch.write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                       fromfile='a/squeezedattention/kernels.py', tofile='b/squeezedattention/kernels.py')))
print(f'Prepared shared-GQA author kernels: {a.output}; patch: {a.patch}')
