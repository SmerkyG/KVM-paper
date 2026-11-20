"""Extract the released split-centroid kernel with stable logit output.

The paper's kernel benchmark stores exponentiated scores without normalizing
across query heads. The generation path uses per-query normalization. Change
only that store to emit logits, allowing stable normalization by the adapter.
"""
import argparse
import ast
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path, help='MultipoleAttention repository root')
parser.add_argument('destination', type=Path, help='directory on PYTHONPATH')
args = parser.parse_args()
source = (args.source / 'benchmark' / 'benchmark_kernels.py').read_text()
tree = ast.parse(source)
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fwd_centroid_kernel_qk_gen_splitkv']
if len(nodes) != 1:
    raise SystemExit('Expected exactly one upstream split kernel')
node = nodes[0]
lines = source.splitlines(keepends=True)
start = min(d.lineno for d in node.decorator_list) - 1
kernel = ''.join(lines[start:node.end_lineno])
old = 'p_store = tl.math.exp2(qk)'
if kernel.count(old) != 1:
    raise SystemExit('Expected upstream score store not found exactly once')
kernel = kernel.replace(old, 'p_store = qk  # preserve logits for stable per-query normalization')
old_mask = 'qk = tl.where(offs_n[None, :] < N_CENTROIDS, qk, float("-inf"))'
if kernel.count(old_mask) != 1:
    raise SystemExit('Expected upstream centroid mask not found exactly once')
kernel = kernel.replace(old_mask, 'qk = tl.where((offs_n[None, :] < N_CENTROIDS) & (nkeys[None, :] > 0), qk, float("-inf"))')
old_norm = """qk = qk - m_curr[:, None]
        p = tl.math.exp2(qk)

        # compute exp_qk * num_c, then sum
        p_tmp = p * nkeys[None, :]
        l_tmp = tl.sum(p_tmp, 1)
        alpha = tl.math.exp2(m_prev - m_curr)"""
new_norm = """safe_m = tl.where(m_curr == -float("inf"), 0.0, m_curr)
        qk = qk - safe_m[:, None]
        p = tl.math.exp2(qk)

        # Empty split blocks have no mass and must not produce NaNs.
        p_tmp = p * nkeys[None, :]
        l_tmp = tl.sum(p_tmp, 1)
        alpha = tl.where(m_prev == -float("inf"), 0.0, tl.math.exp2(m_prev - safe_m))"""
if kernel.count(old_norm) != 1:
    raise SystemExit('Expected upstream normalization block not found exactly once')
kernel = kernel.replace(old_norm, new_norm)
path = args.destination / 'multipole_split_kernel.py'
if path.exists():
    raise SystemExit(f'refusing to overwrite {path}')
path.write_text('import triton\nimport triton.language as tl\n\n' + kernel)
