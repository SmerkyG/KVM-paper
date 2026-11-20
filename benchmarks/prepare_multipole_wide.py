"""Copy Multipole generation kernels and widen sparse KV head addressing.

The author's dynamic sparse kernel calculates head_group_id as int32. At
large batches, multiplying it by the KV-head stride can overflow before the
result is cast to int64. This changes that head index to int64 first and
does not change the selection algorithm or arithmetic on attention values.
"""
import argparse
import shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path, help='author MultipoleAttention checkout')
p.add_argument('destination', type=Path, help='new overlay directory')
a = p.parse_args()
src, dst = a.source / 'multipoleattention', a.destination / 'multipoleattention'
if dst.exists():
    raise SystemExit(f'refusing to overwrite {dst}')
shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))
path = dst / 'kernels.py'
text = path.read_text()
old = '        head_group_id += is_in_head.to(tl.int32)\n\n    # initialize KV block parameters'
new = ('        head_group_id += is_in_head.to(tl.int32)\n'
       '    head_group_id = head_group_id.to(tl.int64)  # avoid KV head-stride overflow\n\n'
       '    # initialize KV block parameters')
if text.count(old) != 1:
    raise RuntimeError('expected sparse head-offset calculation not found exactly once')
path.write_text(text.replace(old, new))
print(f'Wrote 64-bit sparse head-addressing overlay to {dst}', flush=True)
