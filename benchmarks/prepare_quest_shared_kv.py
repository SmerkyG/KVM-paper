"""Patch a current-stream QUEST copy to reference shared GQA KV storage.

The kernels retain one independent selection and attention computation per
query head. Only physical KV/bound addresses change: logical query head h
reads physical KV head h // group_size. Generate an explicit reviewable diff.
"""
import argparse
import difflib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--patch', type=Path, required=True)
    args = parser.parse_args()
    changes = []

    def edit(name, transform):
        path = args.checkout / name
        before = path.read_text()
        after = transform(before)
        path.write_text(after)
        changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                       fromfile='a/' + name, tofile='b/' + name))

    def replace_once(s, old, new):
        if s.count(old) != 1:
            raise ValueError(f'Expected one occurrence of {old!r}, found {s.count(old)}')
        return s.replace(old, new)

    def page(s):
        s = replace_once(s, 'struct paged_kv_t {\n', 'struct paged_kv_t {\n\t// Logical heads may share one physical KV head.\n\tuint32_t storage_group = 1;\n')
        for name in ('get_k_elem_offset', 'get_k_elem_offset_in_page',
                     'get_v_elem_offset', 'get_v_elem_offset_in_page'):
            start = s.index('size_t ' + name + '(')
            body = s.index('{', start)
            end = s.index('\n\t}', body)
            chunk = s[body + 1:end]
            chunk = chunk.replace('num_heads', '(num_heads / storage_group)')
            chunk = '\n\t\thead_idx /= storage_group;' + chunk
            s = s[:body + 1] + chunk + s[end:]
        return replace_once(s, 'return num_heads * page_size * head_dim;',
                            'return (num_heads / storage_group) * page_size * head_dim;')

    def estimate(s):
        s = replace_once(s, 'CHECK_EQ(metadata_data.size(2), num_heads);',
                         'TORCH_CHECK(num_heads % metadata_data.size(2) == 0);')
        s = replace_once(s, 'CHECK_EQ(metadata_data.size(3), num_heads);',
                         'TORCH_CHECK(num_heads % metadata_data.size(3) == 0);')
        marker = '\t\t\tcudaError_t status =\n\t\t\t\tMaxPossibleSampleWithPagedKVCache'
        return replace_once(s, marker, '\t\t\tpaged_kv.storage_group = num_heads / metadata_data.size(kv_layout == QKVLayout::kHND ? 2 : 3);\n' + marker)

    def attention(s):
        marker = '\t\t\tpaged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(\n\t\t\t\tnum_kv_heads,'
        s = replace_once(s, marker, marker.replace('num_kv_heads,', 'num_qo_heads,'))
        marker = '\n\t\t\tcudaError_t status =\n\t\t\t\tBatchDecodeWithPagedKVCacheWrapper'
        return replace_once(s, marker, '\n\t\t\tTORCH_CHECK(num_kv_heads > 0 && num_qo_heads % num_kv_heads == 0);\n\t\t\tpaged_kv.storage_group = num_qo_heads / num_kv_heads;\n' + marker)

    edit('kernels/include/decode/decode_page.cuh', page)
    edit('quest/ops/csrc/estimate.cu', estimate)
    edit('quest/ops/csrc/approx_attn.cu', attention)
    args.patch.parent.mkdir(parents=True, exist_ok=True)
    args.patch.write_text(''.join(changes))
    print(f'Prepared shared-storage QUEST copy at {args.checkout}; patch: {args.patch}')


if __name__ == '__main__':
    main()
