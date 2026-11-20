"""Build query orders for atomics-free shared-region exact attention."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mix(region):
    x = region.to(tl.uint32)
    x = (x ^ (x >> 16)) * tl.full((), 0x7FEB352D, tl.uint32)
    x = (x ^ (x >> 15)) * tl.full((), 0x846CA68B, tl.uint32)
    return x ^ (x >> 16)


@triton.jit
def _frontier_keys(Slots, Counts, Lengths, Keys, Entries, ROWS: tl.constexpr, QKV: tl.constexpr,
                   WIDTH: tl.constexpr, STATE: tl.constexpr, M: tl.constexpr, R: tl.constexpr,
                   ORDER: tl.constexpr, WRITE_ENTRIES: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * M + tl.arange(0, M)
    rank = tl.arange(0, R)
    count = tl.load(Counts + row, mask=row < ROWS, other=0)
    valid = rank[None, :] < count[:, None]
    slot = tl.load(Slots + row[:, None] * WIDTH + rank[None, :], mask=valid, other=0).to(tl.int32)
    if ORDER == "largest" or WRITE_ENTRIES:
        kh = row // QKV
        size = tl.load(Lengths + kh[:, None] * STATE + slot, mask=valid, other=0).to(tl.int32)
    if ORDER == "signature":
        mixed = tl.where(valid, _mix(slot + 1), 0).to(tl.uint32)
        summed = tl.sum(mixed, 1).to(tl.uint64)
        xor = tl.xor_sum(mixed, 1).to(tl.uint64)
        key = ((summed << 32) | xor).to(tl.int64)
    else:
        # The two selected regions with the largest KV payload anchor each query.
        # The size comparison and the region-ID tie break are both integer.
        scored = tl.where(valid, size * STATE + slot, -1)
        first = tl.max(scored, 1)
        second = tl.max(tl.where(scored < first[:, None], scored, -1), 1)
        key = (first % STATE + 1).to(tl.int64) * (STATE + 1) + (second % STATE + 1)
    tl.store(Keys + row, key, mask=row < ROWS)
    if WRITE_ENTRIES:
        tl.store(Entries + row, tl.sum(size, 1), mask=row < ROWS)


def frontier_keys(slots, counts, lengths, *, group, method, write_entries=False):
    """Return one order key and exact-work count per query, by shared KV head."""
    b, h, t, width = slots.shape
    if method not in ("signature", "largest"):
        raise ValueError("prefill order must use a frontier signature or largest regions")
    qkv = group * t
    rows = b * h * t
    keys = torch.empty((rows // qkv, qkv), device=slots.device, dtype=torch.int64)
    entries = torch.empty_like(keys, dtype=torch.int32) if write_entries else None
    _frontier_keys[(triton.cdiv(rows, 8),)](
        slots, counts, lengths, keys, entries, rows, qkv, width, lengths.size(2),
        8, triton.next_power_of_2(width), method, write_entries, num_warps=4,
    )
    return keys, entries
