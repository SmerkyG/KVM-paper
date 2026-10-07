import pytest

from benchmarks.kimi_k3_ipc_extents import extent_summary


def entry(handle, offset, size):
    return dict(transport="cuda_ipc", ipc_args=[None] * 6 + [0, handle, size, offset] + [None] * 5)


def test_extent_bound_merges_aliases_and_counts_internal_not_trailing_gaps():
    result = extent_summary(dict(a=entry(b"one", 100, 100), alias=entry(b"one", 100, 100),
        b=entry(b"one", 250, 50), c=entry(b"two", 0, 32), cpu=dict(transport="cpu")))
    assert result["ipc_allocation_handles"] == 2
    assert result["exported_storage_union_bytes"] == 182
    assert result["minimum_pinned_allocation_extent_bytes"] == 332
    assert result["minimum_unexported_internal_gap_bytes"] == 150


def test_empty_storage_does_not_pin_an_allocation():
    assert extent_summary(dict(empty=entry(None, 0, 0)))["ipc_allocation_handles"] == 0


@pytest.mark.parametrize("offset,size", [(-1, 10), (0, -10)])
def test_negative_extents_fail(offset, size):
    with pytest.raises(ValueError, match="Negative"):
        extent_summary(dict(bad=entry(b"one", offset, size)))
