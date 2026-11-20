import torch
from benchmarks.h100_prefill_overlap import route_statistics


def test_region_sharing_counts_per_head_and_weights_by_region_size():
    # One KV head, two query heads, four queries each. Repeated region ids
    # belonging to different heads must never be grouped together.
    slots = torch.tensor([[[[0, 1], [0, 1], [0, 2], [2, -1]],
                           [[0, 2], [1, -1], [1, -1], [1, -1]]]])
    lengths = torch.tensor([[[10, 20, 30]]])
    r = route_statistics(slots, lengths, state_len=3, group_size=2)
    assert r['query_heads'] == 8
    assert r['routes'] == 12
    assert r['active_head_regions'] == 6
    assert r['shared_routes'] == 10
    assert r['opened_entry_uses'] == 230
    assert r['unique_region_entries'] == 120
    assert r['route_weighted_group_size_sum'] == 28
    assert r['shared_region_query_pairs'] == 8


def test_padded_threshold_slots_are_masked_by_route_counts():
    slots = torch.tensor([[[[0, 2], [1, 2]]]])
    lengths = torch.tensor([[[5, 7, 99]]])
    counts = torch.tensor([[1, 1]])
    r = route_statistics(slots, lengths, state_len=3, group_size=1, route_counts=counts)
    assert r['routes'] == 2
    assert r['opened_entry_uses'] == 12
    assert r['shared_routes'] == 0
