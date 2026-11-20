import math

import torch

from benchmarks.h100_threshold_prefill import compact_routes, select_regions


def test_threshold_prefill_selects_and_compacts_unbalanced_regions():
    sizes = torch.tensor([[[1, 8, 2, 33, 0]]], dtype=torch.int64)
    probability = torch.tensor([[[[0.09, 0.24, 0.16, 0.51, 0.0],
                                  [0.50, 0.10, 0.05, 0.35, 0.0]]]])
    scores = probability.clamp_min(1e-30).log()
    selected, remote = select_regions(scores, sizes, rule='entry', entry_bound=16)
    expected = (probability / sizes.clamp_min(1).unsqueeze(2)) > 1 / (16 * math.sqrt(44))
    expected &= sizes.unsqueeze(2) > 0
    assert torch.equal(selected, expected)
    assert int(remote.item()) == 44
    slots = compact_routes(selected)
    for row in range(2):
        expected_slots = torch.where(expected[0, 0, row])[0]
        actual = slots[0, 0, row]
        assert torch.equal(actual[actual >= 0], expected_slots)


def test_threshold_prefill_cap_uses_prefix_and_preserves_region_control():
    sizes = torch.tensor([[[10, 10, 2, 0]]], dtype=torch.int64)
    probability = torch.tensor([[[[0.4, 0.35, 0.25, 0.0]]]])
    scores = probability.clamp_min(1e-30).log()
    selected, remote = select_regions(scores, sizes, rule='entry', entry_bound=32, work_bound=3)
    assert selected.tolist() == [[[[True, False, False, False]]]]
    opened = (selected * sizes.unsqueeze(2)).sum(-1)
    assert bool((opened.square() < 9 * remote.squeeze(-1)).all())
    region, _ = select_regions(scores, sizes, rule='region', multiplier=1.0, max_region_entries=9)
    assert not bool(region.any())


def test_threshold_prefill_caps_selected_regions_in_region_order():
    sizes = torch.ones((1, 1, 20), dtype=torch.int64)
    scores = torch.zeros((1, 1, 1, 20))
    selected, remote = select_regions(scores, sizes, rule='entry', entry_bound=16,
                                      work_bound=16, max_regions=8)
    assert selected[0, 0, 0].nonzero().flatten().tolist() == list(range(8))
    assert int(remote.item()) == 20
