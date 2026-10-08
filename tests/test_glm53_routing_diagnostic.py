"""CPU checks for independent real-tensor directory and oracle audits."""
import pytest
import torch

from benchmarks._glm53_routing_diagnostic import directory_owners, region_oracle, relative_error, selection_mask


def example_directory():
    # Includes a directory boundary and a partial final page.
    n = 64 * 16 + 3 + 2
    indices = torch.full((1, 1, 67, 16), -1, dtype=torch.int32)
    indices[0, 0, :64] = torch.arange(1024).reshape(64, 16)
    indices[0, 0, 64, :3] = torch.arange(1024, 1027)
    indices[0, 0, 65, :2] = torch.arange(1027, n)
    directory = torch.full((1, 1, 3, 64), -1, dtype=torch.int32)
    directory[0, 0, 0] = torch.arange(64)
    directory[0, 0, 1, 0] = 64
    directory[0, 0, 2, 0] = 65
    return dict(paged_page_directory=True, leaf_count=n,
        slot_lengths=torch.tensor([[[1027, 2]]]),
        slot_pages=torch.tensor([[[[0, 1], [2, -1]]]]),
        overflow_page_values=directory, page_indices=indices)


def test_directory_coverage_crosses_root_boundary_and_ignores_padding():
    owners = directory_owners(example_directory(), 2)
    assert owners.tolist() == [0] * 1027 + [1] * 2


@pytest.mark.parametrize("error", ["duplicate", "missing", "bad_root"])
def test_directory_rejects_damaged_metadata(error):
    cache = example_directory()
    if error == "duplicate":
        cache["page_indices"][0, 0, 65, 0] = 0
    elif error == "missing":
        cache["page_indices"][0, 0, 65, 0] = -1
    else:
        cache["slot_pages"][0, 0, 1, 0] = -1
    with pytest.raises(AssertionError):
        directory_owners(cache, 2)


def test_leaf_oracles_equal_literal_group_calculations():
    scores = torch.tensor([[[[2., -3., 4., 1., -5.], [-2., 7., 1., 0., 3.]]]])
    owners = torch.tensor([1, 0, 1, 0, 2])
    for mode in ("max", "mass"):
        actual = region_oracle(scores, owners, 4, mode=mode)
        for i in range(3):
            selected = scores[..., owners == i]
            expected = selected.amax(-1) if mode == "max" else selected.logsumexp(-1)
            torch.testing.assert_close(actual[..., i], expected)
        assert actual[..., 3].isneginf().all()


def test_output_error_has_per_query_and_global_norms():
    reference = torch.ones(2, 3)
    error = relative_error(reference * 1.5, reference)
    assert error["relative_l2"] == error["max_head_query_relative_l2"] == .5


def test_closed_routes_do_not_erase_region_zero():
    actual = selection_mask(torch.tensor([[0, -1, 2, -1]]), 4)
    assert actual.tolist() == [[True, False, True, False]]
