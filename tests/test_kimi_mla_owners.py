import pytest
import torch

from benchmarks.kimi_k3_mla_owners import consume_attention_rows, owner_rank, real_token_rows, validate_audit


def test_reference_copies_reused_attention_workspace_immediately():
    workspace = torch.empty(2, 3, 4)
    output = torch.empty(8*2, 3, 4)

    def attend(row):
        return workspace.fill_(row)

    consume_attention_rows(output, 2, attend)
    for row in range(8):
        assert torch.equal(output[row*2:(row+1)*2], torch.full((2, 3, 4), row))


def test_owners_cover_each_rank_without_moving_kda():
    for index in range(3, 24, 4):
        assert {owner_rank(row, index) for row in range(8)} == set(range(8))
    assert [owner_rank(0, index) for index in range(3, 24, 4)] == list(range(6))
    with pytest.raises(ValueError):
        owner_rank(-1, 3)


def test_real_tokens_continue_short_documents_without_padding():
    docs = [[1, 2], [3], [4, 5], [6, 7]]
    assert real_token_rows(docs, 5, 2) == [[1, 2, 4, 5, 1], [3, 6, 7, 3, 6]]
    with pytest.raises(ValueError):
        real_token_rows([[], []], 5, 2)
    with pytest.raises(ValueError):
        real_token_rows([[], [1]], 5, 2)


def test_audit_rejects_missing_global_update_and_wrong_owner():
    def audit(rank):
        return {str(i): {str(r): dict(total_len=32768, coverage=32512, state_len=1024)
                        for r in range(8) if owner_rank(r, i) == rank}
                for i in range(3, 24, 4)}
    for rank in range(8):
        validate_audit(audit(rank), variant="owner_mla", length=32768,
                       batch=8, layer_count=24, rank=rank)
    broken = audit(0)
    broken["3"]["0"]["coverage"] = 16384-256
    with pytest.raises(AssertionError, match="global 16K"):
        validate_audit(broken, variant="owner_mla", length=32768,
                       batch=8, layer_count=24, rank=0)
    broken = audit(0)
    broken["3"]["1"] = broken["3"]["0"]
    with pytest.raises(AssertionError, match="non-owner"):
        validate_audit(broken, variant="owner_mla", length=32768,
                       batch=8, layer_count=24, rank=0)
