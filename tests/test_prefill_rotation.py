from types import SimpleNamespace

from benchmarks._prefill_rotation import rotate_prefill_requests


def row(progress, *, prefill=True, eligible=0):
    return SimpleNamespace(
        num_computed_tokens=progress, is_prefill_chunk=prefill,
        next_decode_eligible_step=eligible,
    )


def test_admission_pauses_prefill_without_pausing_decode():
    first, decode = row(16384), row(65536, prefill=False)
    rotate_prefill_requests([first, decode], has_waiting=True, max_running=8, next_step=3)
    assert first.next_decode_eligible_step == 4
    assert decode.next_decode_eligible_step == 0


def test_rotation_preserves_the_existing_request_and_cache_owners():
    rows = [row(32768), row(16384), row(16384)]
    original = rows.copy()
    rotate_prefill_requests(rows, has_waiting=False, max_running=8, next_step=4)
    assert rows == original
    assert [r.next_decode_eligible_step for r in rows] == [5, 0, 5]
    rows[1].num_computed_tokens += 16384
    rotate_prefill_requests(rows, has_waiting=False, max_running=8, next_step=5)
    assert [r.next_decode_eligible_step for r in rows] == [6, 6, 5]


def test_full_cohort_does_not_stall_waiting_for_admission():
    rows = [row(16384), row(32768)]
    rotate_prefill_requests(rows, has_waiting=True, max_running=2, next_step=7)
    assert [r.next_decode_eligible_step for r in rows] == [0, 8]


def test_rotation_never_bypasses_existing_worker_cadence():
    blocked, ready = row(0, eligible=9), row(16384)
    rotate_prefill_requests([blocked, ready], has_waiting=False, max_running=8, next_step=3)
    assert blocked.next_decode_eligible_step == 9
    assert ready.next_decode_eligible_step == 0


def test_rotation_bounds_unfinished_caches_not_the_decode_batch():
    first, second = row(16384), row(32768)
    decode = row(65536, prefill=False)
    rotate_prefill_requests(
        [first, second, decode], has_waiting=True, max_running=8,
        max_prefills=2, next_step=3,
    )
    assert first.next_decode_eligible_step == 0
    assert second.next_decode_eligible_step == 4
    assert decode.next_decode_eligible_step == 0


def test_rotation_admits_up_to_its_unfinished_cache_bound():
    first, decode = row(16384), row(65536, prefill=False)
    rotate_prefill_requests(
        [first, decode], has_waiting=True, max_running=8,
        max_prefills=2, next_step=3,
    )
    assert first.next_decode_eligible_step == 4
    assert decode.next_decode_eligible_step == 0
