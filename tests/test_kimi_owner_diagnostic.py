from types import SimpleNamespace as NS

from benchmarks.kimi_k3_owner_diagnostic import (
    interval_union_us, linear_replication_factor, summarize_profile)


def test_replication_uses_actual_partition_widths():
    assert linear_replication_factor(NS(tp_size=8)) == 1
    assert linear_replication_factor(NS(tp_size=8, input_size=512,
        input_size_per_partition=512, output_size=7168,
        output_size_per_partition=7168)) == 1
    assert linear_replication_factor(NS(input_size=7168,
        input_size_per_partition=896, output_size=7168,
        output_size_per_partition=7168)) == 8


def test_benchmark_reports_distinct_warmup_and_measurement_phases(capsys):
    from benchmarks.kimi_k3_prefill_sweep import report_phase

    result, snapshots = {}, []
    def save():
        snapshots.append(dict(result["current_phase"]))
    report_phase(result, save, length=262144, phase="warmup")
    report_phase(result, save, length=262144, phase="warmup_complete",
                 warmup_elapsed_seconds=341.0)
    report_phase(result, save, length=262144, phase="measurement", repeat_index=0)
    assert [snapshot["phase"] for snapshot in snapshots] == [
        "warmup", "warmup_complete", "measurement"]
    assert result["phase_history"][1]["warmup_elapsed_seconds"] == 341.0
    assert capsys.readouterr().out.count("KIMI_PREFILL_PHASE ") == 3


def test_interval_union_handles_overlap_and_empty():
    assert interval_union_us([]) == 0
    assert interval_union_us([(4, 8), (1, 6), (2, 3), (10, 11)]) == 8


def test_profile_exclusive_correlation_and_overlap():
    outer = NS(id=1, device_type="cpu", name="K3/model_other", cpu_parent=None)
    inner = NS(id=2, device_type="cpu", name="K3/moe", cpu_parent=outer)
    launch = NS(id=3, device_type="cpu", name="launch", cpu_parent=inner)
    def gpu(begin, end, correlation):
        return NS(device_type="gpu", name="kernel", linked_correlation_id=correlation,
                  time_range=NS(start=begin, end=end), device_time_total=end-begin)
    result = summarize_profile([outer, inner, launch,
        gpu(0, 2000, 3), gpu(1000, 3000, 3), gpu(4000, 5000, 0)], "cpu", "gpu")
    assert result["gpu_timeline_span_ms"] == 5
    assert result["gpu_activity_union_ms"] == 4
    assert result["stages"]["K3/moe"] == {
        "gpu_activity_count": 2, "summed_gpu_ms_not_wall_ms": 4,
        "gpu_activity_union_ms": 3}
    assert "K3/model_other" not in result["stages"]


def test_gpu_annotations_are_not_kernel_work():
    annotation = NS(device_type="gpu", name="K3/model_other", is_user_annotation=True,
                    linked_correlation_id=0,
                    time_range=NS(start=0, end=100000), device_time_total=100000)
    assert summarize_profile([annotation], "cpu", "gpu") == {
        "gpu_timeline_span_ms": 0, "gpu_activity_union_ms": 0, "stages": {}}
