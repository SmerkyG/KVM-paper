from pathlib import Path

from benchmarks.kimi_k3_weight_residency import driver_devices, scratch_environment


def test_scratch_environment_does_not_export_unrelated_secrets(tmp_path):
    process = tmp_path / "123"
    process.mkdir()
    (process / "environ").write_bytes(b"HSA_NO_SCRATCH_RECLAIM=0\0SECRET=private\0")
    assert scratch_environment(123, tmp_path) == {
        "HSA_NO_SCRATCH_RECLAIM": "0", "HSA_SCRATCH_SINGLE_LIMIT_ASYNC": ""}


def test_driver_counters_are_deduplicated_and_ignore_connector_paths(tmp_path):
    device = tmp_path / "0000:01:00.0"
    device.mkdir()
    (device / "mem_info_vram_used").write_text("123\n")
    (device / "mem_info_vram_total").write_text("456\n")
    for name in ("card0", "card1", "card0-DP-1"):
        card = tmp_path / name
        card.mkdir()
        (card / "device").symlink_to(device, target_is_directory=True)
    assert driver_devices(tmp_path) == [dict(card="card0", pci_device=device.name,
        device_vram_used_bytes=123, device_vram_total_bytes=456)]


def test_inspection_does_not_import_torch():
    source = Path("benchmarks/kimi_k3_weight_residency.py").read_text()
    assert "import torch" not in source
