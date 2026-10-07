"""Read daemon metadata and Linux VRAM counters without creating a GPU context.

Run after a weight group is ready. Device-wide residency can include clients;
this report deliberately does not attribute that total to the weight daemon.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket


SCRATCH_VARIABLES = ("HSA_NO_SCRATCH_RECLAIM", "HSA_SCRATCH_SINGLE_LIMIT_ASYNC")


def scratch_environment(pid, proc_root=Path("/proc")):
    fields = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    environment = dict(field.split(b"=", 1) for field in fields if b"=" in field)
    return {name: environment.get(name.encode(), b"").decode() for name in SCRATCH_VARIABLES}


def driver_devices(drm_root=Path("/sys/class/drm")):
    records = []
    seen = set()
    for card in sorted(drm_root.glob("card*")):
        if not card.name.removeprefix("card").isdigit():
            continue
        device = (card / "device").resolve()
        if device in seen or not (device / "mem_info_vram_used").exists():
            continue
        seen.add(device)
        records.append(dict(card=card.name, pci_device=device.name,
            device_vram_used_bytes=int((device / "mem_info_vram_used").read_text()),
            device_vram_total_bytes=int((device / "mem_info_vram_total").read_text())))
    return records


def snapshot(cache_id, cache_dir=None):
    from vllm_lod_plugin.weight_cache_protocol import (
        cache_namespace, control_socket_path, receive_message, send_message,
    )

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(10)
        client.connect(str(control_socket_path(cache_dir, cache_id)))
        send_message(client, {"type": "status"})
        status = receive_message(client)
    if status.get("status") != "ok" or not status.get("groups"):
        raise RuntimeError("No ready daemon weight group; do not report a cache miss as residency")
    workers = []
    for ready in sorted(cache_namespace(cache_dir, cache_id).glob("gpu-*.ready.json")):
        data = json.loads(ready.read_text())
        workers.append(dict(pid=data["pid"], tensors=data["tensors"],
            fingerprint=data["fingerprint"], scratch_environment=scratch_environment(data["pid"])))
    return dict(scope="read-only; device totals may include active inference clients",
        timestamp_utc=datetime.now(timezone.utc).isoformat(), cache_id=cache_id,
        daemon=status, workers=workers, devices=driver_devices())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-id", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = snapshot(args.cache_id, args.cache_dir)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), workers=len(report["workers"]),
        devices=len(report["devices"]), active_clients=[group["active_client_pids"]
        for group in report["daemon"]["groups"]])))


if __name__ == "__main__":
    main()
