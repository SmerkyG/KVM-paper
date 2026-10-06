"""Load and audit a genuinely half-resident K3 weight daemon, without timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def preload_kwargs(checkpoint: str, weight_cache_id: str) -> dict:
    from benchmarks._vllm import llm_kwargs

    kwargs = llm_kwargs(checkpoint=checkpoint, mode="full", max_model_len=16385,
        batch_size=8, tensor_parallel_size=8, decode_context_parallel_size=8,
        gpu_memory_utilization=0.8, full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(load_format="ipc_cache", skip_tokenizer_init=True,
        enable_expert_parallel=True, disable_custom_all_reduce=False,
        quantization_config={"moe":{"weight":"int4_per_group_32"}},
        kv_cache_memory_bytes=1073741824,
        model_loader_extra_config=dict(auto_start=False, cache_id=weight_cache_id,
            backing_load_format="auto", broker_timeout=3600.0))
    return kwargs


def audit_half_weights(worker):
    import torch

    model = worker.model_runner.model
    core = next(m for m in model.modules() if type(m).__name__ == "KimiLinearModel")
    layers = list(core.layers)
    indices = [int(l.layer_idx) for l in layers]
    if indices != list(range(48)) or core.start_layer != 0 or core.end_layer != 48:
        raise AssertionError(f"daemon client is not the complete first-48 stage: {indices}")
    kinds = [type(l.self_attn).__name__ for l in layers]
    ffns = [type(l.mlp).__name__ for l in layers]
    if ffns != ["KimiMLP"]+["KimiMoE"]*47:
        raise AssertionError("the actual trained dense/MoE FFNs must remain installed")
    if layers[1].mlp.experts.moe_config.ep_size != 8:
        raise AssertionError("half-stage MoE must use native EP8")
    if kinds.count("KimiMLAAttention") != 12 or kinds.count("KimiK3DeltaAttention") != 36:
        raise AssertionError(f"unexpected first-48 attention geometry: {kinds}")
    storages = {}
    for tensor in model._vllm_weight_cache_imports:
        if tensor.device.type == "cuda":
            storage = tensor.untyped_storage()
            storages[int(storage._cdata)] = int(storage.nbytes())
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return dict(rank=worker.rank, layer_indices=indices, mla_layers=12, kda_layers=36,
        native_moe_layers=47, native_moe_ep=8,
        daemon_resident_bytes=model._vllm_weight_cache_resident_bytes,
        imported_cuda_weight_storage_bytes=sum(storages.values()),
        device_total_bytes=total, device_free_bytes=free,
        client_torch_allocated_bytes=torch.cuda.memory_allocated(),
        scope="weight residency/startup audit only; no speed or quality measurement")


def main():
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weight-cache-id", default="kimi-k3-first48-int4-v1")
    parser.add_argument("--wait-seconds", type=int, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic()+args.wait_seconds
    manifest_path = args.checkpoint / "half-stage-manifest.json"
    while True:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {}
        if manifest.get("status") == "ready":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("half-checkpoint local copy did not finish")
        time.sleep(5)
    config = json.loads((args.checkpoint / "config.json").read_text())
    if config.get("text_config", config)["num_hidden_layers"] != 48:
        raise ValueError("preload requires the first-48 checkpoint, not the full model")
    for name in ("LOD_KIMI_REQUEST_OWNER_PREFILL", "LOD_KIMI_DCP_SHARDED_LEAVES",
                 "LOD_KIMI_DCP_LOCAL_PREFILL", "LOD_KIMI_DCP_SHARED_PREFILL"):
        os.environ.pop(name, None)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from benchmarks._vllm import close_llm
    from vllm import LLM

    kwargs = preload_kwargs(str(args.checkpoint), args.weight_cache_id)
    print("KIMI_HALF_PRELOAD_BEGIN " + json.dumps(dict(checkpoint=str(args.checkpoint),
        cache_id=args.weight_cache_id, layers=48)), flush=True)
    llm = LLM(**kwargs)
    try:
        audits = llm.collective_rpc(audit_half_weights, timeout=300)
        if {a["rank"] for a in audits} != set(range(8)):
            raise AssertionError("missing half-stage audit rank")
        result = dict(checkpoint=str(args.checkpoint), weight_cache_id=args.weight_cache_id,
            trained_prefix_layers=48, audits=audits, checkpoint_manifest=manifest,
            full_model_daemon_modified=False, speed_results=None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print("KIMI_HALF_WEIGHTS_READY " + json.dumps(dict(
            daemon_resident_gib=[a["daemon_resident_bytes"]/1024**3 for a in audits],
            free_device_gib=[a["device_free_bytes"]/1024**3 for a in audits])), flush=True)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()
