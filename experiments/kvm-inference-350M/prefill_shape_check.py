"""Compare Figure 5 prefill at the paper and model shapes, with a cache control."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root), str(root / "scripts")]
from kvm_benchmark.common import (
    DIM, DTYPE, allocate_hardware_cache_eviction_buffer, build_mixer,
    concatenate_heads, distribution, fill_gate_, flash_backend,
    stream_hardware_cache_eviction, write_json_exclusive,
)
from model.statesdictcache import StatesDictCache


def measure_shape(batch, heads, length, eviction):
    mixer = build_mixer("sqrt16", length, heads=heads).eval()
    rng = torch.Generator(device="cuda").manual_seed(0)
    shape = (batch, heads, length, DIM)
    q, k, v = (
        torch.randn(shape, generator=rng, device="cuda", dtype=DTYPE) for _ in range(3)
    )
    gate = torch.empty((*shape[:-1], 1), device="cuda", dtype=DTYPE)
    fill_gate_(gate, rng)
    cache = StatesDictCache()

    def kvm_no_cache():
        return mixer.forward_prefill(q, k, v, gate, None, None, None)

    def kvm_with_cache():
        cache.reset()
        return mixer.forward_prefill(q, k, v, gate, None, None, None, cache)

    def flash():
        return concatenate_heads(F.scaled_dot_product_attention(q, k, v, is_causal=True))

    arms = (kvm_no_cache, kvm_with_cache, flash)
    reference = kvm_no_cache()
    torch.testing.assert_close(kvm_with_cache(), reference, rtol=0.01, atol=0.02)
    assert cache.get_seq_length(0) == length
    del reference
    for _ in range(3):
        for arm in arms:
            output = arm()
            assert output.shape == (batch, length, heads * DIM) and output.dtype == DTYPE
            del output
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = {arm.__name__: [] for arm in arms}
    for round_idx in range(9):
        shift = round_idx % len(arms)
        for arm in (*arms[shift:], *arms[:shift]):
            stream_hardware_cache_eviction(eviction)
            start.record()
            output = arm()
            end.record()
            end.synchronize()
            samples[arm.__name__].append(start.elapsed_time(end))
            del output

    summary = {name: distribution(values) for name, values in samples.items()}
    print(
        f"B{batch} H{heads} T{length} "
        + " ".join(f"{name}={stats['median']:.3f}ms" for name, stats in summary.items()),
        flush=True,
    )
    args = mixer._make_triton_args(batch, length)
    return {
        "batch": batch, "heads": heads, "context": length,
        "workgroups_per_attention_chunk": batch * heads * (args.macro_block // args.attn_block),
        "scan_state_chunk": args.scan_state_chunk, "scan_group_chunks": args.group_chunks,
        "samples_ms": samples, "summary_ms": summary,
    }


@torch.inference_mode()
def main():
    eviction = allocate_hardware_cache_eviction_buffer()
    records = []
    with flash_backend():
        for batch, heads in ((1, 8), (1, 32), (8, 8), (8, 32)):
            for length in (8192, 32768):
                records.append(measure_shape(batch, heads, length, eviction))
                torch.cuda.empty_cache()
    write_json_exclusive(
        Path(__file__).with_name("prefill-shape-check.json"),
        {
            "device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "rocm": torch.version.hip, "head_dim": DIM, "dtype": str(DTYPE),
            "schedule": "sqrt16", "records": records,
        },
    )


if __name__ == "__main__":
    main()
