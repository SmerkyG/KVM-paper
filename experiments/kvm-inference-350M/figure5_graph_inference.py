"""Measure full-model CUDA Graph replay across the Figure 5 context lengths."""

import argparse

from large_batch_inference import (
    CapturedGeneration, Generation, MixerConfig, Path, PreallocatedKVCache,
    RWKV7BackboneForCausalLM, TritonKVMReplayCache, compare, measure, outputs,
    root, torch, utils, write_json_exclusive,
)
from model.gptalpha_mixer import SequenceMixer as GPTA2Mixer
from model.kvm_triton_mixer import SequenceMixer as TritonKVMMixer
from figure5_memory import cache_storages, measure_memory


@torch.inference_mode()
def run(args):
    utils.grad_cp.use_grad_cp = False
    device = torch.device("cuda", 0)
    checkpoint = args.checkpoint.resolve()
    config = MixerConfig.from_pretrained(checkpoint, local_files_only=True)
    is_kvm = args.model in ("sqrt", "fixed256")
    chunked = args.prefill_chunk_size is not None
    series = args.model
    if chunked:
        if is_kvm and args.prefill_chunk_size % config.chunk_len:
            raise ValueError("KVM prefill chunk size must be a multiple of the recurrent chunk length")
        series = f"{args.model}_chunked_{args.prefill_chunk_size}" if is_kvm else "gpta2_chunked"
    if args.model == "fixed256":
        if config.token_mixer != "kvm" or config.state_budget_mode != "fixed" or config.state_min_len != 256:
            raise ValueError("Expected the 350M fixed-256 KVM checkpoint")
        config.token_mixer_class_path = "model.kvm_triton_mixer.SequenceMixer"
    if args.model == "sqrt" and (
        config.state_budget_mode != "power_law"
        or config.state_growth_exponent != 0.5
        or config.state_growth_factor != 16.0
    ):
        raise ValueError("Expected the 350M KVM-sqrt checkpoint")

    model, loading = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16, local_files_only=True,
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"]:
        raise RuntimeError(f"Checkpoint and mixer weights do not match: {loading}")
    model.to(device).eval()
    mixer_types = {type(layer.attn) for layer in model.transformer.h}
    if mixer_types != {TritonKVMMixer if is_kvm else GPTA2Mixer}:
        raise ValueError(f"Unexpected sequence mixers: {mixer_types}")
    metadata = {
        "model": series, "checkpoint": str(checkpoint),
        "mixer": next(iter(mixer_types)).__module__,
        "parameters": sum(param.numel() for param in model.parameters()),
        "batch_size": 32, "new_tokens": args.decode_steps + 1, "rounds": 7,
        "prefill_chunk_size": args.prefill_chunk_size,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rocm": torch.version.hip,
        "dtype": "bfloat16", "seed": 0, "mode": "graph",
    }
    for length in args.contexts:
        print(f"{series} {length}: preparing", flush=True)
        rng = torch.Generator(device=device).manual_seed(0)
        prompt = torch.randint(config.vocab_size, (32, length), generator=rng, device=device)
        cache = (
            TritonKVMReplayCache() if is_kvm
            else PreallocatedKVCache(config.num_hidden_layers, length + args.decode_steps)
        )
        prepared = Generation(
            model, prompt, args.decode_steps + 1, cache, prefill_chunk_size=args.prefill_chunk_size,
        )
        if args.memory_breakdown:
            stats = measure_memory(prepared, prompt, decode_graph_steps=args.decode_graph_steps)
            path = args.output / "memory-raw" / f"{series}-{length}.json"
            write_json_exclusive(path, {**metadata, "context": length, **stats})
            print(
                f"{series} {length}: decode cache={stats['decode_cache_bytes'] / 2**30:.2f} GiB, "
                f"prefill peak={stats['eager_prefill_peak_allocated_bytes'] / 2**30:.2f} GiB, "
                f"graph pool={stats['graph_pool_reserved_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )
            del prepared, cache, prompt
            torch.cuda.empty_cache()
            continue
        other = (prompt + 1) % config.vocab_size
        references = [outputs(prepared, tokens) for tokens in (prompt, other)]
        unchunked_check = None
        if chunked and length > args.prefill_chunk_size:
            unchunked = Generation(model, prompt, args.decode_steps + 1, cache)
            full_first, full_logits = unchunked.prefill(prompt)
            token_ids, logits, _ = references[0]
            unchunked_check = {
                "first_token_mismatches": (full_first != token_ids[:, :1]).sum().item(),
                "first_logit_mismatches": (~torch.isclose(logits, full_logits, atol=0.02, rtol=0.01)).sum().item(),
                "max_abs_logit_difference": (logits - full_logits).abs().max().item(),
            }
            print(f"{series} {length}: unchunked prefill check {unchunked_check}", flush=True)
            del unchunked, full_first, full_logits, token_ids, logits, _
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        graph = CapturedGeneration(prepared, prompt, 1, decode_graph_steps=args.decode_graph_steps)
        torch.cuda.synchronize()
        setup_peak_reserved = torch.cuda.max_memory_reserved()
        state_updates = None
        if is_kvm:
            counts = {
                layer.states_dict[TritonKVMMixer._CACHE_TRITON_DECODE].update_count
                for layer in cache.layers
            }
            if len(counts) != 1:
                raise AssertionError(f"Inconsistent decode update counts across layers: {counts}")
            state_updates = counts.pop()
        if cache.get_seq_length(0) != length + args.decode_steps:
            raise AssertionError("Captured decode did not advance through the full trajectory")
        print(
            f"{series} {length}: captured {args.decode_steps} steps in "
            f"{len(graph.decode_graphs)} graphs, updates={state_updates}", flush=True,
        )

        checks = [
            compare(outputs(graph, tokens), expected)
            for tokens, expected in zip((prompt, other, prompt), (*references, references[0]))
        ]
        del references, other
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        stats = measure(graph, prompt, 7)
        storage, _ = cache_storages(cache)
        path = args.output / "raw" / f"{series}-{length}.json"
        write_json_exclusive(path, {
            **metadata, "context": length, "capture_seconds": graph.capture_seconds,
            "setup_peak_reserved_bytes": setup_peak_reserved, "output_checks": checks, **stats,
            "state_updates_per_layer": state_updates,
            "decode_graphs": len(graph.decode_graphs),
            "decode_cache_bytes": sum(storage.values()),
            "unchunked_prefill_check": unchunked_check,
        })
        print(
            f"{series} {length}: prefill={stats['prefill_ms']['median']:.2f} ms, "
            f"decode={stats['decode_ms']['median'] / args.decode_steps:.3f} ms/step, "
            f"reserved={stats['peak_reserved_bytes'] / 2**30:.2f} GiB",
            flush=True,
        )
        del graph, prepared, cache, prompt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gpta2", "sqrt", "fixed256"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contexts", nargs="+", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, default=31)
    parser.add_argument("--decode-graph-steps", type=int)
    parser.add_argument("--prefill-chunk-size", type=int)
    parser.add_argument("--memory-breakdown", action="store_true")
    parser.add_argument("--output", type=Path, default=root / "experiments/kvm-inference-350M/figure5-b32")
    args = parser.parse_args()
    if args.decode_steps <= 0:
        parser.error("--decode-steps must be positive")
    if args.decode_graph_steps is not None and args.decode_graph_steps <= 0:
        parser.error("--decode-graph-steps must be positive")
    if args.prefill_chunk_size is not None and args.prefill_chunk_size <= 0:
        parser.error("--prefill-chunk-size must be positive")
    run(args)
