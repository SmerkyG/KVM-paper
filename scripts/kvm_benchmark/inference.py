"""Full-model greedy generation with optional CUDA Graph replay."""

from __future__ import annotations

import time

import torch

from .common import distribution, package_version, write_json_exclusive
from model.gptalpha_mixer import SequenceMixer as GPTA2Mixer
from model.kvm_triton_mixer import SequenceMixer as TritonKVMMixer, TritonKVMReplayCache
from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM
from model.statesdictcache import PreallocatedKVCache, StatesDictCache
import utils.grad_cp


class Generation:
    def __init__(self, model, prompt, new_tokens, cache, *, prefill_chunk_size=None):
        if prefill_chunk_size is not None and prefill_chunk_size <= 0:
            raise ValueError("Prefill chunk size must be positive")
        self.model = model
        self.cache = cache
        self.prefill_chunk_size = prefill_chunk_size
        prompt_len = prompt.size(1)
        positions = torch.arange(prompt_len + new_tokens - 1, device=prompt.device)[None, :]
        self.prompt_positions = positions[:, :prompt_len]
        self.decode_positions = positions[:, prompt_len:].split(1, dim=1)

    def next_token(self, tokens, positions, **kwargs):
        logits = self.model(
            input_ids=tokens, position_ids=positions, past_key_values=self.cache,
            use_cache=True, logits_to_keep=1, **kwargs,
        ).logits
        return logits.argmax(dim=-1), logits

    def prefill(self, prompt):
        self.cache.reset()
        chunk = self.prefill_chunk_size
        if chunk is None or prompt.size(1) <= chunk:
            return self.next_token(prompt, self.prompt_positions)
        last_start = (prompt.size(1) - 1) // chunk * chunk
        for start in range(0, last_start, chunk):
            self.model.transformer(
                input_ids=prompt[:, start:start + chunk],
                position_ids=self.prompt_positions[:, start:start + chunk],
                past_key_values=self.cache, use_cache=True, prefill_has_more=True,
            )
        return self.next_token(
            prompt[:, last_start:], self.prompt_positions[:, last_start:], prefill_has_more=False,
        )

    def decode_span(self, first, positions_for_span):
        token, logits = first
        generated = []
        for positions in positions_for_span:
            token, logits = self.next_token(token, positions)
            generated.append(token)
        return generated, (token, logits)

    def decode(self, first):
        generated, last = self.decode_span(first, self.decode_positions)
        return torch.cat([first[0], *generated], dim=1), last[1]


class CapturedGeneration:
    """Replay a fixed-size prefill and growing-length greedy decode trajectory."""

    def __init__(self, generation, prompt, warmups, *, decode_graph_steps=None, after_prefill_capture=None):
        self.generation = generation  # Own the captured model, cache, and position tensors.
        self.prompt = prompt.clone()
        self.prefill_graph = torch.cuda.CUDAGraph()
        self.decode_graphs = []
        stream = torch.cuda.Stream(device=prompt.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmups):
                generation.decode(generation.prefill(self.prompt))
        torch.cuda.current_stream().wait_stream(stream)
        started = time.perf_counter()
        with torch.cuda.graph(self.prefill_graph, stream=stream):
            self.first = generation.prefill(self.prompt)
        if after_prefill_capture is not None:
            after_prefill_capture(self.prefill_graph)
        # The graphs share a pool and must replay in capture order. Every
        # decode segment reads the cache and token written by the prior graph.
        positions = generation.decode_positions
        # Bound graph size so longer trajectories can finalize on ROCm.
        graph_steps = 32 if decode_graph_steps is None else decode_graph_steps
        if graph_steps <= 0:
            raise ValueError("Decode graphs require at least one step")
        last = self.first
        generated = [self.first[0]]
        for start in range(0, len(positions), graph_steps):
            end = min(start + graph_steps, len(positions))
            decode_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(decode_graph, stream=stream, pool=self.prefill_graph.pool()):
                tokens, last = generation.decode_span(last, positions[start:end])
                generated.extend(tokens)
                if end == len(positions):
                    self.generated = torch.cat(generated, dim=1), last[1]
            self.decode_graphs.append(decode_graph)
        self.capture_seconds = time.perf_counter() - started

    def prefill(self, prompt):
        self.prompt.copy_(prompt)
        self.prefill_graph.replay()
        return self.first

    def decode(self, _first):
        for graph in self.decode_graphs:
            graph.replay()
        return self.generated


def outputs(runner, prompt):
    first = runner.prefill(prompt)
    generated, last_logits = runner.decode(first)
    return generated.clone(), first[1].clone(), last_logits.clone()


def validate_outputs(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for logits, reference in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(logits, reference, rtol=0.01, atol=0.02)


def measure(runner, prompt, rounds):
    start, prefill_end, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(prompt.device)
    samples = []
    for _ in range(rounds):
        wall_start = time.perf_counter()
        start.record()
        first = runner.prefill(prompt)
        prefill_end.record()
        generated, last_logits = runner.decode(first)
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1e3
        samples.append({
            "prefill_ms": start.elapsed_time(prefill_end),
            "decode_ms": prefill_end.elapsed_time(end),
            "wall_ms": wall_ms,
        })
        del first, generated, last_logits
    return {
        "samples": samples,
        **{key: distribution([sample[key] for sample in samples]) for key in samples[0]},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(prompt.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(prompt.device),
    }


def benchmark_coordinate(model, prompt, args, backend):
    eager = Generation(
        model, prompt, args.new_tokens, StatesDictCache(), prefill_chunk_size=args.prefill_chunk_size,
    )
    # A different prompt detects stale KV or token-shift state on replay.
    other = (prompt + 1) % model.config.vocab_size
    references = [outputs(eager, sample) for sample in (prompt, other)]
    for _ in range(args.warmup):
        eager.decode(eager.prefill(prompt))
    results = {"eager": measure(eager, prompt, args.rounds)}
    del eager
    torch.cuda.empty_cache()

    if backend == "triton_kvm":
        cache = TritonKVMReplayCache()
        mode_prefix = "prepared"
    else:
        capacity = prompt.size(1) + args.new_tokens - 1
        cache = PreallocatedKVCache(model.config.num_hidden_layers, capacity)
        mode_prefix = "static"
    static = Generation(model, prompt, args.new_tokens, cache, prefill_chunk_size=args.prefill_chunk_size)
    for sample, reference in zip((prompt, other), references):
        validate_outputs(outputs(static, sample), reference)
    for _ in range(args.warmup):
        static.decode(static.prefill(prompt))
    results[f"{mode_prefix}_eager"] = measure(static, prompt, args.rounds)
    if args.cudagraph:
        captured = CapturedGeneration(static, prompt, args.warmup)
        for sample, reference in zip((prompt, other, prompt), (*references, references[0])):
            validate_outputs(outputs(captured, sample), reference)
        results[f"{mode_prefix}_graph"] = {
            **measure(captured, prompt, args.rounds),
            "capture_seconds": captured.capture_seconds,
        }
    return results


@torch.inference_mode()
def run_inference(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Full-model inference timing requires a CUDA/ROCm GPU")
    torch.cuda.set_device(args.device)
    # Inference needs neither recomputation nor checkpoint RNG snapshots.
    utils.grad_cp.use_grad_cp = False
    device = torch.device("cuda", args.device)
    checkpoint = args.checkpoint.resolve()
    config = MixerConfig.from_pretrained(str(checkpoint), local_files_only=True)
    model = RWKV7BackboneForCausalLM.from_pretrained(
        str(checkpoint), config=config, dtype=torch.bfloat16, local_files_only=True,
    ).to(device).eval()
    mixers = {type(layer.attn) for layer in model.transformer.h}
    if mixers == {GPTA2Mixer}:
        backend = "gpta2"
    elif mixers == {TritonKVMMixer}:
        backend = "triton_kvm"
    else:
        raise ValueError("The full-model benchmark requires all GPTA-2 or all Triton KVM layers")
    if args.prefill_chunk_size is not None and backend == "triton_kvm":
        if args.prefill_chunk_size % config.chunk_len:
            raise ValueError("KVM prefill chunk size must be a multiple of the recurrent chunk length")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = {
        "checkpoint": str(checkpoint),
        "backend": backend,
        "parameters": sum(param.numel() for param in model.parameters()),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "hip": torch.version.hip,
        "transformers": package_version("transformers"),
        "dtype": "bfloat16", "batch_size": args.batch_size,
        "new_tokens": args.new_tokens, "warmup": args.warmup, "rounds": args.rounds,
        "prefill_chunk_size": args.prefill_chunk_size,
        "seed": args.seed, "results": {},
    }
    for length in args.contexts:
        prompt = torch.randint(
            config.vocab_size, (args.batch_size, length), device=device, generator=generator,
        )
        modes = benchmark_coordinate(model, prompt, args, backend)
        result["results"][str(length)] = modes
        for mode, stats in modes.items():
            print(
                f"context={length} mode={mode} prefill={stats['prefill_ms']['median']:.2f} ms "
                f"decode={stats['decode_ms']['median'] / (args.new_tokens - 1):.3f} ms/step "
                f"wall={stats['wall_ms']['median']:.2f} ms",
                flush=True,
            )
        torch.cuda.empty_cache()
    write_json_exclusive(args.output, result)
