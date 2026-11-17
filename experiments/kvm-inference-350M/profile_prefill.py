"""Attribute the matched 8K and 32K graph prefill runs to GPU kernels."""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root), str(root / "scripts")]
from kvm_benchmark.inference import CapturedGeneration, Generation, validate_outputs
from model.kvm_triton_mixer import TritonKVMReplayCache
from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM
from model.statesdictcache import PreallocatedKVCache
import utils.grad_cp


@torch.inference_mode()
def profile_prefill(args):
    utils.grad_cp.use_grad_cp = False
    device = torch.device("cuda", 0)
    results = {}
    for backend in ("gpta2", "kvm"):
        previous = root / "experiments" / f"{backend}-inference-350M" / "batch1-final.json"
        checkpoint = json.loads(previous.read_text())["checkpoint"]
        config = MixerConfig.from_pretrained(checkpoint, local_files_only=True)
        model = RWKV7BackboneForCausalLM.from_pretrained(
            checkpoint, config=config, dtype=torch.bfloat16, local_files_only=True,
        ).to(device).eval()
        generator = torch.Generator(device=device).manual_seed(0)
        results[backend] = {}
        for length in args.contexts:
            prompt = torch.randint(
                config.vocab_size, (args.batch_size, length), device=device, generator=generator,
            )
            cache = (
                TritonKVMReplayCache() if backend == "kvm"
                else PreallocatedKVCache(config.num_hidden_layers, length + 31)
            )
            generation = Generation(model, prompt, 2, cache)
            eager = tuple(value.clone() for value in generation.prefill(prompt))
            runner = CapturedGeneration(generation, prompt, 2)
            validate_outputs(runner.prefill(prompt), eager)
            times = []
            for _ in range(7):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                runner.prefill(prompt)
                stop.record()
                stop.synchronize()
                times.append(start.elapsed_time(stop))
            times_ms = statistics.median(times)
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            # ROCm's profiler exposes the graph's prompt copy but omits the
            # replayed kernels. Trace the same prefill calls in prepared eager.
            with torch.profiler.profile(activities=activities) as profile:
                generation.prefill(prompt)
                torch.cuda.synchronize()
            kernel_times = defaultdict(float)
            kernel_calls = defaultdict(int)
            for event in profile.events():
                if event.device_type == torch.autograd.DeviceType.CUDA:
                    kernel_times[event.name] += event.device_time_total / 1e3
                    kernel_calls[event.name] += 1
            if sum(kernel_times.values()) < 1.0:
                raise RuntimeError(f"No GPU kernel trace for {backend} at {length}")
            print(backend, length, f"prefill graph {times_ms:.2f} ms, traced prepared eager", flush=True)
            print(f"GPU total {sum(kernel_times.values()):.2f} ms, {sum(kernel_calls.values())} calls", flush=True)
            for name, ms in sorted(kernel_times.items(), key=lambda x: -x[1])[:18]:
                print(f"{ms:8.2f} ms  {kernel_calls[name]:5d}  {name[:170]}", flush=True)
            results[backend][length] = {
                "graph_prefill_ms": times,
                "trace_mode": "prepared_eager",
                "gpu_kernels": [
                    {"name": name, "ms": ms, "calls": kernel_calls[name]}
                    for name, ms in sorted(kernel_times.items(), key=lambda x: -x[1])
                ],
            }
            del runner, generation, cache, eager
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    with args.output.open("x") as stream:
        json.dump(results, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("prefill-kernel-profile.json"),
    )
    profile_prefill(parser.parse_args())
