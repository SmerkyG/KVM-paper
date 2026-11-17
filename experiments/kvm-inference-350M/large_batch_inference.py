"""Time large batches, retaining eager results if graph capture cannot fit."""

import argparse
import json
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root), str(root / "scripts")]
from kvm_benchmark.common import write_json_exclusive
from kvm_benchmark.inference import CapturedGeneration, Generation, measure, outputs
from model.kvm_triton_mixer import TritonKVMReplayCache
from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM
from model.statesdictcache import PreallocatedKVCache, StatesDictCache
import utils.grad_cp


def compare(actual, expected):
    errors = [
        (~torch.isclose(x, y, atol=0.02, rtol=0.01)).sum().item()
        for x, y in zip(actual[1:], expected[1:])
    ]
    return {
        "greedy_token_mismatches": (actual[0] != expected[0]).sum().item(),
        "first_logit_mismatches": errors[0],
        "last_logit_mismatches": errors[1],
    }


@torch.inference_mode()
def run(args):
    utils.grad_cp.use_grad_cp = False
    device = torch.device("cuda", 0)
    previous = root / "experiments" / f"{args.model}-inference-350M" / "batch1-final.json"
    checkpoint = json.loads(previous.read_text())["checkpoint"]
    config = MixerConfig.from_pretrained(checkpoint, local_files_only=True)
    model = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16, local_files_only=True,
    ).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(0)
    prompt = torch.randint(
        config.vocab_size, (args.batch_size, args.context), device=device, generator=generator,
    )
    other = (prompt + 1) % config.vocab_size
    metadata = {
        "model": args.model, "checkpoint": checkpoint, "batch_size": args.batch_size,
        "context": args.context, "new_tokens": 32, "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(), "device_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__, "rocm": torch.version.hip, "rounds": args.rounds,
    }

    def save(mode, result):
        path = args.output / f"{args.model}-b{args.batch_size}-t{args.context}-{mode}.json"
        write_json_exclusive(path, {**metadata, "mode": mode, **result})
        print(mode, result.get("status"), "saved to", path, flush=True)
        if result["status"] == "ok":
            print(
                f"prefill={result['prefill_ms']['median']:.2f} ms "
                f"wall={result['wall_ms']['median']:.2f} ms "
                f"peak_reserved={result['peak_reserved_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )

    def prepared_generation():
        cache = (
            TritonKVMReplayCache() if args.model == "kvm"
            else PreallocatedKVCache(config.num_hidden_layers, args.context + 31)
        )
        return Generation(model, prompt, 32, cache)

    if args.prepared_eager_only:
        try:
            prepared = prepared_generation()
            outputs(prepared, prompt)
            outputs(prepared, other)
            save("prepared_eager", {"status": "ok", **measure(prepared, prompt, args.rounds)})
        except torch.OutOfMemoryError as exc:
            save("prepared_eager", {"status": "oom", "error": str(exc)})
        return

    print(f"Running eager prefill and generation at B={args.batch_size}, T={args.context}", flush=True)
    try:
        eager = Generation(model, prompt, 32, StatesDictCache())
        references = [outputs(eager, sample) for sample in (prompt, other)]
        save("eager", {"status": "ok", **measure(eager, prompt, args.rounds)})
    except torch.OutOfMemoryError as exc:
        save("eager", {"status": "oom", "error": str(exc)})
        return
    del eager
    torch.cuda.empty_cache()

    print(f"Capturing prefill and generation at B={args.batch_size}, T={args.context}", flush=True)
    try:
        graph = CapturedGeneration(prepared_generation(), prompt, 1)
        checks = [
            compare(outputs(graph, sample), reference)
            for sample, reference in zip((prompt, other, prompt), (*references, references[0]))
        ]
        save("graph", {
            "status": "ok",
            **measure(graph, prompt, args.rounds),
            "capture_seconds": graph.capture_seconds,
            "output_checks": checks,
        })
    except torch.OutOfMemoryError as exc:
        save("graph", {"status": "oom", "error": str(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gpta2", "kvm"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--prepared-eager-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
