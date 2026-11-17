"""Check whether 32K KVM prefill differences also occur between eager calls."""

import json
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root), str(root / "scripts")]
from kvm_benchmark.common import distribution, write_json_exclusive
from kvm_benchmark.inference import CapturedGeneration, Generation
from model.kvm_triton_mixer import TritonKVMReplayCache
from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM
from model.statesdictcache import StatesDictCache
import utils.grad_cp


def compare(logits, reference):
    error = (logits.float() - reference.float()).abs()
    close = error <= 0.02 + 0.01 * reference.float().abs()
    return {
        "max_abs_error": error.amax(dim=(-2, -1)).tolist(),
        "nonclose_fraction": (~close).float().mean(dim=(-2, -1)).tolist(),
        "greedy": logits[:, 0].argmax(dim=-1).tolist(),
    }


@torch.inference_mode()
def main():
    utils.grad_cp.use_grad_cp = False
    checkpoint = root / "experiments" / "kvm-sqrt-triton-350M-20261111" / "runs" / "K0"
    config = MixerConfig.from_pretrained(checkpoint, local_files_only=True)
    device = torch.device("cuda", 0)
    model = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16, local_files_only=True,
    ).to(device).eval()
    rng = torch.Generator(device=device).manual_seed(0)
    prompt = torch.randint(config.vocab_size, (8, 32768), device=device, generator=rng)
    other = (prompt + 1) % config.vocab_size
    eager = Generation(model, prompt, 2, StatesDictCache())
    prepared = Generation(model, prompt, 2, TritonKVMReplayCache())
    references = [eager.prefill(tokens)[1].clone() for tokens in (prompt, other)]
    results = {}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))

    def sample(mode, runner):
        times = []
        comparisons = []
        for index in (0, 1, 0, 1, 0, 1, 0, 1):
            tokens = (prompt, other)[index]
            start.record()
            _, logits = runner.prefill(tokens)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
            comparisons.append({"prompt": index, **compare(logits, references[index])})
        results[mode] = {
            "latency_ms": distribution(times),
            "samples_ms": times,
            "comparisons": comparisons,
        }
        print(mode, json.dumps(results[mode]), flush=True)

    sample("eager", eager)
    del eager
    torch.cuda.empty_cache()
    prepared.prefill(prompt)
    sample("prepared_eager", prepared)
    graph = CapturedGeneration(prepared, prompt, 2)
    sample("prepared_graph", graph)
    results["references_greedy"] = [ref[:, 0].argmax(dim=-1).tolist() for ref in references]
    write_json_exclusive(Path(__file__).with_name("batch8-32k-replay-diagnostic.json"), results)


if __name__ == "__main__":
    main()
