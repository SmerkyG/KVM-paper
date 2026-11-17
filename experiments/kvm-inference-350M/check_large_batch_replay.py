"""Compare repeated KVM generation within and across eager and graph modes."""

import argparse
import itertools
import json
from pathlib import Path

from large_batch_inference import (
    CapturedGeneration, Generation, MixerConfig, RWKV7BackboneForCausalLM,
    StatesDictCache, TritonKVMReplayCache, outputs, root, torch, utils,
    write_json_exclusive,
)


def compare_pairs(left, right, *, same_mode=False):
    pairs = itertools.combinations(left, 2) if same_mode else itertools.product(left, right)
    records = []
    for (ids_a, first_a, last_a), (ids_b, first_b, last_b) in pairs:
        differences = ids_a != ids_b
        same_history = ~differences.any(dim=-1)
        records.append({
            "greedy_token_differences": differences.sum().item(),
            "different_sequences": (~same_history).sum().item(),
            "first_logit_differences": (~torch.isclose(first_a, first_b, atol=0.02, rtol=0.01)).sum().item(),
            "last_logit_differences_with_same_history": (
                ~torch.isclose(last_a[same_history], last_b[same_history], atol=0.02, rtol=0.01)
            ).sum().item(),
        })
    return records


@torch.inference_mode()
def run(batch, context, repeats, prepared_eager):
    utils.grad_cp.use_grad_cp = False
    torch.set_num_threads(2)
    device = torch.device("cuda", 0)
    checkpoint = json.loads(
        (root / "experiments/kvm-inference-350M/batch1-final.json").read_text()
    )["checkpoint"]
    config = MixerConfig.from_pretrained(checkpoint, local_files_only=True)
    model = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16, local_files_only=True,
    ).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(0)
    first_prompt = torch.randint(config.vocab_size, (batch, context), device=device, generator=generator)
    prompts = (first_prompt, (first_prompt + 1) % config.vocab_size)
    eager_cache = TritonKVMReplayCache() if prepared_eager else StatesDictCache()
    eager = Generation(model, first_prompt, 32, eager_cache)

    def sample(runner):
        results = [[], []]
        for _ in range(repeats):
            for prompt, result in zip(prompts, results):
                result.append(tuple(tensor.cpu() for tensor in outputs(runner, prompt)))
        return results

    eager_samples = sample(eager)
    del eager, eager_cache
    torch.cuda.empty_cache()
    print(f"B={batch} finished {2 * repeats} eager generations", flush=True)
    graph = CapturedGeneration(
        Generation(model, first_prompt, 32, TritonKVMReplayCache()), first_prompt, 1,
    )
    graph_samples = sample(graph)
    print(f"B={batch} finished {2 * repeats} graph generations", flush=True)
    results = {
        "batch": batch, "context": context, "new_tokens": 32,
        "repeats_per_prompt": repeats, "prepared_eager": prepared_eager, "prompts": [],
    }
    for eager_group, graph_group in zip(eager_samples, graph_samples):
        results["prompts"].append({
            "eager_to_eager": compare_pairs(eager_group, eager_group, same_mode=True),
            "graph_to_graph": compare_pairs(graph_group, graph_group, same_mode=True),
            "graph_to_eager": compare_pairs(graph_group, eager_group),
        })
    write_json_exclusive(
        Path(__file__).with_name(f"batch{batch}-{context // 1024}k-replay-parity.json"), results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=[16, 32], required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--prepared-eager", action="store_true")
    args = parser.parse_args()
    run(args.batch_size, args.context, args.repeats, args.prepared_eager)
