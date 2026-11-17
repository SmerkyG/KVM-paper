"""Compare same-prompt first-token logits across repeated KVM prefills."""

import argparse
from itertools import combinations, product

from large_batch_inference import (
    Generation, MixerConfig, Path, RWKV7BackboneForCausalLM, TritonKVMReplayCache,
    torch, utils, write_json_exclusive,
)


def compare_runs(left, right):
    tokens_a, logits_a = left
    tokens_b, logits_b = right
    close = torch.isclose(logits_a, logits_b, atol=0.02, rtol=0.01)
    return {
        "different_tokens": (tokens_a != tokens_b).sum().item(),
        "sequences_outside_logit_tolerance": (~close.flatten(1).all(dim=1)).sum().item(),
        "max_abs_logit_difference": (logits_a - logits_b).abs().max().item(),
    }


@torch.inference_mode()
def run(checkpoint, output):
    utils.grad_cp.use_grad_cp = False
    torch.set_num_threads(2)
    device = torch.device("cuda", 0)
    config = MixerConfig.from_pretrained(checkpoint, local_files_only=True)
    config.token_mixer_class_path = "model.kvm_triton_mixer.SequenceMixer"
    model = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16, local_files_only=True,
    ).to(device).eval()
    rng = torch.Generator(device=device).manual_seed(0)
    prompt = torch.randint(config.vocab_size, (32, 32768), generator=rng, device=device)
    cache = TritonKVMReplayCache()
    samples = {name: [] for name in ("one_shot", "chunk_2048", "chunk_8192")}
    for _ in range(3):
        for name, size in (("one_shot", None), ("chunk_2048", 2048), ("chunk_8192", 8192)):
            runner = Generation(model, prompt, 2, cache, prefill_chunk_size=size)
            samples[name].append(tuple(tensor.cpu() for tensor in runner.prefill(prompt)))
            print(name, len(samples[name]), flush=True)
    results = {
        name: [compare_runs(a, b) for a, b in combinations(group, 2)]
        for name, group in samples.items()
    }
    for name in ("chunk_2048", "chunk_8192"):
        results[f"one_shot_vs_{name}"] = [
            compare_runs(a, b) for a, b in product(samples["one_shot"], samples[name])
        ]
    write_json_exclusive(output, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
