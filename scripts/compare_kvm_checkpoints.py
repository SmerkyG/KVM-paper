import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import load_file


def checkpoint_file(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "model.safetensors"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def group_predicates() -> dict[str, Callable[[str], bool]]:
    return {
        "all": lambda _name: True,
        "kvm": lambda name: ".attn." in name,
        "kvm_qkv": lambda name: bool(
            re.search(r"\.attn\.c_[qkv]\.weight$", name)
        ),
        "kvm_output": lambda name: name.endswith(".attn.c_proj.weight"),
        "kvm_routing": lambda name: ".attn.key_weighting.weight" in name,
        "kvm_state_norm": lambda name: ".attn.ln_s_k." in name,
        "kvm_temperatures": lambda name: bool(
            re.search(r"\.attn\.(front|state)_head_temp$", name)
        ),
        "non_kvm": lambda name: ".attn." not in name,
    }


def tensor_sums(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    initial: torch.Tensor | None,
) -> dict[str, float]:
    ref = reference.double().reshape(-1)
    cand = candidate.double().reshape(-1)
    diff = cand - ref
    result = {
        "count": float(ref.numel()),
        "reference_sq": float(torch.dot(ref, ref)),
        "candidate_sq": float(torch.dot(cand, cand)),
        "dot": float(torch.dot(ref, cand)),
        "diff_sq": float(torch.dot(diff, diff)),
        "max_abs": float(diff.abs().max()),
    }
    if initial is not None:
        init = initial.double().reshape(-1)
        ref_update = ref - init
        candidate_update = cand - init
        update_diff = candidate_update - ref_update
        result.update(
            reference_update_sq=float(torch.dot(ref_update, ref_update)),
            candidate_update_sq=float(torch.dot(candidate_update, candidate_update)),
            update_dot=float(torch.dot(ref_update, candidate_update)),
            update_diff_sq=float(torch.dot(update_diff, update_diff)),
        )
    return result


def add_sums(total: dict[str, float], values: dict[str, float]) -> None:
    total["max_abs"] = max(total.get("max_abs", 0.0), values["max_abs"])
    for key, value in values.items():
        if key == "max_abs":
            continue
        total[key] = total.get(key, 0.0) + value


def finish(values: dict[str, float]) -> dict[str, float]:
    eps = 1.0e-30
    result = {
        "parameters": int(values["count"]),
        "relative_l2": (values["diff_sq"] / max(values["reference_sq"], eps))
        ** 0.5,
        "cosine": values["dot"]
        / max((values["reference_sq"] * values["candidate_sq"]) ** 0.5, eps),
        "max_abs": values["max_abs"],
    }
    if "reference_update_sq" in values:
        result.update(
            update_relative_l2=(
                values["update_diff_sq"]
                / max(values["reference_update_sq"], eps)
            )
            ** 0.5,
            update_cosine=values["update_dot"]
            / max(
                (
                    values["reference_update_sq"]
                    * values["candidate_update_sq"]
                )
                ** 0.5,
                eps,
            ),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a KVM training checkpoint with an eager reference."
    )
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument(
        "--initial",
        help="Optional shared initialization checkpoint for update-vector metrics.",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    reference_path = checkpoint_file(args.reference)
    candidate_path = checkpoint_file(args.candidate)
    initial_path = checkpoint_file(args.initial) if args.initial else None
    reference = load_file(str(reference_path), device="cpu")
    candidate = load_file(str(candidate_path), device="cpu")
    initial = load_file(str(initial_path), device="cpu") if initial_path else None

    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise ValueError(f"checkpoint keys differ: missing={missing}, extra={extra}")
    if initial is not None and reference.keys() != initial.keys():
        raise ValueError("initial checkpoint keys differ from reference")

    predicates = group_predicates()
    group_sums: dict[str, dict[str, float]] = {name: {} for name in predicates}
    per_tensor = []
    for name in sorted(reference):
        values = tensor_sums(
            reference[name], candidate[name], None if initial is None else initial[name]
        )
        for group, predicate in predicates.items():
            if predicate(name):
                add_sums(group_sums[group], values)
        tensor_result = finish(values)
        tensor_result["name"] = name
        per_tensor.append(tensor_result)

    per_tensor.sort(key=lambda item: item["relative_l2"], reverse=True)
    report = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "initial": None if initial_path is None else str(initial_path),
        "groups": {
            name: finish(values)
            for name, values in group_sums.items()
            if values.get("count", 0.0) > 0.0
        },
        "largest_tensor_drifts": per_tensor[: max(args.top, 0)],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
