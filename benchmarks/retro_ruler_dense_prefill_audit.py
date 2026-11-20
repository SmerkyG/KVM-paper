"""Replay two 32K retrieval misses on a shared forced decode prefix."""
import argparse
import hashlib
import json
import struct
import types
from pathlib import Path

import numpy as np
import torch

from benchmarks.retro_ruler_frontier_audit import reconstruct
from config import generate_config
from model_hub import load_model, load_tokenizer

from benchmarks.h100_comparison import configure_prefill_mlp_chunk
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_run import LIMITS, save


CASES = (("niah_single_3", 338), ("niah_multikey_1", 223))
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def inspect(pool, query, optimized, positions, context, intervention, log_details):
    state = pool.state
    cache, _, _, _ = reconstruct(pool, context)
    scratch = pool._threshold_scratch
    kh, group, dim = pool.kv_heads, pool.query_heads // pool.kv_heads, pool.head_dim
    q = query.reshape(kh, group, dim).float()
    n = cache.state.size
    count = cache.state.count[0]
    route = scratch["route"].view(kh, group, -1)[..., :n]
    native = scratch["native"].view(kh, group, -1)[..., :n]
    opened = scratch["allowed"].view(kh, group, -1)[..., :n].bool()
    owners = cache.owner[0][:, None].expand(-1, group, -1)
    mean_v = pool._threshold_frontier_cache["mean_v"].reshape(kh, -1, dim)[:, :n].float()

    saved = int(scratch["saved"].item())
    local = min(saved, int(pool.decode_local_limit))
    exact_k, exact_v = [], []
    for field, arrays in (("k", exact_k), ("v", exact_v)):
        recent = state[f"recent_{field}"][0]
        arrays.extend((recent[:, :local], recent[:, saved:saved+1], state[f"sink_{field}"][0]))
    exact_k, exact_v = torch.cat(exact_k, dim=1).float(), torch.cat(exact_v, dim=1).float()
    remote_k, remote_v = cache.archive_key[0].float(), cache.archive_value[0].float()
    remote = torch.einsum("hgd,htd->hgt", q, remote_k) * pool.engine.scaling
    exact = torch.einsum("hgd,htd->hgt", q, exact_k) * pool.engine.scaling
    dense = torch.cat((remote, exact), dim=-1).softmax(-1)
    remote_len = remote.shape[-1]
    sink_len = state["sink_k"].shape[2]
    gold_indices = torch.tensor(positions["answer"], device=q.device) - sink_len
    if not bool(((gold_indices >= 0) & (gold_indices < remote_len)).all()):
        raise RuntimeError("Answer positions must be in the remote archive")

    def frontier(mask):
        remote_open = mask.gather(-1, owners)
        logits = torch.cat((remote.masked_fill(~remote_open, -torch.inf),
                            exact, native.masked_fill(mask, -torch.inf)), dim=-1)
        p = logits.softmax(-1)
        return (torch.einsum("hgt,htd->hgd", p[..., :remote_len], remote_v)
                + torch.einsum("hgt,htd->hgd", p[..., remote_len:remote_len+exact.shape[-1]], exact_v)
                + torch.einsum("hgs,hsd->hgd", p[..., remote_len+exact.shape[-1]:], mean_v))

    reference = frontier(opened)
    if intervention == "reference":
        output = reference
    elif intervention == "open-answer":
        forced = opened.clone().scatter_(-1, owners[..., gold_indices], True)
        # Keep the original kernel's rounding residual when applying the
        # FP32 difference between the two frontiers.
        output = optimized.reshape_as(reference).float() + (frontier(forced) - reference)
    elif intervention == "dense":
        output = (torch.einsum("hgt,htd->hgd", dense[..., :remote_len], remote_v)
                  + torch.einsum("hgt,htd->hgd", dense[..., remote_len:], exact_v))
    else:
        output = optimized

    details = None
    if log_details:
        proxy = route.softmax(-1)
        exact_region = torch.zeros_like(proxy).scatter_add_(-1, owners, dense[..., :remote_len])
        cutoff = (pool.engine._dense_prefill_audit_multiplier / (count > 0).sum(-1))[:, None]
        expected_opened = (proxy > cutoff[..., None]) & (count[:, None] <= 1024)
        if not torch.equal(expected_opened, opened):
            raise RuntimeError("Materialized routing mask disagrees with its threshold")
        details = {
            "reference_max_abs": (optimized.reshape_as(reference).float() - reference).abs().max().item(),
            "reference_relative_l2": ((optimized.reshape_as(reference).float() - reference).norm()
                                      / reference.norm()).item(),
            "opened_regions": opened.sum(-1).flatten().tolist(),
            "opened_entries": (opened * count[:, None]).sum(-1).flatten().tolist(),
            "positions": {},
        }
        for name, tokens in positions.items():
            indices = torch.tensor(tokens, device=q.device) - sink_len
            if not bool(((indices >= 0) & (indices < remote_len)).all()):
                raise RuntimeError(f"{name} positions must be in the remote archive")
            slots = owners[..., indices]
            ranks = (route[..., None] > route.gather(-1, slots)[:, :, None, :]).sum(2) + 1
            details["positions"][name] = dict(
                slots=slots.reshape(kh*group, -1).tolist(),
                size=count[:, None].expand(-1, group, -1).gather(-1, slots).reshape(kh*group, -1).tolist(),
                open=opened.gather(-1, slots).reshape(kh*group, -1).tolist(),
                proxy_mass=proxy.gather(-1, slots).reshape(kh*group, -1).tolist(),
                proxy_rank=ranks.reshape(kh*group, -1).tolist(),
                threshold=cutoff.expand(-1, group).flatten().tolist(),
                exact_mass=dense[..., indices].reshape(kh*group, -1).tolist(),
                exact_region_mass=exact_region.gather(-1, slots).reshape(kh*group, -1).tolist(),
            )
    return output.reshape_as(optimized).to(optimized.dtype), details


def answer_span(prompt, offsets, value, before):
    start = prompt.index(value)
    if prompt.find(value, start+len(value)) != -1:
        raise RuntimeError(f"Expected one prompt occurrence of {value}")
    end = start + len(value)
    indices = [i for i, (lo, hi) in enumerate(offsets) if hi > lo and lo < end and hi > start]
    return indices, prompt[max(0, start-before):end+24]


def replay(model, tokenizer, inputs, prefix, task, positions, gold, dense_prefill, intervention):
    # Reattach using the model's native position method for independent prefills.
    model.position_embedd = model._audit_original_position
    attach_lod(model, "two-tier", dense_prefill=dense_prefill)
    model.attention_type = "LoD"
    model.batch_size = model.prefill_bsz = 1
    model.input_length = inputs.shape[1]
    model.max_new_length = LIMITS[task]
    model.prefill_method = "full"
    model.init_kv_cache(np.zeros(1, dtype=int), generate_config(MODEL, model.input_length, "Full_Flash_Attn"))
    prefill_logits = model.prefill_forward(inputs_ids=inputs)
    prefill_first = model.sampling(prefill_logits).item()
    del prefill_logits
    model.move()

    cache = model.kv_cache
    original = cache.decode_attention
    trace, choices = [], []
    enabled = False

    def decode(this, q, k, v, layer_idx):
        optimized = original(q, k, v, layer_idx)
        if not enabled:
            return optimized
        pool = this.pools[layer_idx]
        pool.engine._dense_prefill_audit_multiplier = this.threshold_multiplier
        out, detail = inspect(pool, q, optimized, positions, this.context, intervention, intervention == "kernel")
        if detail is not None:
            trace.append(dict(layer=layer_idx, **detail))
        return out

    cache.decode_attention = types.MethodType(decode, cache)
    for step, token_id in enumerate(prefix):
        enabled = step == len(prefix)-1
        token = torch.tensor([[token_id]], device=inputs.device)
        logits = model.decode_forward(inputs_ids=token).reshape(-1).float()
        choices.append(logits.argmax().item())
    top = logits.topk(5)
    out = {
        "prefill_first": prefill_first,
        "greedy_through_forced_prefix": [prefill_first, *choices],
        "top5": [{"id": i, "text": tokenizer.decode([i]), "logit": v}
                 for i, v in zip(top.indices.tolist(), top.values.tolist())],
        "gold_logit": logits[gold].item(),
        "gold_rank": 1 + (logits > logits[gold]).sum().item(),
        "layers": trace,
    }
    del cache.decode_attention
    del model.kv_cache
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("cohort", "dense", "sparse-prefill", "dense-prefill", "output"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    args = p.parse_args()

    def read_records(path):
        return {(r["task"], r["source_row"]): r for r in json.loads(path.read_text())["records"]}

    dense = read_records(args.dense)
    sparse_prefill = read_records(args.sparse_prefill)
    dense_prefill = read_records(args.dense_prefill)
    cohort = {(s["task"], s["source_row"]): s for s in json.loads(args.cohort.read_text())["samples"]}
    torch.set_grad_enabled(False)
    tokenizer = load_tokenizer(MODEL)
    model = load_model(MODEL, 65536, torch.bfloat16, "cuda:0", tokenizer)
    configure_prefill_mlp_chunk(model)
    model._audit_original_position = model.position_embedd

    result = {"stage": "running", "model": MODEL, "records": []}
    save(args.output, result)
    for task, row in CASES:
        key = task, row
        sample = cohort[key]
        good, bad, prior = (records[key] for records in (dense, dense_prefill, sparse_prefill))
        if not (good["score"] == prior["score"] == 1 and bad["score"] == 0):
            raise RuntimeError(f"Expected two passes and one miss for {key}")
        prompt = sample["prompt"]
        encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        inputs = encoded["input_ids"].cuda()
        offsets = encoded["offset_mapping"][0].tolist()
        ids = inputs[0].tolist()
        digest = hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()
        if any(record["input_token_sha256"] != digest for record in (good, bad, prior)):
            raise RuntimeError(f"Tokenized prompt mismatch for {key}")
        good_ids, bad_ids = good["output_token_ids"], bad["output_token_ids"]
        first = next(i for i, (a, b) in enumerate(zip(good_ids, bad_ids)) if a != b)
        positions, snippets = {}, {}
        positions["answer"], snippets["answer"] = answer_span(prompt, offsets, sample["answers"][0], 100)
        if task == "niah_multikey_1":
            wrong = tokenizer.decode(bad_ids[3:6])
            positions["distractor"], snippets["distractor"] = answer_span(prompt, offsets, wrong, 100)
        cases = {"first_divergence": (good_ids[:first], good_ids[first], first)}
        if task == "niah_multikey_1":
            cases["after_extra_space"] = (bad_ids[:first+1], good_ids[first], first+1)

        record = dict(task=task, source_row=row, prompt_length=len(ids), input_token_sha256=digest,
                      prompt_end=prompt[-310:], snippets=snippets, positions=positions,
                      position_tokens={k: [tokenizer.decode([ids[i]]) for i in v] for k, v in positions.items()},
                      steps={})
        for case, (prefix, gold, step) in cases.items():
            arms = {}
            record["steps"][case] = dict(index=step, prefix=tokenizer.decode(prefix),
                                         gold_id=gold, gold_text=tokenizer.decode([gold]), arms=arms)
            variants = [("dense-prefill-kernel", True, "kernel"), ("sparse-prefill-kernel", False, "kernel"),
                        ("dense-prefill-reference", True, "reference"),
                        ("dense-prefill-open-answer", True, "open-answer"), ("dense-prefill-dense", True, "dense")]
            for name, use_dense, intervention in variants:
                replayed = replay(model, tokenizer, inputs, prefix, task, positions, gold, use_dense, intervention)
                if intervention == "kernel":
                    orig = bad_ids if use_dense else prior["output_token_ids"]
                    for j, observed in enumerate(replayed["greedy_through_forced_prefix"]):
                        if prefix[:j] == orig[:j] and observed != orig[j]:
                            raise RuntimeError(f"Kernel replay did not reproduce {key}/{j}: {observed} vs {orig[j]}")
                arms[name] = replayed
                print(f"{task} {case} {name}: {replayed['top5'][0]} gold rank {replayed['gold_rank']}", flush=True)
        result["records"].append(record)
        save(args.output, result)
    result["stage"] = "complete"
    save(args.output, result)


if __name__ == "__main__":
    main()
