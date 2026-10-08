"""Small, matched GLM quality panel using the public LongBench prompt/scorer."""
from __future__ import annotations

from collections import defaultdict
import time

from benchmarks import longbench_v2
from benchmarks.prolong import token_digest


def common_quality_prefix(panel):
    """Keep the frozen documents; shorten all to their common available prefix."""
    prompts, metadata = panel
    length = min(len(prompt["prompt_token_ids"]) for prompt in prompts)
    if length < 2:
        raise ValueError("ProLong documents need at least two tokens")
    clipped, records = [], []
    for prompt, record in zip(prompts, metadata, strict=True):
        ids = prompt["prompt_token_ids"][:length]
        clipped.append({"prompt_token_ids": ids})
        records.append(dict(record, available_prefix_tokens=record["tokens"],
            tokens=length, token_sha256=token_digest(ids)))
    return (clipped, records), length


def select_longbench_items(items, samples):
    """Spread a smoke panel across length bands and domains before tokenizing.

    This is deliberately not an estimate of the full benchmark's accuracy.
    Selection uses dataset metadata only, never model answers or attention mode.
    """
    if samples < 3 or samples > len(items):
        raise ValueError("LongBench smoke panel needs 3..dataset-size samples")
    groups = defaultdict(list)
    for item in items:
        groups[item["length"]].append(item)
    labels = sorted(groups)
    selected = []
    for index, label in enumerate(labels):
        group = sorted(groups[label], key=lambda row: (
            row["domain"], row["sub_domain"], row["_id"]))
        count = samples // len(labels) + int(index < samples % len(labels))
        if count > len(group):
            raise ValueError("requested too many samples from a length band")
        for position in range(count):
            selected.append(group[(2 * position + 1) * len(group) // (2 * count)])
    assert len(selected) == samples and len({r["_id"] for r in selected}) == samples
    return selected


def prepare_longbench(tokenizer, *, max_input_tokens, samples):
    from datasets import load_dataset

    items = list(load_dataset(longbench_v2.DATASET,
        revision=longbench_v2.DATASET_REVISION, split="train"))
    rows = []
    for item in select_longbench_items(items, samples):
        prompt, original, truncated = longbench_v2.truncate_prompt(
            tokenizer, longbench_v2.make_prompt(item), max_input_tokens)
        # The same native chat template and closed-thinking assistant boundary
        # as the successful GLM NIAH check; no XTML/Kimi-specific rendering.
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": "", "think": ""}],
            tokenize=True, add_generation_prompt=False,
            continue_final_message=True, enable_thinking=False)
        ids = list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)
        if "<think></think>" not in tokenizer.decode(ids[-128:]):
            raise RuntimeError("GLM assistant thinking was not closed")
        rows.append(dict(_id=item["_id"], domain=item["domain"],
            sub_domain=item["sub_domain"], difficulty=item["difficulty"],
            length=item["length"], answer=item["answer"],
            original_input_tokens=original, truncated=truncated,
            input_tokens=len(ids), token_sha256=token_digest(ids),
            prompt_token_ids=ids))
    return sorted(rows, key=lambda row: (row["input_tokens"], row["_id"]))


def public_rows(rows):
    return [{k: v for k, v in row.items() if k != "prompt_token_ids"} for row in rows]


def evaluate_longbench(llm, rows, *, batch_size, progress):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    if not rows or len(rows) % batch_size:
        raise ValueError("the GLM cohort barrier requires complete batches")
    params = SamplingParams(temperature=0, seed=1234, max_tokens=32,
        structured_outputs=StructuredOutputsParams(
            choice=[f"The correct answer is ({c})" for c in "ABCD"]))
    records = []
    started = time.perf_counter()
    for begin in range(0, len(rows), batch_size):
        batch = rows[begin:begin + batch_size]
        outputs = llm.generate(
            [{"prompt_token_ids": row["prompt_token_ids"]} for row in batch],
            params, use_tqdm=True)
        if len(outputs) != len(batch):
            raise RuntimeError("incomplete LongBench cohort")
        for row, output in zip(batch, outputs, strict=True):
            generated = output.outputs[0]
            prediction = longbench_v2.extract_answer(generated.text)
            if prediction is None:
                raise RuntimeError(f"unparseable constrained answer: {generated.text!r}")
            records.append(dict(public_rows([row])[0], prediction=prediction,
                correct=prediction == row["answer"], response=generated.text,
                output_tokens=len(generated.token_ids), finish_reason=generated.finish_reason))
        result = dict(correct=sum(r["correct"] for r in records), total=len(records),
            metrics=longbench_v2.summarize(records), samples=list(records),
            elapsed_seconds=time.perf_counter() - started)
        progress(result)
    result["accuracy"] = result["correct"] / result["total"]
    return result
