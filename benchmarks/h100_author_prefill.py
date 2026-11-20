"""Measure dense prefill through independent cache construction for the author decoders."""

import argparse
import hashlib
from pathlib import Path
import statistics
import struct
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from benchmarks.h100_comparison import configure_prefill_mlp_chunk
from benchmarks.h100_llama_shared_batch import MODEL_HOOKS, configure_model, release_cache
from benchmarks.h100_qwen_shared_decode import (
    GPU_SQUEEZED_P80, LLAMA, OBS, capture_prefill, install_decode, load_shared_model, read_cases,
)
from benchmarks.h100_squeezed_index import build_batched_gpu
from benchmarks.retro_multipole_adapter import attach_multipole
from benchmarks.retro_multipole_batch_adapter import attach_multipole_batch
from benchmarks.retro_ruler_run import save


def multipole_prefill(model, inputs, batch, centroids):
    configure_model(model, batch, inputs.size(1), 1281)
    attach = attach_multipole if batch == 1 else attach_multipole_batch
    attach(model, budget=512, fixed_centroids=centroids)
    model.attention_type = 'RetroMultipole'
    model.init_kv_cache(np.zeros(batch, dtype=int), {})
    torch.cuda.synchronize()
    start = time.perf_counter()
    tokens = model.sampling(model.prefill_forward(inputs_ids=inputs.expand(batch, -1).contiguous()))
    model.move()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    cache = model.kv_cache
    if cache.context != inputs.size(1):
        raise RuntimeError('incomplete Multipole cache')
    return dict(prefill_s=elapsed, first_token_ids=tokens.flatten().tolist(),
                initial_centroids=cache.fixed_centroids or cache.clustered // cache.tokens_per_centroid)


def squeezed_prefill(model, original, inputs, batch, centroids):
    configure_model(model, 1, inputs.size(1), 1281)
    prefill_s = setup_s = 0.0
    ready_caches, tokens = [], []
    for _ in range(batch):
        torch.manual_seed(0)
        profile = {}
        token, layers, _ = capture_prefill(
            model, inputs, False, False, False, profile, gpu_ipc=True, capture_native_observations=True)
        prefill_s += profile['prefill_and_capture_s'] + profile['prefill_cache_completion_s']
        # The capture holds the prompt KV. Drop the temporary dense cache
        # before making the independent fixed-prefix state ready to decode.
        model.kv_cache = None
        start = time.perf_counter()
        index = build_batched_gpu(layers, OBS, requested_clusters=centroids)
        install_decode(model, GPU_SQUEEZED_P80, layers, index)
        torch.cuda.synchronize()
        setup_s += time.perf_counter() - start
        ready_caches.append(model.kv_cache)
        tokens.append(int(token.item()))
        release_cache(model, original)
    if len(ready_caches) != batch:
        raise RuntimeError('incomplete Squeezed batch')
    return dict(prefill_s=prefill_s + setup_s, dense_prompt_s=prefill_s,
                index_calibration_install_s=setup_s, first_token_ids=tokens,
                initial_centroids=index['clusters'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('multipole', 'squeezed'), required=True)
    p.add_argument('--centroids', type=int, help='Fix index width to the LoD count')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--length', type=int, required=True)
    p.add_argument('--batch-size', type=int, choices=(1, 4), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.cohort = None
    a.timing_prefix_length = a.length
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    row, = read_cases(a, tokenizer)
    ids = row['ids']
    model = load_shared_model(LLAMA, len(ids) + 1281, tokenizer, None)
    configure_prefill_mlp_chunk(model)
    original = {name: getattr(model, name) for name in MODEL_HOOKS}
    inputs = torch.tensor([ids], device='cuda', dtype=torch.long)
    records = []
    result = dict(stage='running', model_name=LLAMA, model_revision=model.config._commit_hash,
                  dtype='bfloat16', batch_size=a.batch_size, prompt_tokens=len(ids),
                  prefill='dense causal', method=a.method, requested_centroids=a.centroids,
                  input_token_sha256=hashlib.sha256(struct.pack(f'<{len(ids)}I', *ids)).hexdigest(),
                  records=records)
    for iteration in range(3):
        torch.manual_seed(0)
        if a.method == 'multipole':
            record = multipole_prefill(model, inputs, a.batch_size, a.centroids)
        else:
            record = squeezed_prefill(model, original, inputs, a.batch_size, a.centroids)
        record.update(iteration=iteration, warmup=iteration == 0)
        print(record, flush=True)
        records.append(record)
        release_cache(model, original)
        save(a.output, result)
    result.update(stage='complete', median_prefill_s=statistics.median(row['prefill_s'] for row in records[1:]))
    save(a.output, result)


if __name__ == '__main__':
    main()
