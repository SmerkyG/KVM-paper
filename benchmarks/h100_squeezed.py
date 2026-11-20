"""Run the released Squeezed Attention evaluation path on frozen prompts.

This path calls the authors' released Triton evaluation kernels, surrounded by
their evaluation pipeline's PyTorch and Python work. The repository lists
integration of efficient generation kernels as future work. Offline clustering
is measured separately.
"""
import argparse
import hashlib
import inspect
import json
import math
import statistics
import textwrap
import time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--prompt-limit", type=int)
    p.add_argument("--tokens", type=int, default=1025)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--percent-clusters", type=int, default=5)
    p.add_argument("--percentile", type=float, default=0.9)
    p.add_argument("--observation-window", type=int, default=100)
    p.add_argument("--index-cache", help="Save or reuse an index for these exact prompt tokens")
    p.add_argument("--require-cached-index", action="store_true")
    p.add_argument("--index-only", action="store_true", help="Prepare the fixed-prefix index without running decode")
    p.add_argument("--replay-prompt-tokens", action="store_true", help="Teacher-force the tail of the prompt")
    p.add_argument("--capture-token-ids", action="store_true")
    p.add_argument("--logits-file", type=Path)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from transformers import LlamaConfig, LlamaForCausalLM
    import transformers.models.llama.modeling_llama as llama
    import squeezedattention.clustering as clustering
    from squeezedattention.clustering import run_global_threshold

    torch.set_grad_enabled(False)
    LlamaConfig._rope_scaling_validation = lambda self: None

    # Current CuPy/cuML use the array DLPack protocol, while the released code
    # passes an already-consumed legacy capsule to PyTorch.
    clustering_source = inspect.getsource(clustering.run_clustering)
    old_dlpack = "            dlpack_labels = cluster_labels.toDlpack()\n            labels = torch.utils.dlpack.from_dlpack(dlpack_labels)"
    assert clustering_source.count(old_dlpack) == 1
    clustering_source = clustering_source.replace(
        old_dlpack, "            labels = torch.from_dlpack(cp.asarray(cluster_labels))")
    clustering_generated = {}
    exec(clustering_source, vars(clustering), clustering_generated)
    run_clustering = clustering_generated["run_clustering"]

    # The bundled Transformers predates Llama 3.1. Install its exact scaled
    # frequency schedule while retaining the authors' attention implementation.
    def init_rope(self):
        rotary = llama.LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=1,
                                           base=self.rope_theta)
        freq = rotary.inv_freq
        wavelength = 2 * math.pi / freq
        scale = self.config.rope_scaling
        old = scale["original_max_position_embeddings"]
        low, high, factor = scale["low_freq_factor"], scale["high_freq_factor"], scale["factor"]
        smooth = (old / wavelength - low) / (high - low)
        scaled = (1 - smooth) * freq / factor + smooth * freq
        freq = torch.where(wavelength > old / low, freq / factor,
                           torch.where(wavelength < old / high, freq, scaled))
        rotary.register_buffer("inv_freq", freq, persistent=False)
        self.rotary_emb = rotary

    llama.LlamaAttention._init_rope = init_rope
    # Released sparse evaluation kernels assume one KV head per query head.
    # Replicate GQA KV heads after the cache update, so the cache stays GQA.
    source = textwrap.dedent(inspect.getsource(llama.LlamaFlashAttention2.forward))
    marker = "    # make sure centroid dtype matches\n"
    assert source.count(marker) == 1
    source = source.replace(marker,
        "    if self.num_key_value_groups > 1:\n"
        "        key_states = repeat_kv(key_states, self.num_key_value_groups)\n"
        "        value_states = repeat_kv(value_states, self.num_key_value_groups)\n\n" + marker)
    # The sparse merge produces FP32 in this runtime; the projection is BF16.
    projection = "    attn_output = self.o_proj(attn_output)"
    assert source.count(projection) == 1
    source = source.replace(projection, "    attn_output = self.o_proj(attn_output.to(query_states.dtype))")
    generated = {}
    exec(source, vars(llama), generated)
    llama.LlamaFlashAttention2.forward = generated["forward"]

    ids = json.loads(Path(args.manifest).read_text())["contexts"][str(args.length)]["prompts"][0]["prompt_token_ids"]
    if args.prompt_limit:
        ids = ids[:args.prompt_limit]
    assert len(ids) > args.observation_window
    prompt_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    cache_path = Path(args.index_cache) if args.index_cache else None
    if args.require_cached_index and (cache_path is None or not cache_path.exists()):
        raise RuntimeError("The requested offline index is not available")
    cached = None
    if cache_path and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        assert cached["prompt_hash"] == prompt_hash
        assert cached["percent_clusters"] == args.percent_clusters
        assert cached["observation_window"] == args.observation_window
        assert cached["checkpoint"] == args.checkpoint

    config = LlamaConfig.from_pretrained(args.checkpoint)
    config._attn_implementation = "flash_attention_2"
    config.return_qkv_states = cached is None
    config.use_centroids = False
    config.different_prefix_index = 0
    config.obs_window = args.observation_window
    model = LlamaForCausalLM.from_pretrained(args.checkpoint, config=config,
                                            torch_dtype=torch.bfloat16).eval().to("cuda")
    inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
    model.model.shared_prefix_length = len(ids)

    queries, keys = [], []
    def hook(_module, _inputs, output):
        q, k, _ = output[1]
        # The authors cluster one layer at a time. Stage Q/K on CPU so the
        # 64K GQA-expanded corpus does not occupy the entire GPU cache.
        queries.append(q.detach().cpu())
        keys.append(k.detach().cpu())
    hooks = ([layer.self_attn.register_forward_hook(hook) for layer in model.model.layers]
             if cached is None else [])
    torch.cuda.synchronize()
    start = time.perf_counter()
    dense = model.model(input_ids=inputs, use_cache=True, output_attentions=False)
    first_token = model.lm_head(dense.last_hidden_state[:, -1:, :]).argmax(-1)
    torch.cuda.synchronize()
    dense_prefill = time.perf_counter() - start
    for h in hooks:
        h.remove()

    n_clusters = max(1, int(args.percent_clusters / 100 * (len(ids) - args.observation_window)))
    if cached is None:
        start = time.perf_counter()
        centroids, labels = run_clustering(keys, n_clusters, observation_window=args.observation_window,
                                           print_log=True, device="cuda:0")
        threshold = run_global_threshold(keys, queries, centroids, labels, n_clusters,
                                         observation_window=args.observation_window, device="cuda:0")
        torch.cuda.synchronize()
        offline_s = time.perf_counter() - start
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"prompt_hash": prompt_hash, "checkpoint": args.checkpoint,
                        "percent_clusters": args.percent_clusters, "observation_window": args.observation_window,
                        "centroids": {k: v.cpu() for k, v in centroids.items()},
                        "labels": {k: v.cpu() for k, v in labels.items()},
                        "threshold": threshold, "offline_clustering_s": offline_s}, cache_path)
    else:
        centroids, labels, threshold = cached["centroids"], cached["labels"], cached["threshold"]
        offline_s = cached["offline_clustering_s"]
    if args.index_only:
        if cache_path is None:
            raise ValueError('--index-only requires --index-cache')
        Path(args.output).write_text(json.dumps({"length": len(ids), "index": str(cache_path),
            "offline_clustering_s": offline_s, "dense_prefill_s": dense_prefill}, indent=2) + "\n")
        return
    del keys, queries
    model.model.use_centroids = True
    model.model.cluster_keys_tensor = centroids
    model.model.centroid_labels = labels
    model.model.global_threshold = threshold[args.percentile]
    model.model.global_threshold_lower = threshold[0.7]
    for layer in model.model.layers:
        layer.self_attn.use_centroids = True

    records = []
    for iteration in range(args.warmups + args.repeats):
        # Reuse the same prefilled fixed prefix and its offline cluster index.
        past = tuple(tuple(t.clone() for t in layer) for layer in dense.past_key_values)
        token = first_token.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        token_ids = []
        logits_by_step = {}
        for step in range(args.tokens - 1):
            if args.replay_prompt_tokens:
                offset = len(ids) - (args.tokens - 1) + step
                token = inputs[:, offset:offset + 1]
            out = model.model(input_ids=token, past_key_values=past, use_cache=True)
            logits = model.lm_head(out.last_hidden_state).float()
            token = logits.argmax(-1)
            if args.capture_token_ids:
                token_ids.append(int(token.item()))
            if args.logits_file:
                logits_by_step[step] = logits.detach().cpu()
            past = out.past_key_values
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - start
        result = {"iteration": iteration, "warmup": iteration < args.warmups,
                  "decode_s": decode_s, "decode_ms_per_token": decode_s * 1000 / (args.tokens - 1)}
        if args.capture_token_ids:
            result['output_token_ids'] = token_ids
        if args.logits_file:
            args.logits_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(logits_by_step, args.logits_file)
        records.append(result)
        output = {"method": "released-squeezed-evaluation", "prompt_tokens": len(ids),
                  "output_tokens": args.tokens, "percent_clusters": args.percent_clusters,
                  "percentile": args.percentile, "offline_clustering_s": offline_s,
                  "prompt_sha256": prompt_hash, "reused_index": cached is not None,
                  "dense_prefill_s": dense_prefill, "records": records}
        measured = [r["decode_ms_per_token"] for r in records if not r["warmup"]]
        if measured:
            output["median_decode_ms_per_token"] = statistics.median(measured)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
