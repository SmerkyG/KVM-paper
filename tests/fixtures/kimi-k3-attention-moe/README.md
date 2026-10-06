# Two-layer K3 attention-stage fixture

This is an untrained geometry/performance fixture, loaded using vLLM's dummy
loader. It retains two native K3 MLA layers, AttnRes, output gating, latent
MoE, and shared experts. Attention width is unchanged: hidden 7168, 96 heads,
Q low-rank 1536, latent 512, direct key 64, expanded K/V 128 per head.
Expert count is reduced to 32, with four selected experts and one shared
expert; expert latent width is 4096 and intermediate width 2048. It contains
no KDA layers and cannot establish trained-model quality or full K3 latency.

The standalone probe directly executes hidden states through both decoder
layers, using native vLLM expert parallelism over eight GPUs. It does not
time embeddings, the LM head, sampling, scheduling, or startup. Each layout
gets an exact-shape warmup and one measured full prefill including LoD
construction. Timings are the maximum synchronized wall interval over all
eight workers. GPU-event stage sums are not used.

Both variants retain one LoD attention/cache owner on GPU 0. The control
keeps TP8 projections and exchanges head slices with that owner. The
candidate executes complete Q/K/V/O projections there and broadcasts its
hidden output before the unchanged native MoE. No request/stage pipeline
overlap is implemented. Gathering complete immutable projection weights is
outside timing, and retained TP shards mean this is not a final stage-layout
VRAM measurement. Numerical agreement is checked before results are accepted.

MoE choices are recorded from an untimed control prefill and reused for both
variants. Router GEMMs, top-k, expert kernels and expert communication still
execute. This isolates the layout: changing from BF16 TP partial sums to one
full W_O GEMM creates approximately 0.4% output roundoff, which flipped some
nearly tied random expert choices and amplified a natural-routing comparison
to 7.5% final relative error. The diagnostic found identical first-layer
Q/K/V and attention outputs. Fixed routes are an explicit fixture limitation,
not an inference change or evidence of trained-model quality equivalence.

The K3 v10 image's **unquantized BF16** AITER instance generator does not
support `situv2`. The probe explicitly uses native vLLM Triton BF16 MoE with
the original SITU activation for both variants. This is not the full model's
INT4 AITER MoE path. The image and its attention kernels are otherwise the
same as the current K3 experiments.

Example on an eight-GPU machine with the unpacked K3 v10 image:

```bash
VLLM_ROCM_USE_AITER_MOE=0 \
LOD_KIMI_SUBTILE64=score LOD_KIMI_CHUNK_TILE_PACK=1 \
LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 LOD_KIMI_SORT_LEAF_ROUTES=1 \
LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1 \
LOD_KIMI_TILE_REFINE=1 LOD_KIMI_DIRECT_LEAF_RESULT=1 \
LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
HSA_NO_SCRATCH_RECLAIM=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_attention_stage_probe \
  --lengths 32768 65536 --chunk-size 2048 \
  --output results/kimi-k3-mla-stack/attention-stage-2layer-2k.json
```

Use `--chunk-size 16384` to test the same computation with full 16K model
chunks. Both choices preserve 16K **per-request global sequence** centroid
update boundaries. The probe requires lengths of at least 32K to exercise
the remote LoD branch, not only its exact first block.
