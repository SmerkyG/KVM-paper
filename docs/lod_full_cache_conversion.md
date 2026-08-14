# Full-attention prefix conversion to LOD

## Goal

Let a serving runtime begin a request with its native full-attention backend,
then reuse the resulting BF16 K/V prefix as LOD state if a later turn or cache
hit makes the logical context long enough to benefit from LOD. Conversion must
not rerun projections, transformer blocks, or attention.

The request's total logical prompt length, before prefix-cache elimination, is
the primary automatic-mode signal. The chosen attention mode remains fixed for
one model execution; a later turn may convert a retained full prefix before its
first LOD execution.

## Non-negotiable storage invariant

A physical cache block and an LOD leaf page are different objects.

- A physical block contains consecutive sequence positions and is useful for
  allocation and prefix sharing.
- An LOD leaf page contains leaves assigned to the same state region, in stable
  arrival order within that region.

INT4 residual quantization is permitted only after LOD routing has established
the second kind of page. Its anchor and scales must be computed from that
region-owned page. Arbitrary consecutive BF16 K/V entries must never be
quantized together merely because they occupy one physical block.

The valid transition is therefore:

```text
FULL_BF16
    -> replay LOD state updates and region ownership from cached BF16 K/V
    -> construct region-owned leaf pages
    -> quantize those semantic pages
    -> LOD_READY_INT4
```

There is deliberately no generic sequential `COMPACT_INT4` tier.

## Implementation stages

1. Provide a state-only engine entry point that consumes post-RoPE BF16 K/V
   and builds the same state, ownership, semantic pages, local tail, and sink as
   normal LOD prefill while skipping all query attention and output materialization.
2. Reject conversion when the configured clustering rule depends on unavailable
   prefill queries. Key-only routing, including the architecture-resolved
   spherical/coherence policies, is directly reconstructible from ordinary K/V
   caches.
3. Expose a cache conversion helper at the Hugging Face cache boundary. A
   serving backend such as vLLM can call the engine-level entry point directly
   for each layer using its paged BF16 source blocks.
4. Build conversion outside CUDA graph replay. Install the completed LOD block
   tables and fixed buffers before the next captured model execution.
5. Keep shared BF16 source blocks immutable. Release only this request's BF16
   references after the LOD cache is ready; retain both representations when
   another native request or cache policy still references BF16.

## Verification

- Compare every persistent state tensor and semantic page index produced by
  ordinary LOD prefill with the state-only conversion of the same post-RoPE K/V.
- Decode the same next token from both caches and compare outputs.
- For INT4, assert that each quantization page contains positions owned by one
  state region and that its anchor/scales are indexed by that semantic page.
- Measure conversion latency separately from the subsequent prefill/decode
  speedup; conversion is useful only after its cost is amortized.

## Current API

The optimized post-QKV engine accepts a contiguous BF16/FP16 cache directly:

```python
lod_cache = recursive_engine.build_cache_from_bf16(key, value)
```

`key` and `value` have shape `[batch, kv_heads, length, dimension]` and must
already contain the model's positional encoding. The call leaves the source
tensors untouched and performs no attention computation.

For ordinary (non-hybrid) Transformers decoders whose native cache is still
available, install the kernel LOD backend and convert every layer with:

```python
from model.hf_pytorch_lod_attention import convert_hf_full_cache_to_lod

lod_cache = convert_hf_full_cache_to_lod(model, full_cache)
```

Hybrid recurrent models need their serving adapter to retain the native
recurrent states while converting only global-attention layers. The generic
helper rejects that case instead of silently losing those states.

## vLLM integration boundary

The production vLLM backend should read BF16 leaves through vLLM block tables
and write LOD state/page pools owned by the scheduler. Request rows must never
own Python-side module state. Allocation, reference-count changes, and optional
full-to-LOD conversion happen between graph replays; attention, routing, and
masked state updates operate on persistent device tables with shape-stable
kernels.

The out-of-tree implementation lives in `integrations/vllm_lod`. Its default
mode makes fixed-address LOD rows authoritative for eligible global-attention
layers and uses direct LOD prefill. vLLM block tables retain only bounded exact
staging for those layers plus the model's native recurrent state. Pool and
graph-scratch allocations happen before vLLM calculates that staging budget.
Graph padding never aliases a scheduled row; an unscheduled row can be used
only after its true tail length is restored, so the fake append remains beyond
the retained prefix and is discarded before reuse.

Cache gathering must use the native backend's canonical K/V views rather than
infer semantic order from the allocation shape. This matters on ROCm: the raw
allocation's nominal dimensions look token-major, while paged attention writes
and reads head-major key and value views over that storage. Conversion indexes
the request block table in chronological order only after constructing those
canonical views.

The current adapter uses one mode for an entire attention batch. Authoritative
mode uses LOD for prefill and decode. An automatic short-context native mode
would require an LOD/native graph discriminator in vLLM's scheduler; a Python
length check cannot change a graph after it has been captured.

`VLLM_LOD_CACHE_OWNERSHIP=dual` preserves the complementary rebuild path for
diagnosis: native K/V remains authoritative and LOD can be reconstructed before
decode. It does not save server cache memory. Both ownership modes use the same
architecture-aware routing geometry: normalized-key attention modules use
coherence-corrected raw centroids, while unnormalized-key modules use spherical
centroids. Exact protected K/V remains outside clustered state.

In dual mode, rebuild conversion batches requests with the same logical prefix
length and leaves the source vLLM blocks untouched. In mixed native
prefill/decode batches, an already-exact one-token LOD copy consumes the current
post-RoPE K/V directly instead of being discarded and rebuilt. Ordinary decode
tokens advance the device-local tail inside the captured graph; eager catch-up
runs only when coverage crosses a state-update boundary.
