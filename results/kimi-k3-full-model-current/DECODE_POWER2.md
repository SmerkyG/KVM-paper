# Kimi K3 four-update decode panel

October 6: LoD rows use the repaired nonreplicated-query DCP dispatch
and authoritative live-tail cache conversion. The earlier `oct4-lod-*`
decode measurements attended to incomplete history and are invalid;
they are not included or resumed. Existing dense controls are unaffected.

Only current LoD kernels are shown: four-wave
routing, parallel split reduction, fused distributed-route bookkeeping
and cached physical centroid means, with 32 splits per physical B1.
B1 uses ordinary DCP8; B8 uses **one request's attention per GPU**
(all 96 heads on its owner, native TP8 projections/KDA/MoE).
Old one-wave and ordinary-B8 LoD timings are removed from this table.
No current run exists at the blank contexts; no gains are extrapolated.
See [cached routing](CACHED_ROUTING.md) for matching audits and commands.

Full model, TP8/DCP8/EP8, real ProLong traces, seed 0, one full-shape warmup
and one measured pass. 1,026 output tokens produce 1,025 timed decode steps.
Each LoD layer/rank must record exactly four catch-ups per request.
Times are end-to-end ms per batched decode step, including state updates.
A blank LoD cell means the current-kernel run has not completed, not the old
invalid measurement. All prompts/forced continuations must match dense.

| Context | B1 dense | B1 LoD | Dense / LoD | B8 dense | B8 row-per-GPU LoD | Dense / LoD |
|--:|--:|--:|--:|--:|--:|--:|
| 16K | 21.397 | 21.631 | 0.989x | 31.649 | 30.134 | 1.050x |
| 32K | 21.620 | — | — | 32.615 | — | — |
| 64K | 21.845 | 21.626 | Therereerw   | 34.198 | 30.872 | 1.108x |
| 128K | 22.333 | 21.695 | 1.029x | 37.994 | 31.938 | 1.190x |
| 256K | 23.360 | — | — | 47.405 | — | — |
| 512K | 24.686 | — | — | 60.323 | — | — |
| 1020K | 27.378 | — | — | — | — | — |

A dash is not an estimate. The replicated archive previously failed
B1/512K and B8/256K warmup on VRAM. B1 dense 512K/1020K values reuse
the already validated 1,024-step controls from README.md; dense has
no LoD catch-ups to amortize. Their old LoD counterparts are not
current-kernel measurements and therefore remain blank here.
The B8/128K memory-safe retry completed all audits, but its prefill
was 202.915 s versus dense's 155.750 s, including allocator
reclamation. Its decode speedup does not imply a prefill speedup.

## Reproduction

With the resident full-model weight daemon and local checkpoint paths
configured in `benchmarks/kimi_k3_decode_power2.py`, from the repo root:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_power2 --run
```

The driver skips completed audited points and reuses unaffected dense controls.
To refresh the Markdown from completed/partial results without GPU work:

```bash
python -m benchmarks.kimi_k3_decode_power2
```

## Sources

- [oct4-full-b1-decode-power2-four-updates.json](oct4-full-b1-decode-power2-four-updates.json)
- [oct4-full-b8-decode-16k64k-four-updates.json](oct4-full-b8-decode-16k64k-four-updates.json)
- [oct4-full-b8-decode-128k128k-four-updates.json](oct4-full-b8-decode-128k128k-four-updates.json)
- [oct4-full-b8-decode-256k512k-four-updates.json](oct4-full-b8-decode-256k512k-four-updates.json)
- [oct6-cached-means-decode-b1-64k.json](oct6-cached-means-decode-b1-64k.json)
- [oct6-cached-means-decode-b1-16k128k.json](oct6-cached-means-decode-b1-16k128k.json)
- [oct6-cached-means-decode-owner-b8-16k.json](oct6-cached-means-decode-owner-b8-16k.json)
- [oct6-cached-means-decode-owner-b8-64k.json](oct6-cached-means-decode-owner-b8-64k.json)
- [oct6-cached-means-decode-owner-b8-128k-memory-r2.json](oct6-cached-means-decode-owner-b8-128k-memory-r2.json)
- [oct4-full-b1-512k1020k-decode1025.json](oct4-full-b1-512k1020k-decode1025.json)
