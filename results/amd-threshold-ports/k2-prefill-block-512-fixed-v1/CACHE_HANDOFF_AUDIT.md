# K2 BF16 prefill cache handoff audit

At 512 tokens, the initial prefill uses the generic cache builder. It keeps the exact tail in the separate recent K/V buffers. The next aligned block can use the cross-layer cached update, which reads the overflow to be clustered from the chronological leaf archive. The previous exact tail was missing from that archive at the handoff.

The first two blocks on the 8K diagnostic had near-dense loss. The large error began at the third block, after the first batched cached update consumed the stale tail. The fix copies the exact tail into the archive before staging the next chunk.

| Setting on one matched 8,192-token ProLong prompt | NLL (nats/token) |
|---|---:|
| Dense, 512-token scheduler | 0.880816 |
| Dense, 1K-token scheduler | 0.880816 |
| 512-token LoD before repair | 11.049342 |
| 512-token LoD, synchronous initial cache | 11.049342 |
| 512-token LoD, near-zero routing threshold | 11.049487 |
| 512-token LoD, synchronous cache updates | 0.880355 |
| 512-token LoD, repaired batched updates | 0.881763 |

[Diagnostic commands](../k2-prefill-512-audit-v1/jobs.json) · [Routing and initial-cache checks](../k2-prefill-512-audit-v2/jobs.json) · [Fixed diagnostic](token-audit/jobs.json) · [64K rerun](REPORT.md)
