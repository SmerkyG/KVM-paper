# K2 Horizon centroid count and decode window at 64K

The five configurations share a c=16 region threshold in prefill and decode, with two-tier BF16 KV. The centroid sweep uses multipliers 8, 16 and 32 with the decode window fixed at 512. The window sweep uses 256, 512 and 1,024 with the centroid multiplier fixed at 16. The scheduler and prefill exact prefix are 16,384 tokens. Decode cache catch-up is 256 tokens. Prefix caching is disabled.

A 64K prefill plus 1,024 decode-step serving check runs each geometry before the matched RULER jobs.

[Settings](settings.json) · [Source](source-manifest.json) · [Serving check jobs](preflight/jobs.json)
