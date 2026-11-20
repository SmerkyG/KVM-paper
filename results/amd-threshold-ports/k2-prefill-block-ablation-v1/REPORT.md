# K2 coupled prefill block ablation at 64K

The first exact block, scheduler chunk and prefill cache catch-up move together across 8K, 16K and 32K. The later exact lookback is 256 tokens. All arms keep the c=16 router, two-tier BF16 KV and 4,096-query internal tiles. The speed pairs record first-token latency and time through the first decode token after cache readiness. The quality jobs score the same eight ProLong documents and record worker PyTorch memory high-water marks.

[Commands](jobs.json) · [Settings](settings.json)
