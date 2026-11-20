# Llama 32K multi-key 3: LoD prefill ablation

The diagnostic constructs the ordinary three-tier LoD cache but uses dense attention outputs during prefill, then uses unchanged LoD decode. All 32 first generated tokens match the recorded dense run. This is an ablation, not a production speed measurement.

First response: dense 100.00%, three-tier with sparse prefill 50.00%, three-tier with dense prefill 56.25%; dense-prefill minus sparse-prefill +6.25 points (paired 95% interval -9.38 to +21.88).

Full fixed-length output: dense 100.00%, three-tier with sparse prefill 53.12%, three-tier with dense prefill 65.62%; dense-prefill minus sparse-prefill +12.50 points (paired 95% interval +0.00 to +28.12).
