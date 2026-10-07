# Kimi K3 decode timings

The fresh matched prefill **and** decode panel is now in [CURRENT_TIMINGS.md](CURRENT_TIMINGS.md). It contains only completed, audited October 7 measurements; pending contexts remain blank.

The superseded October 6 decode panel is preserved in [OCT6_DECODE_POWER2.md](OCT6_DECODE_POWER2.md).

Refresh the current tables without GPU work using `python -m benchmarks.kimi_k3_current_timings`. For new measurements, use `benchmarks.kimi_k3_refresh_panel`, as documented in [README.md](README.md); the old sweep command is historical.
