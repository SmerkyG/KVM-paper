# LoD-matched RULER

Llama 3.1 8B Instruct on H100, one request at a time, with 32 frozen examples per task. The 13-task mean weights tasks equally. Timing and quality use the same adjusted settings.

Multipole and Squeezed match initial centroid counts per KV head. Multipole caps exact remote attention at 512 tokens, while Squeezed uses dynamic p80 selection. [Exact-token counts, Multipole budget check, and held-out CWE](SQUEEZED_WIDTH_AUDIT.md).

## 32K RULER

| Task | Multipole | Squeezed | RetroInfer | QUEST | ClusterKV |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 84.38% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 87.50% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 100.00% | 100.00% | 100.00% | 96.88% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | 56.25% | 84.38% |
| niah_multivalue | 97.66% | 99.22% | 97.66% | 98.44% | 88.28% |
| niah_multiquery | 99.22% | 99.22% | 97.66% | 98.44% | 93.75% |
| vt | 97.50% | 100.00% | 99.38% | 95.00% | 97.50% |
| fwe | 88.54% | 90.62% | 91.67% | 88.54% | 75.00% |
| cwe | 18.44% | 45.31% | 24.06% | 13.12% | 15.62% |
| qa_1 | 81.25% | 81.25% | 81.25% | 78.12% | 78.12% |
| qa_2 | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% |
| **13-task mean** | **87.12%** | **89.66%** | **87.82%** | **82.92%** | **80.88%** |

- Multipole. [Shard 0](llama-lodmatch-32768-ruler-shard0.json) · [Command](provenance/1794816651886930615-llama-lodmatch-32768-ruler-shard0/run.json) · [Shard 1](llama-lodmatch-32768-ruler-shard1.json) · [Command](provenance/1794816651886415596-llama-lodmatch-32768-ruler-shard1/run.json) · [Shard 2](llama-lodmatch-32768-ruler-shard2.json) · [Command](provenance/1794816652112725038-llama-lodmatch-32768-ruler-shard2/run.json) · [Shard 3](llama-lodmatch-32768-ruler-shard3.json) · [Command](provenance/1794816652037419254-llama-lodmatch-32768-ruler-shard3/run.json)
- Squeezed. [Shard 0](llama-lodmatch-32768-ruler-shard0.json) · [Command](provenance/1794816651886930615-llama-lodmatch-32768-ruler-shard0/run.json) · [Shard 1](llama-lodmatch-32768-ruler-shard1.json) · [Command](provenance/1794816651886415596-llama-lodmatch-32768-ruler-shard1/run.json) · [Shard 2](llama-lodmatch-32768-ruler-shard2.json) · [Command](provenance/1794816652112725038-llama-lodmatch-32768-ruler-shard2/run.json) · [Shard 3](llama-lodmatch-32768-ruler-shard3.json) · [Command](provenance/1794816652037419254-llama-lodmatch-32768-ruler-shard3/run.json)
- RetroInfer. [Shard 0](llama-lodmatch-32768-retro-ruler-shard0.json) · [Command](provenance/1794817519674407661-llama-lodmatch-32768-retro-ruler-shard0/run.json) · [Shard 1](llama-lodmatch-32768-retro-ruler-shard1.json) · [Command](provenance/1794817519784123119-llama-lodmatch-32768-retro-ruler-shard1/run.json) · [Shard 2](llama-lodmatch-32768-retro-ruler-shard2.json) · [Command](provenance/1794817519928395998-llama-lodmatch-32768-retro-ruler-shard2/run.json) · [Shard 3](llama-lodmatch-32768-retro-ruler-shard3.json) · [Command](provenance/1794817519485981086-llama-lodmatch-32768-retro-ruler-shard3/run.json)
- QUEST. [Shard 0](llama-lodmatch-32768-quest-ruler-shard0.json) · [Command](provenance/1794817520069521913-llama-lodmatch-32768-quest-ruler-shard0/run.json) · [Shard 1](llama-lodmatch-32768-quest-ruler-shard1.json) · [Command](provenance/1794817520200464524-llama-lodmatch-32768-quest-ruler-shard1/run.json) · [Shard 2](llama-lodmatch-32768-quest-ruler-shard2.json) · [Command](provenance/1794817520335203300-llama-lodmatch-32768-quest-ruler-shard2/run.json) · [Shard 3](llama-lodmatch-32768-quest-ruler-shard3.json) · [Command](provenance/1794817519560820707-llama-lodmatch-32768-quest-ruler-shard3/run.json)
- ClusterKV. [Shard 0](llama-lodmatch-32768-clusterkv-ruler-shard0.json) · [Command](provenance/1794817520448473300-llama-lodmatch-32768-clusterkv-ruler-shard0/run.json) · [Shard 1](llama-lodmatch-32768-clusterkv-ruler-shard1.json) · [Command](provenance/1794817520554189334-llama-lodmatch-32768-clusterkv-ruler-shard1/run.json) · [Shard 2](llama-lodmatch-32768-clusterkv-ruler-shard2.json) · [Command](provenance/1794817520670312224-llama-lodmatch-32768-clusterkv-ruler-shard2/run.json) · [Shard 3](llama-lodmatch-32768-clusterkv-ruler-shard3.json) · [Command](provenance/1794817519634551696-llama-lodmatch-32768-clusterkv-ruler-shard3/run.json)

## 64K RULER

| Task | Multipole | Squeezed | RetroInfer | QUEST | ClusterKV |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 84.38% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 81.25% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% | 90.62% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% | 90.62% |
| niah_multikey_2 | 100.00% | 100.00% | 100.00% | 96.88% | 78.12% |
| niah_multikey_3 | 59.38% | 93.75% | 96.88% | 40.62% | 53.12% |
| niah_multivalue | 98.44% | 98.44% | 96.09% | 98.44% | 53.12% |
| niah_multiquery | 100.00% | 100.00% | 100.00% | 100.00% | 75.00% |
| vt | 93.75% | 99.38% | 98.75% | 93.75% | 87.50% |
| fwe | 85.42% | 84.38% | 76.04% | 83.33% | 76.04% |
| cwe | 0.62% | 11.88% | 0.62% | 1.88% | 0.62% |
| qa_1 | 75.00% | 75.00% | 75.00% | 75.00% | 71.88% |
| qa_2 | 53.12% | 56.25% | 53.12% | 53.12% | 53.12% |
| **13-task mean** | **81.98%** | **86.08%** | **84.35%** | **80.23%** | **68.88%** |

- Multipole. [Shard 0](llama-lodmatch-65536-ruler-shard0.json) · [Command](provenance/1794816895354825533-llama-lodmatch-65536-ruler-shard0/run.json) · [Shard 1](llama-lodmatch-65536-ruler-shard1.json) · [Command](provenance/1794816895450701432-llama-lodmatch-65536-ruler-shard1/run.json) · [Shard 2](llama-lodmatch-65536-ruler-shard2.json) · [Command](provenance/1794816895554335432-llama-lodmatch-65536-ruler-shard2/run.json) · [Shard 3](llama-lodmatch-65536-ruler-shard3.json) · [Command](provenance/1794816895435509509-llama-lodmatch-65536-ruler-shard3/run.json)
- Squeezed. [Shard 0](llama-lodmatch-65536-ruler-shard0.json) · [Command](provenance/1794816895354825533-llama-lodmatch-65536-ruler-shard0/run.json) · [Shard 1](llama-lodmatch-65536-ruler-shard1.json) · [Command](provenance/1794816895450701432-llama-lodmatch-65536-ruler-shard1/run.json) · [Shard 2](llama-lodmatch-65536-ruler-shard2.json) · [Command](provenance/1794816895554335432-llama-lodmatch-65536-ruler-shard2/run.json) · [Shard 3](llama-lodmatch-65536-ruler-shard3.json) · [Command](provenance/1794816895435509509-llama-lodmatch-65536-ruler-shard3/run.json)
- RetroInfer. [Shard 0](llama-lodmatch-65536-retro-ruler-shard0.json) · [Command](provenance/1794817520780650647-llama-lodmatch-65536-retro-ruler-shard0/run.json) · [Shard 1](llama-lodmatch-65536-retro-ruler-shard1.json) · [Command](provenance/1794817520886616463-llama-lodmatch-65536-retro-ruler-shard1/run.json) · [Shard 2](llama-lodmatch-65536-retro-ruler-shard2.json) · [Command](provenance/1794819474279003215-llama-lodmatch-65536-retro-ruler-shard2/run.json) · [Shard 3](llama-lodmatch-65536-retro-ruler-shard3.json) · [Command](provenance/1794817521095941272-llama-lodmatch-65536-retro-ruler-shard3/run.json)
- QUEST. [Shard 0](llama-lodmatch-65536-quest-ruler-shard0.json) · [Command](provenance/1794817521202142415-llama-lodmatch-65536-quest-ruler-shard0/run.json) · [Shard 1](llama-lodmatch-65536-quest-ruler-shard1.json) · [Command](provenance/1794819474198465964-llama-lodmatch-65536-quest-ruler-shard1/run.json) · [Shard 2](llama-lodmatch-65536-quest-ruler-shard2.json) · [Command](provenance/1794817521440692087-llama-lodmatch-65536-quest-ruler-shard2/run.json) · [Shard 3](llama-lodmatch-65536-quest-ruler-shard3.json) · [Command](provenance/1794817521557869539-llama-lodmatch-65536-quest-ruler-shard3/run.json)
- ClusterKV. [Shard 0](llama-lodmatch-65536-clusterkv-ruler-shard0.json) · [Command](provenance/1794819474117116109-llama-lodmatch-65536-clusterkv-ruler-shard0/run.json) · [Shard 1](llama-lodmatch-65536-clusterkv-ruler-shard1.json) · [Command](provenance/1794817521808197807-llama-lodmatch-65536-clusterkv-ruler-shard1/run.json) · [Shard 2](llama-lodmatch-65536-clusterkv-ruler-shard2.json) · [Command](provenance/1794817521917531005-llama-lodmatch-65536-clusterkv-ruler-shard2/run.json) · [Shard 3](llama-lodmatch-65536-clusterkv-ruler-shard3.json) · [Command](provenance/1794817522024853522-llama-lodmatch-65536-clusterkv-ruler-shard3/run.json)
