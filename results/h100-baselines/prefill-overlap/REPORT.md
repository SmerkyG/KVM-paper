# Sharing of selected regions in H100 prefill

BF16 Llama-3.1-8B-Instruct, batch 1, two-tier prefill, exact first chunk 16,384 tokens. The two configurations use the same replayed 32K and 64K prompts and the same source snapshot. They run separately, so numerical differences in preceding layers may change later queries and KV states. The threshold rule is entry-16 with a strict entry-work bound and keeps the first eight surviving regions in region order; it is not an uncapped threshold frontier or a top-eight-by-score frontier.

For each remote prefill chunk and each of the 32 model layers, take the first 512 contiguous query positions per query head. Counts and pair comparisons are confined to one query head, since grouped dispatch groups routes by head and region. The 32K and 64K measurements cover 524,288 and 1,572,864 query heads respectively. This is a sampling of each chunk, not the whole prompt.

| Context | Routing | Selected regions per query head | Entries per selected region | Opened entries per query head | Selected routes shared with another query | Mean regions shared by a query pair |
|---|---|---:|---:|---:|---:|---:|
| 32K | Top-8 | 8.00 | 32.77 | 262.20 | 98.09% | 1.658 |
| 32K | Capped threshold | 7.89 | 3.76 | 29.64 | 98.06% | 1.612 |
| 64K | Top-8 | 8.00 | 41.95 | 335.54 | 98.28% | 2.006 |
| 64K | Capped threshold | 7.95 | 4.69 | 37.28 | 97.88% | 1.609 |

A selected route is shared if at least one other query in the same 512-position group and head selects the same region. The query-pair metric averages the number of shared selected regions over every unordered pair in that group, including pairs with none. Region sizes are weighted by selected routes. In this sample the selections have similar overlap at 32K; top-8 has more at 64K, but both show extensive sharing. The capped threshold selects far smaller regions at both lengths.

A route-weighted view of group size gives the mean number of queries selecting the region associated with a selected route: top-8 and threshold are 106.9 and 105.4 at 32K, and 129.2 and 104.5 at 64K. This weights popular regions more heavily than the mean over distinct regions.

The sum of selected KV entry uses divided by the number of distinct region entries per head and 512-query group is 32.69 for top-8 versus 10.45 for threshold at 32K, and 31.92 versus 10.28 at 64K. This is an idealized opportunity for reusing KV reads, influenced by both region size and popularity; it is not measured memory traffic or a measured kernel speedup.

Actual grouped dispatch in the top-8 path spans the whole 16,384-query remote chunk, whereas the tested threshold path groups within 512-query tiles. The full-chunk top-8 audit records 99.91% and 99.89% of routes shared at 32K and 64K respectively. Those values use a much larger group and should not be compared directly to the 512-query threshold results.

These observations do not by themselves establish what causes the latency differences. In particular, the capped threshold result is heavily shaped by retaining the first eight regions in region order, which tends to select small regions in this run. These diagnostic runs synchronize to collect statistics and are not latency measurements. The separate [top-8 backend timing report](../prefill-top8-query/REPORT.md) and [capped threshold timing report](../prefill-grouped8/REPORT.md) contain the uninstrumented results.

Complete per-layer and per-chunk counts, group-size histograms, source hashes, commands, and logs are in the JSON files and `provenance` directory.
