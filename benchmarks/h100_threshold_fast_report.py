"""Build the fast-threshold report from raw results and immutable run records."""
import argparse,json,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--label',default='v6');a=p.parse_args();root=a.directory;prefix=f'threshold-fast-{a.label}'
 def read(name):
  f=root/(name+'.json');return json.loads(f.read_text()) if f.exists() else None
 def latency(mode,length):
  name=f'{prefix}-{mode}-{length}k';d=read(name)
  return (f'[{d["median_prefill_s"]:.3f} s]({name}.json)' if d else 'pending')
 lines=['# H100 threshold prefill with a fused full-coarse producer','',
  'BF16 Llama-3.1-8B-Instruct, batch 1, same replayed prompts and unchanged 16,384-token exact first chunk. Full-model prefill includes LoD cache updates. Cache initialization and decode graph capture are excluded. Each timing is the median of three runs after one warmup. All GPU experiments run through the logged H100 cluster-run adapter with uv.', '',
  '| Configuration | 32K prefill | 64K prefill |','|---|---:|---:|']
 for mode,label in [('top8','Original top-8 grouped'),('legacy','Prior entry-16 pipeline, uncapped, indexed leaves'),('cap8','Fast threshold, first eight regions, grouped leaves'),('direct','Fast entry-16, uncapped, indexed leaves'),('grouped','Fast entry-16, uncapped, grouped + exact overflow fallback')]:
  lines.append(f'| {label} | {latency(mode,32)} | {latency(mode,64)} |')
 if all(read(f'{prefix}-{mode}-{length}k') for mode in ('top8','legacy','direct','cap8') for length in (32,64)):
  reductions=[100*(1-read(f'{prefix}-direct-{length}k')['median_prefill_s']/read(f'{prefix}-legacy-{length}k')['median_prefill_s']) for length in (32,64)]
  gaps=[100*(read(f'{prefix}-cap8-{length}k')['median_prefill_s']/read(f'{prefix}-top8-{length}k')['median_prefill_s']-1) for length in (32,64)]
  lines += ['',f'The new uncapped indexed path reduces prefill by {reductions[0]:.1f}% at 32K and {reductions[1]:.1f}% at 64K versus the prior entry-16 pipeline in these short runs. It remains substantially slower than top-8. The capped diagnostic is {gaps[0]:.1f}% slower than top-8 at 32K and {gaps[1]:.1f}% slower at 64K. The grouped uncapped consumer is slower than the indexed consumer.']
 lines += ['', 'The capped frontier retains the first eight qualified regions in region order, not the eight highest scores. It is a diagnostic and is not equivalent to uncapped entry-16. Both new uncapped consumers preserve every region selected by the threshold and strict entry-work bound.', '',
  'The new producer computes native query–mean-key products once, scales them by inverse query RMS for routing, and accumulates full coarse attention and the routing normalizer together. It stores native scores for a single selection pass. A linear prefix scan compacts selected regions. The capped selector stops after eight winners. The uncapped selector handles sixteen queries together, and its merge reuses saved native scores; a one-bit-per-region mask supports the numerical fallback. The final merge subtracts their coarse contributions and adds exact attention, local attention, and sinks. This uses the top-8 score-reuse simplification and mean preparation; it does not reproduce the older BF16-normalized-query arithmetic exactly.', '',
  'The fast pipeline uses an 8,192-query internal tile, two per remote prefill chunk. It does not change the exact first chunk or cache update schedule. The grouped consumer uses the existing paged leaf kernel with one extra route-to-query lookup for variable-length route lists. The output workspace reserves 64 routes per query on average. This memory budget is not a routing cap: whole queries beyond it use exact per-query attention on the GPU. The final merge recomputes the closed coarse field on the GPU when subtraction would leave less than 0.0001 of full coarse mass. Both fallbacks count toward the latency.', '',
  'Validation compares against a reference using the new score arithmetic and the actual native dot products produced by the kernel. It independently checks selection, the strict opened-entry bound, and a PyTorch residual-coarse computation with independently dispatched exact leaves. This validates execution, not RULER quality equivalence to the old routing arithmetic.', '',
  '| 64K validation | Sampled query heads | Changed region decisions | Outside 2e-5 threshold tolerance | Maximum attention output error |', '|---|---:|---:|---:|---:|']
 summaries={}
 for mode,label in [('cap8','Capped grouped'),('direct','Uncapped indexed'),('grouped','Uncapped grouped')]:
  d=read(f'{prefix}-{mode}-64k-audit')
  if not d:lines.append(f'| {label} | pending | — | — | — |');continue
  checks=[x for layer in d['records'][0]['prefill_fast_audit'] for x in (layer or [])]
  n=sum(x['query_heads'] for x in checks);changed=sum(x['selection_mismatches'] for x in checks);outside=sum(x['selection_outside_tolerance'] for x in checks);err=max(x['max_abs_output_difference'] for x in checks)
  full=sum(x['full_tile_query_heads'] for x in checks);fallback=sum(x['full_tile_numerical_fallbacks'] for x in checks);overflow=sum(x['full_tile_route_workspace_fallbacks'] for x in checks)
  summaries[mode]={'query_heads':n,'changed':changed,'outside':outside,'max_error':err,'max_changed_margin':max(x['max_changed_threshold_margin'] for x in checks),'full_query_heads':full,'numerical_fallbacks':fallback,'workspace_fallbacks':overflow,'regions_per_head':sum(x['full_tile_opened_regions'] for x in checks)/full,'entries_per_head':sum(x['full_tile_opened_entries'] for x in checks)/full}
  lines.append(f'| {label} | {n:,} | {changed} | {outside} | {err:.6f} |')
 lines += ['', 'The following counts cover the first 8,192-query internal tile of each remote chunk and all layers. They do not cover the second internal tile.', '', '| 64K sampled remote queries | Opened regions/head | Opened entries/head | Numerical fallbacks | Route-workspace fallbacks |','|---|---:|---:|---:|---:|']
 for mode,label in [('cap8','Capped grouped'),('direct','Uncapped indexed'),('grouped','Uncapped grouped')]:
  if mode not in summaries:continue
  x=summaries[mode];lines.append(f'| {label} | {x["regions_per_head"]:.2f} | {x["entries_per_head"]:.2f} | {x["numerical_fallbacks"]:,}/{x["full_query_heads"]:,} | {x["workspace_fallbacks"]:,}/{x["full_query_heads"]:,} |')
 lines += ['', 'Synthetic checks cover deliberately uneven region sizes, the eight-region cap, a tight work bound, and concentrated native mass that triggers coarse recomputation. A separate run forces a route workspace of only one route per query on average, exercising exact overflow fallback.', '',
  'GPU event diagnostics below record stage boundaries without synchronizing the CPU after each stage. They remain instrumented measurements; the table above uses separate uninstrumented runs. Exact grouped timings include dispatch and reduction. The preparation and event stage scopes partition the fast prefill attention path, not the entire model.', '',
  '| Diagnostic | Preparation | Producer | Selection | Exact consumer | Merge/correction |', '|---|---:|---:|---:|---:|---:|']
 for mode,length in [('cap8',32),('grouped',32),('direct',32),('direct',64)]:
  d=read(f'{prefix}-profile-{mode}-{length}k')
  if not d:lines.append(f'| {mode} {length}K | pending | — | — | — | — |');continue
  measured=[r for r in d['records'] if not r['warmup']]
  values=[statistics.median(sum(x.get('fast_'+key,0.) for x in r['prefill_leaf_profile']) for r in measured) for key in ('prepare','produce','select','exact','merge')]
  lines.append(f'| {mode} {length}K | '+' | '.join(f'{v:.3f} s' for v in values)+' |')
 lines += ['', 'Failed preliminary trials are preserved in provenance: these include a numerical-boundary check, compiler errors during merge and early-exit development, and out-of-memory failures during repeated uncapped 64K runs with 16K internal tiles. Their dependent runs were skipped, and superseded queued trials were canceled. The final comparison uses 8K internal tiles and a smaller route workspace. The validation now records threshold-boundary discrepancies explicitly and rejects any outside the stated tolerance. Only successful later runs appear in the tables.', '',
  'The experiment is opt-in: `LOD_H100_PREFILL_RULE=entry LOD_H100_PREFILL_IMPLEMENTATION=fast LOD_H100_FAST_TILE=8192 LOD_H100_FAST_SELECT_M=16 LOD_H100_FAST_SAVED_CORRECTION=1 LOD_H100_FAST_ROUTE_CAPACITY=64`, with `LOD_H100_FAST_LEAF=direct` or `grouped`; `LOD_H100_PREFILL_MAX_REGIONS=8` enables the separate capped diagnostic. Serving defaults remain unchanged. These experiments do not establish a subquadratic bound for complete prefill or amortized cache updates.', '',
  'Commands, hardware identity, source hashes, logs, per-layer validation, and raw timings are retained in the JSON files and `provenance` directory.']
 manifest=read(prefix+'-runs')
 if manifest:
  hashes=[]
  for j in manifest['jobs']:
   f=root/'provenance'/j['id']/'run.json'
   if f.exists():hashes.append(json.loads(f.read_text())['source_files'])
  if hashes and any(x!=hashes[0] for x in hashes):raise RuntimeError('comparison runs use different source snapshots')
 if (root/'CAP16.md').exists():
  lines += ['', 'Follow-up: [entry-16 capped at sixteen regions, with rerun cap-eight and top-eight controls](CAP16.md).']
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n');(root/'SUMMARY.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()
