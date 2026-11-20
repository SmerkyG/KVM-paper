"""Summarize matched native and normalized centroid threshold RULER tests."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',nargs=2,action='append',required=True,metavar=('LABEL','JSON'))
p.add_argument('--output',type=Path,required=True);p.add_argument('--head-csv',type=Path,required=True)
a=p.parse_args(); runs=[]
for label,path in a.run:
 d=json.loads(Path(path).read_text());records={x['row']:x for x in d['records']}
 if d['stage']!='complete' or len(records)!=32:raise ValueError('incomplete '+label)
 checks=records[min(records)]['reference_checks']
 if len(checks)!=9 or max(c['max_abs'] for c in checks)>0.001:raise ValueError('reference check failed '+label)
 runs.append((label,d,records))
base=runs[0][2]
for label,d,r in runs:
 if set(r)!=set(base):raise ValueError('cohort mismatch '+label)
 for k in r:
  if r[k]['input_token_sha256']!=base[k]['input_token_sha256'] or r[k]['answers']!=base[k]['answers']:raise ValueError('input mismatch')
lines=['# Normalized-query threshold routing: Llama 32K multi-key 3','',
 'Paired 32 frozen prompts with Llama-3.1-8B-Instruct, BF16 weights/KV, original LoD sparse prefill and FP32 PyTorch decode. '
 'The threshold opens a region when its score probability is greater than c times the average over nonempty remote regions. '
 'A normalized-query probability is a routing proxy; it is not the region’s native attention mass. '
 'Unopened regions remain summaries. Three-tier opens up to two native-query-scored pages per selected region and retains one residual summary for the other entries. '
 'Two-tier opens all entries in selected regions. The fixed-eight control selects eight regions using normalized-query scores. The 1,024-entry cap applies; local and sink are always exact. '
 'These are quality diagnostics, not optimized speed measurements.','',
 '| Method | First-response accuracy | Open regions / head | Exact remote entries / head | Exact pages / head | Selected routing-score probability |',
 '|---|---:|---:|---:|---:|---:|']
allhead=[];failures=[]
for label,d,r in runs:
 steps=sum(x['decode_steps'] for x in r.values());dim=len(r[min(r)]['routing_sums'][0][0]);vals=[]
 for l in range(32):
  for h in range(32):
   means=[sum(x['routing_sums'][l][h][i] for x in r.values())/steps for i in range(dim)]
   lo=min(x['routing_mins'][l][h] for x in r.values());hi=max(x['routing_maxs'][l][h] for x in r.values())
   vals.append((means,lo,hi));allhead.append([label,l,h,*means[:4],means[4] if dim>4 else '',lo,hi,steps])
 mean=lambda i:sum(x[0][i] for x in vals)/len(vals)
 pages=f'{mean(4):.2f}' if dim>4 else '—'
 value=sum(x['score'] for x in r.values()); misses=[k for k in sorted(r) if r[k]['score']<1]
 lines.append(f'| {label} | {value*100/32:.2f}% ({value:.0f}/32) | {mean(0):.2f} | {mean(1):.2f} | {pages} | {mean(2)*100:.2f}% |')
 failures.append(f'{label}: failed rows {misses}; {steps} decode steps; open-region range {min(x[1] for x in vals):.0f}–{max(x[2] for x in vals):.0f}; average oversized candidates skipped {mean(3):.6f}.')
with a.head_csv.open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['method','layer','head','mean_open_regions','mean_exact_remote_entries','mean_selected_routing_probability','mean_oversize_skipped','mean_open_pages','min_open_regions','max_open_regions','decode_steps']);w.writerows(allhead)
lines+=['',*failures,'','Mean counts weight each run by its own decode steps. All runs passed nine independent first-step selection/frontier checks. '
 'Three-tier additionally checked the selected pages and recomputed residuals from the unopened leaves.','',
 '[Per-layer and per-head statistics]('+a.head_csv.name+').']
a.output.write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
