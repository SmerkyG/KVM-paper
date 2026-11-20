"""Summarize paired centroid threshold and percentile experiments."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',nargs=2,action='append',required=True,metavar=('LABEL','JSON'))
p.add_argument('--output',type=Path,required=True);p.add_argument('--head-csv',type=Path,required=True)
a=p.parse_args();runs=[]
for label,path in a.run:
 d=json.loads(Path(path).read_text());r={r['row']:r for r in d['records']}
 if d['stage']!='complete' or len(r)!=32:raise ValueError('incomplete cohort '+label)
 checks=r[min(r)]['reference_checks']
 if len(checks)!=9 or max(c['max_abs'] for c in checks)>0.001:raise ValueError('reference check failed '+label)
 runs.append((label,d,r))
base=runs[0][2]
for label,d,r in runs:
 if set(r)!=set(base):raise ValueError('different rows')
 for k in r:
  if r[k]['input_token_sha256']!=base[k]['input_token_sha256'] or r[k]['answers']!=base[k]['answers']:raise ValueError('different inputs')
lines=['# Sort-free centroid thresholds on 32K multi-key 3','',
 'All rows use the same 32 frozen Llama-3.1-8B-Instruct examples, original LoD two-tier sparse prefill, BF16 weights and cache, and FP32 PyTorch decode. '
 'Thresholds open each nonempty remote region with native-query centroid probability strictly above the indicated multiple of the mean (1 / number of nonempty remote regions). '
 'The production selection rule requires a reduction and pointwise comparison, not sorting. The percentile row is an order-statistic reference, implemented here by sorting. '
 'Unopened regions remain summaries; local and sink are always exact. The 1,024-entry cap applies. This is a quality diagnostic, not a latency benchmark.','',
 '| Selection | First-response accuracy | Open regions / head | Exact remote entries / head | Selected estimated remote mass |',
 '|---|---:|---:|---:|---:|']
head_rows=[]
for label,d,r in runs:
 steps=sum(x['decode_steps'] for x in r.values())
 vals=[]
 for layer in range(32):
  for head in range(32):
   mean=[sum(x['routing_sums'][layer][head][j] for x in r.values())/steps for j in range(4)]
   mi=min(x['routing_mins'][layer][head] for x in r.values());ma=max(x['routing_maxs'][layer][head] for x in r.values())
   vals.append((mean,mi,ma));head_rows.append([label,layer,head,*mean,mi,ma,steps])
 avg=lambda j:sum(z[0][j] for z in vals)/len(vals)
 lines.append(f'| {label} | {sum(x["score"] for x in r.values())*100/32:.2f}% | {avg(0):.2f} | {avg(1):.2f} | {avg(2)*100:.2f}% |')
 misses=[k for k in sorted(r) if r[k]['score']<1]
 lines += []
 print(f'{label}: {steps} steps, range {min(z[1] for z in vals)}–{max(z[2] for z in vals)}, skipped oversize mean {avg(3):.6f}, failed rows {misses}',flush=True)
with a.head_csv.open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['selection','layer','query_head','mean_open_regions','mean_exact_remote_entries','mean_selected_estimated_remote_mass','mean_oversize_skipped','min_open_regions','max_open_regions','decode_steps']);w.writerows(head_rows)
lines+=['','Mean counts weight each run by its own number of decode steps; no oversized candidate regions were skipped in the listed runs (see logs). '
 'Each selection and explicit frontier passed nine independent first-step checks. '
 'The 70%-mass top-p control and percentile rule use sorting in these diagnostic implementations. '
 'Threshold comparisons use a pointwise rule; the 1x experiment used sorting only in its diagnostic implementation, and 4x/8x did not sort in selection.','',
 '[Per-layer, per-head counts]('+a.head_csv.name+').']
a.output.write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
