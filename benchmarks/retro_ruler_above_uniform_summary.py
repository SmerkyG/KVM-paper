"""Compare above-uniform and top-p routing on paired RULER examples."""
import argparse, csv, json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--above',type=Path,required=True);p.add_argument('--p70',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--csv',type=Path,required=True)
a=p.parse_args(); A=json.loads(a.above.read_text()); B=json.loads(a.p70.read_text())
if A['stage']!='complete' or B['stage']!='complete': raise ValueError('run incomplete')
def by_row(d):return {r['row']:r for r in d['records']}
x,y=by_row(A),by_row(B)
if len(x)!=32 or set(x)!=set(y):raise ValueError('different cohorts')
for k in x:
 if x[k]['input_token_sha256']!=y[k]['input_token_sha256'] or x[k]['answers']!=y[k]['answers']:raise ValueError('input mismatch')
checks=x[min(x)]['reference_checks']
if len(checks)!=9 or max(c['max_abs'] for c in checks)>0.001:raise ValueError('reference check failed')
def summary(records):
 steps=sum(r['decode_steps'] for r in records); rows=[]
 for layer in range(32):
  for h in range(32):
   means=[sum(r['routing_sums'][layer][h][i] for r in records)/steps for i in range(4)]
   lo=min(r['routing_mins'][layer][h] for r in records); hi=max(r['routing_maxs'][layer][h] for r in records)
   rows.append([layer,h,*means,lo,hi])
 return steps,rows
steps,u=summary(list(x.values())); vsteps,v=summary(list(y.values()))
with a.csv.open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['layer','head','mean_open_regions','mean_exact_remote_entries','mean_open_estimated_remote_mass','mean_oversize_skipped','min_open_regions','max_open_regions']);w.writerows(u)
def avg(rows,i):return sum(r[i] for r in rows)/len(rows)
lines=['# Above-uniform centroid mass: Llama 32K multi-key 3','',
 'Same 32 frozen examples; BF16 weights/cache, original two-tier sparse prefill and FP32 PyTorch decode. '
 'The rule opens a region when its native-query count-weighted centroid probability, normalized over nonempty remote regions, '
 'is strictly greater than 1 divided by the number of those regions. Local and sink are always exact; '
 'regions larger than 1024 entries remain summaries. This is a quality diagnostic, not an optimized speed test.','',
 '| Rule | First-response accuracy | Mean open regions per head | Mean exact remote entries per head | Mean selected estimated remote mass |',
 '|---|---:|---:|---:|---:|']
for label,d,rows in [('Above uniform',x,u),('70% top-p',y,v)]:
 lines.append(f'| {label} | {100*sum(r["score"] for r in d.values())/len(d):.2f}% | {avg(rows,2):.2f} | {avg(rows,3):.2f} | {100*avg(rows,4):.2f}% |')
imp=[k for k in sorted(x) if x[k]['score']>y[k]['score']]; worse=[k for k in sorted(x) if x[k]['score']<y[k]['score']]
lines+=['',f'Above-uniform versus top-p: {len(imp)} improved {imp}; {len(worse)} worsened {worse}.',
 f'Above-uniform had {steps} decode steps; counts ranged from {min(r[6] for r in u):.0f} to {max(r[7] for r in u):.0f} per head-step. '
 f'Mean skipped oversized candidate regions per head-step was {avg(u,5):.6f}. '
 f'The top-p run had {vsteps} decode steps. Head averages are weighted by each run’s decode steps.',
 f'Nine independent checks of selection and explicit frontier had maximum absolute error {max(c["max_abs"] for c in checks):.9f}.','',
 '[Per-layer, per-head statistics]('+a.csv.name+').']
a.output.write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
