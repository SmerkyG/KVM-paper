"""Compare adaptive native- and normalized-query routing on the same prompts."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from benchmarks.retro_ruler_compare import paired_interval


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('native','normalized','output','head-csv'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();ds=[json.loads(x.read_text()) for x in (a.native,a.normalized)]
 if any(d['stage']!='complete' or d['top_p']!=0.70 for d in ds):raise ValueError('incomplete or wrong runs')
 if not ds[0]['routing'].startswith('native-query') or not ds[1]['routing'].startswith('normalized-query'):
  raise ValueError('wrong routing geometry')
 groups=[{r['row']:r for r in d['records']} for d in ds];keys=sorted(groups[0])
 if len(keys)!=32 or set(groups[1])!=set(keys):raise ValueError('wrong cohort')
 for k in keys:
  if groups[0][k]['input_token_sha256']!=groups[1][k]['input_token_sha256'] or groups[0][k]['answers']!=groups[1][k]['answers']:
   raise ValueError('unpaired inputs')
  if groups[0][k]['output_token_ids'][0]!=groups[1][k]['output_token_ids'][0]:raise ValueError('first prefill token differs')
  for g in groups:
   if not 0<=g[k]['score']<=1:raise ValueError('invalid score')
 def stats(g):
  records=[g[k] for k in keys];steps=sum(r['decode_steps'] for r in records)
  sums=np.sum([np.array(r['routing_sums'],dtype=float) for r in records],axis=0)
  mins=np.min([np.array(r['routing_mins'],dtype=float) for r in records],axis=0)
  maxs=np.max([np.array(r['routing_maxs'],dtype=float) for r in records],axis=0)
  if sums.shape!=(32,32,4):raise ValueError('invalid stats shape')
  return steps,sums/steps,mins,maxs
 native,normal=[stats(g) for g in groups]
 vals=[[g[k]['score'] for k in keys] for g in groups]
 changes=[x-y for x,y in zip(vals[1],vals[0])];lo,hi=paired_interval(changes,21996)
 improved=[keys[i] for i,v in enumerate(changes) if v>0];worsened=[keys[i] for i,v in enumerate(changes) if v<0]
 N,A,Nmin,Nmax=native;M,B,Bmin,Bmax=normal
 a.head_csv.parent.mkdir(parents=True,exist_ok=True)
 with a.head_csv.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['layer','query_head','native_avg_regions','normalized_avg_regions','native_avg_exact_entries',
    'normalized_avg_exact_entries','native_min','native_max','normalized_min','normalized_max','native_decode_steps','normalized_decode_steps'])
  for l in range(32):
   for h in range(32):w.writerow([l,h,f'{A[l,h,0]:.8f}',f'{B[l,h,0]:.8f}',f'{A[l,h,1]:.8f}',f'{B[l,h,1]:.8f}',
     int(Nmin[l,h]),int(Nmax[l,h]),int(Bmin[l,h]),int(Bmax[l,h]),N,M])
 lines=['# Adaptive 70%: native versus normalized routing scores','',
  'Both use two-tier LoD on the same 32 Llama multi-key 3 examples. '
  'The native-query score is the actual centroid attention logit plus log count. '
  'The normalized-query score is the existing LoD routing score; its softmax is a '
  'routing-proxy distribution, not the actual centroid attention mass. '
  'Each opens the smallest prefix covering 70% of its own distribution, then '
  'retains all other regions as summaries. Both use FP32 PyTorch decode and original sparse prefill; '
  'neither is an optimized timing run.','',
  '| 70% distribution | First-response score | Average open regions / query head | Average exact remote entries / query head |',
  '|---|---:|---:|---:|',
  f'| Native query, actual centroid mass | {sum(vals[0])*100/32:.2f}% | {A[:,:,0].mean():.2f} | {A[:,:,1].mean():.2f} |',
  f'| Normalized-query routing proxy | {sum(vals[1])*100/32:.2f}% | {B[:,:,0].mean():.2f} | {B[:,:,1].mean():.2f} |','',
  f'Normalized minus native: {sum(changes)*100/32:+.2f} percentage points; paired bootstrap 95% interval [{lo:+.2f}, {hi:+.2f}]. '
  f'{len(improved)} improved ({improved}); {len(worsened)} worsened ({worsened}).','',
  f'The native run has {N} decode steps; normalized has {M}. Per-head means are weighted by decode steps separately for each run. '
  'The [layer-by-head CSV]('+a.head_csv.name+') reports all 1,024 actual heads.','',
  '| Head | Native avg | Normalized avg | Head | Native avg | Normalized avg |',
  '|---:|---:|---:|---:|---:|---:|']
 ha=A[:,:,0].mean(axis=0);hb=B[:,:,0].mean(axis=0)
 for i in range(16):lines.append(f'| {i} | {ha[i]:.2f} | {hb[i]:.2f} | {i+16} | {ha[i+16]:.2f} | {hb[i+16]:.2f} |')
 lines+=['',f'Native minimum / maximum individual head-step: {int(Nmin.min())} / {int(Nmax.max())}; '
         f'normalized: {int(Bmin.min())} / {int(Bmax.max())}. '
         f'Mean skipped oversized regions per head-step: {A[:,:,3].mean():.6f} native and '
         f'{B[:,:,3].mean():.6f} normalized.','']
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines));print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
