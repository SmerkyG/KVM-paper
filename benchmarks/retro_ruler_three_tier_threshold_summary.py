"""Summarize matched three-tier c=4 and fixed-eight RULER quality runs."""
import argparse,csv,json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_prompt_compare import score
p=argparse.ArgumentParser(description=__doc__)
for arg in ('adaptive','control','optimized','two-tier','dense','tokenizer','output','head-csv'):p.add_argument('--'+arg,type=Path,required=True)
a=p.parse_args()
paths=[a.adaptive,a.control,a.optimized,a.two_tier,a.dense]
datasets=[json.loads(path.read_text()) for path in paths]
if any(d['stage']!='complete' for d in datasets):raise ValueError('incomplete runs')
def group(d):return {r.get('row',r.get('source_row')):r for r in d['records'] if r['task']=='niah_multikey_3'}
groups=[group(d) for d in datasets]; keys=sorted(groups[0])
if len(keys)!=32 or any(set(g)!=set(keys) for g in groups):raise ValueError('different cohorts')
for k in keys:
 if any(g[k]['input_token_sha256']!=groups[0][k]['input_token_sha256'] or g[k]['answers']!=groups[0][k]['answers'] for g in groups):raise ValueError(f'input mismatch {k}')
for group_ in groups[:2]:
 checks=group_[keys[0]]['reference_checks']
 if len(checks)!=9 or max(x['max_abs'] for x in checks)>0.001:raise ValueError('missing independent frontier checks')
tok=Tokenizer.from_file(str(a.tokenizer)); stop={tok.token_to_id('<|end_of_text|>'),tok.token_to_id('<|eot_id|>')}
if None in stop:raise ValueError('missing stop ids')
def first(r):
 ids=r['output_token_ids']; end=next((i for i,t in enumerate(ids) if t in stop),len(ids))
 text=PreTrainedTokenizerBase.clean_up_tokenization(tok.decode(ids[:end],skip_special_tokens=True))
 return score('niah_multikey_3',text,r['answers'])
scores=[[first(g[k]) for k in keys] for g in groups]
labels=['Three-tier 4x mean, PyTorch','Three-tier fixed 8, PyTorch','Three-tier fixed 8, optimized BF16','Two-tier 4x mean, PyTorch','Dense']
metrics=[]; rows=[]
for label,g in zip(labels,groups):
 if 'routing_sums' not in g[keys[0]]:metrics.append(None);continue
 steps=sum(g[k]['decode_steps'] for k in keys);n=len(g[keys[0]]['routing_sums'][0][0]); heads=[]
 for l in range(32):
  for h in range(32):
   avg=[sum(g[k]['routing_sums'][l][h][i] for k in keys)/steps for i in range(n)]
   lo=min(g[k]['routing_mins'][l][h] for k in keys);hi=max(g[k]['routing_maxs'][l][h] for k in keys)
   heads.append((avg,lo,hi));rows.append([label,l,h,*avg[:4],avg[4] if n>4 else '',lo,hi,steps])
 means=[sum(v[0][i] for v in heads)/len(heads) for i in range(n)]
 metrics.append((means,min(v[1] for v in heads),max(v[2] for v in heads),steps))
with a.head_csv.open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['method','layer','query_head','mean_open_regions','mean_exact_remote_entries','mean_selected_estimated_mass','mean_oversize_skipped','mean_open_pages','min_open_regions','max_open_regions','decode_steps']);w.writerows(rows)
lines=['# Three-tier 4× mean-mass region selection: Llama 32K multi-key 3','',
 'Same 32 frozen Llama-3.1-8B-Instruct prompts, BF16 model and cache. Three-tier selected regions open their two highest-scoring pages, and remaining entries in each opened region are combined in one residual summary; unselected regions remain summaries. '
 'The 4× rule opens regions whose native-query estimated mass is greater than four times the mean remote-region mass. '
 'LoD rows use their respective sparse-prefill modes; PyTorch rows use FP32 reference decode and existing cache updates. '
 'The fixed-eight routes use the original normalized-query score. The optimized baseline uses BF16 decode, and dense uses dense prefill/decode. '
 'This is a quality diagnostic, not an optimized latency run.','',
 '| Mode | First-response accuracy | Mean open regions per head | Mean exact remote entries per head | Mean exact pages per head |',
 '|---|---:|---:|---:|---:|']
for label,sc,m in zip(labels,scores,metrics):
 if m:
  z=m[0];cells=f'{z[0]:.2f} | {z[1]:.2f} | {z[4]:.2f}' if len(z)>4 else f'{z[0]:.2f} | {z[1]:.2f} | —'
 else:cells='— | — | —'
 lines.append(f'| {label} | {100*sum(sc)/32:.2f}% ({sum(sc):.0f}/32) | {cells} |')
imp=[keys[i] for i in range(32) if scores[0][i]>scores[1][i]];worse=[keys[i] for i in range(32) if scores[0][i]<scores[1][i]]
failed=[keys[i] for i in range(32) if scores[0][i]<1]
fixed_diffs=[x['optimized_head_max_abs'] for x in groups[1][keys[0]]['reference_checks']]
lines+=['',f'4× versus fixed-eight FP32: {len(imp)} improved {imp}; {len(worse)} worsened {worse}. 4× failed rows: {failed}.',
 f'The 4× run averaged {metrics[0][0][2]*100:.2f}% estimated remote region mass opened; region counts ranged from {metrics[0][1]:.0f} to {metrics[0][2]:.0f} per head-step, across {metrics[0][3]} decode steps. Mean skipped oversized regions per head-step was {metrics[0][0][3]:.6f}.',
 f'Nine independent first-step checks per PyTorch run validated selection, two-page selection, and the residual frontier. The fixed-eight reference versus optimized output on these nine heads had maximum absolute component difference {max(fixed_diffs):.6f}; this does not validate all heads or later steps.',
 f'First generated token mismatches: three-tier 4× vs three-tier fixed-eight FP32 {sum(groups[0][k]["output_token_ids"][0]!=groups[1][k]["output_token_ids"][0] for k in keys)}; '
 f'three-tier 4× vs two-tier 4× {sum(groups[0][k]["output_token_ids"][0]!=groups[3][k]["output_token_ids"][0] for k in keys)}.', '',
 '[Per-layer, per-head counts]('+a.head_csv.name+').']
a.output.write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
