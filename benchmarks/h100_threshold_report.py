"""Summarize graph-captured H100 threshold routing timings and checks."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for x in ('top32','threshold32','top64','threshold64','quality','reference','real-check','graph-check','synthetic','manifest','tune','output'):p.add_argument('--'+x,type=Path,required=True)
a=p.parse_args();load=lambda path:json.loads(path.read_text())
pairs=[(32768,load(a.top32),load(a.threshold32)),(65536,load(a.top64),load(a.threshold64))]
for length,b,t in pairs:
 for d in (b,t):
  if d['prompt_tokens']!=length or d['batch_size']!=1 or d['timed_decode_steps']!=1024 or d['decode_warmup_steps']!=512 or not d['lod_cuda_graph'] or not d['replay_prompt_tokens']:raise ValueError('benchmark scope mismatch')
  if len([r for r in d['records'] if not r['warmup']])!=2:raise ValueError('expected two timed repetitions')
 if any(b[k]!=t[k] for k in ('model','checkpoint','manifest','dtype','output_tokens')):raise ValueError('paired configuration mismatch')
q=load(a.quality);ref=load(a.reference);qr={x['row']:x for x in q['records']};rr={x['row']:x for x in ref['records']}
if q['stage']!='complete' or ref['stage']!='complete' or len(qr)!=32 or set(qr)!=set(rr):raise ValueError('quality cohort mismatch')
if any(qr[k]['input_token_sha256']!=rr[k]['input_token_sha256'] for k in qr):raise ValueError('quality input mismatch')
if any(qr[k]['output_token_ids']!=rr[k]['output_token_ids'] for k in qr):raise ValueError('quality tokens differ from reference')
real=load(a.real_check);checks=real['records']
if real['stage']!='complete' or len(checks)!=32 or any(x['selection_mismatches'] or x['outside_active'] for x in checks):raise ValueError('real state check failed')
graph=load(a.graph_check); gc=graph['records'][0]['graph_same_cache_checks']
if not gc or any(x['max_abs']!=0 for x in gc.values()):raise ValueError('graph replay differs from eager')
tune=load(a.tune); top8_layer=next(x['microseconds'] for x in tune['records'] if x['kind']=='top8'); threshold_layer=next(x['microseconds'] for x in tune['records'] if x['kind']=='threshold' and x['tile']==64 and x['splits']==64)
edge=load(a.synthetic)
if edge['stage']!='complete' or any(x['selection_mismatches'] or x['max_abs']>0.02 for x in edge['records']):raise ValueError('synthetic check failed')
manifest=load(a.manifest)
lines=['# H100 two-tier average-mass threshold routing','',
 'Experimental opt-in batch-one Llama-3.1-8B-Instruct two-tier BF16 decode in the RetroInfer runtime. '
 'For each query head, region routing uses the normalized query and opens every region with estimated routing-proxy probability greater than 16 times the mean over nonempty remote regions. '
 'Regions larger than 1,024 entries remain summarized, as in the reference. Unopened regions remain summaries; selected entries, local and sink are exact. '
 'Selection has no fixed eight-region cap or top-k sort. It computes a reduction for the average-mass threshold, '
 'compacts candidates deterministically, and distributes exact pages across workers.','',
 'Both comparison paths use whole-model CUDA graphs, BF16 model and cache, the same teacher-forced ProLong prompts, batch one, and one H100 80 GB HBM3. '
 'Each repetition prefills, captures the graph, runs 512 untimed decode steps, and measures the next 1,024 steps; '
 'context grows through the window. Cache catch-up and any periodic updates during the timed window are included. '
 'Initial allocation, prefill and graph capture are excluded from decode timing. One warmup repetition precedes two timed repetitions; values below are their medians.','',
 '| Initial context | Fixed eight ms/token | Threshold c=16 ms/token | Decode latency change | Fixed-eight prefill | Threshold prefill |',
 '|---:|---:|---:|---:|---:|---:|']
for n,b,t in pairs:
 bb=b['median_decode_ms_per_token'];tt=t['median_decode_ms_per_token'];lines.append(f'| {n//1024}K | {bb:.3f} | {tt:.3f} | {(tt/bb-1)*100:+.2f}% | {b["median_prefill_s"]:.3f} s | {t["median_prefill_s"]:.3f} s |')
for n,b,t in pairs:
 br=[round(x['decode_ms_per_token'],4) for x in b['records'] if not x['warmup']];tr=[round(x['decode_ms_per_token'],4) for x in t['records'] if not x['warmup']]
 ids=manifest['contexts'][str(n)]['prompts'][0]['prompt_token_ids'];sha=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
 lines+=['',f'{n//1024}K timed repetitions: fixed eight {br} ms/token; threshold {tr} ms/token. Prompt-token SHA-256: `{sha}`.']
lines+=['', '## Correctness and limitations','',
 f'An isolated graph-captured probe on one real 32K layer measured fixed eight at {top8_layer:.2f} µs and threshold at {threshold_layer:.2f} µs per call, with the same local-length rollback in both. The isolated probe uses one fixed query and omits the model and cache maintenance; it is not a full-model speed comparison.',
 f'The CUDA-graph implementation scored {sum(x["score"] for x in qr.values()):.0f}/32 on the frozen 32K RULER multi-key-3 cohort and reproduced every generated token from the FP32 reference using the same prompts and two-tier sparse prefill. '
 'This is one RULER task, not a full-panel quality result. In the reference, these samples averaged 16.52 open regions and 617.53 exact remote entries per head; those counts are not measurements of the timing prompts.',
 f'On one real 32K prefill/decode step, all {len(checks)} layers matched the FP32 reference region selections; maximum attention component difference was {max(x["max_abs"] for x in checks):.6f}. '
 f'The independent synthetic frontier suite included an oversized region, empty selection, and more than eight open regions; maximum output component difference was {max(x["max_abs"] for x in edge["records"]):.6f}.',
 f'The 64K graph/eager comparison checked {len(gc)} sampled steps, including steps around cache catch-up boundaries; maximum logit difference was {max(x["max_abs"] for x in gc.values()):.6f}.',
 'The H100 path uses 64 exact-attention worker splits by default. It is experimental and limited to batch-one, two-tier BF16 Llama GQA geometry. It is enabled only in the RetroInfer adapter by `LOD_H100_THRESHOLD_MULTIPLIER=16`; it is not a serving default and does not implement three-tier or INT4. '
 'The timing comparison is not quality-matched: fixed eight and threshold routing open different regions and numbers of KV entries.','',
 '[32K fixed eight](h100-threshold-top8-32k-final.json), [32K threshold](h100-threshold-c16-64split-32k.json), '
 '[64K fixed eight](h100-threshold-top8-64k-final.json), [64K threshold](h100-threshold-c16-64split-64k.json), '
 '[GPU quality](h100-threshold-64split-ruler-quality.json), [real-query check](h100-threshold-64split-final-real.json), '
 '[synthetic edge checks](h100-threshold-64split-final-synthetic.json), [64K graph check](h100-threshold-64split-64k-graph-check.json).']
a.output.write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
