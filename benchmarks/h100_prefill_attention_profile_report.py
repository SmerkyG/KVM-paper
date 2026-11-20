"""Compare H100 dense prefill attention and selected-entry GPU events."""
import argparse
import json
import statistics
from pathlib import Path


def read(path):return json.loads(path.read_text())


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args()
 lines=['# H100 dense and threshold prefill attention components','',
        'BF16 Llama-3.1-8B-Instruct, batch one, the same fixed prompt for each '
        'context length, 32 model layers, one warmup and two measured repetitions. '
        'GPU event totals exclude weight projections and FFNs. Dense measures '
        'the complete causal FlashAttention call for all prompt queries; the '
        'selected-entry kernel measures only opened remote KV entries for later '
        'queries. It excludes LoD’s exact first chunk, local and sink attention, '
        'routing, index construction, and merging. These components therefore '
        'do not compute the same amount of work.','',
        '| Context | Dense attention GPU time | LoD selected-entry GPU time | Selected / dense |',
        '|---|---:|---:|---:|']
 summaries={}
 for length in (32,64):
  dense=args.directory/f'prefill-attention-dense-{length}k-gpu-events.json'
  lod=args.directory/f'prefill-attention-entry16-{length}k-gpu-events.json'
  if not dense.exists() or not lod.exists():
   lines.append(f'| {length}K | pending | pending | pending |');continue
  a,b=read(dense),read(lod)
  for k in ('model','checkpoint','dtype','batch_size','prompt_tokens','manifest'):
   if a[k]!=b[k]:raise ValueError(f'mismatched {length}K field {k}')
  if a['prompt_tokens']!=length*1024 or a['dtype']!='bfloat16' or a['batch_size']!=1:
   raise ValueError('unexpected workload')
  ar=[r for r in a['records'] if not r['warmup']];br=[r for r in b['records'] if not r['warmup']]
  if len(ar)!=2 or len(br)!=2 or any(r['dense_prefill_attention_profile']['calls']!=32 for r in ar):
   raise ValueError('incomplete profiling results')
  x=[r['dense_prefill_attention_profile']['gpu_seconds'] for r in ar]
  y=[sum(s.get('selected_attention',0) for s in r['prefill_leaf_profile']) for r in br]
  if any(v<=0 for v in x+y):raise ValueError('missing GPU events')
  dm,lm=statistics.median(x),statistics.median(y)
  stages={key:statistics.median(sum(v.get(key,0) for v in r['prefill_profile']) for r in br)
          for key in ('route_logits','select_and_coarse','compact','expand_indices','exact_leaves','merge')}
  summaries[str(length)]={'dense_gpu_s':dm,'selected_gpu_s':lm,'ratio':lm/dm,'diagnostic_stages_s':stages}
  lines.append(f'| {length}K | [{dm:.3f} s]({dense.name}) | [{lm:.3f} s]({lod.name}) | {lm/dm:.1%} |')
 lines += ['', 'The LoD diagnostic synchronizes between stages; its wall-clock '
           'component times include synchronization and launch overhead. They '
           'cannot be added directly to the GPU event times or used as end-to-end '
           'measurements. The selected kernel event is recorded around its '
           'attention launch, after index expansion. The uninstrumented '
           '[prefill comparison](../prefill-next/REPORT.md) records full-model '
           'latency separately.','']
 for length,row in summaries.items():
  stages=row['diagnostic_stages_s']
  lines.append(f'{length}K synchronized LoD stages, summed across layers: '+
               ', '.join(f'{key.replace("_"," ")} {value:.3f} s' for key,value in stages.items())+'.')
 lines += ['', 'The LoD setting uses entry-16, GPU-packed offsets, reused routing '
           'data, 128-region selection tiles and unchanged 16,384-token exact '
           'prefill chunks. The strict bound covers logical exact remote entries '
           'per query head, not the complete execution.','']
 args.output.write_text('\n'.join(lines))
 args.output.with_suffix('.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':main()
