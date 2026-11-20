
import json,statistics,hashlib,sys
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
r=Path('/home/ubuntu/comparison/results')
p=Path(Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip())/'tokenizer.json'
tok=Tokenizer.from_file(str(p));stop={tok.token_to_id(x) for x in ('<|end_of_text|>','<|eot_id|>')}
inputs={};panels={}
for mode in ('bf16','int4'):
 for variant in ('top8','threshold'):
  f=r/f'three-tier-v2-three-tier-{mode}-32k-quality-{variant}.json';raw=f.read_bytes();x=json.loads(raw);assert x['stage']=='complete';inputs[f.name]=hashlib.sha256(raw).hexdigest();panels[mode,variant]={(v['task'],v['source_row']):v for v in x['records']}
keys=set(next(iter(panels.values())));assert all(set(v)==keys for v in panels.values())
for label,names in [('retroinfer',('ruler-llama-retroinfer-32k-selected.json','ruler-llama-retroinfer-32k-missing5.json')),('dense',('ruler-32k-dense-full.json',))]:
 rows={}
 for name in names:
  raw=(r/name).read_bytes();inputs[name]=hashlib.sha256(raw).hexdigest();x=json.loads(raw);assert x['stage']=='complete'
  for v in x['records']:
   k=v['task'],v['source_row']
   if k in keys:assert k not in rows;rows[k]=v
 assert set(rows)==keys;panels[label,'reference']=rows
base=panels['bf16','threshold'];scores={};records=[];summary=[]
for (mode,variant),rows in panels.items():
 scores[mode,variant]={}
 for k,row in rows.items():
  assert row['input_token_sha256']==base[k]['input_token_sha256'] and row['answers']==base[k]['answers']
  ids=row['output_token_ids'];end=next((i for i,t in enumerate(ids) if t in stop),len(ids));text=PreTrainedTokenizerBase.clean_up_tokenization(tok.decode(ids[:end],skip_special_tokens=True)).lower();value=sum(a.lower() in text for a in row['answers'])/len(row['answers']);scores[mode,variant][k]=value
  records.append(dict(mode=mode,variant=variant,task=k[0],source_row=k[1],first_score=value,full_score=row['score'],first_stop_position=end,input_token_sha256=row['input_token_sha256']))
 for task in ('niah_multikey_3','niah_multivalue'):
  taskkeys=[k for k in rows if k[0]==task];summary.append(dict(mode=mode,variant=variant,task=task,n=len(taskkeys),first_response=100*statistics.mean(scores[mode,variant][k] for k in taskkeys),full_output=100*statistics.mean(rows[k]['score'] for k in taskkeys)))
pairs={}
for mode in ('bf16','int4'):
 a,b=scores[mode,'top8'],scores[mode,'threshold'];pairs[mode]=dict(improved=sum(b[k]>a[k] for k in keys),regressed=sum(b[k]<a[k] for k in keys),unchanged=sum(b[k]==a[k] for k in keys))
print(json.dumps(dict(stage='complete',scope='CPU rescoring of saved generations; first response ends at first EOS/EOT',stop_tokens=sorted(stop),tokenizer_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),input_sha256=inputs,summary=summary,paired_outcomes=pairs,records=records),indent=2))
