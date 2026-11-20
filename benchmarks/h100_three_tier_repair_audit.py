"""CPU-only first-response rescoring of saved three-tier repair generations."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true',help='Label a progress audit partial while a saved run is active')
    a=p.parse_args()
    path=Path(Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip())/'tokenizer.json'
    tok=Tokenizer.from_file(str(path));stop={tok.token_to_id(s) for s in ('<|end_of_text|>','<|eot_id|>')}
    result={'stage':'complete','scope':'CPU rescoring of saved output; first EOS/EOT ends the first response',
            'tokenizer_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'stop_tokens':sorted(stop),'files':{}}
    for file in a.inputs:
        raw=file.read_bytes();data=json.loads(raw)
        if data['stage']!='complete':
            assert a.allow_partial,file
            result['stage']='partial'
        records=[];groups=collections.defaultdict(list)
        for row in data['records']:
            ids=row['output_token_ids'];end=next((i for i,t in enumerate(ids) if t in stop),len(ids))
            text=PreTrainedTokenizerBase.clean_up_tokenization(tok.decode(ids[:end],skip_special_tokens=True)).lower()
            matches=[float(v.lower() in text) for v in row['answers']]
            first=max(matches) if row['task'] in ('qa_1','qa_2') else statistics.mean(matches)
            r={k:row[k] for k in ('task','source_row','input_token_sha256','answers')}
            r.update(first_score=first,full_score=row['score'],stop_position=end);records.append(r);groups[row['task']].append(r)
        scores={task:dict(n=len(rows),first=100*statistics.mean(x['first_score'] for x in rows),full=100*statistics.mean(x['full_score'] for x in rows)) for task,rows in groups.items()}
        result['files'][file.name]=dict(stage=data['stage'],sha256=hashlib.sha256(raw).hexdigest(),cohort_sha256=data.get('cohort_sha256'),
            threshold_routing=data.get('threshold_routing'),task_scores=scores,records=records)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:v['task_scores'] for name,v in result['files'].items()},indent=2))


if __name__=='__main__':main()
