"""Freeze one real Qwen-tokenized ProLong prefix for matched prefill timings."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B');ids=[];domains=[]
    for batch in pq.ParquetFile(a.dataset).iter_batches(batch_size=1,columns=['domain','text']):
        row=batch.to_pylist()[0];ids.extend(tok.encode(row['text'],add_special_tokens=False));domains.append(row['domain'])
        if len(ids)>=128000:break
    assert len(ids)>=128000
    data=dict(identity=dict(model='Qwen/Qwen3-8B',tokenizer_revision=tok.init_kwargs.get('_commit_hash'),
        dataset=str(a.dataset),domains=domains),contexts={})
    for n in (4096,32768,65536,128000):
        data['contexts'][str(n)]=dict(prompts=[dict(prompt_token_ids=ids[:n],
            token_sha256=hashlib.sha256(json.dumps(ids[:n],separators=(',',':')).encode()).hexdigest())])
    raw=json.dumps(data,sort_keys=True)+'\n'
    if a.output.exists():assert a.output.read_text()==raw,'frozen Qwen timing corpus changed'
    else:a.output.write_text(raw)
    print('Frozen Qwen timing manifest:',a.output,flush=True)


if __name__=='__main__':main()
