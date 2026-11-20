"""Inspect differences between raw tokenizer and recorded output decoding."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tokenizer', type=Path, required=True)
p.add_argument('--input', type=Path, required=True)
a = p.parse_args()
tok = Tokenizer.from_file(str(a.tokenizer))
d = json.loads(a.input.read_text())
for r in d['records']:
    t = tok.decode(r['output_token_ids'], skip_special_tokens=True)
    if t != r['prediction']:
        print(json.dumps({'task':r['task'], 'index':r['index'], 'recorded':r['prediction'], 'raw_decode':t, 'answers':r['answers']}))
