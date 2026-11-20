"""Replay real Qwen prefill snapshots without reloading the full model.

This is a fixed-query attention diagnostic; serving and cache-update costs are
measured separately in the full-model sequence.
"""
import argparse
import json
import os
from pathlib import Path
import types
import torch

from benchmarks.amd_threshold_prefill_probe import probe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',type=int)
    a=p.parse_args();modules=[]
    paths=sorted(a.snapshots.glob('layer-*.pt'))
    if not paths:raise ValueError('missing real prefill snapshots')
    for path in paths:
        item=torch.load(path,map_location='cuda',weights_only=False)
        if a.layer is not None and item['layer']!=a.layer:continue
        engine=types.SimpleNamespace(**item['engine'])
        engine._page_lookup_probes=lambda page:-1 if page.get('paged_page_directory') else (8 if page.get('overflow_active') else 0)
        pool=types.SimpleNamespace(engine=engine,kv_heads=item['kv_heads'],_prefill_probe=item['data'],_prefill_probe_layer=item['layer'])
        modules.append(types.SimpleNamespace(_vllm_lod_pool=pool))
    result=probe(types.SimpleNamespace(modules=lambda:modules))
    result.update(scope=__doc__,snapshots=[str(p) for p in paths])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
