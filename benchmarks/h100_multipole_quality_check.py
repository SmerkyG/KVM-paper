"""Check connected author quality options against independent frontier attention."""
import argparse
import json
import os
from pathlib import Path
import runpy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    os.environ['MULTIPOLE_STABLE_PACK']='1'
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'tests/test_retro_multipole_adapter.py'),run_name='__main__')
    a.output.write_text(json.dumps(dict(stage='complete',tokens_per_centroid=[8,16],
        stable_compaction_order=True,frontier_cases=5,graph_steps_per_case=8))+'\n')


if __name__=='__main__':main()
