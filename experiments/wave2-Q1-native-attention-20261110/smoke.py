import json, os, subprocess, sys, tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
m=json.loads((root/'manifest.json').read_text());os.chdir(m['source'])
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TORCHINDUCTOR_COMPILE_THREADS='2')
cache=Path(tempfile.mkdtemp(prefix='q1-native-smoke-'))
os.environ.update(TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
cmd=['uv','run','--no-sync','torchrun','--nnodes=1','--nproc-per-node=8','--master-addr=127.0.0.1','--master-port='+os.environ['MASTER_PORT'],'train.py']
for file in m['experiments']['Q1']['config_files']:cmd.extend(['-c',file])
cmd.extend(['--train.num_iterations','16','--train.save_every','0','--train.output_dir',str(root/'smoke')])
raise SystemExit(subprocess.call(cmd))
