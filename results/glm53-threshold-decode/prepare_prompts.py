"""Rebuild the archived GLM speed panel and check all prompt and trace hashes."""

import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from benchmarks import prolong

folder = Path(sys.argv[1])
snapshot = (
    folder
    / "datasets--Seerkfang--prolong-64k-512-new"
    / "snapshots"
    / prolong.DATASET_REVISION
)
files = sorted(str(p) for p in snapshot.glob("*.parquet"))
assert len(files) == 26
prolong._SPEED_DOCUMENTS = iter(
    load_dataset("parquet", data_files=files, split="train", streaming=True).shuffle(
        seed=prolong.SPEED_SHUFFLE_SEED, buffer_size=1000
    )
)
tok = AutoTokenizer.from_pretrained(sys.argv[2], trust_remote_code=True)
_, rows, _ = prolong.make_speed_trace_panel(
    tok, lengths=[65536], batch_size=8, decode_tokens=1026
)[65536]
archived = json.loads(
    Path(
        "results/glm53-flash-full-model/packed-query-fix-two-tier-tp4-b8-65536.json"
    ).read_text()
)
assert rows == archived["measurements"]["65536"]["prompts"]
torch.save(
    {
        "dataset": prolong.DATASET,
        "revision": prolong.DATASET_REVISION,
        "seed": prolong.SPEED_SHUFFLE_SEED,
        "documents": prolong._SPEED_DOCUMENT_IDS,
    },
    sys.argv[3],
)
print("Matched all eight ProLong prompt and forced-continuation records", flush=True)
