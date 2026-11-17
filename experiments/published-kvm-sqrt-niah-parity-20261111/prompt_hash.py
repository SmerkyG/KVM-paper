"""Hash NIAH-S3 prompts from a fixed dataset and tokenizer under each harness."""
import hashlib
import json
import random
import sys

import numpy as np
from lm_eval.tasks.ruler.niah_utils import niah_single_3

random.seed(0)
np.random.seed(1234)
data = niah_single_3(pretrained=sys.argv[1], max_seq_lengths=[4096, 8192, 16384])["test"]
print(json.dumps({
    "rows": len(data),
    "sha256": hashlib.sha256(
        json.dumps(list(data), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
}))
