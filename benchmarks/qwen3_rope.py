"""Configure Qwen3 YaRN and verify its effective scaling without model weights."""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
from transformers import Qwen3Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def configure_rope(config, scaling):
    config.rope_scaling = dict(scaling)
    if scaling.get('rope_type', scaling.get('type')) == 'yarn' and 'original_max_position_embeddings' in scaling:
        # Transformers 4.51.3 derives the YaRN factor from these lengths.
        config.max_position_embeddings = int(scaling['original_max_position_embeddings'] * scaling['factor'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--factor', type=float, choices=(2.0, 4.0), default=4.0)
    a = p.parse_args()
    original = Qwen3Config.from_pretrained('Qwen/Qwen3-8B')
    scaling = dict(rope_type='yarn', factor=a.factor, original_max_position_embeddings=32768)
    cases = []
    for corrected in (False, True):
        config = copy.deepcopy(original)
        config.rope_scaling = scaling.copy()
        if corrected:
            configure_rope(config, scaling)
        freq, attention = ROPE_INIT_FUNCTIONS['yarn'](config, torch.device('cpu'))
        native, _ = ROPE_INIT_FUNCTIONS['default'](original, torch.device('cpu'))
        effective = float(native[-1] / freq[-1])
        expected = a.factor if corrected else original.max_position_embeddings / 32768
        assert math.isclose(effective, expected, rel_tol=1e-6)
        assert math.isclose(attention, 1 + .1 * math.log(expected), rel_tol=1e-12)
        cases.append(dict(corrected=corrected, max_position_embeddings=config.max_position_embeddings,
                          declared_factor=scaling['factor'], effective_factor=effective,
                          attention_scaling=attention))
    a.output.write_text(json.dumps(dict(stage='complete', cases=cases), indent=2) + '\n')
    print(json.dumps(cases), flush=True)


if __name__ == '__main__':
    main()
