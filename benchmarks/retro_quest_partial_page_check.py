"""Compare shared-KV QUEST with a PyTorch oracle across partial page boundaries."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.retro_quest_author_adapter import RetroAuthorQuestCache


class FixtureModel:
    batch_size = 1
    num_gpus = 1
    num_layers = 3
    num_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    max_new_length = 36
    device_map = 'cuda:0'
    dtype = torch.float16

    def decode_forward(self, inputs_ids):
        output = self.cache.decode_attention(self.query, self.key, self.value, 2)
        self.cache.decode_update_kv_cache(self.key, self.value, 2)
        return output

    def sampling(self, logits):
        return logits.argmax(dim=-1)


def check(start):
    model = FixtureModel()
    model.input_length = start
    cache = model.cache = RetroAuthorQuestCache(model, budget_tokens=64, shared_kv=True)
    length = start + model.max_new_length
    keys = torch.randn(1, length, 8, 128, device='cuda', dtype=model.dtype) * 0.25
    # Separate page scores on one channel so top-k choices have a clear margin.
    keys[..., 0] += (torch.arange(length, device='cuda') // 16)[None, :, None] * 6
    values = torch.randn_like(keys)
    for layer in range(model.num_layers):
        cache.prefill_update_kv_cache(None, keys[:, :start], values[:, :start], layer, 0)
    model.query = torch.randn(1, 1, 32, 128, device='cuda', dtype=model.dtype) * 0.25
    model.query[..., 0] = 2
    model.key, model.value = keys[:, start:start + 1].clone(), values[:, start:start + 1].clone()
    token = torch.zeros(1, 1, device='cuda', dtype=torch.int64)
    cache.capture_decode(model, token)
    kv_head = torch.arange(32, device='cuda') // 4
    max_abs = 0.0
    for position in range(start, length):
        model.key.copy_(keys[:, position:position + 1])
        model.value.copy_(values[:, position:position + 1])
        if position == start or position % 16 == 0:
            _, graph_check = cache.checked_graph_step(model, token)
            assert graph_check['max_abs'] == 0.0
        else:
            cache.graph_step(token)
        page = position // 16
        expected_pages = torch.tensor(cache.archive_ids[page - 3:page], device='cuda')
        selected = cache.top_indices.sort(dim=-1).values
        torch.testing.assert_close(selected, expected_pages.expand(32, -1).sort(dim=-1).values.to(selected.dtype))
        lo = (page - 3) * 16
        exact_keys = keys[0, lo:position + 1, kv_head].transpose(0, 1).float()
        exact_values = values[0, lo:position + 1, kv_head].transpose(0, 1).float()
        query = model.query[0, 0, :, None].float()
        weights = torch.bmm(query, exact_keys.transpose(1, 2)) / model.head_dim**0.5
        expected = torch.bmm(weights.softmax(dim=-1), exact_values).squeeze(1)
        actual = cache.graph_logits[0, 0].float()
        torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.005)
        max_abs = max(max_abs, float((actual - expected).abs().max()))
    return dict(prompt_length=start, tail=start % 16, decode_steps=model.max_new_length, max_abs=max_abs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(17)
    records = []
    # The 16th archived page ends one bounds page and starts another.
    for tail in range(16):
        row = check(240 + tail)
        records.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(stage='complete', records=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
