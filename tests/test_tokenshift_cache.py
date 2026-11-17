import unittest
from unittest.mock import patch

import torch

from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM
from model.statesdictcache import PreallocatedKVCache, StatesDictCache


class TokenShiftCacheTests(unittest.TestCase):
    def test_prefill_caches_only_the_last_token(self):
        config = MixerConfig(
            token_mixer="gptalpha", hidden_size=64, num_attention_heads=2,
            num_hidden_layers=2, vocab_size=128, rope_partial_dim=32, use_tokenshift_ffn=1,
        )
        with torch.random.fork_rng(devices=[]), torch.inference_mode(), patch(
            "utils.grad_cp.use_grad_cp", False
        ):
            torch.manual_seed(0)
            model = RWKV7BackboneForCausalLM(config).eval()
            for layer in model.transformer.h:
                layer.attn.c_proj.weight.normal_(std=0.02)
                layer.mlp.c_proj.weight.normal_(std=0.02)
            prompt = torch.randint(config.vocab_size, (2, 32))
            positions = torch.arange(32)[None, :]
            cache = StatesDictCache()
            model(
                input_ids=prompt[:, :-1], position_ids=positions[:, :-1],
                past_key_values=cache, use_cache=True, logits_to_keep=1,
            )
            for layer in cache.layers:
                final_tokens = [
                    tensor for name, tensor in layer.states_dict.items()
                    if name.endswith("_final_token")
                ]
                self.assertEqual(len(final_tokens), 5)
                for token in final_tokens:
                    self.assertEqual(token.size(1), 1)
                    self.assertEqual(token.untyped_storage().nbytes(), token.numel() * token.element_size())
            cached = model(
                input_ids=prompt[:, -1:], position_ids=positions[:, -1:],
                past_key_values=cache, use_cache=True, logits_to_keep=1,
            ).logits
            full = model(input_ids=prompt, position_ids=positions, use_cache=False, logits_to_keep=1).logits
            torch.testing.assert_close(cached, full, rtol=1e-4, atol=1e-5)

            cache = PreallocatedKVCache(config.num_hidden_layers, prompt.size(1))
            chunks = [
                model(
                    input_ids=prompt[:, start:end], position_ids=positions[:, start:end],
                    past_key_values=cache, use_cache=True,
                ).logits
                for start, end in ((0, 12), (12, 24), (24, 31), (31, 32))
            ]
            full_logits = model(input_ids=prompt, position_ids=positions, use_cache=False).logits
            torch.testing.assert_close(torch.cat(chunks, dim=1), full_logits, rtol=1e-4, atol=1e-5)
