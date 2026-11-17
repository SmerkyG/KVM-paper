"""Check state-only KVM before the full training run."""

import unittest

import torch

from model.kvm_triton_state_only_mixer import SequenceMixer
from model.rwkv7_backbone import MixerConfig
from model.statesdictcache import StatesDictCache


class StateOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        config = MixerConfig(
            hidden_size=128, num_attention_heads=1, num_key_value_heads=1,
            num_hidden_layers=1, rope_partial_dim=64, chunk_len=256,
            n_bswa_chunks=1, n_max_d_chunks=16, sink_len=1,
            state_budget_mode="power_law", state_growth_factor=16,
            state_growth_exponent=0.5, state_min_len=256, token_mixer="kvm",
            kvm_aotriton_forward_attention=1, kvm_apply_merge_gate_to_initial_state=0,
        )
        self.mixer = SequenceMixer(config, 0).to(device="cuda", dtype=torch.bfloat16)
        self.mixer.c_proj = torch.nn.Identity()
        self.streams = tuple(
            torch.randn(1, 1, 769, dim, device="cuda", dtype=torch.bfloat16)
            for dim in (128, 128, 128, 1)
        )
        self.streams = (*self.streams[:3], self.streams[3].sigmoid())

    def prefill(self, length, cache=None):
        return self.mixer.forward_prefill(
            *(tensor[:, :, :length] for tensor in self.streams),
            None, None, None, cache,
        )

    def test_no_local_readout_or_gradient(self):
        q, k, v, gate = (tensor.clone().requires_grad_() for tensor in self.streams)
        args = self.mixer._make_triton_args(1, q.size(2))
        schedule = self.mixer._make_schedule(args)
        output = self.mixer._triton_prefill_out(q, k, v, gate, args, schedule)
        self.assertEqual(output[:, :, :256].count_nonzero().item(), 0)
        self.assertGreater(output[:, :, 256:512].count_nonzero().item(), 0)
        output[:, :, 256:512].float().square().mean().backward()
        for tensor in (q, k, v, gate):
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(v.grad[:, :, :256].count_nonzero().item(), 0)
        self.assertEqual(v.grad[:, :, 256:].count_nonzero().item(), 0)
        self.assertEqual(self.mixer.front_head_temp.grad.count_nonzero().item(), 0)

    @torch.no_grad()
    def test_cached_decode_at_chunk_boundaries(self):
        for length in (37, 255, 256, 512, 768):
            with self.subTest(length=length):
                cache = StatesDictCache()
                self.prefill(length, cache)
                decoded = self.mixer.forward_single(
                    *(tensor[:, :, length:length + 1] for tensor in self.streams),
                    None, None, None, cache,
                )
                expected = self.prefill(length + 1)[:, -1:]
                torch.testing.assert_close(decoded, expected, atol=0.008, rtol=0.04)
                state = cache.get_states(0)[self.mixer._CACHE_TRITON_DECODE]
                self.assertEqual(state.state_coverage_len, length // 256 * 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
