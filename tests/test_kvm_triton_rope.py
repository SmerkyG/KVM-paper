"""Check the local-only RoPE split in Triton prefill, backward, and decode."""

import unittest

import torch

from model.kvm_triton_rope_ablation_mixer import LocalOnlyRoPE
from model.rwkv7_backbone import MixerConfig
from model.statesdictcache import StatesDictCache


@unittest.skipUnless(torch.cuda.is_available(), "requires a ROCm GPU")
class LocalOnlyRoPETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.mixer = LocalOnlyRoPE(MixerConfig(
            hidden_size=128, num_attention_heads=1, num_key_value_heads=1,
            num_hidden_layers=1, rope_partial_dim=64, chunk_len=256,
            n_bswa_chunks=2, n_max_d_chunks=16, sink_len=1,
            state_budget_mode="power_law", state_growth_factor=16,
            state_growth_exponent=0.5, state_min_len=256, token_mixer="kvm",
            kvm_aotriton_forward_attention=1,
        ), 0).to(device="cuda", dtype=torch.bfloat16)
        self.mixer.c_proj = torch.nn.Identity()
        # Front and original rotary channels differ, while the nonrotary
        # channels are shared, exactly as in the packed projection output.
        shape = (1, 1, 769)
        self.q, self.k, self.v, self.gate = [
            torch.randn(*shape, dim, device="cuda", dtype=torch.bfloat16)
            for dim in (192, 192, 128, 1)
        ]
        self.gate = self.gate.sigmoid()

    def raw_prefill(self, length):
        args = self.mixer._make_triton_args(1, length)
        schedule = self.mixer._make_schedule(args)
        return self.mixer._triton_forward_prefill_raw(
            self.q[:, :, :length], self.k[:, :, :length], self.v[:, :, :length],
            self.gate[:, :, :length], args, schedule,
        )

    def test_joint_softmax_and_split_query_gradient(self):
        mixer = self.mixer
        position = 512
        with torch.no_grad():
            raw = self.raw_prefill(position + 1)
            state_temp, front_temp = mixer._head_temperatures(self.q.device)
            state_k = (raw["state_k_attn"][0, :256].float() * state_temp).bfloat16()
            front_k = (self.k[0, 0, 256:position + 1, :128].float() * front_temp).bfloat16()
            values = torch.cat((
                raw["state_v_attn"][0, :256], self.v[0, 0, 256:position + 1],
            )).float()
        query = self.q[0, 0, position].detach().clone().requires_grad_()
        state_q = mixer._unrotated_qk(query).float()
        front_q = query[:128].float()
        scores = torch.cat((state_k.float() @ state_q, front_k.float() @ front_q))
        expected = (scores / 128**0.5).softmax(dim=0) @ values
        expected_grad, = torch.autograd.grad(expected.sum(), query)
        torch.testing.assert_close(raw["out"][0, position].float(), expected, rtol=0.03, atol=0.004)

        args = mixer._make_triton_args(1, self.q.size(2))
        schedule = mixer._make_schedule(args)
        q = self.q.clone().requires_grad_()
        output = mixer._triton_prefill_out(q, self.k, self.v, self.gate, args, schedule)
        output[0, 0, position].float().sum().backward()
        torch.testing.assert_close(output[0, 0, position].float(), expected, rtol=0.03, atol=0.004)
        actual_grad = q.grad[0, 0, position]
        for channels in (slice(0, 64), slice(64, 128), slice(128, 192)):
            grad = expected_grad[channels].float()
            relative_error = (actual_grad[channels].float() - grad).norm() / grad.norm()
            self.assertLess(relative_error.item(), 0.05)

    @torch.no_grad()
    def test_cached_decode_across_state_updates(self):
        for length in (37, 768):
            with self.subTest(length=length):
                cache = StatesDictCache()
                streams = self.q, self.k, self.v, self.gate
                self.mixer.forward_prefill(
                    *(t[:, :, :length] for t in streams), None, None, None, cache,
                )
                decoded = self.mixer.forward_single(
                    *(t[:, :, length:length + 1] for t in streams), None, None, None, cache,
                )
                expected = self.raw_prefill(length + 1)["out"][:, length:length + 1]
                torch.testing.assert_close(decoded, expected, rtol=0.04, atol=0.008)
                decode_cache = cache.get_states(0)[self.mixer._CACHE_TRITON_DECODE]
                recent_k = decode_cache.logical_recent()[0]
                torch.testing.assert_close(recent_k[:, -1:], self.k[0, :, length:length + 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
