import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from config import load_configs
from model.kvm_experiment_mixer import SequenceMixer
from model.kvm_mixer import SequenceMixer as ReferenceMixer
from model.kvm_fp32_routing_reference import SequenceMixer as FP32ReferenceMixer
from model.qwen3_backbone import Qwen3BackboneForCausalLM
from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM


def tiny_config(**overrides):
    settings = dict(
        vocab_size=48, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2,
        rope_partial_dim=4, chunk_len=4, n_bswa_chunks=2,
        state_min_len=4, n_max_d_chunks=100,
        state_budget_mode="power_law", state_growth_factor=2.0,
        state_growth_exponent=0.5, kvm_bipartite_chunk_size=4,
        use_tokenshift_att=0, use_tokenshift_ffn=0, use_value_residual=0,
        token_mixer="kvm", ffn_expansion=2, qwen_intermediate_size=32,
    )
    settings.update(overrides)
    return MixerConfig(**settings)


class Wave1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)

    def test_controls_have_identical_initial_parameters(self):
        controls = []
        for overrides in ({}, {"kvm_append_policy": "random"},
                          {"kvm_merge_strategy": "bipartite"}):
            torch.manual_seed(7)
            controls.append(SequenceMixer(tiny_config(**overrides), 0))
        for other in controls[1:]:
            for name, value in controls[0].state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)

    def test_random_append_is_seeded_and_does_not_consume_global_rng(self):
        mixer = SequenceMixer(tiny_config(kvm_append_policy="random"), 0)
        keys = torch.randn(2, 2, 17, 8)
        state = torch.randn(2, 2, 5, 8)
        rng = torch.get_rng_state().clone()
        first = mixer._split_append_merge_indices(keys, 5, state)
        second = mixer._split_append_merge_indices(keys, 5, state)
        self.assertFalse(torch.equal(first[0], second[0]))
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        mixer.reset_append_rng()
        replay = mixer._split_append_merge_indices(keys, 5, state)
        torch.testing.assert_close(first[0], replay[0], rtol=0, atol=0)
        for appended, merged in (first, second):
            combined = torch.cat([appended, merged], dim=-1).sort(dim=-1).values
            torch.testing.assert_close(combined, torch.arange(17).expand(2, 2, -1))

    def test_evaluation_does_not_advance_training_routing_stream(self):
        mixer = SequenceMixer(tiny_config(kvm_append_policy="random"), 0)
        keys = torch.randn(1, 2, 17, 8)
        ref = keys[..., :4, :]
        mixer._split_append_merge_indices(keys, 3, ref)
        expected = mixer._split_append_merge_indices(keys, 3, ref)[0]
        mixer.reset_append_rng()
        mixer._split_append_merge_indices(keys, 3, ref)
        mixer.eval()
        mixer._split_append_merge_indices(keys, 3, ref)
        mixer.train()
        actual = mixer._split_append_merge_indices(keys, 3, ref)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_count_readout_counts_assignments_and_protects_sink(self):
        for cls in (ReferenceMixer, FP32ReferenceMixer):
            mixer = cls(tiny_config(hidden_size=4, num_attention_heads=2,
                                    num_key_value_heads=2, kvm_use_vlens=0), 0)
            state_k = torch.tensor([[[[10., 10.], [1., -1.], [-1., 1.], [0., 0.]]]])
            incoming_k = torch.tensor([[[[1., -1.], [2., -2.], [-1., 1.]]]])
            state_v = torch.zeros_like(state_k)
            incoming_v = torch.ones_like(incoming_k)
            counts = torch.ones(1, 1, 4, 1)
            _, _, result = mixer._merge_into_state(
                incoming_k, incoming_v, state_k, state_v, counts, 1
            )
            torch.testing.assert_close(result, torch.tensor([[[[1.], [3.], [2.], [1.]]]]))
            self.assertEqual(result.dtype, torch.float32)

    def test_dense_mask_attention_matches_reference_outputs_and_gradients(self):
        reference = ReferenceMixer(tiny_config(), 0)
        candidate = SequenceMixer(tiny_config(), 0)
        values = [torch.randn(1, 2, n, 8, requires_grad=True) for n in (4, 11, 11)]
        with patch("model.kvm_mixer.F.scaled_dot_product_attention",
                   wraps=F.scaled_dot_product_attention) as attention:
            actual = candidate._state_attention(*values)
        self.assertIs(candidate._state_attention.__func__, reference._state_attention.__func__)
        self.assertEqual(attention.call_args.kwargs["attn_mask"].dtype, torch.bool)
        self.assertIs(attention.call_args.kwargs["is_causal"], False)
        expected = reference._state_attention(*values)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        actual_grad = torch.autograd.grad(actual.square().sum(), values, retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), values)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_bipartite_preserves_sums_sinks_and_exact_budget(self):
        mixer = SequenceMixer(tiny_config(kvm_merge_strategy="bipartite"), 0)
        state_k, state_v = torch.randn(2, 2, 4, 8), torch.randn(2, 2, 4, 8)
        radius = state_v.norm(dim=-1, keepdim=True)
        for ctx in range(12, 77, 4):
            incoming_k, incoming_v = torch.randn(2, 2, 4, 8), torch.randn(2, 2, 4, 8)
            expected_k = state_k.sum(dim=2) + mixer._prepare_state_update_k(incoming_k).sum(dim=2)
            expected_v = state_v.sum(dim=2) + incoming_v.sum(dim=2)
            sink = tuple(x[:, :, :1].clone() for x in (state_k, state_v, radius))
            target = mixer._desired_state_len(ctx, ctx - 4, state_k.size(2))
            state_k, state_v, radius = mixer._update_state_from_overflow_tokens(
                state_k, state_v, radius, incoming_k, incoming_v,
                torch.ones(2, 2, 4, 1), ctx, ctx - 4,
            )
            self.assertEqual(state_k.size(2), target)
            self.assertEqual(state_v.size(2), target)
            torch.testing.assert_close(state_k.sum(dim=2), expected_k, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(state_v.sum(dim=2), expected_v, atol=1e-5, rtol=1e-5)
            for value, expected in zip((state_k, state_v, radius), sink):
                torch.testing.assert_close(value[:, :, :1], expected, rtol=0, atol=0)

    def test_bipartite_gates_only_new_sources_and_keeps_destination_radii(self):
        mixer = SequenceMixer(tiny_config(kvm_merge_strategy="bipartite",
                                         state_budget_mode="fixed"), 0)
        state_k, state_v = torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)
        radius = state_v.norm(dim=-1, keepdim=True)
        incoming_k, incoming_v = torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)
        gates = torch.tensor([2., 3., 4., 5.]).view(1, 1, 4, 1)
        src = torch.tensor([[[0, 2, 4, 6]]])
        dst = torch.tensor([[[1, 3, 5, 5]]])
        keep = torch.tensor([[[1, 3, 5]]])
        with patch.object(mixer, "_bipartite_indices", return_value=(src, dst, keep)):
            _, values, radii = mixer._update_state_from_overflow_tokens(
                state_k, state_v, radius, incoming_k, incoming_v, gates, 16, 12
            )
        expected_values = torch.stack([
            state_v[0, 0, 0], state_v[0, 0, 2] + state_v[0, 0, 1],
            incoming_v[0, 0, 0] + state_v[0, 0, 3],
            incoming_v[0, 0, 2] + 3 * incoming_v[0, 0, 1] + 5 * incoming_v[0, 0, 3],
        ]).view(1, 1, 4, 8)
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(radii[:, :, 1], radius[:, :, 2])
        torch.testing.assert_close(radii[:, :, 2:], incoming_v[:, :, [0, 2]].norm(dim=-1, keepdim=True))

    def test_bipartite_backward_is_finite(self):
        mixer = SequenceMixer(tiny_config(kvm_merge_strategy="bipartite"), 0)
        tensors = [torch.randn(1, 2, 4, 8, requires_grad=True) for _ in range(4)]
        state_k, state_v, incoming_k, incoming_v = tensors
        gates = torch.full((1, 2, 4, 1), 0.7, requires_grad=True)
        output = mixer._update_state_from_overflow_tokens(
            state_k, state_v, state_v.norm(dim=-1, keepdim=True),
            incoming_k, incoming_v, gates, 12, 8,
        )
        sum(x.square().sum() for x in output).backward()
        for value in tensors + [gates]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_qwen_loss_uses_already_shifted_labels(self):
        model = Qwen3BackboneForCausalLM(tiny_config(num_key_value_heads=1)).eval()
        tokens = torch.randint(0, 48, (2, 9))
        labels = torch.randint(0, 48, (2, 9))
        labels[0, -2:] = -100
        output = model(tokens, labels=labels)
        expected = F.cross_entropy(output.logits.reshape(-1, 48), labels.reshape(-1))
        torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())

    def test_qwen_cached_decode_matches_full_forward_and_is_causal(self):
        model = Qwen3BackboneForCausalLM(tiny_config(num_key_value_heads=1)).eval()
        tokens = torch.randint(0, 48, (1, 13))
        with torch.no_grad():
            full = model(tokens, use_cache=False).logits
            prefix = model(tokens[:, :8], use_cache=True)
            pieces = [prefix.logits]
            cache = prefix.past_key_values
            for position in range(8, 13):
                result = model(tokens[:, position:position + 1], past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                pieces.append(result.logits)
            changed = tokens.clone()
            changed[:, 8:] = torch.randint(0, 48, (1, 5))
            other = model(changed, use_cache=False).logits
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(other[:, :8], full[:, :8], atol=0, rtol=0)

    def test_qwen_checkpoint_round_trip(self):
        from transformers import AutoModelForCausalLM
        from trainer import write_hf_compatibility_wrapper
        model = Qwen3BackboneForCausalLM(tiny_config(num_key_value_heads=1)).eval()
        tokens = torch.randint(0, 48, (1, 9))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            write_hf_compatibility_wrapper(directory)
            restored = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(tokens).logits, restored(tokens).logits)

    def test_wave1_parameter_budgets_and_labels(self):
        paths = ["configs/prolong/base.yaml", "configs/prolong/tokens_3B.yaml",
                 "configs/prolong/120M/kvm_sqrt16_wd.yaml", "configs/prolong/wave1/common.yaml"]
        counts = []
        for experiment, cls in (("K0", RWKV7BackboneForCausalLM), ("Q0", Qwen3BackboneForCausalLM)):
            raw = load_configs(paths + [f"configs/prolong/wave1/{experiment}.yaml"])
            with torch.device("meta"):
                model = cls(MixerConfig(**raw["model"]))
            counts.append(sum(p.numel() for p in model.parameters()))
            self.assertTrue(all(getattr(p, "label", None) is not None for p in model.parameters()))
        self.assertLess(abs(counts[1] / counts[0] - 1), 0.01)


if __name__ == "__main__":
    unittest.main()
