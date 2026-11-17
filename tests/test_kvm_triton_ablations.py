import unittest

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.kvm_triton_mixer import SequenceMixer
from model.rwkv7_backbone import MixerConfig
from model.statesdictcache import StatesDictCache


@unittest.skipUnless(torch.cuda.is_available(), "requires a ROCm GPU")
class KvmTritonAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.chunk = 256
        self.dim = 128
        self.length = 1537
        self.q, self.k, self.v = (
            torch.randn(1, 1, self.length, self.dim, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
        self.gate = torch.randn_like(self.v[..., :1]).sigmoid().contiguous()

    def mixer(self, **overrides):
        config = dict(
            hidden_size=self.dim, num_attention_heads=1, num_key_value_heads=1,
            num_hidden_layers=1, rope_partial_dim=64, chunk_len=self.chunk,
            n_bswa_chunks=2, n_max_d_chunks=16, sink_len=1, state_budget_mode="power_law",
            state_growth_factor=16, state_growth_exponent=0.5, state_min_len=self.chunk,
            token_mixer="kvm", kvm_aotriton_forward_attention=1,
        )
        return SequenceMixer(MixerConfig(**(config | overrides)), 0).to(
            device="cuda", dtype=torch.bfloat16
        )

    def forward_raw(self, mixer, args, schedule):
        args.capture_state_snapshots = True
        with torch.no_grad():
            return mixer._triton_forward_prefill_raw(
                self.q, self.k, self.v, self.gate, args, schedule
            )

    @torch.inference_mode()
    def test_chunked_prefill_preserves_outputs_and_decode_state(self):
        for mode in ("fixed", "power_law"):
            with self.subTest(schedule=mode):
                mixer = self.mixer(state_budget_mode=mode).eval()
                mixer.c_proj = torch.nn.Identity()
                full_cache, chunked_cache = StatesDictCache(), StatesDictCache()
                streams = (self.q, self.k, self.v, self.gate)
                full = mixer.forward_prefill(*streams, None, None, None, full_cache)
                parts = []
                for start, end in ((0, 768), (768, 1536), (1536, self.length)):
                    parts.append(mixer.forward_prefill(
                        *(tensor[:, :, start:end] for tensor in streams),
                        None, None, None, chunked_cache, keep_prefill_state=end < self.length,
                    ))
                torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=0.01, atol=0.003)
                self.assertEqual(chunked_cache.get_seq_length(0), self.length)
                full_state = full_cache.get_states(0)[mixer._CACHE_TRITON_DECODE].logical_state_dict()
                chunked_state = chunked_cache.get_states(0)[mixer._CACHE_TRITON_DECODE].logical_state_dict()
                torch.testing.assert_close(chunked_state, full_state, rtol=0.01, atol=0.003)

    def test_e2_appends_evenly_spaced_tokens_at_fixed_budget(self):
        mixer = self.mixer(kvm_append_policy="uniform")
        streams = [tensor.repeat(2, 1, 1, 1) for tensor in (self.q, self.k, self.v, self.gate)]
        args = mixer._make_triton_args(2, self.length)
        schedule = mixer._make_schedule(args)
        with torch.no_grad():
            result = mixer._triton_forward_prefill_raw(*streams, args, schedule)
        checked = 0
        for macro, count in enumerate(schedule.n_append_by_macro.tolist()):
            if not count:
                continue
            chosen = ((torch.arange(count, device="cuda") + 0.5) * self.chunk / count).long()
            before = schedule.before_by_macro[macro].item()
            expected = torch.full((self.chunk,), -1, device="cuda", dtype=torch.int32)
            expected[chosen] = before + torch.arange(count, device="cuda", dtype=torch.int32)
            actual = result["append_pos_by_token"][:, macro]
            torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=0)
            torch.testing.assert_close(actual < 0, result["best_idx_by_token"][:, macro] >= 0)
            checked += 1
        self.assertGreater(checked, 1)
        values = streams[2].detach().requires_grad_()
        out = mixer._triton_prefill_out(streams[0], streams[1], values, streams[3], args, schedule)
        (grad,) = torch.autograd.grad(out.float().square().mean(), values)
        self.assertTrue(torch.isfinite(grad).all())

    def test_e1_checkpoint_replays_random_append_choices(self):
        mixer = self.mixer(kvm_append_policy="random")
        args = mixer._make_triton_args(1, self.length)
        schedule = mixer._make_schedule(args)
        first = self.forward_raw(mixer, args, schedule)["append_pos_by_token"]
        second = self.forward_raw(mixer, args, schedule)["append_pos_by_token"]
        self.assertFalse(torch.equal(first, second))

        def forward(v):
            return mixer._triton_prefill_out(self.q, self.k, v, self.gate, args, schedule)

        def run(checkpointed):
            torch.manual_seed(176)
            values = self.v.detach().requires_grad_()
            result = checkpoint(forward, values, use_reentrant=False) if checkpointed else forward(values)
            (grad,) = torch.autograd.grad(result.float().square().sum(), values)
            return result, grad, torch.cuda.get_rng_state()

        direct, direct_grad, direct_rng = run(checkpointed=False)
        replay, replay_grad, replay_rng = run(checkpointed=True)
        self.assertTrue(torch.isfinite(replay_grad).all())
        torch.testing.assert_close(replay, direct, rtol=0.001, atol=0.001)
        torch.testing.assert_close(replay_grad, direct_grad, rtol=0.01, atol=0.002)
        torch.testing.assert_close(replay_rng, direct_rng, rtol=0, atol=0)

    def test_qwen_gqa_triton_training_and_cached_decode(self):
        from model.qwen_kvm_backbone import QwenKVMBackboneForCausalLM

        config = MixerConfig(
            hidden_size=256, num_attention_heads=2, num_key_value_heads=1,
            num_hidden_layers=1, qwen_intermediate_size=768, vocab_size=128,
            rope_partial_dim=64, chunk_len=self.chunk, n_bswa_chunks=2,
            n_max_d_chunks=16, sink_len=1, state_budget_mode="power_law",
            state_growth_factor=16, state_growth_exponent=0.5, state_min_len=self.chunk,
            use_tokenshift_att=0, use_tokenshift_ffn=0, use_value_residual=0,
            token_mixer="kvm", token_mixer_class_path="model.kvm_triton_mixer.SequenceMixer",
            kvm_aotriton_forward_attention=0,
        )
        model = QwenKVMBackboneForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
        self.assertIs(type(model.transformer.layers[0].self_attn.mixer), SequenceMixer)
        tokens = torch.randint(128, (1, 3 * self.chunk + 1), device="cuda")
        result = model(tokens, labels=tokens.roll(-1, dims=1), return_logits=False)
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.eval()
        with torch.no_grad():
            full = model(tokens, use_cache=False).logits[:, -1:]
            prefix = model(tokens[:, :-1], use_cache=True)
            decoded = model(tokens[:, -1:], past_key_values=prefix.past_key_values).logits
        torch.testing.assert_close(decoded, full, rtol=0.03, atol=0.008)

    def test_a6_keeps_append_selection_and_excludes_new_slots(self):
        baseline = self.mixer()
        ablation = self.mixer(kvm_merge_old_slots_only=1)
        ablation.load_state_dict(baseline.state_dict())
        args = baseline._make_triton_args(1, self.length)
        schedule = baseline._make_schedule(args)
        reference = self.forward_raw(baseline, args, schedule)
        args = ablation._make_triton_args(1, self.length)
        result = self.forward_raw(ablation, args, schedule)
        first = int(schedule.n_append_by_macro.nonzero()[0].item())
        torch.testing.assert_close(
            result["append_pos_by_token"][:, first],
            reference["append_pos_by_token"][:, first],
        )
        before = schedule.before_by_macro.to(device="cuda")[None, :, None]
        self.assertTrue((reference["best_idx_by_token"][:, first] >= before[:, first]).any())
        targets = result["best_idx_by_token"]
        self.assertTrue((targets >= ablation.sink_len).any())
        self.assertTrue(((targets == -1) | ((targets >= ablation.sink_len) & (targets < before))).all())
        self.assertGreater(result["append_pos_by_token"].max().item(), args.initial_state_len)

    def test_a3_count_readout_and_backward(self):
        for schedule_mode in ("power_law", "fixed"):
            with self.subTest(schedule=schedule_mode):
                self.check_a3_reference(schedule_mode)

    def check_a3_reference(self, schedule_mode):
        mixer = self.mixer(kvm_use_vlens=0, state_budget_mode=schedule_mode)
        args = mixer._make_triton_args(1, self.length)
        schedule = mixer._make_schedule(args)
        self.assertEqual(schedule.final_state_len > schedule.initial_state_len, schedule_mode == "power_law")
        raw = self.forward_raw(mixer, args, schedule)
        v = self.v.detach().requires_grad_()
        out = mixer._triton_prefill_out(self.q, self.k, v, self.gate, args, schedule)
        reference = self.replay_a3_values(mixer, schedule, raw, v)
        grad_out = torch.randn_like(reference)
        (grad,) = torch.autograd.grad(out[:, :, :self.length], v, grad_out)
        (expected_grad,) = torch.autograd.grad(reference, v, grad_out)
        torch.testing.assert_close(out[:, :, :self.length], reference, atol=0.004, rtol=0.025)
        torch.testing.assert_close(grad, expected_grad, atol=0.018, rtol=0.05)

    def replay_a3_values(self, mixer, schedule, raw, values):
        # The discrete routes are fixed in backward. Replay them with PyTorch
        # autograd to check the count denominator before and after each update.
        v = values[0, 0]
        gate = self.gate[0, 0]
        state_v = v[:self.chunk]
        counts = torch.ones(self.chunk, device=v.device)
        state_temp, front_temp = mixer._head_temperatures(v.device)
        blocks = []
        for macro, active in enumerate(schedule.attention_state_len_by_macro.tolist()):
            start = macro * self.chunk
            end = min(start + self.chunk, self.length)
            front_start = max(0, start - self.chunk)
            keys = (self.k[0, 0, front_start:end].float() * front_temp).bfloat16()
            scores = self.q[0, 0, start:end].float() @ keys.float().T
            future = torch.arange(front_start, end, device=v.device)[None, :] > (
                torch.arange(start, end, device=v.device)[:, None]
            )
            scores = scores.masked_fill(future, -torch.inf)
            attn_values = v[front_start:end]
            if active:
                readout = (state_v.float() / counts[:, None]).bfloat16()
                torch.testing.assert_close(
                    raw["state_v_attn_by_macro"][macro][0, :active], readout,
                    atol=0.001, rtol=0.008,
                )
                keys = (raw["state_k_attn_by_macro"][macro][0, :active].float() * state_temp).bfloat16()
                state_scores = self.q[0, 0, start:end].float() @ keys.float().T
                scores = torch.cat((state_scores, scores), dim=-1)
                attn_values = torch.cat((readout, attn_values))
            weights = (scores / self.dim**0.5).softmax(dim=-1)
            blocks.append((weights @ attn_values.float()).bfloat16())

            overflow = macro - (mixer.n_bswa_chunks - 1)
            if overflow < 0 or not schedule.valid_update_by_macro[overflow]:
                continue
            append = raw["append_pos_by_token"][0, overflow] >= 0
            targets = raw["best_idx_by_token"][0, overflow][~append].long()
            tokens = slice(overflow * self.chunk, (overflow + 1) * self.chunk)
            state_v = torch.cat((state_v, v[tokens][append]))
            counts = torch.cat((counts, torch.ones_like(v[tokens, 0][append], dtype=torch.float32)))
            counts = counts.index_add(0, targets, torch.ones_like(targets, dtype=torch.float32))
            delta = torch.zeros_like(state_v, dtype=torch.float32).index_add(
                0, targets, (v[tokens] * gate[tokens])[~append].float()
            )
            state_v = state_v + delta.bfloat16()

        torch.testing.assert_close(raw["state_vlen"][0, :counts.numel()], counts, rtol=0, atol=0)
        self.assertEqual(counts.sum().item(), schedule.final_state_coverage_len)
        return torch.cat(blocks)[None, None]

    @torch.no_grad()
    def test_cached_decode_matches_prefill(self):
        for flags in (
            {"kvm_use_vlens": 0}, {"kvm_merge_old_slots_only": 1},
            {"kvm_append_policy": "uniform"}, {"kvm_append_policy": "random"},
        ):
            mixer = self.mixer(**flags).eval()
            reference = self.mixer(**flags).eval()
            reference.load_state_dict(mixer.state_dict())
            for length in (37, 3 * self.chunk):
                with self.subTest(flags=flags, length=length):
                    cache = StatesDictCache()
                    mixer.forward_prefill(
                        self.q[:, :, :length], self.k[:, :, :length],
                        self.v[:, :, :length], self.gate[:, :, :length],
                        None, None, None, cache,
                    )
                    actual = mixer.forward_single(
                        self.q[:, :, length:length + 1], self.k[:, :, length:length + 1],
                        self.v[:, :, length:length + 1], self.gate[:, :, length:length + 1],
                        None, None, None, cache,
                    )
                    args = mixer._make_triton_args(1, length + 1)
                    schedule = mixer._make_schedule(args)
                    expected = reference._triton_forward_prefill_raw(
                        self.q[:, :, :length + 1], self.k[:, :, :length + 1],
                        self.v[:, :, :length + 1], self.gate[:, :, :length + 1],
                        args, schedule,
                    )
                    decode = cache.get_states(0)[mixer._CACHE_TRITON_DECODE]
                    if flags.get("kvm_append_policy") == "random" and length == 3 * self.chunk:
                        last_update = int(schedule.valid_update_by_macro.nonzero()[-1].item())
                        torch.testing.assert_close(
                            decode.append_pos_by_token[:, 0],
                            expected["append_pos_by_token"][:, last_update],
                        )
                    torch.testing.assert_close(
                        decode.state_vlen[:, :decode.state_len],
                        expected["state_vlen"][:, :schedule.final_state_len],
                        rtol=0.001, atol=0.001,
                    )
                    torch.testing.assert_close(
                        actual, mixer.c_proj(expected["out"][:, length:length + 1]),
                        rtol=0.03, atol=0.004,
                    )

    @torch.no_grad()
    def test_decode_counts_all_merges_after_growth_stops(self):
        mixer = self.mixer(
            kvm_use_vlens=0, state_growth_factor=32, n_max_d_chunks=2,
        ).eval()
        length = 3 * self.chunk
        cache = StatesDictCache()
        mixer.forward_prefill(
            self.q[:, :, :length], self.k[:, :, :length],
            self.v[:, :, :length], self.gate[:, :, :length],
            None, None, None, cache,
        )
        decode = cache.get_states(0)[mixer._CACHE_TRITON_DECODE]
        for position in range(length, length + self.chunk + 1):
            token = slice(position, position + 1)
            mixer.forward_single(
                self.q[:, :, token], self.k[:, :, token],
                self.v[:, :, token], self.gate[:, :, token],
                None, None, None, cache,
            )
            if position == length:
                self.assertEqual(decode.state_len, mixer.max_state_len)
                self.assertTrue((decode.append_pos_by_token >= 0).any())
        self.assertEqual(decode.update_count, 2)
        self.assertTrue((decode.append_pos_by_token == -1).all())
        self.assertEqual(
            decode.state_vlen[:, :decode.state_len].sum().item(),
            decode.state_coverage_len,
        )
