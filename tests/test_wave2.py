import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from loss_plot.eval import evaluate_batch_loss_by_position
from model.kvm_mixer import SequenceMixer
from model.qwen3_backbone import Qwen3BackboneForCausalLM
from model.qwen_kvm_backbone import QwenKVMBackboneForCausalLM
from tests.test_wave1 import tiny_config
from trainer import write_hf_compatibility_wrapper


class QwenKVMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = tiny_config(num_key_value_heads=1, eos_token_id=47, pad_token_id=47)
        torch.manual_seed(0)
        self.reference = Qwen3BackboneForCausalLM(self.config)
        torch.manual_seed(0)
        self.model = QwenKVMBackboneForCausalLM(self.config)
        self.tokens = torch.randint(0, 47, (2, 21))

    def test_qwen_parameters_and_local_prefix_match(self):
        names = {"q_proj": "c_q", "k_proj": "c_k", "v_proj": "c_v", "o_proj": "c_proj",
                 "q_norm": "ln_q", "k_norm": "ln_k"}
        actual = self.model.state_dict()
        for key, tensor in self.reference.state_dict().items():
            for before, after in names.items():
                key = key.replace(f".self_attn.{before}.", f".self_attn.mixer.{after}.")
            torch.testing.assert_close(actual[key], tensor, rtol=0, atol=0)
        for layer in self.model.transformer.layers:
            self.assertIs(type(layer.self_attn.mixer), SequenceMixer)
        with torch.no_grad():
            actual = self.model(self.tokens, use_cache=False).logits
            expected = self.reference(self.tokens, use_cache=False).logits
        torch.testing.assert_close(actual[:, :8], expected[:, :8], atol=2e-6, rtol=2e-5)

    def test_training_and_cached_decode_across_block_boundary(self):
        labels = torch.roll(self.tokens, -1, dims=1)
        out = self.model(self.tokens, labels=labels, return_logits=False)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        gate = self.model.transformer.layers[0].self_attn.mixer.key_weighting.weight
        self.assertTrue(torch.isfinite(gate.grad).all())
        self.assertGreater(gate.grad.abs().sum(), 0)
        self.model.eval()
        with torch.no_grad():
            full = self.model(self.tokens, use_cache=False).logits
            prefix = self.model(self.tokens[:, :7], use_cache=True)
            pieces = [prefix.logits]
            cache = prefix.past_key_values
            for pos in range(7, self.tokens.shape[1]):
                pieces.append(self.model(self.tokens[:, pos:pos+1], past_key_values=cache).logits)
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=2e-4, rtol=2e-4)

    def test_export_and_greedy_generation(self):
        self.model.eval()
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            write_hf_compatibility_wrapper(directory)
            with patch.dict("os.environ", {"HF_WRAPPER_BASE_PATH": str(Path.cwd())}):
                restored = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True).eval()
            with torch.no_grad():
                torch.testing.assert_close(
                    restored(self.tokens, use_cache=False).logits,
                    self.model(self.tokens, use_cache=False).logits,
                )
            prompt = self.tokens[:, :7].contiguous()
            generated = restored.generate(
                input_ids=prompt, attention_mask=torch.ones_like(prompt),
                max_new_tokens=4, min_new_tokens=4, do_sample=False,
            )
            self.assertEqual(generated.shape, (2, 11))

    def test_textbook_chunked_loss_uses_kvm_forward(self):
        self.model.eval()
        labels = torch.roll(self.tokens, -1, dims=1)
        labels[:, -1] = -100
        loss, counts = evaluate_batch_loss_by_position(
            self.model, self.tokens, labels, loss_chunk_length=7
        )
        with torch.no_grad():
            logits = self.model(self.tokens, use_cache=False).logits
            expected = F.cross_entropy(
                logits.transpose(1, 2), labels, reduction="none"
            ).double().sum(dim=0)
        torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(counts, (labels != -100).sum(dim=0))
