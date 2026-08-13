#!/usr/bin/env python3
"""CPU smoke test for the generation-only DiffusionGemma lm-eval adapter."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from lm_eval.api.instance import Instance
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    DiffusionGemmaConfig,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaTextConfig,
    PreTrainedTokenizerFast,
    SiglipVisionConfig,
)

from model.lm_eval_diffusion_gemma import DiffusionGemmaARLM, DiffusionGemmaLM
from model.diffusion_gemma_acceptance_compare import (
    DiffusionGemmaAcceptanceComparator,
    DiffusionGemmaEarlyNativeController,
)
from model.diffusion_gemma_phase_compare import DiffusionGemmaPhaseComparator
from model.diffusion_gemma_consensus_acceptance import (
    DiffusionGemmaConsensusAcceptance,
)
from model.diffusion_gemma_full_attention_review import (
    DiffusionGemmaFullAttentionReviewer,
)
from model.diffusion_gemma_native_entropy_acceptance import (
    DiffusionGemmaNativeEntropyAcceptance,
)
from model.hf_diffusion_gemma_lod_attention import (
    install_diffusion_gemma_lod_attention,
)
from model.pytorch_lod_attention_paged import PagedLODConfig


def _tiny_model() -> DiffusionGemmaForBlockDiffusion:
    text = DiffusionGemmaTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
        num_experts=2,
        top_k_experts=1,
        moe_intermediate_size=16,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    vision.rms_norm_eps = 1e-6
    vision.output_proj_dims = 16
    config = DiffusionGemmaConfig(
        text_config=text,
        vision_config=vision,
        canvas_length=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return DiffusionGemmaForBlockDiffusion(config).eval()


def main() -> None:
    torch.manual_seed(9)
    lm = object.__new__(DiffusionGemmaLM)
    lm._model = _tiny_model()
    lm.tokenizer = SimpleNamespace(pad_token_id=0)
    lm._diffusion_generation_requests = 0
    lm._diffusion_tokens_per_forward_sum = 0.0

    # The second prompt is left padded.  Ask for only two tokens even though
    # the model generates a complete four-token canvas, then verify truncation.
    context = torch.tensor([[1, 5, 6], [0, 1, 7]])
    attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    output = lm._model_generate(
        context,
        max_length=5,
        stop=[],
        attention_mask=attention_mask,
        max_denoising_steps=1,
    )
    assert output.shape == (2, 5), output.shape
    statistics = lm.diffusion_generation_statistics
    assert statistics["requests"] == 2, statistics
    assert statistics["mean_tokens_per_forward"] is not None, statistics

    try:
        lm.loglikelihood([])
    except NotImplementedError as error:
        assert "only supports generation tasks" in str(error)
    else:
        raise AssertionError("AR likelihood requests must be rejected")

    # Exercise HFLM's real batching, left padding, decoding, and output
    # reordering around the overridden model-generation call.
    vocabulary = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "hello": 4,
        "world": 5,
        "short": 6,
    }
    vocabulary.update({f"t{index}": index for index in range(7, 64)})
    tokenizer_backend = Tokenizer(
        WordLevel(vocab=vocabulary, unk_token="<unk>")
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.padding_side = "left"
    batched_lm = DiffusionGemmaLM(
        pretrained=_tiny_model(),
        tokenizer=tokenizer,
        batch_size=2,
        device="cpu",
        max_length=64,
    )
    generation_kwargs = {
        "until": [],
        "max_gen_toks": 2,
        "max_denoising_steps": 1,
    }
    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(prompt, generation_kwargs),
            idx=index,
        )
        for index, prompt in enumerate(("hello world", "short"))
    ]
    decoded = batched_lm.generate_until(requests, disable_tqdm=True)
    assert len(decoded) == 2, decoded
    assert batched_lm.diffusion_generation_statistics["requests"] == 2

    ar_lm = DiffusionGemmaARLM(
        pretrained=_tiny_model(),
        tokenizer=tokenizer,
        batch_size=2,
        device="cpu",
        max_length=64,
    )
    ar_output = ar_lm._model_generate(
        context,
        max_length=context.shape[1] + 3,
        stop=[],
        attention_mask=attention_mask,
    )
    assert ar_output.shape == (2, context.shape[1] + 3), ar_output.shape
    ar_statistics = ar_lm.ar_generation_statistics
    assert ar_statistics["requests"] == 2, ar_statistics
    assert ar_statistics["forward_passes"] >= 1, ar_statistics

    # Compare native and LOD logits on the exact same denoising trajectory.
    compared_model = _tiny_model()
    install_diffusion_gemma_lod_attention(
        compared_model,
        config=PagedLODConfig(
            chunk_size=4,
            local_window=8,
            state_growth_factor=8,
            state_min_size=8,
            protected_prefix=1,
            max_routes=8,
            leaf_dtype=torch.float32,
            page_size=4,
            kv_bits=0,
            quant_group_size=4,
        ),
        open_count=8,
        engine_backend="torch",
    )
    comparator = DiffusionGemmaAcceptanceComparator(compared_model)
    comparator.install()
    compared_model.generate(
        input_ids=torch.randint(3, 64, (2, 12)),
        attention_mask=torch.ones(2, 12, dtype=torch.long),
        max_new_tokens=2,
        max_denoising_steps=1,
        disable_compile=True,
    )
    comparison = comparator.summary()
    assert comparison["steps"] == 1, comparison
    assert comparison["positions"] == 8, comparison
    assert comparison["accepted"] > 0, comparison

    hybrid_model = _tiny_model()
    install_diffusion_gemma_lod_attention(
        hybrid_model,
        config=PagedLODConfig(
            chunk_size=4,
            local_window=8,
            state_growth_factor=8,
            state_min_size=8,
            protected_prefix=1,
            max_routes=8,
            leaf_dtype=torch.float32,
            page_size=4,
            kv_bits=0,
            quant_group_size=4,
        ),
        open_count=8,
        engine_backend="torch",
        encoder_attention_mode="native",
    )
    hybrid = DiffusionGemmaEarlyNativeController(hybrid_model, early_steps=1)
    hybrid.install()
    hybrid_model.generate(
        input_ids=torch.randint(3, 64, (2, 12)),
        attention_mask=torch.ones(2, 12, dtype=torch.long),
        max_new_tokens=2,
        max_denoising_steps=2,
        disable_compile=True,
    )
    hybrid_summary = hybrid.summary()
    assert hybrid_summary["canvases"] == 1, hybrid_summary
    assert hybrid_summary["native_step_calls"] == 1, hybrid_summary

    phase_model = _tiny_model()
    install_diffusion_gemma_lod_attention(
        phase_model,
        config=PagedLODConfig(
            chunk_size=4,
            local_window=8,
            state_growth_factor=8,
            state_min_size=8,
            protected_prefix=1,
            max_routes=8,
            leaf_dtype=torch.float32,
            page_size=4,
            kv_bits=0,
            quant_group_size=4,
        ),
        open_count=8,
        engine_backend="torch",
    )
    phase_comparator = DiffusionGemmaPhaseComparator(phase_model)
    phase_comparator.install()
    phase_model.generate(
        input_ids=torch.randint(3, 64, (2, 12)),
        attention_mask=torch.ones(2, 12, dtype=torch.long),
        max_new_tokens=2,
        max_denoising_steps=1,
        disable_compile=True,
    )
    phase_summary = phase_comparator.summary()
    assert phase_summary["steps"] == 1, phase_summary
    assert phase_summary["positions"] == 8, phase_summary
    assert phase_summary["reference_accepted"] > 0, phase_summary

    native_entropy_model = _tiny_model()
    install_diffusion_gemma_lod_attention(
        native_entropy_model,
        config=PagedLODConfig(
            chunk_size=4,
            local_window=8,
            state_growth_factor=8,
            state_min_size=8,
            protected_prefix=1,
            max_routes=8,
            leaf_dtype=torch.float32,
            page_size=4,
            kv_bits=0,
            quant_group_size=4,
        ),
        open_count=8,
        engine_backend="torch",
    )
    native_entropy = DiffusionGemmaNativeEntropyAcceptance(native_entropy_model)
    native_entropy.install()
    native_entropy_model.generate(
        input_ids=torch.randint(3, 64, (2, 12)),
        attention_mask=torch.ones(2, 12, dtype=torch.long),
        max_new_tokens=2,
        max_denoising_steps=1,
        disable_compile=True,
    )
    native_entropy_summary = native_entropy.summary()
    assert native_entropy_summary["encoder_calls"] == 1, native_entropy_summary
    assert native_entropy_summary["steps"] == 1, native_entropy_summary
    assert native_entropy_summary["native_masks_applied"] == 1, native_entropy_summary
    assert native_entropy_summary["positions"] == 8, native_entropy_summary
    assert native_entropy_summary["applied_native_accepts"] > 0, native_entropy_summary
    assert native_entropy_summary["sample_source"] == "lod_categorical_logits"

    consensus_model = _tiny_model()
    install_diffusion_gemma_lod_attention(
        consensus_model,
        config=PagedLODConfig(
            chunk_size=4,
            local_window=8,
            state_growth_factor=8,
            state_min_size=8,
            protected_prefix=1,
            max_routes=8,
            leaf_dtype=torch.float32,
            page_size=4,
            kv_bits=0,
            quant_group_size=4,
        ),
        open_count=8,
        engine_backend="torch",
    )
    consensus = DiffusionGemmaConsensusAcceptance(
        consensus_model, probe_open_count=8, mode="apply"
    )
    consensus.install()
    consensus_model.generate(
        input_ids=torch.randint(3, 64, (2, 12)),
        attention_mask=torch.ones(2, 12, dtype=torch.long),
        max_new_tokens=2,
        max_denoising_steps=1,
        disable_compile=True,
    )
    consensus_summary = consensus.summary()
    assert consensus_summary["steps"] == 1, consensus_summary
    assert consensus_summary["positions"] == 8, consensus_summary
    assert consensus_summary["veto_shortfall"] == 0, consensus_summary

    review_model = _tiny_model()
    install_diffusion_gemma_lod_attention(
        review_model,
        config=PagedLODConfig(
            chunk_size=4,
            local_window=8,
            state_growth_factor=8,
            state_min_size=8,
            protected_prefix=1,
            max_routes=8,
            leaf_dtype=torch.float32,
            page_size=4,
            kv_bits=0,
            quant_group_size=4,
        ),
        open_count=8,
        engine_backend="torch",
    )
    reviewer = DiffusionGemmaFullAttentionReviewer(
        review_model, lod_entropy_threshold=100.0, mode="apply"
    )
    reviewer.install()
    review_model.generate(
        input_ids=torch.randint(3, 64, (2, 12)),
        attention_mask=torch.ones(2, 12, dtype=torch.long),
        max_new_tokens=2,
        max_denoising_steps=1,
        disable_compile=True,
    )
    review_summary = reviewer.summary()
    assert review_summary["steps"] == 1, review_summary
    assert review_summary["full_attention_passes"] == 1, review_summary
    assert review_summary["reviewed"] > 0, review_summary
    print("DiffusionGemma lm-eval adapter smoke passed")


if __name__ == "__main__":
    main()
