"""Score the released weights and K0 on the original K0 validation batches."""

import json
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer

from data.preprocess import preprocess_dataset_and_get_dataloader
from model.rwkv7_backbone import RWKV7BackboneForCausalLM
from train import CLI_Config
from utils.defer import apply_deferred


def main():
    output = Path(__file__).resolve().parent
    reference = output.parent / "kvm-sqrt-original-20261110"
    manifest = json.loads((reference / "manifest.json").read_text())
    published = json.loads((output / "textbook-submission.json").read_text())
    checkpoints = {
        "paper_kvm_sqrt": published["checkpoint"],
        "k0": str(reference / "runs/K0"),
    }
    args = CLI_Config(**manifest["experiments"]["K0"]["resolved_config"]).train
    validation = args.val_data
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 0

    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        assert world == 8
        tokens_per_step = world * validation.device_batch_size * min(
            validation.max_document_tokens, validation.sequence_length
        )
        steps, remainder = divmod(args.val_tokens, tokens_per_step)
        assert remainder == 0 and steps > 0
        fingerprint = manifest["datasets"][validation.dataset]["fingerprint"]
        if rank == 0:
            dataset = load_dataset(
                validation.dataset, name=validation.dataset_name, split=validation.split
            )
            assert dataset._fingerprint == fingerprint, dataset._fingerprint
            print(f"Dataset fingerprint {fingerprint}, {steps} batches per rank", flush=True)
            del dataset

        loader = preprocess_dataset_and_get_dataloader(
            rank, world, validation, tokenizer, args.preprocess_batch_size
        )
        models = []
        for checkpoint in checkpoints.values():
            model = RWKV7BackboneForCausalLM.from_pretrained(
                checkpoint, dtype=torch.bfloat16, local_files_only=True
            ).to(device).eval()
            if args.compile:
                apply_deferred(model)
            models.append(model)

        losses = torch.zeros(len(models), device=device, dtype=torch.float32)
        valid_labels = torch.zeros((), device=device, dtype=torch.long)
        batches = iter(loader)
        with torch.no_grad():
            for step in range(steps):
                batch = next(batches)
                inputs = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                valid_labels += labels.ne(-100).sum()
                for index, model in enumerate(models):
                    result = model(
                        input_ids=inputs, labels=labels,
                        cu_seqlens=batch.get("cu_seqlens"), return_logits=True,
                    )
                    losses[index] += result.loss
                    del result
                if rank == 0 and (step == 0 or (step + 1) % 5 == 0):
                    print(f"Scored {step + 1}/{steps} DCLM batches per rank", flush=True)
        dist.all_reduce(losses, op=dist.ReduceOp.AVG)
        dist.all_reduce(valid_labels, op=dist.ReduceOp.SUM)
        if rank == 0:
            scores = dict(zip(checkpoints, (losses / steps).tolist(), strict=True))
            assert all(math.isfinite(score) for score in scores.values())
            logged = json.loads((reference / "pipeline_status.json").read_text())
            result = {
                "avg_batch_nll": scores,
                "k0_training_log_nll": logged["final_validation_loss"],
                "checkpoints": checkpoints,
                "training_reference": str(reference / "manifest.json"),
                "dataset_fingerprint": fingerprint,
                "validation_data": validation.model_dump(),
                "world_size": world,
                "batches_per_rank": steps,
                "padded_token_budget": args.val_tokens,
                "scored_label_tokens": valid_labels.item(),
                "dtype": "bfloat16",
                "compile": bool(args.compile),
            }
            target = output / "dclm-4k.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(target)
            print(json.dumps(scores), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
