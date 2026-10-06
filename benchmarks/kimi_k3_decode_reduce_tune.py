"""Graph-timed split reducers with ragged head tiles and poisoned empty splits.

No model weights or serving timings. The serial implementation is retained
only here as the before-change benchmark control, never as a production path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from benchmarks.kimi_k3_decode_route_tune import compiler_resources, graph_us
from lod_attention.kernels.kimi_gluon_decode import _reduce_head_tiled_mla_splits_kernel


@triton.jit
def _serial_control(P, L, O, F, Lengths, SPLITS: tl.constexpr):
    batch, head = tl.program_id(0), tl.program_id(1)
    length = tl.load(Lengths + batch * 6 + head // 16)
    active = tl.cdiv(length, tl.maximum(1, tl.cdiv(length, SPLITS)))
    d = tl.arange(0, 512)
    acc = tl.full((512,), 0, tl.float32)
    maximum, denominator = -float("inf"), 0.0
    for split in tl.static_range(SPLITS):
        lse = tl.load(L + (batch * 96 + head) * SPLITS + split,
                      mask=split < active, other=-float("inf"))
        next_maximum = tl.maximum(maximum, lse)
        old_scale = tl.where(maximum == -float("inf"), 0., tl.exp(maximum-next_maximum))
        weight = tl.where(lse == -float("inf"), 0., tl.exp(lse-next_maximum))
        value = tl.load(P + ((batch * 96 + head) * SPLITS + split) * 512 + d,
                        mask=split < active, other=0.).to(tl.float32)
        acc = acc * old_scale + value * weight
        denominator = denominator * old_scale + weight
        maximum = next_maximum
    tl.store(O + (batch * 96 + head) * 512 + d,
             tl.where(denominator > 0, acc / denominator, 0))
    tl.store(F + batch * 96 + head, maximum + tl.log(denominator))


def inputs(batch, splits, *, poison_masked=False):
    torch.manual_seed(11)
    p = torch.randn(batch, 96, splits, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.randn(batch, 96, splits, device="cuda") * 30
    lengths = torch.tensor([0, 1, 33, 64, 259, 8193], dtype=torch.int32,
                           device="cuda").repeat(batch)
    # ceil(length / ceil(length / splits)), not floor.
    step = torch.div(lengths + splits - 1, splits, rounding_mode="floor").clamp_min(1)
    active = torch.div(lengths + step - 1, step, rounding_mode="floor").view(batch, 6)
    active = active.repeat_interleave(16, dim=1)
    live = torch.arange(splits, device="cuda")[None, None, :] < active[:, :, None]
    p.masked_fill_(~live[..., None], float("nan"))
    lse.masked_fill_(~live, float("nan"))
    # All-masked heads and a masked split amid otherwise live splits.
    lse[:, 16::13] = -torch.inf
    lse[:, :, 0] = -torch.inf
    if poison_masked:
        p.masked_fill_((live & lse.isneginf())[..., None], float("nan"))
    else:
        p.masked_fill_((live & lse.isneginf())[..., None], 0)
    return p, lse, lengths, live


def reference(p, lse, live):
    safe_lse = lse.masked_fill(~live, -torch.inf).double()
    total_lse = safe_lse.logsumexp(-1)
    weights = (safe_lse - total_lse.nan_to_num(neginf=0)[..., None]).exp()
    values = p.masked_fill(~(live & safe_lse.isfinite())[..., None], 0).double()
    return (values * weights[..., None]).sum(-2).bfloat16(), total_lse.float()


def launch_parallel(p, lse, lengths, out, final, *, block_d=256, warps=4,
                    indices=None, local=None, global_lens=None, rank=0,
                    world=8, interleave=1, advance=False, has_lse=True):
    dummy = lengths
    return _reduce_head_tiled_mla_splits_kernel[
        p.size(0), 96, triton.cdiv(512, block_d)](
        p, lse, out, final, lengths,
        indices if indices is not None else dummy,
        local if local is not None else dummy,
        global_lens if global_lens is not None else dummy,
        *p.stride()[:3], *lse.stride(), *out.stride()[:2], *final.stride(),
        NUM_SPLITS=p.size(2), HEAD_DIM=512, HEAD_TILES=6,
        HAS_FINAL_LSE=has_lse, ADVANCE_DCP_LENGTHS=advance,
        DCP_RANK=rank, DCP_WORLD_SIZE=world, DCP_INTERLEAVE_SIZE=interleave,
        BLOCK_S=triton.next_power_of_2(p.size(2)), BLOCK_D=block_d, num_warps=warps)


def run_case(batch, splits, block_d, warps, serial=False):
    p, lse, lengths, live = inputs(batch, splits)
    out = torch.empty(batch, 96, 512, device="cuda", dtype=torch.bfloat16)
    final = torch.empty(batch, 96, device="cuda")
    def launch():
        if serial:
            return _serial_control[(batch, 96)](p, lse, out, final, lengths,
                                                SPLITS=splits, num_warps=8)
        return launch_parallel(p, lse, lengths, out, final, block_d=block_d, warps=warps)
    compiled = launch()
    expected, expected_lse = reference(p, lse, live)
    torch.testing.assert_close(out, expected, rtol=0.008, atol=0.002)
    torch.testing.assert_close(final, expected_lse, rtol=1e-6, atol=2e-5)
    microseconds, samples = graph_us(launch)
    return dict(batch=batch, splits=splits, block_d=block_d, warps=warps, serial=serial,
                graph_kernel_us=microseconds, samples_us=samples,
                max_output_error=float((out.float()-expected.float()).abs().max()),
                resources=compiler_resources(compiled))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope="isolated split reducer, not serving latency", rows=[], status="in_progress")
    for batch in (1, 8):
        for splits in (64, 128):
            for block_d, warps, serial in ((512, 8, True), (64, 2, False),
                    (64, 4, False), (64, 8, False), (128, 4, False), (256, 4, False)):
                row = run_case(batch, splits, block_d, warps, serial)
                result["rows"].append(row)
                print(json.dumps(row), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2)+"\n")
    result["status"] = "complete"
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
