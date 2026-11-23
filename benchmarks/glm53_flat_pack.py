"""Controlled decode-layer benchmark for physical-ID packing on GLM MLA."""

import json
import random
import statistics
from functools import partial

import torch
import triton
import triton.language as tl

from benchmarks.glm53_threshold_decode import reference
from lod_attention.kernels import glm_threshold_decode as glm


@triton.jit
def _save_bounds(Ends, Masks, Directory, Starts, group, tile, TILES: tl.constexpr,
                 T: tl.constexpr, start, sizes, ends, flags, offsets, src,
                 CAP: tl.constexpr, BLOCK: tl.constexpr):
    cell = group * TILES + tile
    region = cell * T + tl.arange(0, T)
    tl.store(Ends + region, ends)
    tl.store(Masks + region, flags)
    tl.store(Starts + cell, start)


@triton.jit
def _expand(Ends, Masks, Starts, Offsets, Fixed, Slots, Flat, Bits, S: tl.constexpr,
            GROUPS: tl.constexpr, TILES: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
            INDEX_STRIDE: tl.constexpr, LEAF_START: tl.constexpr, CAP: tl.constexpr):
    # Original two-pass flat control, including the six-step region-owner search.
    tile, group = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + group // GROUPS).to(tl.int32)
    cell = group * TILES + tile
    ends = tl.load(Ends + cell * T + tl.arange(0, T))
    size = tl.max(ends, 0)
    if size:
        start = tl.load(Starts + cell)
        for block in range(0, tl.cdiv(size, N)):
            pos = block * N + tl.arange(0, N)
            idx = tl.full((N,), -1, tl.int32)
            tl.static_assert(T == 64)
            for i in tl.static_range(6):
                test = idx + (T >> (i + 1))
                idx = tl.where(tl.gather(ends, test, 0) <= pos, test, idx)
            owner = tl.minimum(idx + 1, T - 1)
            before = tl.where(owner > 0, tl.gather(ends, tl.maximum(owner - 1, 0), 0), 0)
            offset = tl.load(Offsets + slot * (S + 1) + tile * T + owner, mask=pos < size, other=0)
            ids = tl.load(Fixed + slot * INDEX_STRIDE + LEAF_START + offset + pos - before,
                          mask=pos < size, other=0)
            mask = tl.load(Masks + cell * T + owner)
            dest = group * CAP + start + pos
            tl.store(Flat + dest, ids, mask=pos < size)
            tl.store(Bits + dest, mask, mask=pos < size)


@triton.jit
def _gather_ids(Flat, Bits, Fixed, Starts, group, tile, TILES: tl.constexpr, T: tl.constexpr,
                start, sizes, ends, flags, offsets, src, CAP: tl.constexpr, BLOCK: tl.constexpr):
    count = tl.max(ends, 0)
    for first in range(0, count, BLOCK):
        pos = first + tl.arange(0, BLOCK)
        rank = tl.zeros((BLOCK,), tl.int32)
        for bit in tl.static_range(T.bit_length() - 2, -1, -1):
            step = 1 << bit
            rank += tl.where(tl.gather(ends, rank + step - 1, 0) <= pos, step, 0)
        physical = tl.load(Fixed + src + tl.gather(offsets + sizes - ends, rank, 0) + pos,
                           mask=pos < count, other=0)
        dest = group * CAP + start + pos
        tl.store(Flat + dest, physical, mask=pos < count)
        tl.store(Bits + dest, tl.gather(flags, rank, 0), mask=pos < count)


@triton.jit
def _copy_regions(Flat, Bits, Fixed, Starts, group, tile, TILES: tl.constexpr, T: tl.constexpr,
                  start, sizes, ends, flags, offsets, src, CAP: tl.constexpr, BLOCK: tl.constexpr):
    for first in range(0, tl.max(sizes, 0), BLOCK):
        pos = first + tl.arange(0, BLOCK)
        live = pos[None, :] < sizes[:, None]
        physical = tl.load(Fixed + src + offsets[:, None] + pos[None, :], mask=live, other=0)
        dest = group * CAP + start + (ends - sizes)[:, None] + pos[None, :]
        tl.store(Flat + dest, physical, mask=live)
        tl.store(Bits + dest, flags[:, None], mask=live)


@triton.jit
def _copy_four(Flat, Bits, Fixed, Starts, group, tile, TILES: tl.constexpr, T: tl.constexpr,
               start, sizes, ends, flags, offsets, src, CAP: tl.constexpr, BLOCK: tl.constexpr):
    live = sizes > 0
    rank = tl.cumsum(live.to(tl.int32), 0) - 1
    count = tl.sum(live.to(tl.int32), 0)
    for first in range(0, count, 4):
        active = first + tl.arange(0, 4)
        match = live[None, :] & (rank[None, :] == active[:, None])
        owner = tl.max(tl.where(match, tl.arange(0, T)[None, :], 0), 1)
        length = tl.where(active < count, tl.gather(sizes, owner, 0), 0)
        source = tl.gather(offsets, owner, 0)
        dest = group * CAP + start + tl.gather(ends - sizes, owner, 0)
        mask = tl.gather(flags, owner, 0)
        for base in range(0, tl.max(length, 0), BLOCK):
            pos = base + tl.arange(0, BLOCK)
            valid = pos[None, :] < length[:, None]
            ids = tl.load(Fixed + src + source[:, None] + pos[None, :], mask=valid, other=0)
            tl.store(Flat + dest[:, None] + pos[None, :], ids, mask=valid)
            tl.store(Bits + dest[:, None] + pos[None, :], mask[:, None], mask=valid)


@triton.jit
def _copy_chunks(Flat, Bits, Fixed, Starts, group, tile, TILES: tl.constexpr, T: tl.constexpr,
                 start, sizes, ends, flags, offsets, src, CAP: tl.constexpr, BLOCK: tl.constexpr):
    # Resolve the region once per vector of leaves. Descriptors stay in registers.
    WIDTH: tl.constexpr = BLOCK
    CHUNKS: tl.constexpr = 16
    counts = tl.cdiv(sizes, WIDTH)
    chunk_ends = tl.cumsum(counts, 0)
    total_chunks = tl.max(chunk_ends, 0)
    for first in range(0, total_chunks, CHUNKS):
        chunk = first + tl.arange(0, CHUNKS)
        owner = tl.zeros((CHUNKS,), tl.int32)
        for bit in tl.static_range(T.bit_length() - 2, -1, -1):
            step = 1 << bit
            owner += tl.where(tl.gather(chunk_ends, owner + step - 1, 0) <= chunk, step, 0)
        base = (chunk - tl.gather(chunk_ends - counts, owner, 0)) * WIDTH
        pos = base[:, None] + tl.arange(0, WIDTH)[None, :]
        valid = (chunk[:, None] < total_chunks) & (pos < tl.gather(sizes, owner, 0)[:, None])
        physical = tl.load(Fixed + src + tl.gather(offsets, owner, 0)[:, None] + pos,
                           mask=valid, other=0)
        dest = group * CAP + start + tl.gather(ends - sizes, owner, 0)[:, None] + pos
        tl.store(Flat + dest, physical, mask=valid)
        tl.store(Bits + dest, tl.gather(flags, owner, 0)[:, None], mask=valid)


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def compare(a, data, ctx, pool, q, current, out, init):
    batch, heads, _ = q.shape
    groups = triton.cdiv(heads, 16)
    states = pool.state_capacity
    tiles = triton.cdiv(states, 64)
    splits, run_width = (256, 32) if batch == 1 else (64, 8)
    combine = a.pack_deferred and batch == 1
    parts = splits if combine else tiles + splits
    slots = pool.active_indices
    page = pool.state["page_cache"]
    arena = page["unified_page1_k"]
    fixed = page["unified_page1_fixed_indices"]
    offsets = page["unified_page1_fixed_slot_offsets"]
    local, coarse, sink = (int(page[f"unified_page1_{field}_offset"]) for field in ("local", "coarse", "sink"))
    nsink = pool.state["sink_k"].size(2)
    begin = pool.decode_local_limit + nsink + states
    buf = glm.reserve(pool)
    mass = torch.empty_like(buf["lse"]) if a.pack_deferred else None
    def scratch(*shape):
        return torch.empty(shape, device=q.device, dtype=torch.int32)
    ends, masks, starts = scratch(batch, groups, tiles, 64), scratch(batch, groups, tiles, 64), scratch(batch, groups, tiles)
    leaf_ids = scratch(batch, groups, pool.leaf_capacity)
    leaf_flags = torch.empty_like(leaf_ids)
    audit = torch.empty_like(masks)
    cases = {
        "runs": (None, 0), "flat_separate": (_save_bounds, 0),
        "gather128": (_gather_ids, 128), "gather256": (_gather_ids, 256),
        "regions16": (_copy_regions, 16), "regions64": (_copy_regions, 64),
        "regions128": (_copy_regions, 128), "regions256": (_copy_regions, 256),
        "chunks16": (_copy_chunks, 16), "chunks32": (_copy_chunks, 32), "chunks64": (_copy_chunks, 64),
        "chunks128": (_copy_chunks, 128), "four512": (_copy_four, 512),
        "four128": (_copy_four, 128), "four256": (_copy_four, 256),
    }
    for shards in (2, 4, 8):
        for lanes in (32, 64):
            cases[f"regions{lanes}_x{shards}"] = (_copy_regions, lanes)
    if a.pack_variants:
        cases = {name: cases[name] for name in a.pack_variants}
    results, graphs = {}, {}
    def build(name, writer, block):
        width = run_width if writer is None else 1
        separate = name == "flat_separate"
        shards = int(name.split("_x")[1]) if "_x" in name else 1
        ids, flags = (buf["flat"], buf["bits"]) if writer is None else (leaf_ids, leaf_flags)
        cap = ids.size(-1)
        score = partial(glm._score_regions[(tiles, batch * groups)], q, current, arena,
                        pool.state["counts"], pool.state_lens, pool.local_lens, slots, buf["scores"], buf["tlse"],
                        buf["active"], buf["total"], buf["saved"], states, heads, groups, tiles, 64, coarse,
                        local, pool.local_capacity, pool.engine.scaling, num_warps=4)
        def select(audited=False):
            glm._select_and_pack[(tiles, batch * groups, shards)](
                buf["scores"], buf["tlse"], buf["active"], slots, page["slot_lengths"], arena,
                audit if audited or combine else None, buf["total"], buf["partial"], buf["lse"], states, heads, groups,
                tiles, triton.next_power_of_2(tiles), 64, coarse, 16., pool.engine.max_open_centroid_leaves or 0,
                parts, ends if separate else ids, masks if separate else flags, offsets, cap, width,
                COMBINE=combine, Mass=mass, FlatWriter=writer, Directory=fixed if writer else None, TileStarts=starts if separate else None,
                DIR_STRIDE=fixed.size(-1), LEAF_START=begin, COPY=block, COPY_SHARDS=shards, num_warps=4)
        expand = partial(_expand[(tiles, batch * groups)], ends, masks, starts, offsets, fixed, slots, ids,
                         flags, states, groups, tiles, 64, 256, fixed.size(-1), begin, cap, num_warps=4)
        exact = partial(glm._attend_exact[(splits, batch * groups)], q, arena, slots, buf["saved"], ids, flags,
                        buf["total"], buf["partial"], buf["lse"], heads, groups, 32, splits, tiles, parts, local,
                        pool.local_capacity, pool.decode_local_limit, sink, nsink, cap, pool.engine.scaling,
                        fixed if writer is None else None, fixed.size(-1), begin, width, Mass=mass, MaskBits=audit if combine else None,
                        Scores=buf["scores"] if combine else None, S=states, COARSE=coarse, T=64,
                        CONTIGUOUS=combine, num_warps=4, num_stages=1)
        merge = partial(glm._merge_fields[(batch * heads, 16)], buf["partial"], buf["lse"], out, buf["final"],
                        slots, buf["saved"], pool.local_lens, heads, parts, triton.next_power_of_2(parts),
                        *out.stride(), Mass=mass, num_warps=4)
        def step(audited=False):
            pool.local_lens.copy_(init)
            score()
            select(audited)
            if separate:
                expand()
            exact()
            merge()
        def prepare():
            buf["total"].zero_()
            select()
            if separate:
                expand()
        step(True)
        ref, lse, work = reference(pool, q, current, init, 16., audit, packed=(ids, flags),
                                   page_size=run_width if writer is None else 0)
        torch.testing.assert_close(out.float(), ref, atol=.035, rtol=.025)
        torch.testing.assert_close(buf["final"][:batch], lse, atol=.0005, rtol=.0002)
        assert torch.equal(pool.local_lens, init + 1)
        for i, row in enumerate(slots.tolist()):
            assert torch.equal(pool.state["recent_k"][row, 0, int(init[row])], current[i, 0])
        step()
        full = capture(step)
        full.replay()
        torch.testing.assert_close(out.float(), ref, atol=.035, rtol=.025)
        torch.testing.assert_close(buf["final"][:batch], lse, atol=.0005, rtol=.0002)
        return ({"full": full, "prepare": capture(prepare), "exact": capture(exact)},
                dict(exact_leaves=work[1], union_leaves=work[2], relative_error=float((out.float()-ref).norm()/ref.norm()),
                     medians_us={}, timings_us={stage: [] for stage in ("full", "prepare", "exact")}))
    for name, (writer, block) in cases.items():
        graphs[name], results[name] = build(name, writer, block)
    rng = random.Random(a.pack_seed)
    for _ in range(13):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            # Each stage gets a matching frontier. Preparation resets its own atomic counter.
            graphs[name]["full"].replay()
            for stage, graph in graphs[name].items():
                graph.replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):
                    graph.replay()
                end.record()
                end.synchronize()
                results[name]["timings_us"][stage].append(start.elapsed_time(end) * 1000 / 30)
    for row in results.values():
        row["medians_us"] = {stage: statistics.median(times) for stage, times in row["timings_us"].items()}
    result = dict(source=str(a.capture), layer=data["module"], batch=batch, lengths=ctx.tolist(),
                  splits=splits, run_width=run_width, order_seed=a.pack_seed, deferred=a.pack_deferred, gpu=torch.cuda.get_device_name(), results=results)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print({k: v["medians_us"] for k, v in results.items()}, flush=True)
