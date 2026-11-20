"""Benchmark-only QUEST-style page bounds and sparse BF16 attention.

The page cache remains vLLM's native grouped-query cache. Bounds are updated
from the actual post-RoPE cached keys, including partial pages. These kernels
are an independent implementation of QUEST, not the authors' CUDA kernels.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _update(Cache, Slots, Bounds, S0: tl.constexpr, S1: tl.constexpr,
            S2: tl.constexpr, S3: tl.constexpr, NT: tl.constexpr,
            H: tl.constexpr, D: tl.constexpr, P: tl.constexpr):
    i = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(Slots + i)
    prev = tl.load(Slots + i - 1, mask=i > 0, other=-1)
    block = slot // P
    if slot >= 0 and (i == 0 or prev < 0 or prev // P != block):
        j = i + tl.arange(0, P)
        other = tl.load(Slots + j, mask=j < NT, other=-1)
        end = tl.max(tl.where((other >= 0) & (other // P == block), other % P + 1, 0), 0)
        t = tl.arange(0, P)
        d = tl.arange(0, D)
        k = tl.load(Cache + block*S0 + h*S1 + t[:, None]*S2 + d[None, :]*S3,
                    mask=t[:, None] < end, other=0).to(tl.float32)
        lo = tl.min(tl.where(t[:, None] < end, k, float('inf')), 0)
        hi = tl.max(tl.where(t[:, None] < end, k, -float('inf')), 0)
        off = (block*H + h)*2*D + d
        tl.store(Bounds + off, lo)
        tl.store(Bounds + off + D, hi)


@tr.jit
def _score(Q, Bounds, Table, Length, Scores, QB: tl.constexpr, QH: tl.constexpr,
           QD: tl.constexpr, TS: tl.constexpr, MAXP: tl.constexpr,
           H: tl.constexpr, KH: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
           BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    h = tl.program_id(1)
    page = tl.program_id(2)*BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    length = tl.load(Length + batch)
    valid = (page < MAXP) & (page*P < length)
    phys = tl.load(Table + batch*TS + page, mask=valid, other=0)
    kh = h // (H // KH)
    base = (phys[:, None]*KH + kh)*2*D + d[None, :]
    lo = tl.load(Bounds + base, mask=valid[:, None], other=0).to(tl.float32)
    hi = tl.load(Bounds + base + D, mask=valid[:, None], other=0).to(tl.float32)
    q = tl.load(Q + batch*QB + h*QH + d*QD).to(tl.float32)
    score = tl.sum(tl.maximum(q[None, :]*lo, q[None, :]*hi), 1)
    score = tl.where(valid, score, -float('inf'))
    # QUEST always retains the most recent page.
    score = tl.where(valid & (page == (length - 1)//P), float('inf'), score)
    tl.store(Scores + (batch*H+h)*MAXP + page, score, mask=page < MAXP)


@tr.jit
def _attend(Q, Cache, Table, Length, Selected, Partial, QB: tl.constexpr,
            QH: tl.constexpr, QD: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
            S2: tl.constexpr, S3: tl.constexpr, TS: tl.constexpr,
            H: tl.constexpr, KH: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
            K: tl.constexpr, NS: tl.constexpr, PAGES: tl.constexpr,
            SCALE: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    split = tl.program_id(2)
    d = tl.arange(0, D)
    t = tl.arange(0, P)
    q = tl.load(Q + b*QB + h*QH + d*QD).to(tl.float32)
    length = tl.load(Length + b)
    kh = h // (H // KH)
    m = tl.full((), -float('inf'), tl.float32)
    total = tl.full((), 0, tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    for j in range(PAGES):
        ix = split*PAGES + j
        page = tl.load(Selected + (b*H+h)*K + ix, mask=ix < K, other=0)
        valid_page = (ix < K) & (page*P < length)
        phys = tl.load(Table + b*TS + page, mask=valid_page, other=0)
        valid = valid_page & (page*P+t < length)
        ptr = Cache + phys*S0 + kh*S1 + t[:, None]*S2 + d[None, :]*S3
        keys = tl.load(ptr, mask=valid[:, None], other=0).to(tl.float32)
        values = tl.load(ptr + D*S3, mask=valid[:, None], other=0).to(tl.float32)
        logits = tl.sum(keys*q[None, :], 1)*SCALE
        logits = tl.where(valid, logits, -float('inf'))
        new_m = tl.maximum(m, tl.max(logits, 0))
        safe_m = tl.where(new_m == -float('inf'), 0., new_m)
        alpha = tl.where(m == -float('inf'), 0., tl.exp(m-safe_m))
        p = tl.where(valid, tl.exp(logits-safe_m), 0.)
        acc = acc*alpha + tl.sum(p[:, None]*values, 0)
        total = total*alpha + tl.sum(p, 0)
        m = new_m
    base = ((b*H+h)*NS+split)*(D+2)
    tl.store(Partial+base+d, acc)
    tl.store(Partial+base+D, total)
    tl.store(Partial+base+D+1, m)


@tr.jit
def _reduce(Partial, Out, OB: tl.constexpr, OH: tl.constexpr, OD: tl.constexpr,
            H: tl.constexpr, D: tl.constexpr, NS: tl.constexpr, BS: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    base = ((b*H+h)*NS+s)*(D+2)
    m = tl.load(Partial+base+D+1, mask=s < NS, other=-float('inf'))
    mmax = tl.max(m, 0)
    safe = tl.where(mmax == -float('inf'), 0., mmax)
    weight = tl.where(m == -float('inf'), 0., tl.exp(m-safe))
    z = tl.load(Partial+base+D, mask=s < NS, other=0)
    v = tl.load(Partial+base[:, None]+d[None, :], mask=(s < NS)[:, None], other=0)
    numerator = tl.sum(v*weight[:, None], 0)
    denominator = tl.sum(z*weight, 0)
    tl.store(Out+b*OB+h*OH+d*OD, numerator/tl.maximum(denominator, 1.e-20))


def update_bounds(cache, slots, bounds):
    blocks, heads, page_size, twice_dim = cache.shape
    _update[(slots.numel(), heads)](cache, slots, bounds, *cache.stride(),
                                    slots.numel(), heads, twice_dim//2, page_size,
                                    num_warps=4)


def quest_attention(query, cache, table, lengths, bounds, output, budget_tokens, scale):
    batch, heads, dim = query.shape
    blocks, kv_heads, page_size, _ = cache.shape
    assert dim == 128 and page_size == 16 and heads % kv_heads == 0
    max_pages = table.shape[1]
    k = min(max_pages, max(1, (budget_tokens + page_size - 1)//page_size))
    scores = torch.empty((batch, heads, max_pages), device=query.device, dtype=torch.float32)
    _score[(batch, heads, tr.cdiv(max_pages, 16))](query, bounds, table, lengths, scores,
        *query.stride(), table.stride(0), max_pages, heads, kv_heads, dim, page_size, 16,
        num_warps=4)
    selected = torch.topk(scores, k, dim=-1, sorted=False).indices
    pages_per_split = 16
    ns = tr.cdiv(k, pages_per_split)
    partial = torch.empty((batch, heads, ns, dim+2), device=query.device, dtype=torch.float32)
    _attend[(batch, heads, ns)](query, cache, table, lengths, selected, partial,
        *query.stride(), *cache.stride(), table.stride(0), heads, kv_heads, dim, page_size,
        k, ns, pages_per_split, scale, num_warps=4)
    _reduce[(batch, heads)](partial, output, *output.stride(), heads, dim, ns,
                            tr.next_power_of_2(ns), num_warps=4)
