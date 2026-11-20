"""Sort-free BF16 page refinement for the three-tier threshold frontier.

Inside every opened region, page proxy masses are normalized over that region.
Routing can use native or RMS-normalized queries; final attention always uses
native queries. Open page p when mass_p > c / number_of_pages. Single-page
regions are exact. All unselected pages share one disjoint residual summary.

A summary scan builds a 64-bit selection mask. The exact pass reads the
selected pages in ordinal order with one or more pages per attention tile.
The short scan specializes small regions and can reuse their page directory.
The existing 1,024-entry region eligibility cap bounds each list to 64 pages.
This adaptive cutoff does not guarantee sublinear total exact KV work.
"""
import math
import torch
import triton
import triton.language as tl
from lod_attention.kernels._paged_common import _lookup_page_id


@triton.jit
def _first_page_bit(bits, IS_HIP: tl.constexpr):
    if IS_HIP:
        low = bits.to(tl.uint32)
        high = (bits >> 32).to(tl.uint32)
        part = tl.where(low != 0, low, high)
        bit = tl.inline_asm_elementwise(
            'v_ffbl_b32 $0, $1', constraints='=v,v', args=[part],
            dtype=tl.int32, is_pure=True, pack=1)
        return bit + tl.where(low != 0, 0, 32)
    lowest = bits & (~bits + 1)
    return tl.inline_asm_elementwise('bfind.u64 $0, $1;', constraints='=r,l',
                                     args=[lowest], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _page_ids(Table, HashKeys, HashValues, HashUsed, kh, slot, ordinal, valid,
              S:tl.constexpr, ROOTS:tl.constexpr, P:tl.constexpr, HASH:tl.constexpr, PROBES:tl.constexpr):
    # Inline lists are common. Only issue hash loads when a selected ordinal
    # can actually fall outside the inline directory.
    if PROBES > 0:
        inline = ordinal < ROOTS
        if tl.sum((valid & ~inline).to(tl.int32), 0) == 0:
            page = tl.load(Table + (kh*S+slot)*ROOTS + ordinal, mask=valid, other=-1).to(tl.int32)
        else:
            page = _lookup_page_id(Table,HashKeys,HashValues,HashUsed,kh,slot,ordinal,
                                   valid,S,ROOTS,P,HASH,PROBES)
    else:
        page = _lookup_page_id(Table,HashKeys,HashValues,HashUsed,kh,slot,ordinal,
                               valid,S,ROOTS,P,HASH,PROBES)
    return page


@triton.jit
def _select_frontier(SK, SV, SC, PK, PV, PC, Table, HashKeys, HashValues, HashUsed,
                     kh, slot, np, length, query, d, inverse_rms,
                     D:tl.constexpr, S:tl.constexpr, P:tl.constexpr, ROOTS:tl.constexpr,
                     HASH:tl.constexpr, PROBES:tl.constexpr, SCALE:tl.constexpr, C:tl.constexpr,
                     WIDTH:tl.constexpr, FAST:tl.constexpr, PER_ENTRY:tl.constexpr, STABLE:tl.constexpr,
                     FAST_RESIDUAL_LOGIT:tl.constexpr):
    ordinal = tl.arange(0, WIDTH)
    if FAST and PROBES > 0 and WIDTH <= ROOTS:
        page = tl.load(Table + (kh*S + slot)*ROOTS + ordinal, mask=ordinal<np, other=-1).to(tl.int32)
    elif FAST:
        page = _page_ids(Table, HashKeys, HashValues, HashUsed, kh, slot, ordinal, ordinal<np,
                         S, ROOTS, P, HASH, PROBES)
    else:
        page = _lookup_page_id(Table, HashKeys, HashValues, HashUsed, kh, slot, ordinal,
                               ordinal<np, S, ROOTS, P, HASH, PROBES)
    valid = (ordinal<np) & (page>=0) & (page<P)
    count = tl.load(PC + kh*P + page, mask=valid, other=0).to(tl.float32)
    valid &= count>0
    page = tl.where(valid, page, 0)
    sums = tl.full((WIDTH,D), 0, tl.float32)
    dots = tl.full((WIDTH,), 0, tl.float32)
    if (FAST or STABLE) and np==1:
        opened = valid
    else:
        sums = tl.load(PK + (kh*P+page[:,None])*D+d[None,:], mask=valid[:,None], other=0).to(tl.float32)
        dots = tl.sum(sums*query[None,:],1)
        score = dots*SCALE*inverse_rms/tl.maximum(count,1.)+tl.log2(tl.maximum(count,1.))
        score = tl.where(valid,score,-float('inf'))
        top = tl.max(score,0)
        safe = tl.where(top!=-float('inf'),top,0.)
        w = tl.exp2(score-safe)
        z = tl.sum(w,0)
        if PER_ENTRY:
            cutoff = C*z*count/tl.maximum(length,1)*(1.+1.e-6)
        else:
            cutoff = C*z/tl.maximum(np,1)
        opened = valid & ((w>cutoff)|(np==1))
    nopen = tl.sum(tl.where(opened,count,0.),0)
    left = tl.load(SC+kh*S+slot).to(tl.float32)-nopen
    rs = tl.full((),-float('inf'),tl.float32)
    rv = tl.full((D,),0.,tl.float32)
    if left>0:
        keep = (valid & ~opened) if STABLE else opened
        values = tl.load(PV+(kh*P+page[:,None])*D+d[None,:],mask=keep[:,None],other=0).to(tl.float32)
        value_sum = tl.sum(values,0)
        if STABLE:
            rv = value_sum/left
            if FAST_RESIDUAL_LOGIT:
                residual_dot = tl.sum(tl.where(keep, dots, 0.), 0) / left
            else:
                key_sum = tl.sum(tl.where(keep[:,None],sums,0.),0)
                residual_dot = tl.sum((key_sum/left)*query,0)
        else:
            sk = tl.load(SK+(kh*S+slot)*D+d).to(tl.float32)
            sv = tl.load(SV+(kh*S+slot)*D+d).to(tl.float32)
            rv = (sv-value_sum)/left
            if FAST_RESIDUAL_LOGIT:
                # Region and page sums are BF16. Keep the subtraction and
                # the reused query dot products in FP32.
                residual_dot = (tl.sum(sk*query,0)-tl.sum(tl.where(keep, dots, 0.),0))/left
            else:
                key_sum = tl.sum(tl.where(keep[:,None],sums,0.),0)
                residual_dot = tl.sum(((sk-key_sum)/left)*query,0)
        rs = residual_dot*SCALE+tl.log2(left)
    bits = tl.sum(tl.where(opened,tl.full((WIDTH,),1,tl.uint64)<<ordinal,0),0)
    return bits, page, count, rs, rv, tl.sum(opened.to(tl.int32),0), nopen.to(tl.int32)


@triton.jit
def _pages(Q, SK, SV, SC, LK, LV, Indices, PK, PV, PC, Table, HashKeys, HashValues, HashUsed,
           Lengths, Selected, RouteCount, Out, LSE, Stats, CacheIndices,
           H:tl.constexpr, G:tl.constexpr, D:tl.constexpr, S:tl.constexpr, P:tl.constexpr,
           L:tl.constexpr, ROOTS:tl.constexpr, HASH:tl.constexpr, PROBES:tl.constexpr,
           WORKERS:tl.constexpr, SCALE:tl.constexpr, C:tl.constexpr, PER_ENTRY:tl.constexpr,
           NORMALIZE_QUERY:tl.constexpr, STABLE_RESIDUAL:tl.constexpr,
           BATCHED:tl.constexpr, HAS_CACHE_INDICES:tl.constexpr, FAST_BF16:tl.constexpr,
           EXACT_PAGES:tl.constexpr, REUSE_DIRECTORY:tl.constexpr, SHORT_SCAN:tl.constexpr,
           FAST_RESIDUAL_LOGIT:tl.constexpr, IS_HIP:tl.constexpr):
    head=tl.program_id(0);worker=tl.program_id(1);kh=head//G
    if BATCHED:
        request = head // H
        cache_row = tl.load(CacheIndices + request).to(tl.int32) if HAS_CACHE_INDICES else request
        kh = cache_row * (H // G) + (head % H) // G
    d=tl.arange(0,D);ordinal=tl.arange(0,64);token=tl.arange(0,16*EXACT_PAGES)
    query=tl.load(Q+head*D+d).to(tl.float32)
    inverse_rms=tl.full((),1.,tl.float32)
    if NORMALIZE_QUERY:
        inverse_rms=tl.rsqrt(tl.maximum(tl.sum(query*query,0)/D,1.e-12))
    maximum=tl.full((),-float('inf'),tl.float32);den=tl.full((),0.,tl.float32);acc=tl.full((D,),0.,tl.float32)
    nr=tl.load(RouteCount+head);allpages=tl.full((),0,tl.int32);selectedpages=tl.full((),0,tl.int32);exactentries=tl.full((),0,tl.int32)
    for route in tl.range(worker,nr,WORKERS,num_stages=1):
        slot=tl.load(Selected+head*S+route).to(tl.int32)
        length=tl.load(Lengths+kh*S+slot);np=(length+15)//16
        allpages+=np
        if SHORT_SCAN and np <= SHORT_SCAN:
            bits, short_page, short_count, rs, rv, npage, nentry = _select_frontier(
                SK, SV, SC, PK, PV, PC, Table, HashKeys, HashValues, HashUsed,
                kh, slot, np, length, query, d, inverse_rms, D, S, P, ROOTS, HASH, PROBES,
                SCALE, C, SHORT_SCAN, FAST_BF16, PER_ENTRY, STABLE_RESIDUAL, FAST_RESIDUAL_LOGIT)
            if REUSE_DIRECTORY:
                page = tl.gather(short_page, ordinal % SHORT_SCAN, 0)
                count = tl.gather(short_count, ordinal % SHORT_SCAN, 0)
        else:
            bits, page, count, rs, rv, npage, nentry = _select_frontier(
                SK, SV, SC, PK, PV, PC, Table, HashKeys, HashValues, HashUsed,
                kh, slot, np, length, query, d, inverse_rms, D, S, P, ROOTS, HASH, PROBES,
                SCALE, C, 64, FAST_BF16, PER_ENTRY, STABLE_RESIDUAL, FAST_RESIDUAL_LOGIT)
        selectedpages += npage
        exactentries += nentry
        if rs != -float('inf'):
            nm=tl.maximum(maximum,rs);correction=tl.exp2(maximum-nm);prob=tl.exp2(rs-nm)
            acc=acc*correction+prob*rv;den=den*correction+prob;maximum=nm
        while bits!=0:
            if EXACT_PAGES > 1:
                groups = tl.arange(0, EXACT_PAGES)
                indices = tl.full((EXACT_PAGES,), -1, tl.int32)
                for i in tl.static_range(EXACT_PAGES):
                    next_page = tl.where(bits != 0, _first_page_bit(bits, IS_HIP), -1)
                    indices = tl.where(groups == i, next_page, indices)
                    bits &= bits - 1
                present = indices >= 0
                if REUSE_DIRECTORY:
                    chosen = tl.gather(page, tl.maximum(indices, 0), 0)
                    nc = tl.gather(count, tl.maximum(indices, 0), 0).to(tl.int32)
                elif FAST_BF16 and PROBES > 0 and np <= ROOTS:
                    chosen = tl.load(Table + (kh * S + slot) * ROOTS + indices, mask=present, other=0).to(tl.int32)
                    nc = tl.load(PC + kh * P + chosen, mask=present, other=0)
                else:
                    chosen = _lookup_page_id(Table,HashKeys,HashValues,HashUsed,kh,slot,indices,
                                             present,S,ROOTS,P,HASH,PROBES)
                    nc = tl.load(PC + kh * P + chosen, mask=present, other=0)
                chosen = tl.gather(chosen, token // 16, 0)
                nc = tl.gather(tl.where(present, nc, 0), token // 16, 0)
                within = token % 16
            else:
                which = _first_page_bit(bits, IS_HIP)
                bits &= bits - 1
                if REUSE_DIRECTORY:
                    chosen = tl.sum(tl.gather(page, which[None], 0), 0)
                    nc = tl.sum(tl.gather(count, which[None], 0), 0).to(tl.int32)
                else:
                    if FAST_BF16 and PROBES > 0 and which < ROOTS:
                        chosen=tl.load(Table+(kh*S+slot)*ROOTS+which).to(tl.int32)
                    else:
                        chosen=_lookup_page_id(Table,HashKeys,HashValues,HashUsed,kh,slot,which,True,S,ROOTS,P,HASH,PROBES)
                    nc=tl.load(PC+kh*P+chosen)
                within = token
            idx=tl.load(Indices+(kh*P+chosen)*16+within,mask=within<nc,other=-1)
            ok=(within<nc)&(idx>=0)&(idx<L);safeidx=tl.where(ok,idx,0)
            key=tl.load(LK+(kh*L+safeidx[:,None])*D+d[None,:],mask=ok[:,None],other=0).to(tl.float32)
            value=tl.load(LV+(kh*L+safeidx[:,None])*D+d[None,:],mask=ok[:,None],other=0).to(tl.float32)
            score=tl.where(ok,tl.sum(key*query[None,:],1)*SCALE,-float('inf'))
            mx=tl.max(score,0);nm=tl.maximum(maximum,mx);correction=tl.exp2(maximum-nm)
            probability=tl.where(ok,tl.exp2(score-nm),0.)
            acc=acc*correction+tl.sum(probability[:,None]*value,0);den=den*correction+tl.sum(probability,0);maximum=nm
    row=head*WORKERS+worker
    tl.store(Out+row*D+d,tl.where(den>0,acc/tl.maximum(den,1.e-30),0.))
    tl.store(LSE+row,tl.where(den>0,maximum+tl.log2(tl.maximum(den,1.e-30)),-float('inf'))*math.log(2.))
    tl.store(Stats+row*4,nr if worker==0 else 0)
    tl.store(Stats+row*4+1,allpages);tl.store(Stats+row*4+2,selectedpages);tl.store(Stats+row*4+3,exactentries)


def page_threshold_attention(q,state_k,state_v,state_counts,leaf_k,leaf_v,page_indices,page_sum_k,page_sum_v,page_counts,
        slot_pages,overflow_page_keys,overflow_page_values,overflow_used,slot_lengths,top_slots,*,
        multiplier,stats_buffer,kv_group_size,scale,hash_probes,route_counts,route_workers,output_buffer,lse_buffer,
        page_threshold_warps=4,rule='page',query_score='native',residual='region',cache_indices=None,
        fast_bf16=False,exact_pages=1,reuse_directory=False,short_scan=0,
        fast_residual_logit=False,**kwargs):
    if not math.isfinite(multiplier) or multiplier<=0:
        raise ValueError('page mass multiplier must be finite and positive')
    if rule not in ('page','entry'):
        raise ValueError('page threshold rule must be page or entry')
    if query_score not in ('native','normalized'):
        raise ValueError('page query score must be native or normalized')
    if residual not in ('region','pages'):
        raise ValueError('page residual must use region subtraction or remaining-page sums')
    if exact_pages not in (1, 2, 4):
        raise ValueError('exact page tile must be 1, 2 or 4')
    if short_scan not in (0, 8, 16, 32):
        raise ValueError('short page scan must be disabled or use 8, 16 or 32 pages')
    if (q.ndim != 4 or q.size(2) != 1 or q.dtype != torch.bfloat16
            or (q.size(-1), kv_group_size) not in ((128, 4), (128, 8), (256, 6))
            or q.size(1) != state_k.size(1) * kv_group_size):
        raise ValueError('page thresholds require supported single-token BF16 GQA geometry')
    if 'quantized_leaf_k' in kwargs:
        raise ValueError('page threshold refinement currently requires BF16 KV')
    if not leaf_k.is_contiguous() or not leaf_v.is_contiguous():
        raise ValueError('page threshold KV must use contiguous indexed storage')
    B,H,_,D=q.shape;S=state_k.shape[2];P=page_counts.shape[2];L=leaf_k.shape[2]
    if (B != state_k.size(0) and cache_indices is None) or (cache_indices is not None and cache_indices.shape != (B,)):
        raise ValueError('page thresholds require a cache row for each request')
    _pages[(B*H,route_workers)](q,state_k,state_v,state_counts,leaf_k,leaf_v,page_indices,page_sum_k,page_sum_v,page_counts,
        slot_pages,overflow_page_keys,overflow_page_values,overflow_used,slot_lengths,top_slots,route_counts,
        output_buffer,lse_buffer,stats_buffer,cache_indices,H,kv_group_size,D,S,P,L,slot_pages.shape[3],overflow_page_values.shape[2],
        hash_probes,route_workers,float(scale)*math.log2(math.e),float(multiplier),rule=='entry',query_score=='normalized',
        residual=='pages',B>1 or B!=state_k.size(0),cache_indices is not None,fast_bf16,exact_pages,reuse_directory,
        short_scan,fast_residual_logit,torch.version.hip is not None,
        num_warps=page_threshold_warps)
    return output_buffer,lse_buffer
