"""One-token owner MLA projections, with fixed-address collective buffers."""

import triton
import triton.language as tl


@triton.jit
def _owner_query(Q, Key, UK, Index, Absorbed, SelectedKey,
                 Q_ROW:tl.constexpr, Q_HEAD:tl.constexpr, KEY_ROW:tl.constexpr,
                 UK_HEAD:tl.constexpr, UK_K:tl.constexpr, UK_N:tl.constexpr):
    head, tile = tl.program_id(0), tl.program_id(1)
    row = tl.load(Index)
    col = tile*64+tl.arange(0,64)
    if tile < 8:
        inner = tl.arange(0,128)
        query = tl.load(Q+row*Q_ROW+head*Q_HEAD+inner).to(tl.float32)
        weight = tl.load(UK+head*UK_HEAD+inner[:,None]*UK_K+col[None,:]*UK_N).to(tl.float32)
        latent = tl.sum(query[:,None]*weight,axis=0)
        tl.store(Absorbed+head*576+col,latent)
    else:
        direct = tl.load(Q+row*Q_ROW+head*Q_HEAD+128+tl.arange(0,64))
        tl.store(Absorbed+head*576+512+tl.arange(0,64),direct)
    if head==0 and tile==0:
        channels=tl.arange(0,1024)
        current=tl.load(Key+row*KEY_ROW+channels,channels<576,other=0)
        tl.store(SelectedKey+channels,current,channels<576)


@triton.jit
def _owner_value(Output, UV, Index, Send,
                 UV_HEAD:tl.constexpr, UV_K:tl.constexpr, UV_N:tl.constexpr):
    head, tile=tl.program_id(0),tl.program_id(1)
    col=tile*32+tl.arange(0,32)
    inner=tl.arange(0,128)
    value=tl.full((32,),0,tl.float32)
    for block in range(4):
        k=block*128+inner
        x=tl.load(Output+head*512+k).to(tl.float32)
        w=tl.load(UV+head*UV_HEAD+k[:,None]*UV_K+col[None,:]*UV_N).to(tl.float32)
        value+=tl.sum(x[:,None]*w,axis=0)
    rows=tl.arange(0,8)
    owner_row=tl.load(Index)
    # Head-owner-major reduce-scatter input, all inactive request rows zero.
    pointer=Send+(head//12)*8*12*128+rows[:,None]*12*128+(head%12)*128+col[None,:]
    tl.store(pointer,tl.where(rows[:,None]==owner_row,value[None,:],0))


def owner_query_projection(query,record,uk,index,absorbed,selected_key):
    if query.shape!=(8,96,192) or record.shape!=(8,1,576) or uk.shape!=(96,128,512):
        raise ValueError("one-token owner query geometry mismatch")
    _owner_query[(96,9)](query,record,uk,index,absorbed,selected_key,
        query.stride(0),query.stride(1),record.stride(0),*uk.stride(),num_warps=4)


def owner_value_projection(output,uv,index,send):
    if output.shape!=(1,96,512) or uv.shape!=(96,512,128) or send.shape!=(8,8,12,128):
        raise ValueError("one-token owner value geometry mismatch")
    _owner_value[(96,4)](output,uv,index,send,*uv.stride(),num_warps=4)


__all__=["owner_query_projection","owner_value_projection"]
