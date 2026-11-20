"""Register-bounded final attention reduction and cache append."""
import triton
import triton.language as tl


@triton.jit
def merge_split_d(Co,Cl,Eo,El,Lo,Ll,NewK,NewV,LocalK,LocalV,LocalLens,Saved,Out,
                  D:tl.constexpr,G:tl.constexpr,KV:tl.constexpr,LOCAL_CAP:tl.constexpr,
                  NC:tl.constexpr,NE:tl.constexpr,NL:tl.constexpr,BLOCK:tl.constexpr,BD:tl.constexpr,APPEND:tl.constexpr=True,
    BATCH_KV_HEADS:tl.constexpr=0, RouteCount=None):
    head=tl.program_id(0);kh=head//G;db=tl.program_id(1)
    row=kh//BATCH_KV_HEADS if BATCH_KV_HEADS else 0
    x=tl.arange(0,BLOCK);dim=db*BD+tl.arange(0,BD);dm=dim<D
    ci=x<NC;ei=(x>=NC)&(x<NC+NE);li=(x>=NC+NE)&(x<NC+NE+NL)
    if RouteCount is not None:
        ei = ei & (x - NC < tl.load(RouteCount + head))
    l=tl.load(Cl+head*NC+x,mask=ci,other=-float('inf'))
    l=tl.where(ei,tl.load(El+head*NE+tl.maximum(x-NC,0),mask=ei,other=-float('inf')),l)
    l=tl.where(li,tl.load(Ll+head*NL+tl.maximum(x-NC-NE,0),mask=li,other=-float('inf')),l)
    mx=tl.max(l,0);w=tl.exp(l-mx);z=tl.sum(w,0)
    coarse=tl.load(Co+(head*NC+x[:,None])*D+dim[None,:],mask=ci[:,None]&dm[None,:],other=0)
    exact=tl.load(Eo+(head*NE+tl.maximum(x-NC,0)[:,None])*D+dim[None,:],mask=ei[:,None]&dm[None,:],other=0)
    local=tl.load(Lo+(head*NL+tl.maximum(x-NC-NE,0)[:,None])*D+dim[None,:],mask=li[:,None]&dm[None,:],other=0)
    value=tl.sum(w[:,None]*(coarse+exact+local),0)/z
    tl.store(Out+head*D+dim,value,mask=dm)
    if APPEND and head%G==0:
        n=tl.load(Saved+row);nk=tl.load(NewK+kh*D+dim,mask=dm,other=0);nv=tl.load(NewV+kh*D+dim,mask=dm,other=0)
        tl.store(LocalK+(kh*LOCAL_CAP+n)*D+dim,nk,mask=dm);tl.store(LocalV+(kh*LOCAL_CAP+n)*D+dim,nv,mask=dm)
    if APPEND and (head%(BATCH_KV_HEADS*G)==0 if BATCH_KV_HEADS else head==0) and db==0:
        tl.store(LocalLens+row,tl.load(Saved+row)+1)
