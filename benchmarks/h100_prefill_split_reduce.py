"""Keep rare large-frontier register requirements out of the common reducer."""
import triton
import triton.language as tl
from lod_attention.kernels.paged_prefill import _reduce_small_frontier,_reduce_variable_route_attention_kernel


@triton.jit
def reduce(Values,LSE,Counts,Output,OutputLSE,WIDTH:tl.constexpr,R:tl.constexpr,D:tl.constexpr,
           SMALL:tl.constexpr,MIN_COUNT:tl.constexpr=32):
    row=tl.program_id(0).to(tl.int64);count=tl.load(Counts+row)
    if SMALL:
        if count<=32:_reduce_small_frontier(Values,LSE,Output,OutputLSE,row,count,WIDTH,32,D)
    else:
        if count>MIN_COUNT:
            if count<=32:_reduce_small_frontier(Values,LSE,Output,OutputLSE,row,count,WIDTH,32,D)
            elif count<=64:_reduce_small_frontier(Values,LSE,Output,OutputLSE,row,count,WIDTH,64,D)
            elif count<=128:_reduce_small_frontier(Values,LSE,Output,OutputLSE,row,count,WIDTH,128,D)
            else:_reduce_variable_route_attention_kernel(Values,LSE,Counts,Output,OutputLSE,WIDTH,R,D)
