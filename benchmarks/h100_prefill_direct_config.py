"""Dependency-free dispatch settings for the experimental direct-summary path."""
import os


def producer_stages(state_len):
    """Use deeper pipelining for small states and lower shared memory for larger ones."""
    policy=os.environ.get('LOD_H100_FAST_PRODUCE_STAGES','2')
    if policy=='auto':return 2 if state_len<=2048 else 1
    if policy not in ('1','2'):raise ValueError('producer stages must be 1, 2, or auto')
    return int(policy)


def scratch_queries(state_len,default=4096):
    """Bound score scratch growth while retaining the larger small-state tile."""
    policy=os.environ.get('LOD_H100_FAST_TILE',str(default))
    if policy=='auto':return 16384 if state_len<=2048 else 8192 if state_len<=4096 else 4096
    size=int(policy)
    if size<=0:raise ValueError('fast tile must be positive')
    return size


def fused_selection_enabled():
    """Resolve the optimized selector only inside the opt-in FP16 direct path."""
    policy=os.environ.get('LOD_H100_DIRECT_FUSE_SELECT','auto')
    if policy not in ('0','1','auto'):raise ValueError('direct selection policy must be 0, 1, or auto')
    return policy=='1' or (policy=='auto' and os.environ.get('LOD_H100_FAST_PREFIX_CLOSED')=='1'
        and os.environ.get('LOD_H100_CLOSED_PRECISION')=='scaled-fp16')
