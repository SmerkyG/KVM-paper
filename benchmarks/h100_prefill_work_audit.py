"""Diagnostic full-prefill work counts, without count readbacks during prefill.

Counts and CUDA events are collected only after timed model execution. The
instrumented run is not a replacement for uninstrumented latency measurements.
"""
import types
import os
import torch


def timed(engine,label,fn,*args,**kwargs):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();out=fn(*args,**kwargs);end.record()
    engine._lod_scaling_events.setdefault(label,[]).append((start,end))
    return out


def record(engine,q,slots,lengths,state_len,counts=None,entries=None):
    h,t,width=map(int,slots.shape[1:]);g=engine.num_key_value_groups
    if counts is None:
        ids=slots[0].long();valid=(ids>=0)&(ids<state_len)
        sizes=lengths[0,:,:state_len].repeat_interleave(g,0)
        gathered=sizes[:,None,:].expand(-1,t,-1).gather(2,torch.where(valid,ids,0))
        counts=valid.sum(-1);entries=torch.where(valid,gathered,0).sum(-1)
    stats=torch.stack((counts.sum(dtype=torch.int64),entries.sum(dtype=torch.int64),
                       counts.max().to(torch.int64),entries.max().to(torch.int64)))
    row=dict(query_heads=h*t,queries=t,regions=state_len,route_capacity=width,stats=stats)
    engine._lod_scaling_rows.append(row)
    return row


def workspace_bytes(buffers):
    return {name:t.numel()*t.element_size() for name,t in buffers.items()
            if isinstance(t,torch.Tensor)}


def install(model):
    from benchmarks import h100_threshold_prefill_closed as closed
    original_exact=closed.exact_attention
    active_update=[None]
    if os.environ.get('LOD_H100_FULL_PREFILL_AUDIT')=='1':
        import lod_attention._core as core
        for name in ('merge_state_in_place','streaming_state_maxsim','tiled_dot_maxsim','prepare_state_clustering_keys','_gather_by_idx','state_score_max'):
            original=getattr(core,name)
            def component(*aa,_original=original,_name=name,**kw):
                engine=active_update[0]
                if engine is None:return _original(*aa,**kw)
                return timed(engine,'state_'+_name,_original,*aa,**kw)
            setattr(core,name,component)

    def exact(engine,q,page,state_len,routed,buffers,**kwargs):
        row=record(engine,q,routed[0],page['slot_lengths'],state_len,
                   routed[1].view(q.size(1),q.size(2)),routed[2].view(q.size(1),q.size(2)))
        out=timed(engine,'exact_total',original_exact,engine,q,page,state_len,routed,buffers,**kwargs)
        row['workspace_bytes']=workspace_bytes(buffers)
        return out
    closed.exact_attention=exact
    original_init=model.init_kv_cache

    def init(this,*args,**kwargs):
        original_init(*args,**kwargs)
        for pool in this.kv_cache.pools:
            engine=pool.engine
            engine._lod_scaling_rows=[];engine._lod_scaling_events={}
            engine._lod_leaf_timing_events={}
            if os.environ.get('LOD_H100_FULL_PREFILL_AUDIT')=='1':
                for method,label,token_arg in (('_update_state','state_update',4),
                        ('_append_page_cache','page_cache_append',1),
                        ('_prefill_local_attention','local_or_initial_exact',0),
                        ('_split_append_merge_indices','state_split_indices',0),
                        ('_state_clustering_key','state_clustering_key',0),
                        ('_state_clustering_similarity','state_clustering_similarity',0)):
                    original=getattr(engine,method)
                    def auxiliary(self,*aa,_original=original,_label=label,_token=token_arg,**kw):
                        if len(aa)<=_token or not isinstance(aa[_token],torch.Tensor)or aa[_token].size(2)<=1:
                            return _original(*aa,**kw)
                        previous=active_update[0]
                        if _label=='state_update':active_update[0]=self
                        try:return timed(self,_label,_original,*aa,**kw)
                        finally:active_update[0]=previous
                    setattr(engine,method,types.MethodType(auxiliary,engine))
            original_route=engine._route_top_slots
            original_leaf=engine._paged_leaf_attention

            def route(self,q,*aa,_original=original_route,**kw):
                if q.size(2)<=1:return _original(q,*aa,**kw)
                return timed(self,'producer_coarse_selection',_original,q,*aa,**kw)

            def leaf(self,q,slots,page,*,active_slots,_original=original_leaf,**kw):
                if q.size(2)<=1:return _original(q,slots,page,active_slots=active_slots,**kw)
                row=record(self,q,slots,page['slot_lengths'],active_slots)
                out=timed(self,'exact_total',_original,q,slots,page,active_slots=active_slots,**kw)
                row['workspace_bytes']=workspace_bytes(getattr(self,'_lod_prefill_attention_buffers',{}))
                return out
            engine._route_top_slots=types.MethodType(route,engine)
            engine._paged_leaf_attention=types.MethodType(leaf,engine)
    model.init_kv_cache=types.MethodType(init,model)


def collect(cache):
    layers=[]
    for index,pool in enumerate(cache.pools):
        engine=pool.engine;rows=[]
        for saved in engine._lod_scaling_rows:
            row={k:v for k,v in saved.items() if k!='stats'}
            vals=saved['stats'].cpu().tolist()
            row.update(zip(('opened_regions','opened_entries','maximum_regions','maximum_entries'),vals))
            rows.append(row)
        events={k:sum(a.elapsed_time(b) for a,b in pairs)/1000 for k,pairs in engine._lod_scaling_events.items()}
        layers.append(dict(layer=index,rows=rows,gpu_seconds=events))
    return dict(scope='all remote prefill query heads; exact initial field and local/sink excluded',
                auxiliary_scope='optional state_update, page_cache_append, local_or_initial_exact timings are separate from remote work counts',
                instrumented=True,layers=layers)
