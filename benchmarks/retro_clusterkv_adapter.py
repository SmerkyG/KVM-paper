"""Use the released ClusterKV controller and kernels with RetroInfer's Llama."""
from types import MethodType

import torch

from clusterkv.clusterkv_utils import (
    ClusterKVController,
    append_kv,
    build_cluster,
    decode_sparse_attn,
    prefill_forward,
    update_sel_indices,
)


class RetroClusterKVCache:
    def __init__(self, model, *, clusters, budget, iterations, sink, window, window_clusters, full=False):
        if model.batch_size != 1 or model.num_gpus != 1 or model.dtype != torch.float16:
            raise ValueError('ClusterKV requires batch-one single-GPU FP16')
        capacity = model.input_length + model.max_new_length
        if not full and (not 0 < clusters <= 512 or not 1 < budget < model.input_length - sink
                         or clusters + (model.max_new_length - 1) // window * window_clusters > 512):
            raise ValueError('ClusterKV geometry exceeds its 512-centroid selector or prompt token budget')
        self.layer_count = model.num_layers
        self.context = 0
        self.update_count = 0
        self.device_decode = None
        self.controller = ClusterKVController(
            num_layers=model.num_layers, num_heads=model.num_heads,
            num_kv_heads=model.num_key_value_heads, head_dim=model.head_dim,
            nlist=clusters, niter=iterations, token_budget=budget,
            max_seq_len=capacity,
            dtype=model.dtype, device=torch.device(model.device_map),
            full=full, sink=sink, window=window, window_nlist=window_clusters, offload=False,
        )
        self.budget = budget

    def prefill_update_kv_cache(self, q, k, v, layer_idx, start_bdx):
        ctrl = self.controller
        if start_bdx != 0 or k.size(0) != 1 or self.context:
            raise ValueError('ClusterKV requires one unpadded prefill')
        if layer_idx == 0:
            ctrl.prepare_metadata(k.size(1))
            ctrl.begin_forward(k.size(1))
        if layer_idx >= 2 and not ctrl.full:
            keys = k[0, ctrl.sink:]
            stream = ctrl.build_cluster_stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                keys.record_stream(stream)
                build_cluster(ctrl, layer_idx, keys, 0, ctrl.nlist, stream)
                ctrl.build_cluster_events[layer_idx].record(stream)
        append_kv(k[0].contiguous(), v[0].contiguous(), ctrl, layer_idx)
        return k, v

    def prefill_attention(self, q, k, v, layer_idx):
        # The release uses causal dense attention for the prompt.
        return prefill_forward(q[0].contiguous(), self.controller, layer_idx).unsqueeze(0)

    def sync(self, layer_idx, start_bdx):
        if layer_idx == self.layer_count - 1:
            self.controller.end_forward()
            self.context = self.controller.kv_seqlen

    def decode_update_kv_cache(self, k, v, layer_idx):
        if self.device_decode is not None:
            if layer_idx == 0:
                self.device_decode.prepare()
            self.device_decode.append(k, v, layer_idx)
            return k, v
        ctrl = self.controller
        if layer_idx == 0:
            ctrl.prepare_metadata(1)
            ctrl.set_token_budget(ctrl._max_page_limit)
            ctrl.begin_forward(1)
        elif layer_idx == 2 and not ctrl.full:
            ctrl.end_forward()
            ctrl.set_token_budget(self.budget)
            ctrl.begin_forward(1, updateTensor=False)
        append_kv(k[0].contiguous(), v[0].contiguous(), ctrl, layer_idx)
        return k, v

    def decode_attention(self, q, layer_idx):
        if self.device_decode is not None:
            return self.device_decode.attention(q, layer_idx)
        ctrl = self.controller
        q = q[0].contiguous()
        indices = None
        if layer_idx >= 2 and not ctrl.full:
            if not ctrl.build_cluster_finish[layer_idx]:
                torch.cuda.current_stream().wait_event(ctrl.build_cluster_events[layer_idx])
                ctrl.build_cluster_finish[layer_idx] = True
            update_sel_indices(q, ctrl, layer_idx)
            indices = ctrl.sel_token_indices
        out = decode_sparse_attn(q, ctrl, layer_idx, indices)
        if indices is not None and ctrl.generated_len % ctrl.window == 0:
            # This step already attends to the completed window exactly. Publish
            # its clusters for the next step, when it leaves the exact window.
            keys = ctrl.get_app_k_clustering(layer_idx)
            build_cluster(ctrl, layer_idx, keys, ctrl.kv_seqlen - ctrl.sink - ctrl.window,
                          ctrl.window_nlist, torch.cuda.default_stream())
            if layer_idx == 2:
                self.update_count += 1
        if layer_idx == self.layer_count - 1:
            ctrl.end_forward()
            self.context = ctrl.kv_seqlen
        return out.unsqueeze(0)

    def init_device_decode(self, model, initial_token, *, capture):
        from benchmarks.retro_clusterkv_graph import ClusterKVDeviceDecode
        if self.device_decode is not None or self.context != self.controller.prompt_len:
            raise ValueError('initialize device decode once, immediately after prefill')
        torch.cuda.current_stream().wait_stream(self.controller.build_cluster_stream)
        device = self.device_decode = ClusterKVDeviceDecode(self.controller, self.budget)
        device.position.fill_(self.context)
        device.input.copy_(initial_token)
        if capture:
            # Warm the Triton kernels and CUDA libraries, then capture the same
            # step. Only the pending token's KV slots are written.
            model.sampling(model.decode_forward(inputs_ids=device.input))
            torch.cuda.synchronize()
            device.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(device.graph):
                device.logits = model.decode_forward(inputs_ids=device.input)
                device.output = model.sampling(device.logits)

    def device_step(self, model, token, *, check=False):
        device = self.device_decode
        device.position.fill_(self.context)
        stats = None
        if check:
            expected = model.decode_forward(inputs_ids=token)
            expected_token = model.sampling(expected)
        if device.graph is None:
            device.logits = model.decode_forward(inputs_ids=token)
            out = model.sampling(device.logits)
        else:
            device.input.copy_(token)
            device.graph.replay()
            out = device.output
        if check:
            difference = (expected.float() - device.logits.float()).abs()
            if not torch.isfinite(difference).all():
                raise RuntimeError('nonfinite ClusterKV graph/eager logit difference')
            stats = {'max_abs': float(difference.max()),
                     'graph_argmax': int(out.item()), 'eager_argmax': int(expected_token.item())}
            if stats['max_abs'] != 0 or stats['graph_argmax'] != stats['eager_argmax']:
                raise RuntimeError(f'ClusterKV graph/eager mismatch at context {self.context}: {stats}')
        self.context += 1
        if not self.controller.full and (self.context - device.prompt) % self.controller.window == 0:
            device.publish_window(self.context)
            self.update_count += 1
        return out, device.logits, stats


def attach_clusterkv(model, *, clusters=400, budget=1024, iterations=20,
                     sink=16, window=320, window_clusters=8, full=False):
    def init_cache(this, valid_start, config):
        if len(valid_start) != 1 or int(valid_start[0]) != 0:
            raise ValueError('ClusterKV requires an unpadded request')
        this.kv_cache = RetroClusterKVCache(
            this, clusters=clusters, budget=budget, iterations=iterations,
            sink=sink, window=window, window_clusters=window_clusters, full=full)

    def prefill(this, q, k, v, layer_idx):
        return this.kv_cache.prefill_attention(q, k, v, layer_idx)

    def decode(this, q, k, v, layer_idx):
        return this.kv_cache.decode_attention(q, layer_idx)

    def move(this):
        # Cache and index are already resident.
        torch.cuda.current_stream().wait_stream(this.kv_cache.controller.build_cluster_stream)

    original_position = model.position_embedd

    def position(this, q, k):
        device = this.kv_cache.device_decode
        return original_position(q, k) if device is None else this.apply_rotary_pos_emb(q, k, device.position)

    model.init_kv_cache = MethodType(init_cache, model)
    model.prefill_attention = MethodType(prefill, model)
    model.decode_attention = MethodType(decode, model)
    model.move = MethodType(move, model)
    model.position_embedd = MethodType(position, model)
