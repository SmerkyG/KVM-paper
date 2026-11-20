"""Released QUEST FP16 CUDA operators inside RetroInfer's batch-one Llama.

CUDA graphs capture 16 variants, one for each length of the current 16-token
page. A fixed physical page holds the in-progress page. At page boundaries,
its KV and author-maintained bounds are copied into the archive outside the
graph, with this work included in the decode timer. Author score, top-k,
KV append, and paged sparse attention kernels are all used unchanged apart
from the current-stream launch patch built by build_quest_author_graph.sh.
"""
import hashlib
from pathlib import Path
import types

import torch
import triton
import triton.language as tl
import quest.utils as quest
import quest._kernels as author
from flash_attn import flash_attn_with_kvcache


@triton.jit
def _mask_inactive(Scores, Count, N: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(Count)
    score = tl.load(Scores + h * N + i, mask=i < N, other=0)
    score = tl.where(i < count, score, -float('inf'))
    tl.store(Scores + h * N + i, score, mask=i < N)


class RetroAuthorQuestCache:
    def __init__(self, model, budget_tokens, shared_kv=False):
        if model.batch_size != 1 or model.num_gpus != 1 or model.dtype != torch.float16:
            raise ValueError('author QUEST comparison requires batch-one single-GPU FP16')
        if shared_kv:
            binary = Path(author.__file__)
            marker = Path(str(binary) + '.shared-kv.sha256')
            if not marker.exists() or marker.read_text().strip() != hashlib.sha256(binary.read_bytes()).hexdigest():
                raise RuntimeError('shared KV requires the matching patched author kernel build')
        if model.input_length % 16:
            raise ValueError('author graph comparison requires a page-aligned prompt')
        self.context = 0
        self.prompt_length = model.input_length
        self.layer_num = model.num_layers
        self.heads = model.num_heads
        self.kv_heads = model.num_key_value_heads
        self.dim = model.head_dim
        self.device = torch.device(model.device_map)
        self.group = self.heads // self.kv_heads
        self.shared_kv = shared_kv
        self.budget_pages = max(2, (budget_tokens + 15) // 16)
        # Extra archive pages align the fixed selector input and provide an
        # independent physical sentinel page for the current partial page.
        needed = (model.input_length + model.max_new_length + 15) // 16
        self.archive_capacity = ((needed + 15) // 16) * 16
        if self.budget_pages >= self.prompt_length // 16:
            raise ValueError('author graph path requires a sparse page budget')
        self.controller = quest.InferenceController(
            num_layers=self.layer_num, num_heads=self.kv_heads if shared_kv else self.heads, head_dim=self.dim,
            page_size=16, page_budget=self.budget_pages,
            max_seq_len=(self.archive_capacity + 16) * 16,
            dtype=torch.float16, device=self.device)
        self.graph_mode = False
        self.graphs = []
        self.phase = 1
        self.graph_position = torch.zeros((1, 1), dtype=torch.int64, device=self.device)
        self.lengths = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.archive_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.dense_k = [torch.empty((1, self.archive_capacity * 16, self.kv_heads, self.dim),
                                    dtype=torch.float16, device=self.device) for _ in range(2)]
        self.dense_v = [torch.empty_like(k) for k in self.dense_k]
        self.prefill_initialized = False

    def prefill_update_kv_cache(self, q, k, v, layer_idx, start_bdx):
        if start_bdx != 0 or k.shape[0] != 1:
            raise ValueError('author QUEST requires one unpadded request')
        if not self.prefill_initialized:
            self.controller.prepare_metadata(k.shape[1])
            self.controller.begin_forward(k.shape[1])
            self.prefill_initialized = True
        k32 = k[0].contiguous() if self.shared_kv else k[0].repeat_interleave(self.group, dim=1).contiguous()
        v32 = v[0].contiguous() if self.shared_kv else v[0].repeat_interleave(self.group, dim=1).contiguous()
        quest.append_kv(k32, v32, self.controller, layer_idx)
        if layer_idx < 2:
            self.dense_k[layer_idx][:, :k.shape[1]].copy_(k)
            self.dense_v[layer_idx][:, :v.shape[1]].copy_(v)
        if layer_idx == self.layer_num - 1:
            self.context += k.shape[1]
        return k, v

    def sync(self, layer_idx, start_bdx):
        pass

    def _initialize_decode(self):
        c = self.controller
        # Preserve the author's existing prefill page mapping; append physical
        # pages and metadata pages ahead of time to stabilize graph addresses.
        self.archive_ids = list(c.kv_cache.indicies)
        while len(self.archive_ids) < self.archive_capacity:
            self.archive_ids.append(c.kv_cache.pool.alloc_block())
        self.sentinel = c.kv_cache.pool.alloc_block()
        self.meta_ids = list(c.metadata_cache.indicies)
        required_meta = self.archive_capacity // 16
        while len(self.meta_ids) < required_meta:
            self.meta_ids.append(c.metadata_cache.pool.alloc_block())
        self.meta_sentinel = c.metadata_cache.pool.alloc_block()
        tensor = lambda data: torch.tensor(data, dtype=torch.int32, device=self.device)
        self.append_indices = tensor([self.sentinel])
        self.append_indptr = tensor([0, 1])
        self.meta_append_indices = tensor([self.meta_sentinel])
        self.meta_append_indptr = tensor([0, 1])
        self.score_indices = tensor(self.meta_ids + [self.meta_sentinel])
        self.score_indptr = tensor([0, len(self.meta_ids) + 1])
        self.candidate_indices = tensor(self.archive_ids).repeat(self.heads, 1)
        first_archived = self.context // 16
        self.archive_count.fill_(first_archived)
        self.scores = torch.empty((self.heads, self.archive_capacity),
                                  dtype=torch.float16, device=self.device)
        selected = self.budget_pages - 1
        self.top_values = torch.empty((self.heads, selected), dtype=torch.float16, device=self.device)
        self.top_indices = torch.empty((self.heads, selected), dtype=torch.int32, device=self.device)
        self.top_workspace = torch.zeros((self.heads, 8192 * 2 * (2 + 4) // 2 // 48),
                                         dtype=torch.float16, device=self.device)
        self.attn_indptr = tensor([0, selected])
        c._decode_handler.begin_forward(self.attn_indptr, self.heads, self.heads,
                                         self.dim, 16, torch.float16)

    def _archive_completed_page(self):
        # Called immediately before the next page's first token. No query can
        # observe the sentinel until its first append has reset its bounds.
        logical = self.context // 16 - 1
        physical = self.archive_ids[logical]
        c = self.controller
        c.kv_cache.pool.buf[:, physical].copy_(c.kv_cache.pool.buf[:, self.sentinel])
        meta_page = self.meta_ids[logical // 16]
        c.metadata_cache.pool.buf[:, meta_page, :, logical % 16].copy_(
            c.metadata_cache.pool.buf[:, self.meta_sentinel, :, 0])
        self.archive_count.fill_(logical + 1)

    def _set_position(self, previous):
        self.graph_position.fill_(previous)
        self.lengths.fill_(previous + 1)

    def decode_update_kv_cache(self, k, v, layer_idx):
        if layer_idx == self.layer_num - 1:
            self.context += 1
        return k, v

    def decode_attention(self, q, k, v, layer_idx):
        c = self.controller
        k32 = k.reshape(1, self.kv_heads, self.dim)
        v32 = v.reshape(1, self.kv_heads, self.dim)
        if not self.shared_kv:
            k32 = k32.repeat_interleave(self.group, dim=1)
            v32 = v32.repeat_interleave(self.group, dim=1)
        k32, v32 = k32.contiguous(), v32.contiguous()
        kv = c.kv_cache.buf_layer(layer_idx)
        meta = c.metadata_cache.buf_layer(layer_idx)
        author.append_kv_cache_decode(k32, v32, kv, self.append_indices, self.append_indptr,
            self.phase, self.sentinel, meta, self.meta_append_indices,
            self.meta_append_indptr, 1, self.meta_sentinel, c.layout)
        if layer_idx < 2:
            self.dense_k[layer_idx].index_copy_(1, self.graph_position.view(-1), k)
            self.dense_v[layer_idx].index_copy_(1, self.graph_position.view(-1), v)
            return flash_attn_with_kvcache(q, self.dense_k[layer_idx],
                self.dense_v[layer_idx], cache_seqlens=self.lengths)
        query = q.reshape(1, self.heads, self.dim).contiguous()
        author.estimate_attn_score(query, self.scores, meta, self.score_indices,
            self.score_indptr, 1, self.meta_sentinel, c.layout)
        _mask_inactive[(self.heads, triton.cdiv(self.archive_capacity, 128))](
            self.scores, self.archive_count, self.archive_capacity, 128, num_warps=4)
        author.topk_filtering(self.scores, self.candidate_indices, self.top_values,
            self.top_indices, self.top_workspace, self.budget_pages - 1)
        out = torch.empty_like(query)
        c._decode_handler.forward(query, out, kv, self.top_indices, self.attn_indptr,
                                  self.phase, self.sentinel)
        return out.reshape_as(q)

    def capture_decode(self, model, initial_token):
        self._initialize_decode()
        previous = self.context
        self.graph_mode = True
        self.graph_input = initial_token.clone()
        self._set_position(previous)
        self.phase = 1
        model.sampling(model.decode_forward(inputs_ids=self.graph_input))
        self.context = previous
        torch.cuda.synchronize()
        self.graph_logits_variants = []
        self.graph_outputs = []
        pool = None
        for phase in range(1, 17):
            self.phase = phase
            self._set_position(previous)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                logits = model.decode_forward(inputs_ids=self.graph_input)
                output = model.sampling(logits)
            if pool is None:
                pool = graph.pool()
            self.graphs.append(graph)
            self.graph_logits_variants.append(logits)
            self.graph_outputs.append(output)
            self.context = previous
        self.phase = 1

    def graph_step(self, token):
        if self.context > self.prompt_length and self.context % 16 == 0:
            self._archive_completed_page()
        self._set_position(self.context)
        self.phase = self.context % 16 + 1
        self.graph_input.copy_(token)
        self.graphs[self.phase - 1].replay()
        self.context += 1
        self.graph_logits = self.graph_logits_variants[self.phase - 1]
        return self.graph_outputs[self.phase - 1]

    def checked_graph_step(self, model, token):
        previous = self.context
        graph_token = self.graph_step(token)
        graph_logits = self.graph_logits.clone()
        self.context = previous
        self._set_position(previous)
        eager_logits = model.decode_forward(inputs_ids=token)
        difference = (graph_logits - eager_logits).abs()
        return graph_token, dict(max_abs=float(difference.max()),
            mean_abs=float(difference.mean()), graph_argmax=int(graph_logits.argmax()),
            eager_argmax=int(eager_logits.argmax()))


def attach_author_quest(model, budget_tokens, shared_kv=False):
    original_position = model.position_embedd

    def init_cache(this, valid_start, config):
        if len(valid_start) != 1 or int(valid_start[0]) != 0:
            raise ValueError('author QUEST requires one unpadded request')
        this.kv_cache = None
        this.kv_cache = RetroAuthorQuestCache(this, budget_tokens, shared_kv=shared_kv)

    def decode(this, q, k, v, layer_idx):
        return this.kv_cache.decode_attention(q, k, v, layer_idx)

    def move(this):
        torch.cuda.empty_cache()

    def position(this, q, k):
        if this.kv_cache.graph_mode:
            return this.apply_rotary_pos_emb(q, k, this.kv_cache.graph_position)
        return original_position(q, k)

    model.init_kv_cache = types.MethodType(init_cache, model)
    model.decode_attention = types.MethodType(decode, model)
    model.move = types.MethodType(move, model)
    model.position_embedd = types.MethodType(position, model)
