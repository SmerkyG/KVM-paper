"""Model-agnostic, pure-PyTorch levels-of-detail attention.

The public modules in this file consume query, key, and value tensors *after*
the model has applied its QKV projections, normalizers, and positional
encoding.  They therefore contain no Hugging Face or model-specific code.

``CoarseLODAttention`` keeps only count-corrected state summaries plus an exact
local window.  ``TwoLevelLODAttention`` additionally archives the original KV
leaves (BF16 by default), opens up to eight routed state regions, computes one
exact attention and log-sum-exp per opened region, and LSE-merges those regions
with the coarse remainder and local window.

Tensor layout follows current Hugging Face attention implementations:

    query: [batch, query_heads, query_length, key_head_dim]
    key:   [batch, key_value_heads, key_length, key_head_dim]
    value: [batch, key_value_heads, key_length, value_head_dim]

GQA is supported when ``query_heads`` is divisible by ``key_value_heads``.
Inputs are assumed to be unpadded causal sequences.  The returned output is
still head-separated; the caller remains responsible for transposing,
flattening, gating, and applying the output projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LODConfig:
    """Chunk, state-growth, and routing settings for LOD attention.

    Protected prefix slots remain exact in the coarse field and are excluded
    from both state merging and detailed-region routing.
    """

    chunk_size: int = 256
    local_window: int = 512
    state_growth_factor: float = 16.0
    state_min_size: int = 256
    protected_prefix: int = 1
    max_routes: int = 8
    leaf_dtype: torch.dtype = torch.bfloat16

    # Keep all extension fields after the original positional constructor
    # surface so older callers using positional LODConfig arguments retain
    # their meaning. New code should continue to pass these options by name.
    state_size_offset: int = 0
    state_premerge_factor: int = 1
    # Experimental overcomplete state: keep every posting list at or below
    # this many exact leaves by routing overflow assignments into additional
    # affinity-stratified child centroids. Scheduled centroid appends continue
    # independently at the ordinary state-growth rate.
    state_split_max_leaves: int | None = None
    state_clustering_policy: str = "manual"
    state_clustering_normalization: str = "none"
    state_clustering_radial_bias: float = 0.0
    state_clustering_radial_scope: str = "all"
    state_clustering_centroid_rescale: str = "none"
    state_clustering_centroid_rescale_scope: str = "all"
    state_clustering_query_metric: str = "none"
    state_clustering_rope_filter: str = "none"
    state_clustering_rope_dim: int = 0
    state_clustering_rope_fast_pairs: int = 0
    coherence_single_matmul: bool = True
    routing_normalization: str = "none"
    routing_rope_filter: str = "none"
    routing_rope_cutoff_factor: float = 1.0
    routing_rope_dim: int = 0
    routing_rope_fast_pairs: int = 0
    routing_rope_jensen_pairs: int = 0
    routing_rope_jensen: bool = False
    routing_count_bias: float = 1.0
    routing_variance_bias: float = 0.0
    routing_page_mass_candidates: int = 0
    routing_leaf_mass_candidates: int = 0
    routing_leaf_mass_objective: str = "exact"
    routing_leaf_mass_review_top_p: float | None = None
    routing_leaf_mass_top_p: float | None = None
    routing_leaf_mass_min_routes: int = 1
    mla_state_key_normalization: str = "none"
    mla_recursive_page_key_normalization: bool = False
    # Flat kernel-backend archive controls. Sealing stops exact-leaf growth
    # without freezing the centroid sum/count. Native INT8 uses signed K/V
    # pages and real INT8 MMA in the expert-layout prefill kernels.
    leaf_paged_directory: bool = True
    leaf_seal_capacity: int | None = None
    prefill_int8_leaf_mma: bool = False
    prefill_int8_coarse_mma: bool = False
    prefill_int8_pv_mma: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.local_window < 2 * self.chunk_size:
            raise ValueError("local_window must contain at least two chunks")
        if self.local_window % self.chunk_size:
            raise ValueError("local_window must be a multiple of chunk_size")
        if self.state_growth_factor < 0:
            raise ValueError("state_growth_factor cannot be negative")
        if self.state_min_size < 0:
            raise ValueError("state_min_size cannot be negative")
        if self.state_size_offset < 0:
            raise ValueError("state_size_offset cannot be negative")
        if self.state_premerge_factor not in {1, 2, 4}:
            raise ValueError("state_premerge_factor must be one, two, or four")
        if (
            self.state_split_max_leaves is not None
            and self.state_split_max_leaves <= 0
        ):
            raise ValueError("state_split_max_leaves must be positive")
        if self.state_split_max_leaves is not None and self.state_premerge_factor != 1:
            raise ValueError(
                "state posting-list splitting requires unmerged token leaves"
            )
        if self.protected_prefix < 0:
            raise ValueError("protected_prefix cannot be negative")
        if self.leaf_seal_capacity is not None and self.leaf_seal_capacity <= 0:
            raise ValueError("leaf_seal_capacity must be positive")
        if self.state_clustering_policy not in {
            "manual",
            "qk_norm_aware",
            "rope_aware",
            "rnope_nope_spherical",
            "rnope_rope_spherical",
        }:
            raise ValueError(
                "state_clustering_policy must be manual, qk_norm_aware, "
                "rope_aware, rnope_nope_spherical, or "
                "rnope_rope_spherical"
            )
        if self.state_clustering_policy != "manual" and (
            self.state_clustering_normalization != "none"
            or self.state_clustering_radial_bias
            or self.state_clustering_centroid_rescale != "none"
            or self.state_clustering_query_metric != "none"
            or self.state_clustering_rope_filter != "none"
        ):
            raise ValueError(
                "automatic state clustering requires the manual geometry "
                "controls to remain disabled"
            )
        if self.state_clustering_normalization not in {
            "none",
            "leaf_cosine",
            "centroid_cosine",
            "cosine",
            "l2",
        }:
            raise ValueError(
                "state_clustering_normalization must be none, leaf_cosine, "
                "centroid_cosine, cosine, or l2"
            )
        if self.state_clustering_radial_bias < 0:
            raise ValueError("state_clustering_radial_bias cannot be negative")
        if (
            self.state_clustering_radial_bias
            and self.state_clustering_normalization != "cosine"
        ):
            raise ValueError(
                "state_clustering_radial_bias requires cosine state clustering"
            )
        if self.state_clustering_radial_scope not in {
            "all",
            "append",
            "assignment",
        }:
            raise ValueError(
                "state_clustering_radial_scope must be all, append, or assignment"
            )
        if self.state_clustering_centroid_rescale not in {
            "none",
            "mean_leaf_norm",
            "coherence",
            "spherical_coherence",
            "rope_coherence",
            "direction_l2",
        }:
            raise ValueError(
                "state_clustering_centroid_rescale must be none, "
                "mean_leaf_norm, coherence, spherical_coherence, "
                "rope_coherence, or direction_l2"
            )
        if self.state_clustering_centroid_rescale_scope not in {
            "all",
            "append",
            "assignment",
        }:
            raise ValueError(
                "state_clustering_centroid_rescale_scope must be all, append, "
                "or assignment"
            )
        if (
            self.state_clustering_centroid_rescale != "none"
            and self.state_clustering_normalization != "none"
        ):
            raise ValueError(
                "centroid rescaling requires raw leaf clustering"
            )
        if (
            self.state_clustering_centroid_rescale != "none"
            and (
                self.state_clustering_radial_bias
                or self.state_clustering_query_metric != "none"
                or self.state_clustering_rope_filter != "none"
            )
        ):
            raise ValueError(
                "centroid rescaling requires unmodified key geometry"
            )
        if (
            self.state_clustering_centroid_rescale == "direction_l2"
            and self.state_clustering_centroid_rescale_scope != "all"
        ):
            raise ValueError("direction_l2 clustering requires all-purpose scope")
        if self.state_clustering_query_metric not in {
            "none",
            "diagonal",
            "full",
        }:
            raise ValueError(
                "state_clustering_query_metric must be none, diagonal, or full"
            )
        if (
            self.state_clustering_query_metric == "full"
            and self.state_clustering_rope_fast_pairs
        ):
            raise ValueError(
                "full query-metric clustering cannot be combined with a RoPE filter"
            )
        if self.state_clustering_rope_filter not in {"none", "local_window"}:
            raise ValueError(
                "state_clustering_rope_filter must be none or local_window"
            )
        if self.state_clustering_rope_dim < 0 or self.state_clustering_rope_dim % 2:
            raise ValueError(
                "state_clustering_rope_dim must be a nonnegative even integer"
            )
        if not 0 <= self.state_clustering_rope_fast_pairs <= (
            self.state_clustering_rope_dim // 2
        ):
            raise ValueError(
                "state_clustering_rope_fast_pairs exceeds the rotary geometry"
            )
        if self.routing_normalization not in {
            "none",
            "query",
            "key",
            "both",
            "qk_norm_aware",
        }:
            raise ValueError(
                "routing_normalization must be none, query, key, both, or "
                "qk_norm_aware"
            )
        if self.routing_rope_filter not in {"none", "local_window"}:
            raise ValueError("routing_rope_filter must be none or local_window")
        if (
            not math.isfinite(self.routing_rope_cutoff_factor)
            or self.routing_rope_cutoff_factor <= 0
        ):
            raise ValueError("routing_rope_cutoff_factor must be finite and positive")
        if self.routing_rope_dim < 0 or self.routing_rope_dim % 2:
            raise ValueError("routing_rope_dim must be a nonnegative even integer")
        if not 0 <= self.routing_rope_fast_pairs <= self.routing_rope_dim // 2:
            raise ValueError("routing_rope_fast_pairs exceeds the rotary geometry")
        if not 0 <= self.routing_rope_jensen_pairs <= self.routing_rope_dim // 2:
            raise ValueError(
                "routing_rope_jensen_pairs exceeds the rotary geometry"
            )
        if not math.isfinite(self.routing_count_bias) or self.routing_count_bias < 0:
            raise ValueError("routing_count_bias must be finite and nonnegative")
        if (
            not math.isfinite(self.routing_variance_bias)
            or self.routing_variance_bias < 0
        ):
            raise ValueError("routing_variance_bias must be finite and nonnegative")
        if self.routing_variance_bias and self.routing_normalization != "none":
            raise ValueError(
                "routing_variance_bias currently requires unnormalized routing"
            )
        if self.routing_variance_bias and self.routing_rope_fast_pairs:
            raise ValueError(
                "routing_variance_bias is not calibrated for filtered RoPE routing"
            )
        if self.routing_rope_jensen and self.routing_variance_bias:
            raise ValueError("RoPE-pair and scalar variance corrections conflict")
        if self.routing_rope_jensen and self.routing_rope_fast_pairs:
            raise ValueError("RoPE-pair correction cannot be combined with filtering")
        if self.routing_page_mass_candidates not in {0, 16, 32, 64, 128}:
            raise ValueError(
                "routing_page_mass_candidates must be 0, 16, 32, 64, or 128"
            )
        if self.routing_leaf_mass_candidates not in {0, 16, 32, 64, 128}:
            raise ValueError(
                "routing_leaf_mass_candidates must be 0, 16, 32, 64, or 128"
            )
        if self.routing_page_mass_candidates and self.routing_leaf_mass_candidates:
            raise ValueError("page-mass and leaf-mass routing are mutually exclusive")
        if (
            self.routing_leaf_mass_candidates > 32
            and self.routing_leaf_mass_objective == "output"
        ):
            raise ValueError(
                "output-error leaf routing supports at most 32 candidates"
            )
        if self.routing_leaf_mass_objective not in {
            "exact",
            "additional",
            "deficit",
            "output",
            "rope_jensen",
            "fast_rope_jensen",
            "slow_rope_jensen",
        }:
            raise ValueError(
                "routing_leaf_mass_objective must be exact, additional, deficit, "
                "output, rope_jensen, fast_rope_jensen, or slow_rope_jensen"
            )
        if self.routing_leaf_mass_review_top_p is not None:
            if not 0.0 < self.routing_leaf_mass_review_top_p <= 1.0:
                raise ValueError(
                    "routing_leaf_mass_review_top_p must lie in (0, 1]"
                )
            if not self.routing_leaf_mass_candidates:
                raise ValueError(
                    "routing_leaf_mass_review_top_p requires leaf-mass candidates"
                )
            if self.max_routes == 0:
                raise ValueError(
                    "routing_leaf_mass_review_top_p requires at least one route"
                )
        if self.routing_leaf_mass_top_p is not None:
            if not 0.0 < self.routing_leaf_mass_top_p <= 1.0:
                raise ValueError("routing_leaf_mass_top_p must lie in (0, 1]")
            if not self.routing_leaf_mass_candidates:
                raise ValueError(
                    "routing_leaf_mass_top_p requires leaf-mass candidates"
                )
            if self.max_routes == 0:
                raise ValueError(
                    "routing_leaf_mass_top_p requires at least one route"
                )
            if not 1 <= self.routing_leaf_mass_min_routes <= self.max_routes:
                raise ValueError(
                    "routing_leaf_mass_min_routes must be between one and max_routes"
                )
            if self.routing_leaf_mass_objective not in {
                "exact",
                "rope_jensen",
                "fast_rope_jensen",
                "slow_rope_jensen",
            }:
                raise ValueError(
                    "routing_leaf_mass_top_p requires an attention-mass objective"
                )
        elif self.routing_leaf_mass_min_routes != 1:
            raise ValueError(
                "routing_leaf_mass_min_routes requires routing_leaf_mass_top_p"
            )
        if not 0 <= self.max_routes <= 128:
            raise ValueError("max_routes must be between zero and 128")
        if self.mla_state_key_normalization not in {
            "none",
            "latent",
            "whole",
            "raw",
        }:
            raise ValueError(
                "MLA state-key normalization must be none, latent, whole, or raw"
            )
        if (
            self.mla_recursive_page_key_normalization
            and self.mla_state_key_normalization != "latent"
        ):
            raise ValueError(
                "recursive MLA page normalization requires latent state normalization"
            )
        if (
            self.routing_leaf_mass_candidates
            and self.max_routes > self.routing_leaf_mass_candidates
        ):
            raise ValueError("max_routes cannot exceed leaf-mass candidates")


@dataclass
class LODState:
    """Low-LOD state. Keys and values are sums; counts recover their means.

    Chunk-aligned left padding uses a negative count only on an inert owner
    slot. Its magnitude records the number of padding leaves for fixed-shape
    posting lists, while attention and state updates mask all nonpositive
    counts.
    """

    key_sum: torch.Tensor
    value_sum: torch.Tensor
    count: torch.Tensor

    @property
    def slot_count(self) -> int:
        return int(self.key_sum.size(2))

    @property
    def mean_key(self) -> torch.Tensor:
        return self.key_sum / self.count.to(self.key_sum.dtype).clamp_min(1).unsqueeze(-1)

    @property
    def mean_value(self) -> torch.Tensor:
        return self.value_sum / self.count.to(self.value_sum.dtype).clamp_min(1).unsqueeze(-1)

    def detached(self) -> LODState:
        return LODState(
            key_sum=self.key_sum.detach(),
            value_sum=self.value_sum.detach(),
            count=self.count.detach(),
        )


@dataclass
class LODCache:
    """Explicit cache returned to an HF attention wrapper for later decode."""

    state: LODState
    coverage: int
    recent_key: torch.Tensor
    recent_value: torch.Tensor
    total_length: int
    owner: torch.Tensor | None = None
    leaf_key: torch.Tensor | None = None
    leaf_value: torch.Tensor | None = None

    def detached(self) -> LODCache:
        return LODCache(
            state=self.state.detached(),
            coverage=self.coverage,
            recent_key=self.recent_key.detach(),
            recent_value=self.recent_value.detach(),
            total_length=self.total_length,
            owner=None if self.owner is None else self.owner.detach(),
            leaf_key=None if self.leaf_key is None else self.leaf_key.detach(),
            leaf_value=None if self.leaf_value is None else self.leaf_value.detach(),
        )


@dataclass
class LODAttentionResult:
    """Attention result and optional two-level routing diagnostics."""

    output: torch.Tensor
    logsumexp: torch.Tensor
    top_slots: torch.Tensor | None = None
    open_mask: torch.Tensor | None = None


def _validate_qkv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must all be rank-four tensors")
    if query.size(0) != key.size(0) or key.shape[:3] != value.shape[:3]:
        raise ValueError("query, key, and value batch/sequence shapes disagree")
    if query.size(-1) != key.size(-1):
        raise ValueError("query and key head dimensions must match")
    query_heads = int(query.size(1))
    key_value_heads = int(key.size(1))
    if key_value_heads <= 0 or query_heads % key_value_heads:
        raise ValueError(
            "query head count must be divisible by key/value head count"
        )
    return query_heads, key_value_heads


def _repeat_kv(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    key_value_heads = int(tensor.size(1))
    if query_heads % key_value_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    groups = query_heads // key_value_heads
    return tensor if groups == 1 else tensor.repeat_interleave(groups, dim=1)


def _empty_attention(
    query: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = value.new_zeros(*query.shape[:-1], int(value.size(-1)))
    lse = torch.full(
        query.shape[:-1],
        float("-inf"),
        device=query.device,
        dtype=torch.float32,
    )
    return output, lse


def _attention_from_scores(
    scores: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized attention and FP32 LSE, including all-masked rows."""
    if int(scores.size(-1)) == 0:
        return _empty_attention(scores, value)
    valid = torch.isfinite(scores).any(dim=-1)
    safe_scores = torch.where(valid.unsqueeze(-1), scores, torch.zeros_like(scores))
    probability = torch.softmax(safe_scores, dim=-1)
    probability = torch.where(
        valid.unsqueeze(-1), probability, torch.zeros_like(probability)
    )
    output = torch.matmul(probability.to(value.dtype), value)
    lse = torch.logsumexp(scores, dim=-1)
    return output, lse


def _scaled_scores(
    query: torch.Tensor, key: torch.Tensor, scale: float
) -> torch.Tensor:
    # FP32 score/LSE math makes this file a stable reference implementation;
    # values and returned outputs retain their original dtype.
    return torch.matmul(
        query.float(), key.float().transpose(-1, -2)
    ) * float(scale)


def _routing_rms_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Remove vector-norm temperature while preserving the usual sqrt(d) scale."""
    inverse_rms = torch.rsqrt(
        tensor.detach().float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
    )
    return tensor.detach().float() * inverse_rms


def _routing_query_key(
    query: torch.Tensor,
    key: torch.Tensor,
    normalization: str,
    rope_dim: int = 0,
    rope_fast_pairs: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Query-only ranking is exactly equivalent to leaving q unchanged and
    # multiplying log(slot_count) by RMS(q). It removes the pretrained
    # attention temperature from lossy-centroid visibility selection without
    # changing the closed or opened attention calculation.
    if normalization == "qk_norm_aware":
        raise ValueError(
            "qk_norm_aware routing must be resolved from the attention module "
            "by the Hugging Face installer"
        )
    if normalization not in {"none", "query", "key", "both"}:
        raise ValueError("routing normalization must be none, query, key, or both")
    original_query_rms = None
    if rope_fast_pairs:
        if rope_dim > int(query.size(-1)) or rope_dim > int(key.size(-1)):
            raise ValueError("routing RoPE dimension exceeds the attention head")
        if normalization not in {"query", "both"}:
            original_query_rms = (
                query.detach().float().square().mean(-1, keepdim=True).sqrt()
            )
        half = rope_dim // 2
        query = query.detach().clone()
        key = key.detach().clone()
        query[..., :rope_fast_pairs] = 0
        query[..., half : half + rope_fast_pairs] = 0
        key[..., :rope_fast_pairs] = 0
        key[..., half : half + rope_fast_pairs] = 0
    if normalization in {"query", "both"}:
        query = _routing_rms_normalize(query)
    elif original_query_rms is not None:
        # Removing dimensions must not also retune the centroid-score
        # temperature relative to log(slot_count). Preserve the query's
        # pre-mask RMS while changing only its routing direction.
        query = _routing_rms_normalize(query) * original_query_rms
    if normalization in {"key", "both"}:
        key = _routing_rms_normalize(key)
    return query, key


def _routing_state_scores(
    query: torch.Tensor,
    state: LODState,
    scale: float,
    normalization: str,
    count_bias: float = 1.0,
    reference_key: torch.Tensor | None = None,
    variance_bias: float = 0.0,
    rope_dim: int = 0,
    rope_fast_pairs: int = 0,
    rope_jensen: bool = False,
) -> torch.Tensor:
    query_heads = int(query.size(1))
    mean_key = _repeat_kv(state.mean_key, query_heads)
    count = _repeat_kv(state.count, query_heads)
    route_query, route_key = _routing_query_key(
        query,
        mean_key,
        normalization,
        rope_dim,
        rope_fast_pairs,
    )
    scores = _scaled_scores(route_query, route_key, scale)
    scores = scores + (
        count.clamp_min(1).log().float().unsqueeze(2) * float(count_bias)
    )
    if variance_bias:
        if reference_key is None or int(reference_key.size(2)) == 0:
            raise ValueError("routing variance correction requires reference keys")
        # A second-order approximation to the omitted within-slot log-mass:
        #   log mean_i exp(scale q.(k_i - mean_k)) ~= variance(score) / 2.
        # Estimate the constituent-key second moment from the exact local
        # field.  The centroid norm deficit then estimates the trace of the
        # within-slot covariance without adding anything to persistent state.
        reference_sq = reference_key.detach().float().square().sum(-1).mean(-1)
        mean_sq = state.mean_key.detach().float().square().sum(-1)
        variance_trace = (reference_sq.unsqueeze(-1) - mean_sq).clamp_min(0)
        variance_trace = variance_trace.masked_fill(state.count.le(1), 0)
        variance_trace = _repeat_kv(variance_trace, query_heads)
        query_sq = route_query.detach().float().square().sum(-1)
        correction = (
            0.5
            * float(variance_bias)
            * float(scale) ** 2
            * query_sq.unsqueeze(-1)
            * variance_trace.unsqueeze(2)
            / float(query.size(-1))
        )
        scores = scores + correction
    if rope_jensen and rope_dim:
        if normalization in {"key", "both"}:
            raise ValueError("RoPE Jensen routing does not support key normalization")
        if reference_key is None or int(reference_key.size(2)) == 0:
            raise ValueError("RoPE Jensen routing requires reference keys")
        half = rope_dim // 2
        query_pairs = (
            route_query[..., :half].detach().float().square()
            + route_query[..., half:rope_dim].detach().float().square()
        )
        reference_pairs = (
            reference_key[..., :half].detach().float().square()
            + reference_key[..., half:rope_dim].detach().float().square()
        ).mean(-2)
        mean_pairs = (
            state.mean_key[..., :half].detach().float().square()
            + state.mean_key[..., half:rope_dim].detach().float().square()
        )
        pair_variance = (reference_pairs.unsqueeze(-2) - mean_pairs).clamp_min(0)
        pair_variance.masked_fill_(state.count.unsqueeze(-1).le(1), 0)
        pair_variance = _repeat_kv(pair_variance, query_heads)
        # Within each rotary plane, use an isotropic covariance whose trace is
        # the observed pair-energy deficit. The second-order Jensen term is
        # 0.5 Var(score), hence the 0.25 factor below.
        correction = 0.25 * float(scale) ** 2 * torch.matmul(
            query_pairs, pair_variance.transpose(-1, -2)
        )
        scores = scores + correction
    scores.masked_fill_(count.le(0.5).unsqueeze(2), -torch.inf)
    return scores


def _local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    query_offset: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_heads, _ = _validate_qkv(query, key, value)
    key = _repeat_kv(key, query_heads)
    value = _repeat_kv(value, query_heads)
    query_length = int(query.size(2))
    key_length = int(key.size(2))
    if query_offset is None:
        query_offset = key_length - query_length
    if query_offset < 0 or query_offset + query_length > key_length:
        raise ValueError("query_offset does not place queries inside local keys")
    scores = _scaled_scores(query, key, scale)
    query_index = torch.arange(query_length, device=query.device).unsqueeze(-1)
    key_index = torch.arange(key_length, device=query.device).unsqueeze(0)
    visible = key_index <= query_index + query_offset
    scores = scores.masked_fill(~visible.view(1, 1, query_length, key_length), -torch.inf)
    return _attention_from_scores(scores, value)


def _masked_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    valid_starts: torch.Tensor,
) -> torch.Tensor:
    """Causal exact attention with one contiguous left-pad prefix per row."""
    batch, _, query_length, _ = query.shape
    key_length = int(key.size(2))
    valid_starts = valid_starts.to(device=query.device, dtype=torch.long)
    if tuple(valid_starts.shape) != (batch,):
        raise ValueError("prefill valid starts must have one entry per row")
    if bool(
        (valid_starts.lt(0) | valid_starts.ge(key_length)).any().item()
    ):
        raise ValueError("every prefill row must contain at least one valid token")
    query_position = torch.arange(query_length, device=query.device)
    key_position = torch.arange(key_length, device=query.device)
    visible = key_position.view(1, 1, 1, key_length) >= valid_starts.view(
        -1, 1, 1, 1
    )
    visible = visible & (
        key_position.view(1, 1, 1, key_length)
        <= query_position.view(1, 1, query_length, 1)
    )
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=visible,
        enable_gqa=int(query.size(1)) != int(key.size(1)),
        scale=scale,
    )
    query_valid = query_position.view(1, 1, query_length, 1) >= valid_starts.view(
        -1, 1, 1, 1
    )
    return torch.where(query_valid, output, torch.zeros_like(output))


def _state_scores_and_value(
    query: torch.Tensor, state: LODState, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.slot_count == 0:
        value = state.value_sum.new_empty(
            int(query.size(0)), int(query.size(1)), 0, int(state.value_sum.size(-1))
        )
        scores = torch.empty(
            *query.shape[:-1], 0, device=query.device, dtype=torch.float32
        )
        return scores, value
    query_heads = int(query.size(1))
    mean_key = _repeat_kv(state.mean_key, query_heads)
    mean_value = _repeat_kv(state.mean_value, query_heads)
    count = _repeat_kv(state.count, query_heads)
    scores = _scaled_scores(query, mean_key, scale)
    scores = scores + count.clamp_min(1).log().float().unsqueeze(2)
    scores.masked_fill_(count.le(0.5).unsqueeze(2), -torch.inf)
    return scores, mean_value


def _state_attention(
    query: torch.Tensor,
    state: LODState,
    *,
    scale: float,
    excluded_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores, value = _state_scores_and_value(query, state, scale)
    if excluded_slots is not None:
        if excluded_slots.shape != scores.shape:
            raise ValueError("excluded state-slot mask has the wrong shape")
        scores = scores.masked_fill(excluded_slots, -torch.inf)
    return _attention_from_scores(scores, value)


def _merge_lse_branches(
    outputs: list[torch.Tensor], lses: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    if not outputs or len(outputs) != len(lses):
        raise ValueError("outputs and LSEs must contain the same nonzero branches")
    branch_lse = torch.stack([lse.float() for lse in lses], dim=-1)
    total_lse = torch.logsumexp(branch_lse, dim=-1)
    valid = torch.isfinite(total_lse)
    safe_total = torch.where(valid, total_lse, torch.zeros_like(total_lse))
    weight = torch.exp(branch_lse - safe_total.unsqueeze(-1))
    weight = torch.where(valid.unsqueeze(-1), weight, torch.zeros_like(weight))
    output = torch.stack(outputs, dim=-2)
    merged = (output * weight.to(output.dtype).unsqueeze(-1)).sum(dim=-2)
    return merged, total_lse


def coarse_lod_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    state: LODState,
    *,
    scale: float | None = None,
    query_offset: int | None = None,
) -> LODAttentionResult:
    """Attend to count-corrected state summaries and an exact local field."""
    _validate_qkv(query, local_key, local_value)
    if scale is None:
        scale = 1.0 / math.sqrt(float(query.size(-1)))
    state_output, state_lse = _state_attention(query, state, scale=scale)
    local_output, local_lse = _local_attention(
        query,
        local_key,
        local_value,
        scale=scale,
        query_offset=query_offset,
    )
    output, lse = _merge_lse_branches(
        [state_output, local_output], [state_lse, local_lse]
    )
    return LODAttentionResult(output=output, logsumexp=lse)


def _normalize_open_count(
    open_count: int | torch.Tensor,
    *,
    shape: tuple[int, int, int],
    route_count: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(open_count, int):
        result = torch.full(shape, open_count, dtype=torch.long, device=device)
    elif isinstance(open_count, torch.Tensor):
        if open_count.is_floating_point() and bool(
            (open_count != open_count.round()).any().item()
        ):
            raise ValueError("open_count tensor must contain integers")
        try:
            result = torch.broadcast_to(open_count.to(device=device, dtype=torch.long), shape)
        except RuntimeError as exc:
            raise ValueError(
                "open_count must be scalar or broadcastable to [batch, heads, queries]"
            ) from exc
    else:
        raise TypeError("open_count must be an integer or tensor")
    if bool((result < 0).any().item()):
        raise ValueError("open_count cannot be negative")
    return result.clamp_max(route_count)


def two_level_lod_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    state: LODState,
    owner: torch.Tensor,
    leaf_key: torch.Tensor,
    leaf_value: torch.Tensor,
    *,
    max_routes: int = 8,
    open_count: int | torch.Tensor = 8,
    route_protected_prefix: int = 1,
    routing_normalization: str = "none",
    routing_rope_dim: int = 0,
    routing_rope_fast_pairs: int = 0,
    routing_count_bias: float = 1.0,
    routing_variance_bias: float = 0.0,
    routing_rope_jensen: bool = False,
    scale: float | None = None,
    query_offset: int | None = None,
) -> LODAttentionResult:
    """Replace opened coarse regions with independently normalized exact leaves.

    ``max_routes`` controls how many ranked regions are identified (at most
    128). ``route_protected_prefix`` leaves exact protected prefix entries in
    the coarse softmax but prevents them from consuming detailed routes.
    ``open_count`` selects how many of that ordered list are actually opened
    and may be either a scalar or a per-query tensor broadcastable to
    ``[batch, query_heads, query_length]``.
    """
    query_heads, key_value_heads = _validate_qkv(query, local_key, local_value)
    _validate_qkv(query, leaf_key, leaf_value)
    if int(leaf_key.size(1)) != key_value_heads:
        raise ValueError("local and leaf archives use different KV-head counts")
    if owner.ndim != 3 or owner.shape[:2] != leaf_key.shape[:2]:
        raise ValueError("owner must have shape [batch, key_value_heads, leaves]")
    history_length = int(owner.size(2))
    if history_length > int(leaf_key.size(2)):
        raise ValueError("owner archive is longer than the leaf archive")
    if history_length and (
        bool((owner < 0).any().item())
        or bool((owner >= state.slot_count).any().item())
    ):
        raise ValueError("owner archive contains an invalid state slot")
    if not 0 <= max_routes <= 128:
        raise ValueError("max_routes must be between zero and 128")
    if route_protected_prefix < 0:
        raise ValueError("route_protected_prefix cannot be negative")
    if scale is None:
        scale = 1.0 / math.sqrt(float(query.size(-1)))

    state_scores, state_value = _state_scores_and_value(query, state, scale)
    protected = min(route_protected_prefix, state.slot_count)
    route_count = min(max_routes, state.slot_count - protected)
    open_counts = _normalize_open_count(
        open_count,
        shape=(int(query.size(0)), query_heads, int(query.size(2))),
        route_count=route_count,
        device=query.device,
    )
    if route_count:
        with torch.no_grad():
            route_scores = (
                state_scores.detach()
                if (
                    routing_normalization == "none"
                    and routing_rope_fast_pairs == 0
                    and routing_count_bias == 1.0
                    and routing_variance_bias == 0.0
                    and not routing_rope_jensen
                )
                else _routing_state_scores(
                    query,
                    state,
                    scale,
                    routing_normalization,
                    routing_count_bias,
                    local_key,
                    routing_variance_bias,
                    routing_rope_dim,
                    routing_rope_fast_pairs,
                    routing_rope_jensen,
                )
            )
            if protected:
                route_scores = route_scores.clone()
                route_scores[..., :protected] = -torch.inf
            top_slots = route_scores.topk(
                route_count, dim=-1, largest=True, sorted=True
            ).indices
        route_rank = torch.arange(route_count, device=query.device)
        open_mask = route_rank.view(1, 1, 1, route_count) < open_counts.unsqueeze(-1)
        selected_count = _repeat_kv(state.count, query_heads).unsqueeze(2).expand(
            -1, -1, int(query.size(2)), -1
        ).gather(-1, top_slots)
        open_mask = open_mask & selected_count.gt(0.5)
        excluded = torch.zeros_like(state_scores, dtype=torch.bool)
        for route_index in range(route_count):
            excluded.scatter_(
                -1,
                top_slots[..., route_index : route_index + 1],
                open_mask[..., route_index : route_index + 1],
            )
    else:
        top_slots = torch.empty(
            *query.shape[:-1], 0, dtype=torch.long, device=query.device
        )
        open_mask = torch.empty(
            *query.shape[:-1], 0, dtype=torch.bool, device=query.device
        )
        excluded = torch.zeros_like(state_scores, dtype=torch.bool)

    coarse_scores = state_scores.masked_fill(excluded, -torch.inf)
    coarse_output, coarse_lse = _attention_from_scores(coarse_scores, state_value)
    local_output, local_lse = _local_attention(
        query,
        local_key,
        local_value,
        scale=scale,
        query_offset=query_offset,
    )
    outputs = [coarse_output, local_output]
    lses = [coarse_lse, local_lse]

    if route_count:
        repeated_key = _repeat_kv(leaf_key[..., :history_length, :], query_heads)
        repeated_value = _repeat_kv(
            leaf_value[..., :history_length, :], query_heads
        )
        repeated_owner = _repeat_kv(owner, query_heads)
        leaf_scores = _scaled_scores(query, repeated_key, scale)
        for route_index in range(route_count):
            slot = top_slots[..., route_index]
            selected = repeated_owner.unsqueeze(2) == slot.unsqueeze(-1)
            selected = selected & open_mask[..., route_index].unsqueeze(-1)
            route_scores = leaf_scores.masked_fill(~selected, -torch.inf)
            route_output, route_lse = _attention_from_scores(
                route_scores, repeated_value
            )
            outputs.append(route_output)
            lses.append(route_lse)

    output, lse = _merge_lse_branches(outputs, lses)
    return LODAttentionResult(
        output=output,
        logsumexp=lse,
        top_slots=top_slots,
        open_mask=open_mask,
    )


def _gather_sequence(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return tensor.gather(
        2, index.unsqueeze(-1).expand(*index.shape, int(tensor.size(-1)))
    )


def _sum_adjacent_groups(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    """Sum consecutive sequence entries into fixed, non-overlapping groups."""
    if factor == 1:
        return tensor
    length = int(tensor.size(2))
    groups = (length + factor - 1) // factor
    padded_length = groups * factor
    if padded_length != length:
        tensor = F.pad(tensor, (0, 0, 0, padded_length - length))
    return tensor.reshape(
        *tensor.shape[:2], groups, factor, int(tensor.size(-1))
    ).sum(dim=3)


def _premerge_adjacent_state_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Form atomic adjacent-token centroids before learned state routing.

    Returned K/V tensors are sums, counts recover their means, and membership
    maps every original leaf to its fixed adjacent group. The exact leaf
    archive is intentionally not modified by this operation.
    """
    if factor not in {1, 2, 4}:
        raise ValueError("adjacent state premerge factor must be one, two, or four")
    length = int(key.size(2))
    if length != int(value.size(2)):
        raise ValueError("adjacent state premerge K/V lengths differ")
    grouped_key = _sum_adjacent_groups(key, factor)
    grouped_value = _sum_adjacent_groups(value, factor)
    groups = int(grouped_key.size(2))
    count = torch.full(
        (*key.shape[:2], groups, 1),
        float(factor),
        dtype=torch.float32,
        device=key.device,
    )
    if groups and length % factor:
        count[..., -1, 0] = float(length % factor)
    membership = (
        torch.div(
            torch.arange(length, device=key.device, dtype=torch.long),
            factor,
            rounding_mode="floor",
        )
        .view(1, 1, length)
        .expand(*key.shape[:2], length)
    )
    return grouped_key, grouped_value, count, membership


class _PytorchLODAttention(nn.Module):
    """Shared causal state/cache lifecycle for the two public variants."""

    def __init__(
        self,
        config: LODConfig | None,
        *,
        store_leaves: bool,
        default_open_count: int,
    ) -> None:
        super().__init__()
        self.config = LODConfig() if config is None else config
        self.store_leaves = store_leaves
        self.default_open_count = default_open_count
        if not store_leaves and default_open_count:
            raise ValueError("coarse-only attention cannot open leaf regions")
        if not 0 <= default_open_count <= self.config.max_routes:
            raise ValueError("default_open_count exceeds configured max_routes")

    @staticmethod
    def _empty_state(key: torch.Tensor, value: torch.Tensor) -> LODState:
        return LODState(
            key_sum=key[..., :0, :],
            value_sum=value[..., :0, :],
            count=torch.empty(
                *key.shape[:2], 0, dtype=torch.float32, device=key.device
            ),
        )

    def _desired_state_size(
        self, context_length: int, available_context: int, current_size: int
    ) -> int:
        target = max(
            math.floor(
                self.config.state_growth_factor
                * math.sqrt(max(context_length, 0))
            ),
            self.config.state_min_size,
        ) + self.config.state_size_offset + int(
            getattr(self, "_lod_padding_state_reserve", 0)
        )
        return max(current_size, min(target, available_context))

    def _bswa_begin(self, total_length: int) -> int:
        rounded_end = (
            (total_length + self.config.chunk_size - 1)
            // self.config.chunk_size
            * self.config.chunk_size
        )
        return max(0, rounded_end - self.config.local_window)

    def _initialize_state(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_starts: torch.Tensor | None = None,
    ) -> tuple[LODState, torch.Tensor]:
        length = int(key.size(2))
        if valid_starts is not None:
            valid_starts = valid_starts.to(device=key.device, dtype=torch.long)
            if tuple(valid_starts.shape) != (int(key.size(0)),):
                raise ValueError("prefill valid starts must have one entry per row")
            if bool(
                (valid_starts.lt(0) | valid_starts.ge(length)).any().item()
            ):
                raise ValueError(
                    "chunk-aligned padding must fit entirely in the first state block"
                )
            position = torch.arange(length, device=key.device)
            valid_count = length - valid_starts
            valid = position.unsqueeze(0) < valid_count.unsqueeze(1)
            source = (valid_starts.unsqueeze(1) + position).clamp_max(length - 1)
            key = _gather_sequence(
                key,
                source[:, None, :].expand(-1, int(key.size(1)), -1),
            ).masked_fill(~valid[:, None, :, None], 0)
            value = _gather_sequence(
                value,
                source[:, None, :].expand(-1, int(value.size(1)), -1),
            ).masked_fill(~valid[:, None, :, None], 0)
            if self.config.state_premerge_factor > 1:
                key, value, _, _ = _premerge_adjacent_state_inputs(
                    key,
                    value,
                    self.config.state_premerge_factor,
                )
                grouped_count = _sum_adjacent_groups(
                    valid[:, None, :, None].float(),
                    self.config.state_premerge_factor,
                )[..., 0]
                grouped_count = grouped_count.expand(
                    -1, int(key.size(1)), -1
                )
                grouped_slots = int(key.size(2))
                key = torch.cat((key, torch.zeros_like(key[..., :1, :])), dim=2)
                value = torch.cat(
                    (value, torch.zeros_like(value[..., :1, :])), dim=2
                )
                dummy_count = -valid_starts[:, None, None].expand(
                    -1, int(key.size(1)), 1
                ).float()
                count = torch.cat((grouped_count, dummy_count), dim=2)
                physical = position.unsqueeze(0).expand(int(key.size(0)), -1)
                owner = torch.where(
                    physical >= valid_starts.unsqueeze(1),
                    torch.div(
                        physical - valid_starts.unsqueeze(1),
                        self.config.state_premerge_factor,
                        rounding_mode="floor",
                    ),
                    torch.full_like(physical, grouped_slots),
                )[:, None, :].expand(-1, int(key.size(1)), -1)
                return LODState(key_sum=key, value_sum=value, count=count), owner
            count = valid[:, None, :].expand(-1, int(key.size(1)), -1).float()
            dummy_slot = valid_count.clamp_max(length - 1)
            dummy = F.one_hot(dummy_slot, num_classes=length).to(count.dtype)
            count = count - dummy[:, None, :] * valid_starts[:, None, None]
            physical = position.unsqueeze(0).expand(int(key.size(0)), -1)
            owner = torch.where(
                physical >= valid_starts.unsqueeze(1),
                physical - valid_starts.unsqueeze(1),
                valid_count.unsqueeze(1),
            )[:, None, :].expand(-1, int(key.size(1)), -1)
            return LODState(key_sum=key, value_sum=value, count=count), owner
        if self.config.state_premerge_factor > 1:
            key, value, count, owner = _premerge_adjacent_state_inputs(
                key,
                value,
                self.config.state_premerge_factor,
            )
            return LODState(key_sum=key, value_sum=value, count=count[..., 0]), owner
        count = torch.ones(
            *key.shape[:2], length, dtype=torch.float32, device=key.device
        )
        owner = (
            torch.arange(length, dtype=torch.long, device=key.device)
            .view(1, 1, length)
            .expand(int(key.size(0)), int(key.size(1)), length)
        )
        return LODState(key_sum=key, value_sum=value, count=count), owner

    def _update_state(
        self,
        state: LODState,
        overflow_key: torch.Tensor,
        overflow_value: torch.Tensor,
        *,
        context_length: int,
        available_context: int,
    ) -> tuple[LODState, torch.Tensor]:
        overflow_length = int(overflow_key.size(2))
        if overflow_length == 0:
            owner = torch.empty(
                *overflow_key.shape[:3], dtype=torch.long, device=overflow_key.device
            )
            return state, owner
        (
            overflow_key,
            overflow_value,
            overflow_count,
            overflow_membership,
        ) = _premerge_adjacent_state_inputs(
            overflow_key,
            overflow_value,
            self.config.state_premerge_factor,
        )
        overflow_mean_key = overflow_key / overflow_count.to(
            overflow_key.dtype
        ).clamp_min(1)
        grouped_overflow_length = int(overflow_key.size(2))
        current_size = state.slot_count
        desired_size = self._desired_state_size(
            context_length, available_context, current_size
        )
        append_count = min(
            max(desired_size - current_size, 0), grouped_overflow_length
        )

        with torch.no_grad():
            similarity = torch.matmul(
                overflow_mean_key.detach(),
                state.mean_key.detach().transpose(-1, -2),
            )
            similarity.masked_fill_(state.count.le(0.5).unsqueeze(-2), -torch.inf)
            max_similarity = similarity.max(dim=-1).values
            order = max_similarity.argsort(dim=-1, descending=False)
            append_index = torch.sort(order[..., :append_count], dim=-1).values
            merge_index = torch.sort(order[..., append_count:], dim=-1).values

        grouped_owner = torch.full(
            overflow_key.shape[:3], -1, dtype=torch.long, device=overflow_key.device
        )
        if append_count:
            append_key = _gather_sequence(overflow_key, append_index)
            append_value = _gather_sequence(overflow_value, append_index)
            append_count_tensor = _gather_sequence(
                overflow_count, append_index
            )[..., 0]
            state = LODState(
                key_sum=torch.cat((state.key_sum, append_key), dim=2),
                value_sum=torch.cat((state.value_sum, append_value), dim=2),
                count=torch.cat((state.count, append_count_tensor), dim=2),
            )
            append_slot = (
                torch.arange(
                    current_size,
                    current_size + append_count,
                    dtype=torch.long,
                    device=overflow_key.device,
                )
                .view(1, 1, append_count)
                .expand_as(append_index)
            )
            grouped_owner.scatter_(2, append_index, append_slot)

        merge_key = _gather_sequence(overflow_key, merge_index)
        merge_value = _gather_sequence(overflow_value, merge_index)
        if int(merge_key.size(2)) == 0:
            return state, grouped_owner.gather(2, overflow_membership)
        merge_count = _gather_sequence(overflow_count, merge_index)
        merge_mean_key = merge_key / merge_count.to(merge_key.dtype).clamp_min(1)

        protected = min(self.config.protected_prefix, state.slot_count)
        if protected >= state.slot_count:
            raise ValueError("all state slots are protected from merging")
        with torch.no_grad():
            route_score = torch.matmul(
                merge_mean_key.detach(),
                state.mean_key.detach().transpose(-1, -2),
            )
            route_score.masked_fill_(state.count.le(0.5).unsqueeze(-2), -torch.inf)
            route_score[..., :protected] = -torch.inf
            destination = route_score.argmax(dim=-1)
        assignment = F.one_hot(destination, num_classes=state.slot_count)
        assignment_key = assignment.to(merge_key.dtype).transpose(-1, -2)
        assignment_value = assignment.to(merge_value.dtype).transpose(-1, -2)
        state = LODState(
            key_sum=state.key_sum + torch.matmul(assignment_key, merge_key),
            value_sum=state.value_sum + torch.matmul(
                assignment_value, merge_value
            ),
            count=state.count
            + torch.matmul(
                assignment.float().transpose(-1, -2),
                merge_count.float(),
            )[..., 0],
        )
        grouped_owner.scatter_(2, merge_index, destination)
        return state, grouped_owner.gather(2, overflow_membership)

    def _compress_to(
        self,
        state: LODState,
        owner: torch.Tensor | None,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        coverage: int,
        target: int,
        context_length: int,
        coverage_offset: int = 0,
        initial_valid_starts: torch.Tensor | None = None,
    ) -> tuple[LODState, torch.Tensor | None, int]:
        if target < coverage or target > int(key.size(2)):
            raise ValueError("invalid LOD compression target")
        while coverage < target:
            block_end = min(target, coverage + self.config.chunk_size)
            block_key = key[..., coverage:block_end, :]
            block_value = value[..., coverage:block_end, :]
            if state.slot_count == 0:
                state, block_owner = self._initialize_state(
                    block_key, block_value, initial_valid_starts
                )
            else:
                state, block_owner = self._update_state(
                    state,
                    block_key,
                    block_value,
                    context_length=context_length,
                    available_context=coverage_offset + block_end,
                )
            if owner is not None:
                owner = torch.cat((owner, block_owner), dim=2)
            coverage = block_end
        return state, owner, coverage

    @staticmethod
    def _slice_open_count(
        open_count: int | torch.Tensor, begin: int, end: int
    ) -> int | torch.Tensor:
        if (
            isinstance(open_count, int)
            or open_count.ndim == 0
            or int(open_count.size(-1)) == 1
        ):
            return open_count
        return open_count[..., begin:end]

    def _attend(
        self,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        state: LODState,
        *,
        owner: torch.Tensor | None,
        leaf_key: torch.Tensor | None,
        leaf_value: torch.Tensor | None,
        open_count: int | torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        query_offset = int(local_key.size(2)) - int(query.size(2))
        if not self.store_leaves:
            if isinstance(open_count, int):
                nonzero = open_count != 0
            else:
                nonzero = bool((open_count != 0).any().item())
            if nonzero:
                raise ValueError("coarse-only attention requires open_count=0")
            return coarse_lod_attention(
                query,
                local_key,
                local_value,
                state,
                scale=scale,
                query_offset=query_offset,
            ).output
        if owner is None or leaf_key is None or leaf_value is None:
            raise ValueError("two-level attention requires an exact leaf archive")
        return two_level_lod_attention(
            query,
            local_key,
            local_value,
            state,
            owner,
            leaf_key,
            leaf_value,
            max_routes=self.config.max_routes,
            open_count=open_count,
            # A protected singleton is already exact in the coarse branch.
            # A protected fixed group is not, so it must remain eligible for
            # detailed opening even though state updates cannot merge into it.
            route_protected_prefix=(
                self.config.protected_prefix
                if self.config.state_premerge_factor == 1
                else 0
            ),
            routing_normalization=self.config.routing_normalization,
            routing_rope_dim=self.config.routing_rope_dim,
            routing_rope_fast_pairs=self.config.routing_rope_fast_pairs,
            routing_count_bias=self.config.routing_count_bias,
            routing_variance_bias=self.config.routing_variance_bias,
            routing_rope_jensen=self.config.routing_rope_jensen,
            scale=scale,
            query_offset=query_offset,
        ).output

    def _local(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scale: float,
        query_offset: int,
    ) -> torch.Tensor:
        return _local_attention(
            query,
            key,
            value,
            scale=scale,
            query_offset=query_offset,
        )[0]

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        open_count: int | torch.Tensor,
        scale: float,
        use_cache: bool,
        prefill_valid_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LODCache | None]:
        sequence_length = int(query.size(2))
        if prefill_valid_starts is not None:
            prefill_valid_starts = prefill_valid_starts.to(
                device=query.device, dtype=torch.long
            )
            if bool(
                (
                    prefill_valid_starts.lt(0)
                    | prefill_valid_starts.ge(self.config.chunk_size)
                ).any().item()
            ):
                raise ValueError(
                    "chunk-aligned padding must fit entirely in the first chunk"
                )
            self._lod_padding_state_reserve = int(
                prefill_valid_starts.max().item()
            )
        else:
            self._lod_padding_state_reserve = 0
        front_length = min(sequence_length, self.config.local_window)
        front_query = query[..., :front_length, :]
        front_key = key[..., :front_length, :]
        front_value = value[..., :front_length, :]
        front_output = (
            self._local(
                front_query,
                front_key,
                front_value,
                scale=scale,
                query_offset=0,
            )
            if prefill_valid_starts is None
            else _masked_causal_attention(
                front_query,
                front_key,
                front_value,
                scale=scale,
                valid_starts=prefill_valid_starts,
            )
        )
        outputs = [front_output]
        state = self._empty_state(key, value)
        coverage = 0
        owner = (
            torch.empty(
                *key.shape[:2], 0, dtype=torch.long, device=key.device
            )
            if self.store_leaves
            else None
        )
        leaf_key = key.to(self.config.leaf_dtype) if self.store_leaves else None
        leaf_value = value.to(self.config.leaf_dtype) if self.store_leaves else None
        exact_lookback = self.config.local_window - self.config.chunk_size

        for query_begin in range(
            front_length, sequence_length, self.config.chunk_size
        ):
            query_end = min(sequence_length, query_begin + self.config.chunk_size)
            local_begin = max(0, query_begin - exact_lookback)
            state, owner, coverage = self._compress_to(
                state,
                owner,
                key,
                value,
                coverage=coverage,
                target=local_begin,
                context_length=query_begin,
                initial_valid_starts=prefill_valid_starts,
            )
            outputs.append(
                self._attend(
                    query[..., query_begin:query_end, :],
                    key[..., local_begin:query_end, :],
                    value[..., local_begin:query_end, :],
                    state,
                    owner=owner,
                    leaf_key=leaf_key,
                    leaf_value=leaf_value,
                    open_count=self._slice_open_count(
                        open_count, query_begin, query_end
                    ),
                    scale=scale,
                )
            )

        if use_cache:
            decode_coverage = self._bswa_begin(sequence_length + 1)
            state, owner, coverage = self._compress_to(
                state,
                owner,
                key,
                value,
                coverage=coverage,
                target=decode_coverage,
                context_length=sequence_length,
                initial_valid_starts=prefill_valid_starts,
            )
            cache = LODCache(
                state=state,
                coverage=coverage,
                recent_key=key[..., coverage:, :],
                recent_value=value[..., coverage:, :],
                total_length=sequence_length,
                owner=owner,
                leaf_key=leaf_key,
                leaf_value=leaf_value,
            ).detached()
        else:
            cache = None
        return torch.cat(outputs, dim=2), cache

    def _decode_one(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: LODCache,
        *,
        open_count: int | torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, LODCache]:
        if int(query.size(2)) != 1 or int(key.size(2)) != 1:
            raise ValueError("_decode_one requires one query/key/value token")
        recent_key = torch.cat((cache.recent_key, key), dim=2)
        recent_value = torch.cat((cache.recent_value, value), dim=2)
        leaf_key = cache.leaf_key
        leaf_value = cache.leaf_value
        if self.store_leaves:
            if leaf_key is None or leaf_value is None:
                raise ValueError("two-level decode cache has no leaf archive")
            leaf_key = torch.cat((leaf_key, key.to(self.config.leaf_dtype)), dim=2)
            leaf_value = torch.cat(
                (leaf_value, value.to(self.config.leaf_dtype)), dim=2
            )

        total_length = cache.total_length + 1
        target_coverage = self._bswa_begin(total_length)
        overflow_length = target_coverage - cache.coverage
        if overflow_length < 0 or overflow_length > int(recent_key.size(2)):
            raise AssertionError("LOD decode coverage drifted")
        state, owner, coverage = self._compress_to(
            cache.state,
            cache.owner,
            recent_key,
            recent_value,
            coverage=0,
            target=overflow_length,
            context_length=cache.total_length,
            coverage_offset=cache.coverage,
        )
        # _compress_to used coordinates relative to the recent buffer.
        coverage = cache.coverage + coverage
        recent_key = recent_key[..., overflow_length:, :]
        recent_value = recent_value[..., overflow_length:, :]

        if state.slot_count == 0:
            output = self._local(
                query,
                recent_key,
                recent_value,
                scale=scale,
                query_offset=int(recent_key.size(2)) - 1,
            )
        else:
            output = self._attend(
                query,
                recent_key,
                recent_value,
                state,
                owner=owner,
                leaf_key=leaf_key,
                leaf_value=leaf_value,
                open_count=open_count,
                scale=scale,
            )
        next_cache = LODCache(
            state=state,
            coverage=coverage,
            recent_key=recent_key,
            recent_value=recent_value,
            total_length=total_length,
            owner=owner,
            leaf_key=leaf_key,
            leaf_value=leaf_value,
        )
        return output, next_cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cache: LODCache | None = None,
        use_cache: bool = False,
        open_count: int | torch.Tensor | None = None,
        scale: float | None = None,
        prefill_valid_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LODCache | None]:
        """Apply causal LOD attention directly to post-RoPE Q/K/V tensors."""
        _validate_qkv(query, key, value)
        if int(query.size(2)) != int(key.size(2)):
            raise ValueError("query and new key/value lengths must match")
        if scale is None:
            scale = 1.0 / math.sqrt(float(query.size(-1)))
        if open_count is None:
            open_count = self.default_open_count
        if cache is None:
            return self._prefill(
                query,
                key,
                value,
                open_count=open_count,
                scale=scale,
                use_cache=use_cache,
                prefill_valid_starts=prefill_valid_starts,
            )

        if prefill_valid_starts is not None:
            raise ValueError("prefill padding metadata is valid only for initial prefill")

        outputs = []
        next_cache = cache
        for token_index in range(int(query.size(2))):
            token_open_count = self._slice_open_count(
                open_count, token_index, token_index + 1
            )
            output, next_cache = self._decode_one(
                query[..., token_index : token_index + 1, :],
                key[..., token_index : token_index + 1, :],
                value[..., token_index : token_index + 1, :],
                next_cache,
                open_count=token_open_count,
                scale=scale,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=2), next_cache.detached() if use_cache else None


class CoarseLODAttention(_PytorchLODAttention):
    """Simple low-LOD state plus exact-local attention with no leaf archive."""

    def __init__(self, config: LODConfig | None = None) -> None:
        super().__init__(config, store_leaves=False, default_open_count=0)


class TwoLevelLODAttention(_PytorchLODAttention):
    """Low LOD plus independently normalized exact BF16 leaf regions."""

    def __init__(
        self,
        config: LODConfig | None = None,
        *,
        default_open_count: int = 8,
    ) -> None:
        super().__init__(
            config, store_leaves=True, default_open_count=default_open_count
        )


__all__ = [
    "CoarseLODAttention",
    "LODAttentionResult",
    "LODCache",
    "LODConfig",
    "LODState",
    "TwoLevelLODAttention",
    "coarse_lod_attention",
    "two_level_lod_attention",
]
