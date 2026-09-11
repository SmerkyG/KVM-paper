"""The fixed LoD Attention configurations used by the paper release."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

ROUTE_COUNT = 4
PAGE_SIZE = 16
CHUNK_SIZE = 256
LOCAL_WINDOW = 512
PREFIX_CACHE_LOCAL_WINDOW = 1_024
PREFILL_CHUNK_SIZE = 16_384
PREFILL_LOCAL_WINDOW = PREFILL_CHUNK_SIZE + CHUNK_SIZE
# Very short decode scans every retained leaf. Published 4K-and-longer results
# therefore exercise routed LoD rather than a full-cache fallback.
EXACT_DECODE_LIMIT = 2_048


class LODMode(str, Enum):
    """Supported cache organizations.

    Two-tier LoD keeps every exact leaf in BF16. Three-tier LoD adds
    centroid-owned 16-token pages; those pages can be BF16 or residual INT4.
    """

    TWO_TIER = "two-tier"
    THREE_TIER_BF16 = "three-tier-bf16"
    THREE_TIER_INT4 = "three-tier-int4"

    @classmethod
    def parse(cls, value: str | LODMode) -> LODMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(value.strip().lower())
        except (AttributeError, ValueError) as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"LoD mode must be one of: {choices}") from exc

    @property
    def levels(self) -> int:
        return 2 if self is LODMode.TWO_TIER else 3

    @property
    def kv_bits(self) -> int:
        return 4 if self is LODMode.THREE_TIER_INT4 else 0


class ModelFamily(str, Enum):
    QWEN38 = "qwen3.8"
    K2 = "k2-horizon"


def _text_config(config: Any) -> Any:
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            return getter(decoder=True)
        except TypeError:
            return getter()
    return getattr(config, "text_config", None) or config


def model_family(config_or_model: Any) -> ModelFamily:
    """Identify one of the two model families supported by this release."""

    config = getattr(config_or_model, "config", config_or_model)
    text = _text_config(config)
    root_type = str(getattr(config, "model_type", "")).lower()
    text_type = str(getattr(text, "model_type", "")).lower()
    architectures = {
        str(name).lower()
        for name in (
            list(getattr(config, "architectures", None) or ())
            + list(getattr(text, "architectures", None) or ())
        )
    }
    if root_type == "k2_horizon" or "k2horizonforcausallm" in architectures:
        return ModelFamily.K2

    q_heads = int(getattr(text, "num_attention_heads", 0) or 0)
    kv_heads = int(getattr(text, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(text, "head_dim", 0) or 0)
    qwen38_geometry = (q_heads, kv_heads, head_dim) == (24, 4, 256)
    if (
        root_type in ("qwen3_5", "qwen3_5_text")
        and text_type == "qwen3_5_text"
        and qwen38_geometry
    ):
        return ModelFamily.QWEN38

    raise ValueError(
        "This LoD paper release supports only Qwen3.8 and K2 Horizon; "
        f"received model_type={root_type or text_type or '<unknown>'!r}."
    )


@dataclass(frozen=True)
class LODConfig:
    """Small internal value object for the fixed two-tier calculation."""

    chunk_size: int = CHUNK_SIZE
    local_window: int = LOCAL_WINDOW
    state_growth_factor: float = 16.0
    state_min_size: int = 256
    protected_prefix: int = 1
    max_routes: int = ROUTE_COUNT
    state_clustering_policy: str = "manual"
    state_clustering_normalization: str = "none"
    state_clustering_centroid_rescale: str = "none"
    state_clustering_centroid_rescale_scope: str = "assignment"
    routing_normalization: str = "none"
    leaf_paged_directory: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size != CHUNK_SIZE or self.local_window not in (
            LOCAL_WINDOW,
            PREFIX_CACHE_LOCAL_WINDOW,
        ):
            raise ValueError(
                "the paper release uses a 256-token chunk and either its "
                "512-token base window or 1024-token vLLM prefix rollback window"
            )
        if self.max_routes != ROUTE_COUNT:
            raise ValueError("the paper release has exactly four routes")
        if self.protected_prefix != 1:
            raise ValueError("the paper release keeps exactly one protected sink")
        if self.state_clustering_policy not in ("manual", "qk_norm_aware"):
            raise ValueError("unsupported routing-geometry policy")


@dataclass(frozen=True)
class PagedLODConfig(LODConfig):
    """Internal configuration for semantic 16-token pages."""

    page_size: int = PAGE_SIZE
    kv_bits: int = 0
    quant_group_size: int = 32
    page_summary_quant_bits: int = 8
    recursive_materialize_page_scores: bool = False
    recursive_page_score_block_n: int = 16
    recursive_page_score_num_warps: int = 2
    recursive_page_select_block_n: int = 64
    recursive_state_route_backend: str = "fused"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.page_size != PAGE_SIZE:
            raise ValueError("the paper release uses 16-token semantic pages")
        if self.kv_bits not in (0, 4):
            raise ValueError("three-tier leaves must use BF16 or INT4")
        expected_group = 4 if self.kv_bits == 4 else 32
        if self.quant_group_size != expected_group:
            raise ValueError(
                f"{self.kv_bits or 16}-bit leaves require group size {expected_group}"
            )


def kernel_config(mode: str | LODMode) -> LODConfig:
    """Build the fixed internal config for a public LoD mode."""

    resolved = LODMode.parse(mode)
    # Hugging Face resolves normalized-key geometry per installed attention
    # module. vLLM constructs this internal object directly with ``manual``.
    common = dict(
        state_clustering_policy="qk_norm_aware",
        routing_normalization="qk_norm_aware",
    )
    if resolved is LODMode.TWO_TIER:
        return LODConfig(**common)
    int4 = resolved is LODMode.THREE_TIER_INT4
    return PagedLODConfig(
        **common,
        kv_bits=4 if int4 else 0,
        quant_group_size=4 if int4 else 32,
        recursive_state_route_backend="fused",
    )


__all__ = [
    "EXACT_DECODE_LIMIT",
    "LODConfig",
    "LODMode",
    "ModelFamily",
    "PagedLODConfig",
]
