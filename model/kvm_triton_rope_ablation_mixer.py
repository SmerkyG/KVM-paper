"""Positional encoding variants of the Triton KVM attention layer."""

from .kvm_triton_mixer import SequenceMixer


class RoPEEverywhere(SequenceMixer):
    """Keep the rotated key channels in recurrent-state routing and readout."""

    state_position_mode = "rotated"


class LocalOnlyRoPE(SequenceMixer):
    """Use unrotated queries and keys for the recurrent state."""

    state_position_mode = "unrotated"
