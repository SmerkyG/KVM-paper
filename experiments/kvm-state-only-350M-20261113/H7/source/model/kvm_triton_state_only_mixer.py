"""KVM with a single update chunk and attention over completed state only."""

from .kvm_triton_mixer import SequenceMixer as TritonSequenceMixer


class SequenceMixer(TritonSequenceMixer):
    attend_bswa = False
