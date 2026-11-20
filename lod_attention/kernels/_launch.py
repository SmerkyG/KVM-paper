"""Platform-specific compiler options for the shared Triton kernels."""

import torch


def hip_launch_kwargs(*, waves_per_eu: int) -> dict[str, int]:
    """NVIDIA rejects AMD's occupancy hint instead of ignoring it."""
    return {"waves_per_eu": waves_per_eu} if torch.version.hip else {}
