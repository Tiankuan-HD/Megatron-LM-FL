# Copyright (c) 2026, FlagOS contributors. All rights reserved.

"""Recompute mHC projection inputs instead of retaining their FP32 copies."""

import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.hyper_connection import HyperConnectionModule


def projection_and_get_norm(
    self: HyperConnectionModule, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint before the activation upcast, preserving Core arithmetic."""
    original = HyperConnectionModule._projection_and_get_norm.__wrapped__
    if (
        x.device.type != "npu"
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not torch.is_grad_enabled()
    ):
        return original(self, x)
    return checkpoint(original, self, x, use_reentrant=False, preserve_rng_state=False)
