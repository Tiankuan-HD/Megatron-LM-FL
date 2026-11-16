# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""Recompute DSv4 Q up-projection and normalization on Ascend."""

import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
)


def q_up_proj_and_norm(self: DSv4HybridSelfAttention, q_compressed: torch.Tensor) -> torch.Tensor:
    """Checkpoint the raw Q intermediate while retaining normal output lifetimes."""
    original = DSv4HybridSelfAttention._q_up_proj_and_norm.__wrapped__
    config = self.config
    if (
        q_compressed.device.type != "npu"
        or q_compressed.dtype not in (torch.bfloat16, torch.float16)
        or not self.training
        or not torch.is_grad_enabled()
        or config.recompute_granularity != "full"
        or config.fp8
        or config.fp4
        or config.context_parallel_size != 1
        or config.cpu_offloading
        or config.fine_grained_activation_offloading
        or config.cuda_graph_impl != "none"
    ):
        return original(self, q_compressed)

    return checkpoint(original, self, q_compressed, use_reentrant=False)
