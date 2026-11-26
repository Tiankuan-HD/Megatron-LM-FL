# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""Ascend Hadamard rotation for DSA and CSA indexers."""

import torch


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension, using the Ascend provider."""
    if x.device.type != "npu":
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            rotate_activation as original,
        )

        return original.__wrapped__(x)

    from transformer_engine.plugin.core.manager import get_default_manager

    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    if x.numel() == 0:
        return x.clone()
    return get_default_manager().call("hadamard_transform", x, scale=x.size(-1) ** -0.5)


def get_packed_csa_provider(device: torch.device):
    """Resolve device compute through the TE-FL registry for Ascend tensors."""
    if device.type == "npu":
        from transformer_engine.plugin.core.manager import get_default_manager

        return get_default_manager()
    return None
