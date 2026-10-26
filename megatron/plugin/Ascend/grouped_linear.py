"""Dense expert-parameter ownership in MG, grouped GEMM execution in TE-FL."""

import os
from math import prod

import torch

from megatron.plugin.platform import get_platform


def supports_dense(config):
    """Select the MG-owned dense layout only for the supported NPU contract."""
    return (
        get_platform().device_name() == "npu"
        and not config.fp8
        and not config.fp4
        and config.params_dtype in (torch.bfloat16, torch.float16)
        and not config.gradient_accumulation_fusion
        and not config.add_bias_linear
        and not config.moe_single_grouped_bias
        and not config.delay_wgrad_compute
        and not config.overlap_dispatch_backward_with_experts_wgrad
        and not config.cpu_offloading
        and os.getenv("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", "0") == "1"
    )


def supports_dense_grouped_weight(config):
    from megatron.core.transformer.transformer_config import _supports_dense_grouped_weight

    return supports_dense(config) or _supports_dense_grouped_weight.__wrapped__(config)


def make_grouped_weights(self, defer_init=False):
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    if not supports_dense(self.config):
        return TEGroupedLinear.make_grouped_weights.__wrapped__(self, defer_init)
    if defer_init:
        return
    if not 0 < self.num_gemms <= 128:
        raise ValueError("NPU dense grouped GEMM requires 1..128 local experts")
    weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
    packed = torch.stack([w.detach() for w in weights])
    self.register_parameter(
        "weight",
        torch.nn.Parameter(packed),
        init_fn=self.init_method,
        get_rng_state_tracker=self.get_rng_state_tracker,
        fp8_meta_index=self._offsets["weight"],
    )
    for i in range(self.num_gemms):
        self.register_parameter(f"weight{i}", None)
    self._mg_dense_grouped_weight = True
    self.set_tensor_parallel_attributes(defer_init=False)
    self.weight.partition_dim = 2 if self.parallel_mode == "row" else 1


def get_weight_tensors(self):
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    if getattr(self, "_mg_dense_grouped_weight", False):
        # Recreate views after DDP or checkpoint storage rebinding.
        return list(self.weight.unbind(0))
    return TEGroupedLinear._get_weight_tensors.__wrapped__(self)


def _metadata(data, groups, splits=None):
    from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import (
        GroupedTensorStorage,
    )

    return GroupedTensorStorage(
        shape=(data.numel() // data.shape[-1], data.shape[-1]),
        dtype=data.dtype,
        num_tensors=groups,
        quantizer=None,
        data=data.reshape(-1),
        first_dims=splits,
    )


def _gemm(a, b, shape, groups, layout, a_splits=None, b_splits=None, d_splits=None):
    from transformer_engine.plugin import tefl

    from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import (
        GroupedTensorStorage,
    )

    output = GroupedTensorStorage(
        shape=(prod(shape[:-1]), shape[-1]),
        dtype=a.dtype,
        num_tensors=groups,
        quantizer=None,
        first_dims=d_splits,
    )
    workspace = torch.empty(0, dtype=torch.uint8, device=a.device)
    tefl.te_general_grouped_gemm_for_grouped_tensor(
        _metadata(a, groups, a_splits),
        layout[0] == "T",
        _metadata(b, groups, b_splits),
        layout[1] == "T",
        output,
        None,
        None,
        None,
        None,
        workspace,
        workspace,
        False,
        0,
    )
    return output.rowwise_data.view(shape)


class _DenseGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, splits):
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        weight = weight.contiguous()
        groups, out_features, _ = weight.shape
        output = _gemm(
            weight,
            flat,
            (flat.shape[0], out_features),
            groups,
            "TN",
            b_splits=splits,
            d_splits=splits,
        )
        ctx.save_for_backward(flat, weight, splits)
        ctx.input_shape = x.shape
        return output.reshape(*x.shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, splits = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, weight.shape[1]).contiguous()
        groups = weight.shape[0]
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = _gemm(
                weight,
                grad_output,
                x.shape,
                groups,
                "NN",
                b_splits=splits,
                d_splits=splits,
            ).reshape(ctx.input_shape)
        if ctx.needs_input_grad[1]:
            dw = _gemm(
                x,
                grad_output,
                weight.shape,
                groups,
                "NT",
                a_splits=splits,
                b_splits=splits,
            )
        return dx, dw, None


def forward_grouped_linear(self, x, m_splits, is_first_microbatch=None):
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    if not getattr(self, "_mg_dense_grouped_weight", False):
        return TEGroupedLinear._forward_grouped_linear.__wrapped__(
            self, x, m_splits, is_first_microbatch
        )
    if not supports_dense(self.config) or self.use_bias or self.primary_weights_in_fp8:
        raise RuntimeError("Dense NPU grouped layout cannot be used with this configuration")
    if x.device.type != "npu" or x.device != self.weight.device:
        raise ValueError("Dense grouped input and weight must share an NPU device")
    if x.shape[-1] != self.in_features:
        raise ValueError("Dense grouped input feature dimension does not match the weight")
    splits = torch.as_tensor(m_splits, dtype=torch.int64, device=x.device)
    if splits.shape != (self.num_gemms,):
        raise ValueError("Expected one split per local expert")
    x = self.prepare_forward(x, num_gemms=self.num_gemms)
    try:
        if self.fp8 or self.fp8_calibration or self.is_debug_iter():
            raise NotImplementedError(
                "Dense NPU adapter does not support quantized/debug execution"
            )
        # Cast outside the custom Function so the parent Parameter receives its
        # own dtype's gradient through autograd, rather than detached TE storage.
        weight = self.weight.to(dtype=x.dtype)
        return _DenseGroupedLinear.apply(x, weight, splits)
    finally:
        self.end_forward()
