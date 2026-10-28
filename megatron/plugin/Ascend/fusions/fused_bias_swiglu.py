import torch

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _native_op(y, name):
    if not (
        y.device.type == "npu"
        and y.layout == torch.strided
        and y.dtype in _FLOAT_DTYPES
        and 1 <= y.ndim <= 8
        and y.numel() > 0
        and y.shape[-1] % 2 == 0
    ):
        return None
    import torch_npu

    op = getattr(torch_npu, name, None)
    return op if callable(op) else None


def swiglu(y):
    op = _native_op(y, "npu_swiglu")
    if op is None:
        from megatron.core.fusions.fused_bias_swiglu import swiglu as original

        return original.__wrapped__(y)
    return op(y.contiguous(), dim=-1)


def swiglu_back(g, y):
    op = _native_op(y, "npu_swiglu_backward")
    if op is None or not (
        g.device == y.device
        and g.layout == torch.strided
        and g.dtype == y.dtype
        and tuple(g.shape) == (*y.shape[:-1], y.shape[-1] // 2)
        and y.shape[-1] * y.element_size() % 64 == 0
    ):
        from megatron.core.fusions.fused_bias_swiglu import swiglu_back as original

        return original.__wrapped__(g, y)
    return op(g.contiguous(), y.contiguous(), dim=-1)


def weighted_swiglu_back(g, y, weights):
    from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu_back as original

    if (
        _native_op(y, "npu_swiglu") is None
        or any(
            t.device != y.device or t.layout != torch.strided or t.dtype not in _FLOAT_DTYPES
            for t in (g, weights)
        )
        or (torch.is_grad_enabled() and any(t.requires_grad for t in (g, y, weights)))
        or (weights.ndim > 0 and weights.shape[-1] != 1)
    ):
        return original.__wrapped__(g, y, weights)
    shape = (*y.shape[:-1], y.shape[-1] // 2)
    weighted_shape = torch.broadcast_shapes(g.shape, weights.shape)
    if torch.broadcast_shapes(weighted_shape, shape) != shape:
        return original.__wrapped__(g, y, weights)
    try:
        from .weighted_swiglu_kernels import weighted_swiglu_backward
    except ModuleNotFoundError as error:
        if error.name != "triton":
            raise
        return original.__wrapped__(g, y, weights)

    import torch_npu

    native_grad_rounding = (
        torch.promote_types(g.dtype, weights.dtype) == y.dtype
        and weighted_shape == shape
        and y.shape[-1] * y.element_size() % 64 == 0
        and callable(getattr(torch_npu, "npu_swiglu_backward", None))
    )
    return weighted_swiglu_backward(g, y, weights, native_grad_rounding)
