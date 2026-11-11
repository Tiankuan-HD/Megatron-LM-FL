"""Dense expert-parameter ownership in MG, grouped GEMM execution in TE-FL."""

import os
from dataclasses import replace
from math import prod

import torch

from megatron.plugin.platform import get_platform

_GEMM_OP = "te_general_grouped_gemm_for_grouped_tensor"
_contract_checked = set()


def _npu_grouped_gemm():
    """Return TE-FL's NPU grouped-tensor GEMM, or None if the policy selects another one.

    The dense layout calls the operator with alpha=beta=None and an output without
    storage, a call that only the NPU vendor backend defines. tefl.<op> retries a
    failed call on the next registered backend, so the layout binds this
    implementation when it is built and calls it directly.
    """
    from transformer_engine.plugin.core.manager import get_default_manager

    manager = get_default_manager()
    try:
        if manager.get_selected_impl_id(_GEMM_OP) != "vendor.npu":
            return None
        return manager.resolve(_GEMM_OP)
    except RuntimeError:  # nothing is available under the current policy
        return None


def _dense_config(config):
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
        and int(os.getenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")) > 0
        and bool(int(os.getenv("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", "0")))
    )


def supports_dense(config):
    """Select the MG-owned dense layout only for the supported NPU contract."""
    return _dense_config(config) and _npu_grouped_gemm() is not None


def _expert_partition_dim(self):
    """TP axis of one expert's 2-D weight, or None when the layer has no TP layout."""
    return {"column": 0, "row": 1}.get(self._original_parallel_mode)


def set_parallel_attributes(self):
    """Mark expert weights as TP shards so they are not treated as duplicates.

    When explicit_expert_comm clears TE's parallel_mode, the indexed weights receive
    no tensor_model_parallel, so every one of them looks like a duplicate to
    param_is_not_tensor_parallel_duplicate(). Gradient norm, clipping and the zero
    count then keep the expert weights of tensor-parallel rank 0 only. The other
    ranks do not hold copies of them: they hold another shard of each expert when
    the expert TP size is above one, or other experts under expert parallelism.

    The dense parent Parameter needs the same treatment, and its TP axis is shifted
    by one because the expert index occupies dimension 0. The Core default has
    already set allreduce on it, since it iterates over every parameter.
    """
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    TEGroupedLinear._set_parallel_attributes.__wrapped__(self)
    part_dim = _expert_partition_dim(self)
    if part_dim is None:
        return

    if getattr(self, "_mg_dense_grouped_weight", False):
        self.weight.tensor_model_parallel = self.explicit_expert_comm
        self.weight.partition_dim = part_dim + 1
        self.weight.partition_stride = 1
    elif self.explicit_expert_comm:
        for i in range(self.num_gemms):
            indexed = getattr(self, f"weight{i}", None)
            if indexed is not None:
                indexed.tensor_model_parallel = True


def make_grouped_weights(self, defer_init=False):
    """Replace TE's per-expert Parameters with one dense parent Parameter.

    TE calls this from reset_parameters() only, and TEGroupedLinear.reset_parameters()
    re-applies the parallel attributes right after, so they are not set here.
    """
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    if getattr(self, "_mg_dense_grouped_weight", False):
        # Already packed. TE re-initializes the parent Parameter in place, and the
        # indexed Parameters no longer exist, so there is nothing to rebuild.
        return
    gemm = _npu_grouped_gemm() if _dense_config(self.config) else None
    if gemm is None:
        return TEGroupedLinear.make_grouped_weights.__wrapped__(self, defer_init)
    if defer_init:
        return
    if not 0 < self.num_gemms <= 128:
        raise ValueError("NPU dense grouped GEMM requires 1..128 local experts")
    weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
    self.register_parameter(
        "weight",
        torch.nn.Parameter(torch.stack([w.detach() for w in weights])),
        init_fn=self.init_method,
        get_rng_state_tracker=self.get_rng_state_tracker,
        fp8_meta_index=self._offsets["weight"],
    )
    for i in range(self.num_gemms):
        self.register_parameter(f"weight{i}", None)
    self._mg_grouped_gemm = gemm
    self._mg_dense_grouped_weight = True


def get_weight_tensors(self):
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    if getattr(self, "_mg_dense_grouped_weight", False):
        # Recreate views after DDP or checkpoint storage rebinding.
        return list(self.weight.unbind(0))
    return TEGroupedLinear._get_weight_tensors.__wrapped__(self)


def _gemm(gemm, a, b, shape, groups, layout, a_splits=None, b_splits=None, d_splits=None):
    from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import (
        GroupedTensorStorage,
    )

    def wrap(data, dims, splits):
        return GroupedTensorStorage(
            shape=(prod(dims[:-1]), dims[-1]),
            dtype=a.dtype if data is None else data.dtype,
            num_tensors=groups,
            quantizer=None,
            data=None if data is None else data.reshape(-1),
            first_dims=splits,
        )

    # alpha=beta=None selects the identity epilogue, so the backend hands back its
    # native result instead of writing into a destination we would have to allocate.
    output = wrap(None, shape, d_splits)
    workspace = torch.empty(0, dtype=torch.uint8, device=a.device)
    gemm(
        wrap(a, a.shape, a_splits),
        layout[0] == "T",
        wrap(b, b.shape, b_splits),
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


def _require_identity_epilogue(gemm, like):
    """Reject, once per backend, an NPU GEMM that predates the alpha=beta=None call."""
    if gemm in _contract_checked:
        return
    one = like.new_ones(1, 1, 1)
    count = torch.ones(1, dtype=torch.int64, device=like.device)
    try:
        _gemm(gemm, one, one[0], (1, 1), 1, "TN", b_splits=count, d_splits=count)
    except (NotImplementedError, TypeError, ValueError) as error:
        raise RuntimeError(
            "The dense grouped layout needs a TE-FL NPU backend whose grouped-tensor GEMM "
            "accepts alpha=beta=None with an output without storage"
        ) from error
    _contract_checked.add(gemm)


class _DenseGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, splits, gemm):
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        weight = weight.contiguous()
        groups, out_features, _ = weight.shape
        output = _gemm(
            gemm,
            weight,
            flat,
            (flat.shape[0], out_features),
            groups,
            "TN",
            b_splits=splits,
            d_splits=splits,
        )
        # dgrad needs the weight; wgrad needs the input. Save each only when the
        # corresponding gradient is actually requested.
        ctx.save_for_backward(
            flat if ctx.needs_input_grad[1] else None,
            weight if ctx.needs_input_grad[0] else None,
            splits,
        )
        ctx.gemm = gemm
        ctx.input_shape = x.shape
        ctx.groups = groups
        ctx.out_features = out_features
        return output.reshape(*x.shape[:-1], out_features)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        x, weight, splits = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, ctx.out_features).contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            flat_shape = (grad_output.shape[0], ctx.input_shape[-1])
            dx = _gemm(
                ctx.gemm,
                weight,
                grad_output,
                flat_shape,
                ctx.groups,
                "NN",
                b_splits=splits,
                d_splits=splits,
            ).reshape(ctx.input_shape)
        if ctx.needs_input_grad[1]:
            weight_shape = (ctx.groups, ctx.out_features, ctx.input_shape[-1])
            dw = _gemm(
                ctx.gemm,
                x,
                grad_output,
                weight_shape,
                ctx.groups,
                "NT",
                a_splits=splits,
                b_splits=splits,
            )
        return dx, dw, None, None


def forward_grouped_linear(self, x, m_splits, is_first_microbatch=None):
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    if not getattr(self, "_mg_dense_grouped_weight", False):
        return TEGroupedLinear._forward_grouped_linear.__wrapped__(
            self, x, m_splits, is_first_microbatch
        )
    # The layout and its GEMM backend were selected when the parent Parameter was
    # built; re-reading the environment or the TE-FL policy on every microbatch would
    # be both wasted work in the hot path and a chance to diverge from that choice.
    if self.use_bias or self.primary_weights_in_fp8:
        raise RuntimeError("Dense NPU grouped layout cannot be used with this configuration")
    if x.device.type != "npu" or x.device != self.weight.device:
        raise ValueError("Dense grouped input and weight must share an NPU device")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("Dense NPU grouped GEMM requires BF16 or FP16 input")
    if x.shape[-1] != self.in_features:
        raise ValueError("Dense grouped input feature dimension does not match the weight")
    splits = torch.as_tensor(m_splits, dtype=torch.int64, device=x.device)
    if splits.shape != (self.num_gemms,):
        raise ValueError("Expected one split per local expert")
    _require_identity_epilogue(self._mg_grouped_gemm, x)
    x = self.prepare_forward(x, num_gemms=self.num_gemms)
    try:
        if self.fp8 or self.fp8_calibration or self.is_debug_iter():
            raise NotImplementedError(
                "Dense NPU adapter does not support quantized/debug execution"
            )
        # Cast outside the custom Function so the parent Parameter receives its
        # own dtype's gradient through autograd, rather than detached TE storage.
        weight = self.weight.to(dtype=x.dtype)
        return _DenseGroupedLinear.apply(x, weight, splits, self._mg_grouped_gemm)
    finally:
        self.end_forward()


def sharded_state_dict_grouped(self, tp_axis_map, prefix="", sharded_offsets=(), metadata=None):
    """Keep the dense parent Parameter visible to distributed optimizer checkpoints."""
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    sharded_state_dict = TEGroupedLinear._sharded_state_dict_grouped.__wrapped__(
        self, tp_axis_map, prefix, sharded_offsets, metadata
    )
    # Keep the real dense parent identity for distributed optimizer state
    # mapping. Returning only detached expert views loses that identity.
    weight = getattr(self, "weight", None)
    if getattr(self, "_mg_dense_grouped_weight", False):
        from megatron.core.dist_checkpointing.mapping import (
            LocalNonpersistentObject,
            ShardedTensorFactory,
        )

        if (metadata or {}).get('singleton_local_shards', False):
            raise NotImplementedError("Dense grouped weights require singleton_local_shards=False")
        shape = tuple(weight.shape)
        templates = [
            replace(sharded_state_dict.pop(f"{prefix}weight{i}"), data=None)
            for i in range(self.num_gemms)
        ]

        def build(key, tensor, replica_id, flattened_range):
            full = flattened_range is None
            # Express full weights and optimizer slices as rectangular shards;
            # this backend does not accept ShardedTensor.flattened_range.
            start, stop, step = (flattened_range or slice(None)).indices(prod(shape))
            if step != 1:
                raise ValueError("Grouped optimizer shards must be contiguous")
            if start == stop:
                return {"flat": [LocalNonpersistentObject(tensor.reshape(-1))]}
            flat = tensor.reshape(-1)
            expert_size, width = prod(shape[1:]), shape[-1]
            shards = []
            for i, template in enumerate(templates):
                first = max(start, i * expert_size)
                end = min(stop, (i + 1) * expert_size)
                while first < end:
                    row, col = divmod(first - i * expert_size, width)
                    count = min(end - first, width - col)
                    if col == 0 and end - first >= width:
                        count = (end - first) // width * width
                    local_shape = (count // width, width) if count >= width else (1, count)
                    local_shape = (1,) * template.prepend_axis_num + local_shape
                    offset = (
                        *template.global_offset[:-2],
                        template.global_offset[-2] + row,
                        template.global_offset[-1] + col,
                    )
                    shards.append(
                        replace(
                            template,
                            key=key,
                            data=flat[first - start : first - start + count].view(local_shape),
                            dtype=tensor.dtype,
                            replica_id=replica_id,
                            local_shape=local_shape,
                            global_offset=offset,
                            axis_fragmentations=None,
                            prepend_axis_num=0,
                        )
                    )
                    first += count
            return {"full" if full else "flat": shards}

        def merge(shards):
            parts = shards["full"] if "full" in shards else shards["flat"]
            tensor = torch.cat([part.reshape(-1) for part in parts])
            return tensor.reshape(shape) if "full" in shards else tensor

        sharded_state_dict[f"{prefix}weight"] = ShardedTensorFactory(
            templates[0].key, weight, build, merge, templates[0].replica_id
        )
    return sharded_state_dict
