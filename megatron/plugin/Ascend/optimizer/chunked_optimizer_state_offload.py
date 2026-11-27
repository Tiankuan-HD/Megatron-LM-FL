"""Stage selected optimizer tensor states on NPU one parameter chunk at a time."""

import math

import torch


class ChunkedOptimizerStateOffloader:
    """Run the existing precision-aware FusedAdam on subsets of local parameter shards."""

    _STATE_NAMES = ("exp_avg", "exp_avg_sq", "master_param")

    def __init__(self, optimizer, chunk_size_bytes, fraction):
        from transformer_engine.pytorch.optimizers import FusedAdam

        if not isinstance(optimizer, FusedAdam):
            raise ValueError("Chunked NPU offload requires TE FusedAdam")
        self.optimizer = optimizer
        if (
            not getattr(optimizer, "master_weights", False)
            or getattr(optimizer, "capturable", False)
            or optimizer.exp_avg_dtype != torch.bfloat16
            or optimizer.exp_avg_sq_dtype != torch.bfloat16
            or getattr(optimizer, "master_weight_dtype", None) != torch.float32
        ):
            raise ValueError("Chunked NPU offload requires non-capturable precision-aware FusedAdam "
                             "with BF16 moments and FP32 master weights")
        if chunk_size_bytes <= 0 or not 0 < fraction <= 1:
            raise ValueError("Expected a positive chunk size and an offload fraction in (0, 1]")

        entries = []
        for group_index, group in enumerate(optimizer.param_groups):
            for param in group["params"]:
                if param.device.type != "npu" or param.dtype not in (torch.bfloat16, torch.float32):
                    raise ValueError("Chunked NPU offload supports ordinary BF16/FP32 NPU parameters")
                if not param.is_contiguous():
                    raise ValueError("Chunked NPU offload requires contiguous local parameter shards")
                master_bytes = 0
                if param.dtype == torch.bfloat16:
                    master_bytes = 2 if optimizer.store_param_remainders else 4
                state_bytes = param.numel() * (4 + master_bytes)
                entries.append((group_index, param, state_bytes))

        target_bytes = math.ceil(sum(size for _, _, size in entries) * fraction)
        selected_bytes = 0
        chunk_bytes = 0
        chunk = {}
        self.chunks = []
        self.resident = {}
        for group_index, param, size in entries:
            if size == 0 or selected_bytes >= target_bytes:
                self.resident.setdefault(group_index, []).append(param)
                continue
            if chunk and chunk_bytes + size > chunk_size_bytes:
                self.chunks.append(chunk)
                chunk, chunk_bytes = {}, 0
            chunk.setdefault(group_index, []).append(param)
            selected_bytes += size
            chunk_bytes += size
        if chunk:
            self.chunks.append(chunk)
        self.host_state = {}

    def _restore_state(self, params):
        for param in params:
            for name, tensor in self.host_state.get(param, {}).items():
                self.optimizer.state[param][name] = tensor.to(param.device, non_blocking=True)

    def _offload_state(self, params):
        for param in params:
            state = self.optimizer.state.get(param, {})
            for name in self._STATE_NAMES:
                tensor = state.get(name)
                if tensor is None:
                    continue
                host_state = self.host_state.setdefault(param, {})
                if name not in host_state:
                    host_state[name] = torch.empty_like(tensor, device="cpu", pin_memory=True)
                host_tensor = host_state[name]
                # Wait for the update and copy before freeing the NPU allocation.
                host_tensor.copy_(tensor, non_blocking=False)
                state[name] = host_tensor

    @torch.no_grad()
    def step(self):
        if not self.chunks:
            return self.optimizer.step()

        original_groups = self.optimizer.param_groups
        step_groups = [dict(group) for group in original_groups]
        resulting_steps = {}
        try:
            for groups in (self.resident, *self.chunks):
                if not groups:
                    continue
                params = [param for group_params in groups.values() for param in group_params]
                stage_state = groups is not self.resident
                if stage_state:
                    self._restore_state(params)
                subset_groups = [
                    dict(step_groups[index], params=group_params)
                    for index, group_params in groups.items()
                ]
                self.optimizer.param_groups = subset_groups
                self.optimizer.step()
                for index, subset in zip(groups, subset_groups):
                    resulting_steps[index] = subset["step"]
                if stage_state:
                    self._offload_state(params)
        finally:
            self.optimizer.param_groups = original_groups

        for index, step in resulting_steps.items():
            original_groups[index]["step"] = step


def validate_optimizer_state_offload_args(args):
    """Validate Ascend-specific requirements before constructing the optimizer."""
    from megatron.training.arguments import _validate_optimizer_state_offload_args

    if not getattr(args, "chunked_optimizer_state_offload", False):
        return _validate_optimizer_state_offload_args.__wrapped__(args)

    from megatron.plugin.platform import get_platform

    if get_platform().device_name() != "npu":
        return _validate_optimizer_state_offload_args.__wrapped__(args)
    if args.use_megatron_fsdp or args.use_torch_fsdp2:
        raise ValueError("Chunked optimizer state offload does not support FSDP")
    if not (
        args.use_distributed_optimizer
        and args.use_precision_aware_optimizer
        and args.optimizer == "adam"
        and args.bf16
    ):
        raise ValueError("Chunked NPU offload requires distributed precision-aware BF16 Adam")
    if args.optimizer_cpu_offload or args.optimizer_cuda_graph or args.fp8_param_gather:
        raise ValueError(
            "Chunked NPU offload is incompatible with CPU Adam, CUDA graph, or FP8 parameter gather"
        )
    if not args.no_save_optim or not args.no_load_optim:
        raise ValueError(
            "Chunked NPU offload currently requires --no-save-optim and --no-load-optim"
        )
    if args.optimizer_state_offload_chunk_size_mb <= 0:
        raise ValueError("Optimizer-state offload chunk size must be positive")
    if not 0.0 <= args.optimizer_state_offload_fraction <= 1.0:
        raise ValueError("Optimizer-state offload fraction must be between 0 and 1")


def configure_optimizer_state_offload(self):
    """Attach the offloader after the distributed optimizer builds local shards."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.plugin.platform import get_platform

    config = self.config
    if not config.chunked_optimizer_state_offload or not config.optimizer_state_offload_fraction:
        return DistributedOptimizer._configure_optimizer_state_offload.__wrapped__(self)
    if get_platform().device_name() != "npu":
        return DistributedOptimizer._configure_optimizer_state_offload.__wrapped__(self)
    if not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
        raise ValueError("Chunked NPU offload requires optimizer-owned master weights")
    self._chunked_state_offloader = ChunkedOptimizerStateOffloader(
        self.optimizer,
        config.optimizer_state_offload_chunk_size_mb * 1024**2,
        config.optimizer_state_offload_fraction,
    )


def step_inner_optimizer(self):
    """Use the Ascend state scheduler when configured; otherwise retain the original step."""
    from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer

    offloader = getattr(self, "_chunked_state_offloader", None)
    if offloader is None:
        return MixedPrecisionOptimizer._step_inner_optimizer.__wrapped__(self)
    return offloader.step()
