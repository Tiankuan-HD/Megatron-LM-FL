import json
import logging

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable


_LOGITS_LIMIT_BYTES = 4 * 1024**3
_CHUNK_LOGITS_BYTES = 1024**3


def _logits_save_limit_bytes(device):
    total = torch.npu.get_device_properties(device).total_memory
    return min(_LOGITS_LIMIT_BYTES, total // 8)


def _chunk_size(weight):
    budget = min(_CHUNK_LOGITS_BYTES, max(1, _logits_save_limit_bytes(weight.device)))
    rows = max(1, budget // (weight.shape[0] * weight.element_size()))
    return max(1, rows // 256 * 256) if rows >= 256 else rows


def _project(hidden, weight, bias, scale):
    logits = torch.mm(hidden, weight.t())
    if bias is not None:
        logits = logits + bias
    if scale != 1.0:
        logits = logits * scale
    return logits


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, tp_group, bias=None, scale=1.0, chunked=False):
        from .linear_cross_entropy_kernels import global_statistics, local_statistics

        x = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        labels = target.transpose(0, 1).reshape(-1).contiguous()
        world = tp_group.size() if tp_group is not None else 1
        rank = tp_group.rank() if tp_group is not None else 0
        vocab = weight.shape[0]
        ctx.chunked = chunked
        ctx.has_bias = bias is not None
        ctx.scale = scale
        if chunked:
            ctx.chunk = _chunk_size(weight)
            local = torch.empty((3, x.shape[0]), device=x.device, dtype=torch.float32)
            for start in range(0, x.shape[0], ctx.chunk):
                end = min(start + ctx.chunk, x.shape[0])
                logits = _project(x[start:end], weight, bias, scale)
                local[:, start:end] = local_statistics(logits, labels[start:end], rank * vocab)
                del logits
        else:
            logits = _project(x, weight, bias, scale)
            local = local_statistics(logits, labels, rank * vocab)
            ctx.logits_consumed = False
        if world > 1 and x.shape[0]:
            gathered = torch.empty((world * 3, x.shape[0]), device=x.device, dtype=torch.float32)
            dist.all_gather_into_tensor(gathered, local, group=tp_group)
        else:
            gathered = local
        norm, loss = global_statistics(gathered, labels, world, vocab)
        ctx.save_for_backward(
            x,
            weight,
            labels,
            norm,
            *((bias,) if bias is not None else ()),
            *(() if chunked else (logits,)),
        )
        ctx.hidden_shape = hidden.shape
        ctx.rank_start, ctx.world, ctx.tp_group = rank * vocab, world, tp_group
        _log_dispatch("chunked_forward" if chunked else "forward", x, weight, tp_group)
        return loss.reshape(target.shape[1], target.shape[0]).transpose(0, 1).contiguous()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        from .linear_cross_entropy_kernels import backward_inplace

        x, weight, labels, norm, *saved_logits = ctx.saved_tensors
        bias = saved_logits.pop(0) if ctx.has_bias else None
        grad = grad_loss.transpose(0, 1).reshape(-1).float().contiguous()
        needs_bias = bias is not None and len(ctx.needs_input_grad) > 4 and ctx.needs_input_grad[4]
        db = torch.zeros_like(bias, dtype=torch.float32) if needs_bias else None
        if ctx.chunked:
            dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
            dw = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
            for start in range(0, x.shape[0], ctx.chunk):
                end = min(start + ctx.chunk, x.shape[0])
                logits = _project(x[start:end], weight, bias, ctx.scale)
                backward_inplace(
                    logits,
                    labels[start:end],
                    norm[:, start:end].contiguous(),
                    grad[start:end],
                    ctx.rank_start,
                    ctx.world,
                )
                if ctx.scale != 1.0:
                    logits.mul_(ctx.scale)
                if db is not None:
                    db.add_(logits.float().sum(dim=0))
                dlinear = logits.to(weight.dtype)
                if dx is not None:
                    torch.mm(dlinear, weight, out=dx[start:end])
                if dw is not None:
                    partial = torch.mm(dlinear.t(), x[start:end])
                    dw.add_(partial)
                    del partial
                del logits, dlinear
            dw = dw.to(weight.dtype) if dw is not None else None
        else:
            logits = (
                _project(x, weight, bias, ctx.scale) if ctx.logits_consumed else saved_logits[0]
            )
            backward_inplace(logits, labels, norm, grad, ctx.rank_start, ctx.world)
            ctx.logits_consumed = True
            if ctx.scale != 1.0:
                logits = logits * ctx.scale
            if db is not None:
                db.add_(logits.float().sum(dim=0))
            dlinear = logits.to(weight.dtype)
            dx = torch.mm(dlinear, weight) if ctx.needs_input_grad[0] else None
            dw = None
            if ctx.needs_input_grad[1]:
                transposed = dlinear.t()
                if (
                    dlinear.numel() * dlinear.element_size()
                    <= _logits_save_limit_bytes(x.device) // 2
                ):
                    transposed = transposed.contiguous()
                dw = torch.mm(transposed, x)
        result = (
            dx.reshape(ctx.hidden_shape) if dx is not None else None,
            dw,
            None,
            None,
            db.to(bias.dtype) if db is not None else None,
            None,
            None,
        )
        _log_dispatch("chunked_backward" if ctx.chunked else "backward", x, weight, ctx.tp_group)
        return result[: len(ctx.needs_input_grad)]


class _ChunkedLinearCrossEntropy(_LinearCrossEntropy):
    @staticmethod
    def forward(ctx, hidden, weight, target, tp_group, bias=None, scale=1.0):
        return _LinearCrossEntropy.forward(ctx, hidden, weight, target, tp_group, bias, scale, True)


def compute_output_loss(
    self,
    hidden_states,
    output_layer,
    output_weight,
    labels,
    runtime_gather_output=None,
    scale_logits_fn=None,
):
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_sequence_parallel_region,
    )

    def fallback():
        return LanguageModule.compute_output_loss.__wrapped__(
            self,
            hidden_states,
            output_layer,
            output_weight,
            labels,
            runtime_gather_output,
            scale_logits_fn,
        )

    config = self.config
    if (
        not isinstance(output_layer, ColumnParallelLinear)
        or not config.cross_entropy_loss_fusion
        or config.cross_entropy_fusion_impl != "native"
    ):
        return fallback()
    gather = output_layer.gather_output if runtime_gather_output is None else runtime_gather_output
    if gather or output_layer.gradient_accumulation_fusion or config.defer_embedding_wgrad_compute:
        return fallback()
    if scale_logits_fn is not None and (
        getattr(scale_logits_fn, "__self__", None) is not self
        or getattr(scale_logits_fn, "__func__", None) is not LanguageModule._scale_logits
    ):
        return fallback()
    scale = config.mup_output_mult if scale_logits_fn is not None and config.use_mup else 1.0
    group = output_layer.tp_group

    def processor(hidden, live_weight):
        if output_layer.sequence_parallel and not output_layer.explicit_expert_comm:
            hidden = gather_from_sequence_parallel_region(
                hidden, group=group, tensor_parallel_output_grad=True
            )
        elif not output_layer.disable_grad_reduce and not output_layer.explicit_expert_comm:
            hidden = copy_to_tensor_model_parallel_region(hidden, group=group)
        bias = output_layer.bias if not output_layer.skip_bias_add else None
        element_size = max(
            live_weight.element_size(), bias.element_size() if bias is not None else 0
        )
        implementation = (
            _LinearCrossEntropy
            if labels.numel() * live_weight.shape[0] * element_size
            <= _logits_save_limit_bytes(live_weight.device)
            else _ChunkedLinearCrossEntropy
        )
        return implementation.apply(hidden, live_weight, labels, group, bias, scale)

    loss, _ = output_layer(
        hidden_states,
        weight=output_weight,
        runtime_gather_output=runtime_gather_output,
        output_processor=processor,
    )
    return loss


_DISPATCH_LOGGED = set()


def _log_dispatch(stage, x, weight, group):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    key = (stage, tuple(x.shape), tuple(weight.shape))
    if key in _DISPATCH_LOGGED:
        return
    _DISPATCH_LOGGED.add(key)
    logging.getLogger(__name__).info(
        "NPU_LINEAR_CE %s",
        json.dumps(
            dict(
                stage=stage,
                input_shape=list(x.shape),
                weight_shape=list(weight.shape),
                backend=str(dist.get_backend(group)) if group is not None else "local",
                statistics="fp32_max_denominator",
                implementation="liger_tp",
            )
        ),
    )


def column_parallel_linear_forward(
    self, input_, weight=None, runtime_gather_output=None, *, output_processor=None
):
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    if output_processor is None:
        return ColumnParallelLinear.forward.__wrapped__(
            self, input_, weight=weight, runtime_gather_output=runtime_gather_output
        )
    if weight is None:
        weight = self.weight
        if weight is None:
            raise RuntimeError("weight was not supplied to ColumnParallelLinear forward pass")
    elif tuple(weight.shape) != (self.output_size_per_partition, self.input_size):
        raise RuntimeError(f"supplied weight has unexpected shape {tuple(weight.shape)}")
    context = self.config._cpu_offloading_context
    if context is not None and context.inside_context:
        input_.activation_offloading = self.config.cpu_offloading_activations
    return output_processor(input_, weight), self.bias if self.skip_bias_add else None
