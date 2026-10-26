from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.plugin.Ascend.fusions import linear_cross_entropy as ce
from megatron.plugin.Ascend.fusions.linear_cross_entropy_kernels import (
    backward_inplace,
    global_statistics,
    local_statistics,
)
from megatron.plugin.tests.test_linear_cross_entropy import tp_group


def close_gradient(actual, expected):
    diff = actual.float() - expected.float()
    relative = diff.norm() / expected.float().norm().clamp_min(1e-12)
    assert torch.isfinite(actual).all()
    assert relative < 0.012, relative.item()
    assert diff.abs().max() < 0.035 * expected.float().abs().max() + 2e-6


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("variant", ["strided", "bias_scale", "frozen_weight", "empty"])
def test_general_linear_loss(tp_group, monkeypatch, dtype, chunked, variant):
    torch.manual_seed(119)
    world, rank = tp_group.size(), tp_group.rank()
    s, b, h, v = 7, 2, 33, 131
    x = torch.randn(b, s, h, device="npu", dtype=dtype).transpose(0, 1) / 8
    full_weight = torch.randn(h, world * v, device="npu", dtype=dtype).t() / 8
    bias = (
        torch.randn(world * v, device="npu", dtype=dtype) / 8 if variant == "bias_scale" else None
    )
    scale = 0.37 if variant == "bias_scale" else 1.0
    if variant == "empty":
        x = x[:0]
        s = 0
    x = x.detach().requires_grad_()
    xr = x.detach().clone().requires_grad_()
    w = full_weight[rank * v : (rank + 1) * v].detach().requires_grad_(variant != "frozen_weight")
    wr = full_weight.detach().clone().requires_grad_(variant != "frozen_weight")
    local_bias = (
        bias[rank * v : (rank + 1) * v].detach().requires_grad_() if bias is not None else None
    )
    br = bias.detach().clone().requires_grad_() if bias is not None else None
    labels = torch.arange(s * b, device="npu").reshape(b, s) * 43 % (world * v)
    if s:
        labels[0, 0] = -100
    upstream = torch.linspace(0.0001, 0.73, max(1, s * b), device="npu")[: s * b].reshape(b, s)
    monkeypatch.setattr(
        ce, "_CHUNK_LOGITS_BYTES", 3 * v * torch.empty((), dtype=dtype).element_size()
    )
    implementation = ce._ChunkedLinearCrossEntropy if chunked else ce._LinearCrossEntropy
    actual = implementation.apply(x, w, labels, tp_group, local_bias, scale)
    logits = xr.reshape(-1, h) @ wr.T
    if br is not None:
        logits = logits + br
    if scale != 1.0:
        logits = logits * scale
    expected = (
        F.cross_entropy(logits.float(), labels.T.reshape(-1), reduction="none").reshape(s, b).T
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for repeat in range(2):
        actual.backward(upstream, retain_graph=repeat == 0)
        expected.backward(upstream, retain_graph=repeat == 0)
        if not s:
            continue
        dx = x.grad.clone()
        dist.all_reduce(dx, group=tp_group)
        close_gradient(dx, xr.grad)
        if w.requires_grad:
            close_gradient(w.grad, wr.grad[rank * v : (rank + 1) * v])
        if br is not None:
            close_gradient(local_bias.grad, br.grad[rank * v : (rank + 1) * v])


@pytest.mark.parametrize("offset", [0.0, 10000.0])
def test_fp32_statistics_and_bf16_external_scaling(tp_group, offset):
    world, rank = tp_group.size(), tp_group.rank()
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        v = 7
        values = torch.full((4, v), offset, device="npu", dtype=dtype)
        labels = torch.tensor([0, world * v - 1, -100, world * v // 2], device="npu")
        upstream = torch.tensor([0.01, 0.03, 0.1, 0.7], device="npu")
        local = local_statistics(values, labels, rank * v)
        stats = torch.empty(world * 3, 4, device="npu", dtype=torch.float32)
        dist.all_gather_into_tensor(stats, local, group=tp_group)
        norm, loss = global_statistics(stats, labels, world, v)
        backward_inplace(values, labels, norm, upstream, rank * v, world)
        reference_logits = torch.full(
            (4, world * v), offset, device="npu", dtype=torch.float32, requires_grad=True
        )
        expected = F.cross_entropy(reference_logits, labels, reduction="none")
        expected.backward(upstream)
        torch.testing.assert_close(loss, expected, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(
            values,
            reference_logits.grad[:, rank * v : (rank + 1) * v].to(dtype),
            rtol=2e-6,
            atol=2e-7,
        )


def test_vocab_above_previous_limit(tp_group):
    world, rank = tp_group.size(), tp_group.rank()
    v = 131073
    values = torch.zeros(2, v, device="npu", dtype=torch.bfloat16)
    labels = torch.tensor([0, world * v - 1], device="npu")
    local = local_statistics(values, labels, rank * v)
    stats = torch.empty(world * 3, 2, device="npu", dtype=torch.float32)
    dist.all_gather_into_tensor(stats, local, group=tp_group)
    norm, loss = global_statistics(stats, labels, world, v)
    backward_inplace(values, labels, norm, torch.ones(2, device="npu"), rank * v, world)
    torch.testing.assert_close(
        loss, torch.full_like(loss, float(torch.log(torch.tensor(world * v))))
    )
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("mode", ["shared_bias_mup", "cp", "disable_reduce", "expert"])
def test_module_general_dispatch(tp_group, monkeypatch, dtype, mode):
    from megatron.core import parallel_state
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig

    world = tp_group.size()
    monkeypatch.setattr(parallel_state, "_GLOBAL_MEMORY_BUFFER", None)
    parallel_state._set_global_memory_buffer()
    monkeypatch.setattr(parallel_state, "_TENSOR_MODEL_PARALLEL_GROUP", tp_group)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=world,
        tensor_model_parallel_size=world,
        params_dtype=dtype,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="native",
        context_parallel_size=2 if mode == "cp" else 1,
    )
    config.use_mup = mode == "shared_bias_mup"
    config.mup_output_mult = 0.5

    class Model(LanguageModule):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = config
            self.share_embeddings_and_output_weights = True
            self.pg_collection = SimpleNamespace(tp=tp_group)

    torch.manual_seed(117)
    layer = ColumnParallelLinear(
        32, world * 67, config=config, init_method=lambda x: x, bias=True, tp_group=tp_group
    )
    with torch.no_grad():
        layer.weight.normal_(0, 0.1)
        layer.bias.normal_(0, 0.1)
    layer.disable_grad_reduce = mode == "disable_reduce"
    layer.explicit_expert_comm = mode == "expert"
    if mode in ["disable_reduce", "expert"]:
        layer.allreduce_dgrad = False
    model = Model()
    x = torch.randn(5, 2, 32, device="npu", dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    labels = torch.arange(10, device="npu").reshape(2, 5) * 17 % (world * 67)
    upstream = torch.linspace(0.001, 0.9, 10, device="npu").reshape(2, 5)
    hooks = []
    handle = layer.register_forward_pre_hook(lambda *unused: hooks.append(1))
    actual = model.compute_output_loss(
        x, layer, layer.weight, labels, scale_logits_fn=model._scale_logits
    )
    assert actual.grad_fn.__class__.__name__ == "_LinearCrossEntropyBackward"
    actual.backward(upstream)
    gx, gw, gb = x.grad.clone(), layer.weight.grad.clone(), layer.bias.grad.clone()
    layer.weight.grad = None
    layer.bias.grad = None
    expected = LanguageModule.compute_output_loss.__wrapped__(
        model, xr, layer, layer.weight, labels, scale_logits_fn=model._scale_logits
    )
    expected.backward(upstream)
    handle.remove()
    assert len(hooks) == 2
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    close_gradient(gx, xr.grad)
    close_gradient(gw, layer.weight.grad)
    close_gradient(gb, layer.bias.grad)
