import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")

from megatron.plugin.Ascend.fusions import linear_cross_entropy as ce


@pytest.fixture(scope="session")
def tp_group():
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_group = not dist.is_initialized()
    if owns_group:
        backend = os.environ.get("MEGATRON_TEST_BACKEND", "hccl")
        if backend == "flagcx":
            __import__("flagcx")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29791")
        dist.init_process_group(
            backend,
            rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
    yield dist.group.WORLD
    if owns_group:
        dist.destroy_process_group()


def test_cache_budget_scales_with_capacity(monkeypatch):
    for gib, expected_gib in ((8, 1), (16, 2), (64, 4)):
        monkeypatch.setattr(
            torch.npu,
            "get_device_properties",
            lambda device, gib=gib: SimpleNamespace(total_memory=gib * 1024**3),
        )
        assert ce._logits_save_limit_bytes(torch.device("npu")) == expected_gib * 1024**3


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("frozen_input", [False, True])
def test_tp_loss_gradients_and_repeated_backward(tp_group, monkeypatch, chunked, frozen_input):
    torch.manual_seed(20261022)
    world, rank = tp_group.size(), tp_group.rank()
    s, b, h, v = 11, 2, 64, 259
    hidden = torch.randn(s, b, h, device="npu", dtype=torch.bfloat16) / 8
    full_weight = torch.randn(world * v, h, device="npu", dtype=torch.bfloat16) / 8
    labels = torch.arange(s * b, device="npu").reshape(b, s) * 257 % (world * v)
    labels[0, 0] = -100
    upstream = torch.randn(b, s, device="npu")
    upstream[:, ::3] = 0
    x = hidden.clone().requires_grad_(not frozen_input)
    xr = hidden.clone().requires_grad_(not frozen_input)
    w = full_weight[rank * v : (rank + 1) * v].clone().requires_grad_()
    wr = full_weight.clone().requires_grad_()
    monkeypatch.setattr(ce, "_CHUNK_LOGITS_BYTES", 7 * v * 2)
    implementation = ce._ChunkedLinearCrossEntropy if chunked else ce._LinearCrossEntropy
    actual = implementation.apply(x, w, labels, tp_group)
    logits = xr.reshape(-1, h) @ wr.T
    reference = (
        torch.nn.functional.cross_entropy(
            logits.float(), labels.T.reshape(-1), reduction="none", ignore_index=-100
        )
        .reshape(s, b)
        .T.contiguous()
    )
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-6)
    for repeat in range(2):
        actual.backward(upstream * (repeat + 1), retain_graph=repeat == 0)
        reference.backward(upstream * (repeat + 1), retain_graph=repeat == 0)
        pairs = [(w.grad, wr.grad[rank * v : (rank + 1) * v])]
        if not frozen_input:
            reduced = x.grad.clone()
            dist.all_reduce(reduced, group=tp_group)
            pairs.append((reduced, xr.grad))
        for value, expected in pairs:
            diff = (value.float() - expected.float()).norm()
            assert diff / expected.float().norm().clamp_min(1e-12) < 0.01
            assert (
                value.float() - expected.float()
            ).abs().max() < 0.03 * expected.abs().max().float()


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_module_hooks_reference_bypass_and_mtp(tp_group, monkeypatch, sequence_parallel):
    from megatron.core import parallel_state
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossLoggingHelper,
        process_mtp_loss,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    monkeypatch.setattr(parallel_state, "_GLOBAL_MEMORY_BUFFER", None)
    parallel_state._set_global_memory_buffer()
    world = tp_group.size()
    if sequence_parallel and world == 1:
        pytest.skip("SP requires multiple tensor parallel ranks")
    monkeypatch.setattr(parallel_state, "_TENSOR_MODEL_PARALLEL_GROUP", tp_group)
    monkeypatch.setattr(parallel_state, "_DATA_PARALLEL_GROUP_WITH_CP", tp_group)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=world,
        tensor_model_parallel_size=world,
        sequence_parallel=sequence_parallel,
        bf16=True,
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="native",
        mtp_num_layers=2,
    )

    class Model(LanguageModule):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = config
            self.share_embeddings_and_output_weights = False
            self.pg_collection = SimpleNamespace(tp=tp_group)

    model = Model()
    layer = ColumnParallelLinear(
        64, 1024, config=config, init_method=lambda x: x, bias=False, tp_group=tp_group
    )
    with torch.no_grad():
        layer.weight.normal_(0, 0.125)
    calls = {"optimized": 0, "hooks": 0}
    original = ce._ChunkedLinearCrossEntropy.forward

    def counted(*args, **kwargs):
        calls["optimized"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ce._ChunkedLinearCrossEntropy, "forward", staticmethod(counted))
    monkeypatch.setattr(ce, "_LOGITS_LIMIT_BYTES", 4096)

    def hook(*unused):
        calls["hooks"] += 1

    handle = layer.register_forward_pre_hook(hook)
    saved_tracker = MTPLossLoggingHelper.tracker
    try:
        for enabled in (False, True):
            config.cross_entropy_loss_fusion = enabled
            before = calls.copy()
            values = []
            for optimized in (True, False):
                torch.manual_seed(223)
                s = 16 if not sequence_parallel else 16 // world
                x = torch.randn(
                    s * 3, 2, 64, device="npu", dtype=torch.bfloat16, requires_grad=True
                )
                labels = torch.arange(32, device="npu").reshape(2, 16) * 13
                layer.weight.grad = None
                MTPLossLoggingHelper.tracker = {}
                hidden = process_mtp_loss(
                    x,
                    labels,
                    None,
                    layer,
                    None,
                    None,
                    True,
                    model.compute_language_model_loss,
                    config,
                    compute_output_loss=(model.compute_output_loss if optimized else None),
                )
                f = (
                    model.compute_output_loss
                    if optimized
                    else lambda *a: LanguageModule.compute_output_loss.__wrapped__(model, *a)
                )
                loss = f(hidden, layer, None, labels)
                loss.mean().backward()
                values.append((loss.detach(), x.grad, layer.weight.grad.clone()))
            assert calls["optimized"] - before["optimized"] == (3 if enabled else 0)
            assert calls["hooks"] - before["hooks"] == 6
            torch.testing.assert_close(values[0][0], values[1][0], rtol=1e-5, atol=2e-6)
            for actual, expected in zip(values[0][1:], values[1][1:]):
                assert (
                    actual.float() - expected.float()
                ).norm() / expected.float().norm().clamp_min(1e-12) < 0.01
    finally:
        handle.remove()
        MTPLossLoggingHelper.tracker = saved_tracker
