from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
from megatron.plugin import decorators
from megatron.training.arguments import _validate_optimizer_state_offload_args


@pytest.fixture
def select_backend(monkeypatch):
    monkeypatch.setattr(decorators, "_original_impl_cache", set())
    monkeypatch.setattr(decorators, "_plugin_impl_cache", {})
    monkeypatch.setattr(decorators, "_plugin_registry", {})
    return lambda vendor: monkeypatch.setenv("MG_FL_PREFER", vendor)


@pytest.mark.parametrize("vendor", ["npu", "cpu"])
def test_disabled_and_unsupported_backend(select_backend, vendor):
    select_backend(vendor)
    cfg = SimpleNamespace(chunked_optimizer_state_offload=False)
    owner = SimpleNamespace(config=cfg, optimizer=SimpleNamespace(step=lambda: 123))
    _validate_optimizer_state_offload_args(cfg)
    DistributedOptimizer._configure_optimizer_state_offload(owner)
    assert not hasattr(owner, "_chunked_state_offloader")
    assert MixedPrecisionOptimizer._step_inner_optimizer(owner) == 123
    if vendor == "cpu":
        cfg.chunked_optimizer_state_offload = True
        cfg.optimizer_state_offload_fraction = 1.0
        with pytest.raises(ValueError, match="not supported by this backend"):
            _validate_optimizer_state_offload_args(cfg)


@pytest.fixture
def ascend(select_backend):
    from megatron.plugin.platform import get_platform

    if get_platform().device_name() != "npu":
        pytest.skip("Requires Ascend NPU")
    select_backend("npu")


@pytest.mark.parametrize("mode", ["use_megatron_fsdp", "use_torch_fsdp2"])
def test_reject_fsdp(ascend, mode):
    args = SimpleNamespace(
        chunked_optimizer_state_offload=True, use_megatron_fsdp=False, use_torch_fsdp2=False
    )
    setattr(args, mode, True)
    with pytest.raises(ValueError, match="does not support FSDP"):
        _validate_optimizer_state_offload_args(args)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_fraction_and_adam_updates(ascend, fraction):
    from transformer_engine.pytorch.optimizers import FusedAdam

    # Four equal shards, each with >1 MiB of states, all in the same Adam group.
    initial = torch.linspace(-1, 1, 196608).to(device="npu", dtype=torch.bfloat16)
    params = [torch.nn.Parameter(initial.clone()) for _ in range(4)]
    refs = [torch.nn.Parameter(initial.clone()) for _ in params]
    options = dict(
        master_weights=True, store_param_remainders=True, use_decoupled_grad=True,
        exp_avg_dtype=torch.bfloat16, exp_avg_sq_dtype=torch.bfloat16,
    )
    actual, reference = FusedAdam(params, **options), FusedAdam(refs, **options)
    cfg = SimpleNamespace(
        chunked_optimizer_state_offload=True, optimizer_state_offload_fraction=fraction,
        optimizer_state_offload_chunk_size_mb=1, use_precision_aware_optimizer_no_fp8_or_ds_fp8=True,
    )
    owner = SimpleNamespace(config=cfg, optimizer=actual)
    DistributedOptimizer._configure_optimizer_state_offload(owner)
    assert hasattr(owner, "_chunked_state_offloader") == bool(fraction)
    for step in range(1, 5):
        for i, (p, ref) in enumerate(zip(params, refs)):
            grad = initial.float().mul((i + 1) * (-1) ** step).add_(step / 16)
            p.decoupled_grad, ref.decoupled_grad = grad.clone(), grad.clone()
        reference.step()
        MixedPrecisionOptimizer._step_inner_optimizer(owner)
        assert actual.param_groups[0]["step"] == reference.param_groups[0]["step"] == step
        for i, (p, ref) in enumerate(zip(params, refs)):
            torch.testing.assert_close(p.cpu(), ref.cpu(), rtol=0, atol=0)
            for name in ("exp_avg", "exp_avg_sq", "master_param"):
                state = actual.state[p][name]
                assert state.device.type == ("cpu" if i < len(params) * fraction else "npu")
                torch.testing.assert_close(state.cpu(), reference.state[ref][name].cpu(), rtol=0, atol=0)


def test_direct_core_rejects_megatron_fsdp(ascend):
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    param = torch.nn.Parameter(torch.ones(1))
    inner = torch.optim.AdamW([param])
    cfg = OptimizerConfig(bf16=True, chunked_optimizer_state_offload=True)
    model = SimpleNamespace(ddp_config=SimpleNamespace(use_megatron_fsdp=True))
    with pytest.raises(ValueError, match="does not support Megatron-FSDP"):
        DistributedOptimizer(
            optimizer=inner, config=cfg, grad_scaler=None, init_state_fn=None,
            model_chunks=[model], per_model_buffers={}, data_parallel_group=None,
            data_parallel_group_gloo=None, data_parallel_group_idx=0, distributed_optimizer_instance_id=0,
        )
