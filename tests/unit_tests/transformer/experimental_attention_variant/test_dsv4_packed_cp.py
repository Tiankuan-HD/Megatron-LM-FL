# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""DSv4 backend, packed-layout and CP parity; run with two, four or eight ranks."""

import json
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.extensions.transformer_engine import HAVE_TE, TELinear
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_utils,
    packed_layout,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    HAVE_HADAMARD,
    _build_attention,
    _make_config,
)

pytestmark = pytest.mark.experimental


@pytest.fixture(autouse=True)
def fresh_packed_compile_cache():
    """Isolate independent layouts without changing the production recompile limit."""
    # Keep compiled state shared between CP/reference or replay arms within one case.
    # Carrying every parameterized layout into the next case can exhaust Dynamo's limit.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires compiled CUDA metadata")
@pytest.mark.parametrize("cu_values", [(0, 128, 256), (0, 0, 128, 128, 256), (0, 96, 256)])
@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("torch_deterministic", [False, True])
def test_compiled_packed_metadata_matches_document_positions(cu_values, ratio, torch_deterministic):
    """Compiled metadata must preserve real document positions, including empty segments."""
    cu = torch.tensor(cu_values, dtype=torch.int32, device="cuda")
    capacity = 40 if ratio == 4 else 2
    halo = 8 if ratio == 4 else ratio
    prior = torch.are_deterministic_algorithms_enabled()
    prior_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(torch_deterministic)
    try:
        for start in (0, 128):
            hidden = torch.arange(start, start + 128, device="cuda").float().view(128, 1, 1)
            boundary = (
                torch.arange(start - 128, start, device="cuda").float().view(128, 1, 1)
                if start
                else hidden[:0]
            )
            args = (hidden, boundary, cu, start, ratio, halo, capacity, 2)
            expected = packed_layout._compact_compressor_input(*args)
            actual = packed_layout._compiled_compactor(*args)
            for reference, result in zip(expected, actual):
                assert torch.equal(reference, result)
            # An independent integer oracle also protects the eager reference.
            ids, positions, source_rows = [], [], []
            for begin, end in zip(cu_values, cu_values[1:]):
                first = (max(start - halo - begin, 0) + ratio - 1) // ratio
                stop = (min(end, start + 128) - begin) // ratio
                if begin < start + 128 and end > start:
                    for group in range(first, max(first, stop)):
                        ids.append(group)
                        positions.append(group * ratio)
                        assert 0 <= group * ratio < end - begin
                        source_rows.extend(
                            range(begin + group * ratio, begin + (group + 1) * ratio)
                        )
            valid_groups = len(ids)
            ids.extend([-1] * (capacity - valid_groups))
            positions.extend([0] * (capacity - valid_groups))
            source_rows.extend([0] * ((capacity - valid_groups) * ratio))
            assert actual[1].tolist() == ids
            assert actual[2].tolist() == positions
            assert actual[0].flatten().tolist() == source_rows
            # The other compiled bucketize consumer must retain every window row.
            index_args = (cu, start, 128, boundary.shape[0], 128, ratio, 0)
            index_expected = packed_layout._build_attention_indices(*index_args)
            index_actual = packed_layout._compiled_attention_indices(*index_args)
            for reference, result in zip(index_expected, index_actual):
                if reference is None:
                    assert result is None
                else:
                    assert torch.equal(reference, result)
            assert (index_actual[1] > 0).all()
    finally:
        torch.use_deterministic_algorithms(prior, warn_only=prior_warn)


class _CP1:
    @staticmethod
    def size():
        return 1

    @staticmethod
    def rank():
        return 0


def test_packed_deterministic_mode_is_rejected():
    """Do not silently run external nondeterministic backward for an explicit mode request."""
    cu = torch.tensor([0, 4], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4
    )
    config = SimpleNamespace(
        dsa_kernel_backend="cudnn", deterministic_mode=True, attention_cp_layout="contiguous"
    )
    with pytest.raises(NotImplementedError, match="does not yet support deterministic_mode"):
        cp_utils.validate_packed_inputs(packed, config, _CP1())
    config.deterministic_mode = False
    cp_utils.validate_packed_inputs(packed, config, _CP1())


def _similarities(actual, expected):
    a, b = actual.flatten().double(), expected.flatten().double()
    cosine = 1.0 if torch.equal(a, b) else torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    denominator = (a.square() + b.square()).sum()
    tensor_sim = (2 * (a * b).sum() / denominator).item() if denominator else 1.0
    return cosine, tensor_sim


def _assert_match(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    # Match dev's CP criteria: direction and magnitude, reduced in FP64.
    cosine, tensor_sim = _similarities(actual, expected)
    assert cosine > 0.999, f"cosine similarity {cosine:.8f} must exceed 0.999"
    assert tensor_sim > 0.999, f"tensor similarity {tensor_sim:.8f} must exceed 0.999"


def _forward_per_document(model, hidden, physical, real):
    """Run the configured SBHD backend without packed layout or CP helpers."""
    outputs = []
    for i, (start, end) in enumerate(zip(physical, physical[1:])):
        length = real[i + 1] - real[i]
        if length:
            output, _ = model(hidden[start : start + length], attention_mask=None)
            outputs.append(output)
        if start + length < end:
            # Keep padding gradients explicitly zero in the reference input.
            outputs.append(hidden[start + length : end] * 0)
    return torch.cat(outputs, dim=0)


def _run_attention(model, hidden, grad, packed, *, documents=None, cp_group=None):
    """Collect real TE FP32 wgrads, or ordinary grads when fusion is disabled."""
    fused_weights = set()
    for module in model.modules():
        if isinstance(module, TELinear) and module.fuse_wgrad_accumulation:
            module.weight.main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
            fused_weights.add(id(module.weight))
    hidden = hidden.detach().clone().requires_grad_()
    if documents is None:
        output, _ = model(hidden, attention_mask=None, packed_seq_params=packed)
        output.backward(grad)
    else:
        # TE's first-microbatch wgrad overwrites main_grad. Backpropagate each
        # independent document before the next forward, so the first document
        # initializes the buffer and subsequent documents accumulate into it.
        physical, real = documents
        outputs = []
        for i, (start, end) in enumerate(zip(physical, physical[1:])):
            length = real[i + 1] - real[i]
            if length:
                part, _ = model(hidden[start : start + length], attention_mask=None)
                part.backward(grad[start : start + length])
                outputs.append(part.detach())
            if start + length < end:
                outputs.append(hidden[start + length : end].detach() * 0)
        output = torch.cat(outputs, dim=0)
    result = {"output": output.detach(), "input_grad": hidden.grad.detach()}
    for name, param in model.named_parameters():
        value = param.main_grad if id(param) in fused_weights else param.grad
        assert value is not None, f"Missing gradient: {name}"
        value = value.detach().to(dtype=torch.float32, copy=True)
        if cp_group is not None:
            dist.all_reduce(value, group=cp_group)
        result[f"param:{name}"] = value
    return result


def _compare_results(actual, expected, label, failures, rows=None):
    """Report every tensor before failing either of dev's similarity gates."""
    assert actual.keys() == expected.keys(), label
    for name, value in actual.items():
        reference = expected[name]
        if rows is not None and not name.startswith("param:"):
            reference = reference[rows]
        a, b = value.float(), reference.float()
        delta = a - b
        energy = b.square().sum().clamp_min(1e-12)
        cosine, tensor_sim = _similarities(value, reference)
        stats = {
            "comparison": label,
            "tensor": name,
            "dtype": str(value.dtype),
            "reference_dtype": str(reference.dtype),
            "relative_l2": (delta.square().sum() / energy).sqrt().item(),
            "max_abs": delta.abs().max().item(),
            "mismatch_fraction": (delta.abs() > 0.06 + 0.06 * b.abs()).float().mean().item(),
            "reference_norm": b.norm().item(),
            "actual_norm": a.norm().item(),
            "cosine": cosine,
            "tensor_sim": tensor_sim,
        }
        try:
            _assert_match(value, reference)
        except AssertionError as error:
            failures.append(f"{label}/{name}: {error}")
            stats["passed"] = False
        else:
            stats["passed"] = True
        print("DSV4_CP_PARITY " + json.dumps(stats), flush=True)


@pytest.mark.flaky
@pytest.mark.flaky_in_dev
@pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TE and HAVE_HADAMARD),
    reason="needs CUDA, TE and the real Hadamard kernel",
)
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize(
    "ratio,sparse,coeff,recompute,rope_fusion",
    [
        (0, True, 0.0, False, True),
        (4, True, 0.2, True, True),
        (4, False, 0.0, False, True),
        (4, False, 0.2, False, True),
        (128, True, 0.0, True, True),
        (4, True, 0.2, False, False),
    ],
)
def test_packed_cp_matches_full_attention_and_gradients(
    cp_size, ratio, sparse, coeff, recompute, rope_fusion
):
    if Utils.world_size < cp_size:
        pytest.skip(f"requires {cp_size} ranks")
    pytest.importorskip("flash_mla")
    pytest.importorskip("cudnn.deepseek_sparse_attention")
    if torch.cuda.get_device_capability()[0] < 10 and ratio == 4 and not sparse and coeff > 0:
        pytest.skip("Dense fused indexer loss requires SM100; sparse H100 cases run normally")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp_size
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg = copy(pg)
        ref_pg.cp = _CP1()
        torch.manual_seed(198)
        model_parallel_cuda_manual_seed(198)
        # A rank boundary cuts a document and a ratio-4 group. Real lengths
        # exclude internal padding; a repeated prefix exercises empty documents.
        physical_cu = [0, 133, 133, 1157, 2048]
        real_cu = [0, 129, 129, 1141, 2016]
        cfg = _make_config(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=64,
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            q_lora_rank=128,
            csa_compress_ratios=[ratio],
            csa_window_size=128,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=16 if ratio == 4 and coeff > 0 else 512,
            dsa_indexer_loss_coeff=coeff,
            dsa_indexer_use_sparse_loss=sparse,
            dsa_kernel_backend="cudnn",
            context_parallel_size=cp_size,
            attention_cp_layout="contiguous",
            qk_layernorm=True,
            apply_rope_fusion=rope_fusion,
            gradient_accumulation_fusion=True,
            recompute_granularity="selective" if recompute else None,
            recompute_modules=["mla_up_proj"] if recompute else [],
        )
        # Isolate CP: keep the packed layout, backend and recompute settings identical.
        ref_cfg = replace(cfg, context_parallel_size=1)
        model = _build_attention(cfg, 1, pg).cuda()
        reference = _build_attention(ref_cfg, 1, ref_pg).cuda()
        reference.load_state_dict(model.state_dict())
        physical = torch.tensor(physical_cu, dtype=torch.int32, device="cuda")
        real = torch.tensor(real_cu, dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=real,
            cu_seqlens_kv=real,
            cu_seqlens_q_padded=physical,
            cu_seqlens_kv_padded=physical,
            max_seqlen_q=1024,
            max_seqlen_kv=1024,
        )
        whole = torch.randn(2048, 1, 256, dtype=torch.bfloat16, device="cuda")
        grad = torch.randn_like(whole)
        count = whole.shape[0] // cp_size
        rows = slice(pg.cp.rank() * count, (pg.cp.rank() + 1) * count)
        actual = _run_attention(model, whole[rows], grad[rows], packed, cp_group=pg.cp)
        expected = _run_attention(reference, whole, grad, packed)
        failures = []
        label = f"ratio={ratio}:cp={cp_size}:rope_fusion={rope_fusion}:fp32_wgrad"
        _compare_results(actual, expected, f"{label}:cp_vs_fused_cp1", failures, rows)
        if ratio == 4 and coeff == 0:
            for name, value in actual.items():
                if ".indexer." in name:
                    assert torch.count_nonzero(value) == 0, name

        # Check the preceding stages once per configuration, not again for CP4.
        if cp_size == 2:
            state = model.state_dict()

            def run_control(config, *, packed_params=packed, documents=None):
                attention = _build_attention(config, 1, ref_pg).cuda()
                attention.load_state_dict(state)
                return _run_attention(attention, whole, grad, packed_params, documents=documents)

            # The SBHD indexer requires S_q <= S_k * ratio. Use complete ratio-4
            # groups for this comparison, while retaining partial groups in the
            # original THD/CP case above. Both fixtures have 2016 real tokens.
            sbhd_real_cu = [0, 128, 128, 1140, 2016] if ratio == 4 else real_cu
            thd_reference = expected
            if ratio == 4:
                sbhd_real = real.new_tensor(sbhd_real_cu)
                sbhd_packed = replace(packed, cu_seqlens_q=sbhd_real, cu_seqlens_kv=sbhd_real)
                thd_reference = run_control(ref_cfg, packed_params=sbhd_packed)
            documents = (physical_cu, sbhd_real_cu)
            # Sum each document's auxiliary loss using the packed global real-token count.
            sbhd_cfg = replace(
                ref_cfg,
                calculate_per_token_loss=True,
                dsa_indexer_loss_coeff=coeff / sbhd_real_cu[-1],
            )
            native_sbhd = run_control(
                replace(sbhd_cfg, dsa_kernel_backend="none"), documents=documents
            )
            fused_sbhd = run_control(sbhd_cfg, documents=documents)
            _compare_results(fused_sbhd, native_sbhd, f"{label}:sbhd_fused_vs_native", failures)
            _compare_results(thd_reference, fused_sbhd, f"{label}:thd_cp1_vs_fused_sbhd", failures)
            if recompute:
                eager = run_control(
                    replace(ref_cfg, recompute_granularity=None, recompute_modules=[])
                )
                _compare_results(expected, eager, f"{label}:recompute_vs_eager", failures)
        assert not failures, "\n\n".join(failures)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TE and HAVE_HADAMARD),
    reason="requires CUDA, TE and Hadamard",
)
def test_packed_cp2_optimizer_steps_match_cp1():
    """Two real cuDNN training steps preserve replicated parameters and the CP1 update."""
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("requires a multiple of two ranks for independent CP2 groups")
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg = copy(pg)
        ref_pg.cp = _CP1()
        torch.manual_seed(410)
        model_parallel_cuda_manual_seed(410)
        cfg = _make_config(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=64,
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            q_lora_rank=128,
            csa_compress_ratios=[4],
            csa_window_size=16,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=512,
            dsa_indexer_loss_coeff=0.2,
            dsa_indexer_use_sparse_loss=True,
            dsa_kernel_backend="cudnn",
            context_parallel_size=2,
            attention_cp_layout="contiguous",
            qk_layernorm=True,
            apply_rope_fusion=True,
            gradient_accumulation_fusion=False,
        )
        model = _build_attention(cfg, 1, pg).cuda()
        reference = _build_attention(replace(cfg, context_parallel_size=1), 1, ref_pg).cuda()
        reference.load_state_dict(model.state_dict())
        actual_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        physical = torch.tensor([0, 67, 128, 256], dtype=torch.int32, device="cuda")
        real = torch.tensor([0, 65, 123, 250], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=real,
            cu_seqlens_kv=real,
            cu_seqlens_q_padded=physical,
            cu_seqlens_kv_padded=physical,
            max_seqlen_q=128,
            max_seqlen_kv=128,
        )
        rows = slice(pg.cp.rank() * 128, (pg.cp.rank() + 1) * 128)
        for step in range(2):
            whole = torch.randn(256, 1, 256, dtype=torch.bfloat16, device="cuda")
            actual_optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
            output, _ = model(whole[rows], attention_mask=None, packed_seq_params=packed)
            expected, _ = reference(whole, attention_mask=None, packed_seq_params=packed)
            actual_loss = output.float().square().sum() / expected.numel()
            expected_loss = expected.float().square().mean()
            assert torch.isfinite(actual_loss) and torch.isfinite(expected_loss)
            actual_loss.backward()
            expected_loss.backward()
            model.backward_dw()
            reference.backward_dw()
            for (name, param), (ref_name, ref_param) in zip(
                model.named_parameters(), reference.named_parameters()
            ):
                assert name == ref_name and param.grad is not None and ref_param.grad is not None
                dist.all_reduce(param.grad, group=pg.cp)
                _assert_match(param.grad, ref_param.grad)
            actual_optimizer.step()
            reference_optimizer.step()
            for (name, param), (_, ref_param) in zip(
                model.named_parameters(), reference.named_parameters()
            ):
                _assert_match(param, ref_param)
                replica = param.detach().clone()
                dist.broadcast(replica, src=dist.get_global_rank(pg.cp, 0), group=pg.cp)
                torch.testing.assert_close(param, replica, rtol=0, atol=0)
        assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "with_indexer,fused,ops",
    [
        (False, False, ["packed_csa_attention"]),
        (
            True,
            True,
            [
                "packed_csa_attention_with_loss",
                "packed_csa_indexer_topk",
                "hadamard_transform",
                "mla_interleaved_rope",
            ],
        ),
    ],
)
def test_packed_provider_checks_only_active_ops(monkeypatch, with_indexer, fused, ops):
    from unittest.mock import Mock

    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
    )
    from megatron.plugin.Ascend.transformer.experimental_attention_variant import csa

    layer = Mock(spec=CompressedSparseAttention)
    layer.training = True
    layer.indexer = object() if with_indexer else None
    config = SimpleNamespace(dsa_indexer_loss_coeff=0.2, apply_rope_fusion=fused)
    manager = Mock()
    monkeypatch.setattr(csa, "_manager", lambda: manager)
    csa.validate_backend(config, layer)
    assert [call.args[0] for call in manager.resolve.call_args_list] == ops
    manager.resolve.side_effect = RuntimeError("backend excluded by policy")
    with pytest.raises(RuntimeError, match=ops[0]) as error:
        csa.validate_backend(config, layer)
    assert "excluded by policy" in str(error.value.__cause__)


def test_dsv4_rope_default_forwards_original_call():
    from unittest.mock import Mock

    operation = Mock()
    inputs = (object(), object(), object(), 32, 64, object(), 3, 4)
    kwargs = dict(inverse=True, remove_interleaving=True, position_ids=object())
    result = cp_utils.apply_dsv4_rope.__wrapped__(operation, *inputs, **kwargs)
    operation.assert_called_once_with(*inputs, **kwargs)
    assert result is operation.return_value


def test_packed_backend_non_npu_preserves_upstream_operators():
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        packed_sparse_attention,
    )
    from megatron.plugin.Ascend.transformer.experimental_attention_variant import csa

    backend = csa.get_packed_csa_backend(torch.device("cuda"))
    assert backend is packed_sparse_attention
    assert backend.attention_with_loss == backend.FusedCSAIndexerSparseAttnFromTopkFunc.apply
    assert backend.topk_alignment is backend.get_flash_mla_topk_alignment
    from unittest.mock import Mock

    gathered = Mock()
    assert backend.gathered_for_loss(gathered) is gathered.detach.return_value
    gathered.detach.assert_called_once_with()


@pytest.mark.parametrize("device,cp_size", [("cuda", 4), ("npu", 4)])
def test_ascend_rope_delegates_implicit_sharded_positions(device, cp_size):
    from unittest.mock import Mock

    from megatron.plugin.Ascend.transformer.experimental_attention_variant import csa

    operation = Mock()
    t = SimpleNamespace(device=SimpleNamespace(type=device))
    inputs = (t, object(), object(), 32, 64, object(), 2, cp_size, False, True, True, None)
    result = csa.apply_dsv4_rope(operation, *inputs)
    operation.assert_called_once_with(*inputs)
    assert result is operation.return_value


@pytest.mark.parametrize("with_indexer", [False, True])
def test_packed_provider_accepts_builder_compatible_module(monkeypatch, with_indexer):
    from unittest.mock import Mock

    from megatron.plugin.Ascend.transformer.experimental_attention_variant import csa

    layer = SimpleNamespace(indexer=object() if with_indexer else None, training=True)
    config = SimpleNamespace(dsa_indexer_loss_coeff=0.2, apply_rope_fusion=False)
    manager = Mock()
    monkeypatch.setattr(csa, "_manager", lambda: manager)
    csa.validate_backend(config, layer)
    expected = (
        ["packed_csa_attention_with_loss", "packed_csa_indexer_topk", "hadamard_transform"]
        if with_indexer
        else ["packed_csa_attention"]
    )
    assert [call.args[0] for call in manager.resolve.call_args_list] == expected


def test_packed_validation_selects_compute_device_and_returns_backend(monkeypatch):
    from unittest.mock import Mock

    cu = torch.tensor([0, 4], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4
    )
    config = SimpleNamespace(deterministic_mode=False, attention_cp_layout="contiguous")
    attention = object()
    backend = Mock()
    select = Mock(return_value=backend)
    monkeypatch.setattr(cp_utils, "get_packed_csa_backend", select)
    compute_device = torch.device("cuda", 1)
    assert (
        cp_utils.validate_packed_inputs(packed, config, _CP1(), attention, compute_device)
        is backend
    )
    select.assert_called_once_with(compute_device)
    backend.validate_backend.assert_called_once_with(config, attention)


def test_ascend_gather_gradient_contract_uses_original_autograd_edge():
    from megatron.plugin.Ascend.transformer.experimental_attention_variant import csa

    tensor = object()
    assert csa.gathered_for_loss(tensor) is tensor
    assert csa.defer_reduce_scatter_wait(tensor, "consumer_wait") == (None, None)


def test_direct_packed_attention_preflights_before_execution(monkeypatch):
    from unittest.mock import Mock

    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
    )

    layer = SimpleNamespace(config=object(), pg_collection=SimpleNamespace(cp=_CP1()))
    query = SimpleNamespace(device=torch.device("cuda", 1))
    packed = object()
    validate = Mock(side_effect=RuntimeError("missing active provider"))
    monkeypatch.setattr(cp_utils, "validate_packed_inputs", validate)
    with pytest.raises(RuntimeError, match="missing active provider"):
        CompressedSparseAttention.forward(layer, query, None, None, None, packed_seq_params=packed)
    validate.assert_called_once_with(
        packed, layer.config, layer.pg_collection.cp, layer, query.device
    )


def test_cp_indexer_topk_reuses_selected_backend(monkeypatch):
    from unittest.mock import Mock

    q = torch.empty(2, 1, 4)
    k = torch.empty(1, 4)
    weights = torch.empty(2, 1)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 1], dtype=torch.int32)
    offsets = torch.tensor([0], dtype=torch.int32)
    backend = Mock()
    backend.indexer_topk.return_value = (object(), None)
    select = Mock(side_effect=AssertionError("backend was selected twice"))
    monkeypatch.setattr(cp_utils, "get_packed_csa_backend", select)
    monkeypatch.setattr(cp_utils, "build_cp_indexer_layout", lambda *args: (cu_q, cu_k, offsets))
    result, _ = cp_utils.compute_cp_indexer_topk(
        q, weights, k, cu_q, cu_k, 0, 4, 1, 1.0, 4, backend=backend
    )
    assert result is backend.indexer_topk.return_value[0]
    backend.indexer_topk.assert_called_once()
    select.assert_not_called()
