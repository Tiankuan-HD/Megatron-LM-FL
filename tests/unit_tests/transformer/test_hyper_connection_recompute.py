# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for HyperConnection block-level recomputation.

Tests the following functionality:
1. HyperConnectionModule._forward_with_checkpoint correctness
2. HyperConnectionModule.apply_h_post with CheckpointWithoutOutputManager
3. Multiple HyperConnectionModules chained with a single CheckpointWithoutOutputManager
4. Partial checkpoint (last layer not checkpointed)
5. TransformerConfig 'mhc' in recompute_modules option
"""

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutputManager,
    checkpoint,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.plugin.platform import get_platform
from tests.unit_tests.test_utilities import Utils


class TestHyperConnectionModuleCheckpoint:
    """Test HyperConnectionModule checkpoint functionality."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_hyper_connection_module(self, hidden_size=64, num_residual_streams=4):
        """Create a HyperConnectionModule for testing."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_residual_streams,
            mhc_sinkhorn_iterations=5,  # Fewer iterations for faster tests
            mhc_init_gating_factor=0.01,
        )
        module = HyperConnectionModule(config=config, layer_number=1)
        module.cuda()
        return module

    def test_apply_h_res_uses_h_res_transpose(self):
        """apply_h_res should compute H_res.T @ residual."""
        module = self._create_hyper_connection_module(hidden_size=4, num_residual_streams=2)
        h_res = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device='cuda')
        residual = torch.tensor([[[10.0, 100.0, 3.0, 4.0, 1.0, 2.0, 5.0, 6.0]]], device='cuda')
        expected = torch.tensor(
            [[[13.0, 106.0, 18.0, 22.0, 24.0, 208.0, 26.0, 32.0]]], device='cuda'
        )

        mixed = module.apply_h_res(h_res, residual)

        torch.testing.assert_close(mixed, expected, atol=0.0, rtol=0.0)

    def test_forward_preserves_three_tuple_api_and_hybrid_can_request_residual(self):
        module = self._create_hyper_connection_module(hidden_size=8, num_residual_streams=2)
        hidden_states = torch.randn(4, 1, 16, device='cuda', requires_grad=True)

        compatible_output = module(hidden_states)
        hybrid_output = module(hidden_states, return_residual=True)

        assert len(compatible_output) == 3
        assert len(hybrid_output) == 4
        for compatible, hybrid in zip(compatible_output, hybrid_output[:3]):
            torch.testing.assert_close(compatible, hybrid)
        assert hybrid_output[3].shape == hidden_states.shape

    def test_forward_normal_vs_checkpoint_correctness(self):
        """
        Test that _forward_with_checkpoint produces the same outputs as _forward_normal.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        module = self._create_hyper_connection_module(hidden_size, num_streams)

        # Create input tensors
        hidden_states = torch.randn(
            seq_len, batch_size, num_streams * hidden_size, device='cuda', requires_grad=True
        )
        residual = torch.randn(
            seq_len, batch_size, num_streams * hidden_size, device='cuda', requires_grad=True
        )

        # Clone inputs for comparison
        hidden_states_ckpt = hidden_states.detach().clone().requires_grad_(True)
        residual_ckpt = residual.detach().clone().requires_grad_(True)

        # Forward without checkpoint (reference)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        aggregated_ref, h_res_ref, h_post_ref, residual_ref = module._forward_normal(hidden_states)
        mixed_ref = module.apply_h_res(h_res_ref, residual)
        loss_ref = aggregated_ref.sum() + mixed_ref.sum() + h_post_ref.sum()
        loss_ref.backward()
        grad_hidden_ref = hidden_states.grad.clone()
        grad_residual_ref = residual.grad.clone()

        # Forward with checkpoint
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointWithoutOutputManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt, residual_ckpt_out = (
            module._forward_with_checkpoint(hidden_states_ckpt, manager)
        )
        mixed_ckpt = module.apply_h_res(h_res_ckpt, residual_ckpt)
        # Calculate loss before discarding outputs
        loss_ckpt = aggregated_ckpt.sum() + mixed_ckpt.sum() + h_post_ckpt.sum()

        # Register unified recompute hook
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

        # Backward pass
        loss_ckpt.backward()
        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()
        grad_residual_ckpt = residual_ckpt.grad.clone()

        # Verify gradients match
        assert torch.allclose(grad_hidden_ckpt, grad_hidden_ref, atol=1e-5), (
            f"Hidden states gradients mismatch:\n"
            f"Checkpoint: {grad_hidden_ckpt}\n"
            f"Reference: {grad_hidden_ref}"
        )
        assert torch.allclose(grad_residual_ckpt, grad_residual_ref, atol=1e-5), (
            f"Residual gradients mismatch:\n"
            f"Checkpoint: {grad_residual_ckpt}\n"
            f"Reference: {grad_residual_ref}"
        )

    def test_apply_h_post_with_checkpoint(self):
        """
        Test that apply_h_post with manager produces correct gradients.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        module = self._create_hyper_connection_module(hidden_size, num_streams)

        # Create input tensors
        x = torch.randn(seq_len, batch_size, hidden_size, device='cuda', requires_grad=True)
        bias = torch.randn(hidden_size, device='cuda')
        h_post = torch.randn(seq_len, batch_size, num_streams, device='cuda', requires_grad=True)

        # Clone inputs
        x_ckpt = x.detach().clone().requires_grad_(True)
        h_post_ckpt = h_post.detach().clone().requires_grad_(True)

        # Reference: without checkpoint (manager=None)
        torch.manual_seed(42)
        x_out_ref, bias_out_ref = module.apply_h_post((x, bias), h_post, manager=None)
        loss_ref = x_out_ref.sum()
        if bias_out_ref is not None:
            loss_ref = loss_ref + bias_out_ref.sum()
        loss_ref.backward()
        grad_x_ref = x.grad.clone()
        grad_h_post_ref = h_post.grad.clone()

        # With checkpoint (manager provided)
        torch.manual_seed(42)
        manager = CheckpointWithoutOutputManager()
        x_out_ckpt, bias_out_ckpt = module.apply_h_post(
            (x_ckpt, bias), h_post_ckpt, manager=manager
        )
        loss_ckpt = x_out_ckpt.sum()
        if bias_out_ckpt is not None:
            loss_ckpt = loss_ckpt + bias_out_ckpt.sum()

        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)
        loss_ckpt.backward()
        grad_x_ckpt = x_ckpt.grad.clone()
        grad_h_post_ckpt = h_post_ckpt.grad.clone()

        # Verify gradients
        assert torch.allclose(grad_x_ckpt, grad_x_ref, atol=1e-5)
        assert torch.allclose(grad_h_post_ckpt, grad_h_post_ref, atol=1e-5)

    def test_forward_with_manager_parameter(self):
        """
        Test forward() method with mhc_recompute_manager parameter.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        module = self._create_hyper_connection_module(hidden_size, num_streams)

        # Create input tensors
        hidden_states = torch.randn(
            seq_len, batch_size, num_streams * hidden_size, device='cuda', requires_grad=True
        )

        # Clone inputs
        hidden_states_ckpt = hidden_states.detach().clone().requires_grad_(True)

        # Reference: forward without manager (uses _forward_normal)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        aggregated_ref, h_res_ref, h_post_ref = module.forward(
            hidden_states, mhc_recompute_manager=None
        )
        loss_ref = aggregated_ref.sum() + h_res_ref.sum() + h_post_ref.sum()
        loss_ref.backward()
        grad_hidden_ref = hidden_states.grad.clone()

        # With manager (uses _forward_with_checkpoint)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointWithoutOutputManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt = module.forward(
            hidden_states_ckpt, mhc_recompute_manager=manager
        )
        loss_ckpt = aggregated_ckpt.sum() + h_res_ckpt.sum() + h_post_ckpt.sum()

        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)
        loss_ckpt.backward()
        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()

        # Verify gradients match
        assert torch.allclose(grad_hidden_ckpt, grad_hidden_ref, atol=1e-5)


class TestMHCBlockRecomputeIntegration:
    """Test CheckpointWithoutOutputManager integration with HyperConnection."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_multiple_hyper_connections_in_chain(self):
        """
        Test that multiple HyperConnectionModules can be chained together
        with a single CheckpointWithoutOutputManager.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2
        n_channels = num_streams * hidden_size

        # Create multiple HyperConnection modules (simulating multiple layers)
        config = TransformerConfig(
            num_layers=4,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
        )

        modules = [
            HyperConnectionModule(config=config, layer_number=i + 1).cuda() for i in range(3)
        ]

        # Create input tensors
        hidden_states_ref = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        residual_ref = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )

        hidden_states_ckpt = hidden_states_ref.detach().clone().requires_grad_(True)
        residual_ckpt = residual_ref.detach().clone().requires_grad_(True)

        # Reference: forward without checkpoint
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        h = hidden_states_ref
        r = residual_ref
        for module in modules:
            agg, h_res, h_post = module.forward(h, mhc_recompute_manager=None)
            agg, _ = module.apply_h_post((0.1 * agg, None), h_post, manager=None)
            mixed = module.apply_h_res(h_res, r)  # Apply h_res to get mixed [s, b, n*C]
            h = agg + mixed
            r = h

        loss_ref = h.sum()
        loss_ref.backward()
        grad_hidden_ref = hidden_states_ref.grad.clone()
        grad_residual_ref = residual_ref.grad.clone()

        # With checkpoint using single manager
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        manager = CheckpointWithoutOutputManager()

        h = hidden_states_ckpt
        r = residual_ckpt
        for module in modules:
            agg, h_res, h_post = module.forward(h, mhc_recompute_manager=manager)
            agg, _ = module.apply_h_post((0.1 * agg, None), h_post, manager=manager)
            mixed = module.apply_h_res(h_res, r)  # Apply h_res to get mixed [s, b, n*C]
            h = agg + mixed
            r = h

        loss_ckpt = h.sum()
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)
        loss_ckpt.backward()

        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()
        grad_residual_ckpt = residual_ckpt.grad.clone()

        # Verify gradients
        assert torch.allclose(
            grad_hidden_ckpt, grad_hidden_ref, atol=1e-4
        ), f"Chained HyperConnection hidden gradients mismatch"
        assert torch.allclose(
            grad_residual_ckpt, grad_residual_ref, atol=1e-4
        ), f"Chained HyperConnection residual gradients mismatch"

    def test_partial_checkpoint_last_layer_not_checkpointed(self):
        """
        Test that when is_last_layer_in_block=True, the final output is NOT checkpointed.
        This simulates the TransformerBlock behavior where the last layer's MLP BDA
        serves as the hook_tensor for unified recompute.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
        )

        module = HyperConnectionModule(config=config, layer_number=1).cuda()

        hidden_states_ref = torch.randn(
            seq_len, batch_size, num_streams * hidden_size, device='cuda', requires_grad=True
        )
        residual_ref = torch.randn(
            seq_len, batch_size, num_streams * hidden_size, device='cuda', requires_grad=True
        )

        hidden_states_ckpt = hidden_states_ref.detach().clone().requires_grad_(True)
        residual_ckpt = residual_ref.detach().clone().requires_grad_(True)

        # Reference
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        aggregated_ref, h_res_ref, h_post_ref = module.forward(
            hidden_states_ref, mhc_recompute_manager=None
        )
        aggregated_ref, _ = module.apply_h_post(
            (0.1 * aggregated_ref, None), h_post_ref, manager=None
        )
        mixed_ref = module.apply_h_res(
            h_res_ref, residual_ref
        )  # Apply h_res to get mixed [s, b, n*C]
        # Simulate BDA that is NOT checkpointed (last layer)
        output_ref = aggregated_ref + 0.5 * mixed_ref
        loss_ref = output_ref.sum()
        loss_ref.backward()
        grad_hidden_ref = hidden_states_ref.grad.clone()

        # With manager - checkpoint everything except final output
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointWithoutOutputManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt = module.forward(
            hidden_states_ckpt, mhc_recompute_manager=manager
        )

        aggregated_ckpt, _ = module.apply_h_post(
            (0.1 * aggregated_ckpt, None), h_post_ckpt, manager=manager
        )
        mixed_ckpt = module.apply_h_res(
            h_res_ckpt, residual_ckpt
        )  # Apply h_res to get mixed [s, b, n*C]
        # Simulate BDA that is NOT checkpointed (last layer) - this is the hook_tensor
        output_ckpt = aggregated_ckpt + 0.5 * mixed_ckpt

        # Register unified recompute on the output (which is not checkpointed)
        manager.discard_all_outputs_and_register_unified_recompute(output_ckpt)

        loss_ckpt = output_ckpt.sum()
        loss_ckpt.backward()
        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()

        # Verify gradients match
        assert torch.allclose(grad_hidden_ckpt, grad_hidden_ref, atol=1e-5)


class TestAscendMHCMappingCheckpoint:
    """Check Core arithmetic, nested full recompute and saved storage on NPU."""

    @pytest.fixture(autouse=True)
    def initialize_parallel(self):
        if get_platform().device_name() != "npu":
            pytest.skip("Ascend NPU required")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def _module(self, hidden_size=64):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=4,
            mhc_sinkhorn_iterations=5,
        )
        return HyperConnectionModule(config, layer_number=1).to("npu")

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("nested_full", [False, True])
    @pytest.mark.parametrize("scale", [0.0, 1e-7, 1.0])
    @pytest.mark.parametrize("hidden_size", [64, 4096])
    def test_mapping_matches_core(self, dtype, nested_full, scale, hidden_size):
        module = self._module(hidden_size)
        x = torch.randn(16, 2, 4 * hidden_size, device="npu", dtype=dtype) * scale
        original = HyperConnectionModule._projection_and_get_norm.__wrapped__
        optimized = module._projection_and_get_norm
        coefficients = [
            torch.randn(16, 2, 4, device="npu"),
            torch.randn(16, 2, 4, device="npu"),
            torch.randn(16, 2, 4, 4, device="npu"),
        ]

        def run(projection):
            module.zero_grad(set_to_none=True)
            module._projection_and_get_norm = projection
            hidden = x.detach().clone().requires_grad_(True)
            outputs = (
                checkpoint(module.compute_mappings, False, hidden)
                if nested_full
                else module.compute_mappings(hidden)
            )
            loss = sum((out.float() * coef).sum() for out, coef in zip(outputs, coefficients))
            loss.backward()
            return (
                [out.detach().clone() for out in outputs],
                [hidden.grad.clone()] + [p.grad.clone() for p in module.parameters()],
            )

        # Repeat after a parameter update so the recompute closure cannot use stale weights.
        for _ in range(2):
            reference = run(original.__get__(module, type(module)))
            candidate = run(optimized)
            for expected, actual in zip(reference, candidate):
                for ref, opt in zip(expected, actual):
                    torch.testing.assert_close(opt, ref, atol=1e-5, rtol=1e-5)
            with torch.no_grad():
                for parameter in module.parameters():
                    parameter.add_(0.001)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_projection_does_not_save_fp32_input(self, dtype):
        module = self._module()
        x = torch.randn(16, 2, 256, device="npu", dtype=dtype, requires_grad=True)
        original = HyperConnectionModule._projection_and_get_norm.__wrapped__

        def saved_inputs(projection):
            records = []

            def pack(tensor):
                records.append((tensor.dtype, tensor.numel(), tensor.untyped_storage().data_ptr()))
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                outputs = projection(x)
            del outputs
            return records

        reference = saved_inputs(original.__get__(module, type(module)))
        candidate = saved_inputs(module._projection_and_get_norm)
        assert any(dtype == torch.float32 and size == x.numel() for dtype, size, _ in reference)
        assert not any(dtype == torch.float32 and size == x.numel() for dtype, size, _ in candidate)
        assert any(pointer == x.untyped_storage().data_ptr() for _, _, pointer in candidate)

    @pytest.mark.parametrize(
        "device,dtype,grad_enabled",
        [
            ("cpu", torch.bfloat16, True),
            ("npu", torch.float32, True),
            ("npu", torch.bfloat16, False),
        ],
    )
    def test_projection_fallback(self, device, dtype, grad_enabled):
        module = self._module().to(device)
        x = torch.randn(8, 2, 256, device=device, dtype=dtype, requires_grad=True)
        original = HyperConnectionModule._projection_and_get_norm.__wrapped__
        with torch.set_grad_enabled(grad_enabled):
            reference = original(module, x)
            candidate = module._projection_and_get_norm(x)
        for ref, opt in zip(reference, candidate):
            torch.testing.assert_close(opt, ref, atol=0, rtol=0)

    @pytest.mark.parametrize("nested_full", [False, True])
    @pytest.mark.parametrize("local_checkpoint", [False, True])
    def test_native_dispatch_and_recompute_count(self, nested_full, local_checkpoint, monkeypatch):
        from megatron.plugin.Ascend.transformer import hyper_connection as ascend_mhc

        module = self._module()
        assert not module.config.use_fused_mhc
        assert module._proj_rms_compute_h_op is None
        hidden = torch.randn(16, 2, 256, device="npu", dtype=torch.bfloat16, requires_grad=True)
        projection_grad_modes = []
        checkpoint_inputs = []
        native_projection = module._proj_rms_op
        pytorch_checkpoint = ascend_mhc.checkpoint

        def record_projection(*args):
            projection_grad_modes.append(torch.is_grad_enabled())
            return native_projection(*args)

        def record_checkpoint(function, owner, x, **kwargs):
            checkpoint_inputs.append((x.device.type, x.dtype, torch.is_grad_enabled()))
            return pytorch_checkpoint(function, owner, x, **kwargs)

        monkeypatch.setattr(module, "_proj_rms_op", record_projection)
        monkeypatch.setattr(ascend_mhc, "checkpoint", record_checkpoint)
        if not local_checkpoint:
            original = HyperConnectionModule._projection_and_get_norm.__wrapped__
            monkeypatch.setattr(
                module, "_projection_and_get_norm", original.__get__(module, type(module))
            )
        outputs = (
            checkpoint(module.compute_mappings, False, hidden)
            if nested_full
            else module.compute_mappings(hidden)
        )
        sum(out.float().square().sum() for out in outputs).backward()

        expected_modes = ([False] if nested_full else []) + [True]
        if local_checkpoint:
            expected_modes.append(True)
        assert projection_grad_modes == expected_modes
        assert checkpoint_inputs == ([("npu", torch.bfloat16, True)] if local_checkpoint else [])
        assert torch.isfinite(hidden.grad).all()
