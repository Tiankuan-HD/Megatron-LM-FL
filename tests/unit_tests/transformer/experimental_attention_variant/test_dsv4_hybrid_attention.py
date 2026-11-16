# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False
    _hadamard_transform = None

_SEED = 42


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Patch hadamard_transform in dsa/csa modules if the library is not installed."""
    if not HAVE_HADAMARD:
        with (
            patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
                _mock_hadamard_transform,
            ),
            patch(
                'megatron.core.transformer.experimental_attention_variant.csa.rotate_activation',
                lambda x: x * (x.size(-1) ** -0.5),
            ),
        ):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Config / spec helpers
# ---------------------------------------------------------------------------


def _make_config(
    num_layers=4,
    hidden_size=256,
    num_attention_heads=16,
    v_head_dim=64,
    qk_pos_emb_head_dim=32,
    q_lora_rank=64,
    o_groups=8,
    o_lora_rank=64,
    csa_compress_ratios=None,
    csa_window_size=8,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    dsa_indexer_n_heads=8,
    dsa_indexer_head_dim=64,
    dsa_indexer_topk=8,
    dsa_indexer_loss_coeff=0.0,
    **extra_config_kwargs,
):
    """Create an MLATransformerConfig for DSv4 hybrid attention tests."""
    if csa_compress_ratios is None:
        csa_compress_ratios = [0, 4, 128, 4]
    return MLATransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=v_head_dim - qk_pos_emb_head_dim,
        qk_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        csa_compress_ratios=csa_compress_ratios,
        csa_window_size=csa_window_size,
        dsa_indexer_n_heads=dsa_indexer_n_heads,
        dsa_indexer_head_dim=dsa_indexer_head_dim,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        **extra_config_kwargs,
    )


def _make_attention_spec(config):
    """Build the full DSv4HybridSelfAttention ModuleSpec using the canonical spec builder."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_dsv4_hybrid_module_spec_for_backend,
    )

    return get_dsv4_hybrid_module_spec_for_backend(config=config, backend=TESpecProvider())


def _build_attention(config, layer_number, pg_collection):
    """Instantiate a DSv4HybridSelfAttention from config."""
    from megatron.core.transformer.spec_utils import build_module

    spec = _make_attention_spec(config)
    return build_module(spec, config=config, layer_number=layer_number, pg_collection=pg_collection)


# ===========================================================================
# Constructor tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionConstructor:
    """Test construction of DSv4HybridSelfAttention across TP sizes."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_basic_construction(self):
        """Verify the layer builds and has the expected sub-modules."""
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            DSv4HybridSelfAttention,
        )

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        assert isinstance(attn, DSv4HybridSelfAttention)
        assert hasattr(attn, 'linear_q_down_proj')
        assert hasattr(attn, 'linear_q_up_proj')
        assert hasattr(attn, 'linear_kv_proj')
        assert hasattr(attn, 'linear_proj')
        assert hasattr(attn, 'linear_o_group_proj')
        assert hasattr(attn, 'core_attention')
        assert hasattr(attn, 'q_layernorm')
        assert hasattr(attn, 'kv_layernorm')

    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_float16_wrapper_preserves_attention_fp32_parameters(self, compress_ratio):
        """Model conversion must preserve sink/APE values as well as their dtype."""
        from megatron.core.transformer.module import Float16Module

        model_parallel_cuda_manual_seed(_SEED)
        config = _make_config(num_layers=1, csa_compress_ratios=[compress_ratio])
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg).cuda()
        protected = {
            name: parameter
            for name, parameter in attn.named_parameters()
            if name.endswith("attn_sink") or name.endswith("ape")
        }
        assert len(protected) == {0: 1, 4: 3, 128: 2}[compress_ratio]
        with torch.no_grad():
            for parameter in protected.values():
                parameter.fill_(0.1234567)
        expected = {name: parameter.detach().clone() for name, parameter in protected.items()}
        wrapped = Float16Module(config, attn)
        for name, parameter in wrapped.module.named_parameters():
            if name in expected:
                assert parameter.dtype == torch.float32, name
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
            else:
                assert parameter.dtype == torch.bfloat16, name

    def test_q_head_dim_equals_v_head_dim(self):
        """q_head_dim must equal v_head_dim for DSv4 hybrid."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        assert attn.q_head_dim == config.v_head_dim

    @pytest.mark.parametrize("layer_number", [1, 2, 3, 4])
    def test_rope_base_varies_with_compress_ratio(self, layer_number):
        """Layers with compress_ratio > 1 should use csa_compress_rotary_base."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        ratios = [0, 4, 128, 4]
        config = _make_config(csa_compress_ratios=ratios)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=layer_number, pg_collection=pg)

        ratio = ratios[layer_number - 1]
        if ratio > 1:
            expected_base = config.csa_compress_rotary_base
        else:
            expected_base = config.rotary_base

        # inv_freq is derived from rotary_base; verify the correct base was used
        dim = config.qk_pos_emb_head_dim
        recomputed_inv_freq = 1.0 / (
            expected_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        assert torch.allclose(
            attn.rotary_pos_emb.inv_freq.cpu(), recomputed_inv_freq, rtol=1e-5, atol=1e-5
        )


# ===========================================================================
# Forward / backward tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionForwardBackward:
    """Test forward and backward passes of DSv4HybridSelfAttention."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_config(dsa_indexer_loss_coeff=1.0)
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_deferred_weight_gradients_match_eager(self, compress_ratio, monkeypatch):
        """Flushing delayed wgrads must include compressor and indexer projections."""
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            DSAIndexerLossAutoScaler,
            DSAIndexerLossLoggingHelper,
        )
        from megatron.core.transformer.module import Float16Module

        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {})
        monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)

        def build(delayed):
            config = _make_config(
                num_layers=1,
                csa_compress_ratios=[compress_ratio],
                dsa_indexer_loss_coeff=0.1,
                gradient_accumulation_fusion=False,
                hidden_dropout=0.0,
                attention_dropout=0.0,
            )
            # Exercise attention wgrad flushing in isolation, without an EP scheduler.
            config.delay_wgrad_compute = delayed
            return Float16Module(
                config, _build_attention(config, layer_number=1, pg_collection=self.pg).cuda()
            ).module

        eager = build(False)
        delayed = build(True)
        delayed.load_state_dict(eager.state_dict())
        hidden = torch.randn(256, 1, eager.config.hidden_size, device="cuda", dtype=torch.bfloat16)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        outputs = []
        input_grads = []
        for model in (eager, delayed):
            inputs = hidden.detach().clone().requires_grad_()
            output, _ = model(inputs, attention_mask=None)
            output.float().square().mean().backward()
            model.backward_dw()
            outputs.append(output.detach())
            input_grads.append(inputs.grad)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(input_grads[0], input_grads[1], rtol=0, atol=0)
        for (name, expected), (actual_name, actual) in zip(
            eager.named_parameters(), delayed.named_parameters()
        ):
            assert name == actual_name
            assert expected.grad is not None, name
            assert actual.grad is not None, name
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-2, atol=1e-5, msg=name)

    @pytest.mark.parametrize("layer_number", [1, 2, 3, 4])
    def test_forward_output_shape(self, layer_number):
        """Forward should produce [sq, b, hidden_size] output."""
        seq_len = 256
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()

        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        output, bias = attn(hidden_states=hidden, attention_mask=None)

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("layer_number", [1, 2])
    def test_backward_gradient_flow(self, layer_number):
        """Backward should produce gradients for all trainable parameters."""
        seq_len = 256
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.train()

        hidden = (
            torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        output, bias = attn(hidden_states=hidden, attention_mask=None)
        loss = output.sum()
        loss.backward()

        assert hidden.grad is not None, "No gradient on hidden_states"
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"

    def test_eval_mode(self):
        """Forward should work in eval mode."""
        seq_len = 128
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        attn.eval()

        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        with torch.no_grad():
            output, bias = attn(hidden_states=hidden, attention_mask=None)

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert not torch.isnan(output).any()

    def test_different_seq_lengths(self):
        """Forward should handle various sequence lengths."""
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=2, pg_collection=self.pg).cuda()

        for seq_len in [64, 128, 256]:
            hidden = torch.randn(
                seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
            ).cuda()
            output, bias = attn(hidden_states=hidden, attention_mask=None)
            assert output.shape == (seq_len, batch_size, self.config.hidden_size)


# ===========================================================================
# get_query_key_value_tensors tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridQKV:
    """Test get_query_key_value_tensors internals."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_qkv_shapes(self):
        """Query, key, value should have correct shapes."""
        seq_len = 64
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        q, k, v, q_compressed, kv_compressed = attn.get_query_key_value_tensors(hidden)

        n_heads = self.config.num_attention_heads
        v_dim = self.config.v_head_dim

        assert q.shape == (seq_len, batch_size, n_heads, v_dim)
        # key and value are single-head (MQA-style) with an extra head dim
        assert k.shape[-1] == v_dim
        assert v.shape[-1] == v_dim

    def test_key_equals_value(self):
        """In the wkv path, key and value should be the same tensor."""
        seq_len = 64
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        q, k, v, _, _ = attn.get_query_key_value_tensors(hidden)
        assert torch.equal(k, v), "key and value should be identical in wkv path"


# ===========================================================================
# Grouped output projection tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridGroupedOutput:
    """Test that grouped output projection (wo_a) parameters are created."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_o_group_proj_shape(self):
        """linear_o_group_proj should have the correct shape."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        o_groups = 8
        o_lora_rank = 64
        config = _make_config(o_groups=o_groups, o_lora_rank=o_lora_rank)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        expected_out = o_groups * o_lora_rank
        expected_in = (config.v_head_dim * config.num_attention_heads) // o_groups
        assert attn.linear_o_group_proj.shape == (expected_out, expected_in)
        assert attn.linear_o_group_proj.requires_grad


# ===========================================================================
# DSv4 Hybrid Attention + Hash MoE integration tests
# ===========================================================================


def _make_dsv4_hash_moe_config():
    """Create a compact DSv4 config that combines CSA/HCA with hash MoE."""
    return _make_config(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        v_head_dim=32,
        qk_pos_emb_head_dim=16,
        q_lora_rank=32,
        o_groups=4,
        o_lora_rank=32,
        csa_compress_ratios=[4, 128],
        csa_window_size=16,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
        ffn_hidden_size=256,
        num_moe_experts=4,
        moe_ffn_hidden_size=256,
        moe_layer_freq=1,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_router_score_function="sqrtsoftplus",
        moe_n_hash_layers=1,
        actual_vocab_size=128,
        activation_func=F.silu,
        gated_linear_unit=True,
        activation_func_clamp_value=10.0,
        bias_activation_fusion=False,
        moe_grouped_gemm=False,
    )


def _build_dsv4_moe_layer(config, layer_number, pg_collection):
    """Instantiate a TransformerLayer from the DSv4 experimental attention spec."""
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.transformer.spec_utils import build_module

    block_submodules = get_transformer_block_with_experimental_attention_variant_spec(config=config)
    return build_module(
        block_submodules.layer_specs[layer_number - 1],
        config=config,
        layer_number=layer_number,
        pg_collection=pg_collection,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridHashMoEIntegration:
    """Integration coverage for DSv4 hybrid attention with hash MoE and clamped SwiGLU."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_dsv4_hash_moe_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_csa_hash_moe_layer_forward_backward(self):
        """Layer 1 should combine DSv4 CSA, hash routing, and clamped SwiGLU."""
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            DSv4HybridSelfAttention,
        )
        from megatron.core.transformer.moe.moe_layer import MoELayer

        seq_len = 256
        batch_size = 1

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        layer = _build_dsv4_moe_layer(self.config, layer_number=1, pg_collection=self.pg).cuda()
        layer.train()

        assert isinstance(layer.self_attention, DSv4HybridSelfAttention)
        assert layer.self_attention.core_attention.compress_ratio == 4
        assert isinstance(layer.mlp, MoELayer)
        assert layer.mlp.router.is_hash_layer is True
        assert layer.mlp.router.tid2eid is not None
        assert layer.config.activation_func_clamp_value == 10.0
        assert layer.config.activation_func is F.silu
        assert layer.config.gated_linear_unit is True

        hidden = torch.randn(
            seq_len,
            batch_size,
            self.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        input_ids = torch.randint(
            0, self.config.actual_vocab_size, (batch_size, seq_len), device="cuda"
        )

        output, context = layer(hidden_states=hidden, attention_mask=None, input_ids=input_ids)
        loss = output.float().square().mean()
        loss.backward()

        assert context is None
        assert output.shape == hidden.shape
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.self_attention.parameters()
            if p.requires_grad
        )
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.mlp.parameters()
            if p.requires_grad
        )

    def test_hash_moe_layer_requires_input_ids_but_hca_layer_does_not(self):
        """Hash routing is limited to leading layers while later HCA MoE layers remain runnable."""
        seq_len = 256
        batch_size = 1

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        hash_layer = _build_dsv4_moe_layer(
            self.config, layer_number=1, pg_collection=self.pg
        ).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        with pytest.raises(AssertionError, match="input_ids is required for hash-based routing"):
            hash_layer(hidden_states=hidden, attention_mask=None)

        hca_layer = _build_dsv4_moe_layer(self.config, layer_number=2, pg_collection=self.pg).cuda()
        assert hca_layer.self_attention.core_attention.compress_ratio == 128
        assert hca_layer.mlp.router.is_hash_layer is False

        output, context = hca_layer(hidden_states=hidden, attention_mask=None)

        assert context is None
        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()


# ===========================================================================
# apply_rope_fusion tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridRopeFusion:
    """Test that apply_rope_fusion=True works for both yarn and non-yarn layers.

    DSv4 Hybrid uses YarnRotaryEmbedding for layers with compress_ratio > 1
    and standard RotaryEmbedding for layers with compress_ratio <= 1. The
    fused RoPE path must obtain cos/sin from both embedding classes via
    get_cached_cos_sin.

    compress_ratios=[0, 4, 128, 4]: layer 1 has ratio 0 (standard
    RotaryEmbedding), layers 2-4 have ratio > 1 (YarnRotaryEmbedding).
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_rope_fusion_forward_backward_parity(self):
        """Fused RoPE forward/backward succeeds and matches the unfused path."""
        seq_len = 128
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        fused_config = _make_config(apply_rope_fusion=True)
        attn_fused = _build_attention(fused_config, layer_number=4, pg_collection=self.pg).cuda()
        attn_fused.train()

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        unfused_config = _make_config(apply_rope_fusion=False)
        attn_unfused = _build_attention(
            unfused_config, layer_number=4, pg_collection=self.pg
        ).cuda()
        attn_unfused.train()

        hidden = torch.randn(
            seq_len, batch_size, fused_config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        out_fused, _ = attn_fused(hidden_states=hidden, attention_mask=None)
        out_unfused, _ = attn_unfused(hidden_states=hidden, attention_mask=None)

        assert out_fused.shape == (seq_len, batch_size, fused_config.hidden_size)
        assert torch.isfinite(out_fused).all()
        # Production code forces ``mscale=1.0`` (DSv4 contract) in both
        # fused and unfused paths, so the only residual is bf16 noise from
        # the fused Triton kernel's different accumulation order vs the
        # PyTorch eager ops. The residual concentrates at output positions
        # whose values are near zero (sign flips on tiny magnitudes drive
        # the worst-case max-abs-diff).
        torch.testing.assert_close(out_fused, out_unfused, atol=3e-2, rtol=3e-2)

        hidden_fused = hidden.detach().clone().requires_grad_(True)
        hidden_unfused = hidden.detach().clone().requires_grad_(True)

        attn_fused(hidden_states=hidden_fused, attention_mask=None)[0].sum().backward()
        attn_unfused(hidden_states=hidden_unfused, attention_mask=None)[0].sum().backward()

        assert hidden_fused.grad is not None
        for name, param in attn_fused.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"


pytestmark = pytest.mark.experimental


@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridNPUUpProjectionRecompute:
    """Exercise narrow Q-projection recomputation through the real Ascend attention path."""

    @pytest.fixture(autouse=True)
    def setup_npu(self, monkeypatch, patch_hadamard_if_needed):
        torch_npu = pytest.importorskip("torch_npu")
        if not torch_npu.npu.is_available():
            pytest.skip("NPU not available")
        from megatron.core.transformer.experimental_attention_variant import csa, dsa

        # Restore the native override hidden by this file's optional CUDA-library fixture.
        monkeypatch.setattr(csa, "rotate_activation", dsa.rotate_activation)
        monkeypatch.setattr(dsa.DSAIndexerLossLoggingHelper, "tracker", {})
        monkeypatch.setattr(dsa.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        self.pg = ProcessGroupCollection.use_mpu_process_groups()
        self.device = torch.device("npu", torch_npu.npu.current_device())
        dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device=self.device))
        yield
        Utils.destroy_model_parallel()

    def _build(self, ratio, dtype=torch.bfloat16, granularity="full", native=True):
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            DSv4HybridSelfAttention,
        )
        from megatron.core.transformer.module import Float16Module

        config = _make_config(
            num_layers=1,
            csa_compress_ratios=[ratio],
            dsa_indexer_head_dim=128,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
            apply_dsa_kernel_fusion=False,
            apply_rope_fusion=False,
            gradient_accumulation_fusion=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            recompute_granularity=granularity,
            recompute_method="uniform" if granularity == "full" else None,
            recompute_num_layers=1 if granularity == "full" else None,
            recompute_modules=["mla_up_proj"] if granularity == "selective" else [],
        )
        config.bf16 = dtype == torch.bfloat16
        config.fp16 = dtype == torch.float16
        config.params_dtype = dtype
        attention = _build_attention(config, layer_number=1, pg_collection=self.pg).to(self.device)
        if dtype in (torch.bfloat16, torch.float16):
            attention = Float16Module(config, attention).module
        if not native:
            # The binding must also bypass the plugin during outer checkpoint replay.
            original = DSv4HybridSelfAttention._q_up_proj_and_norm.__wrapped__
            attention._q_up_proj_and_norm = original.__get__(attention)
        return attention

    @staticmethod
    def _forward(attention, hidden, outer_checkpoint=False):
        from megatron.core import tensor_parallel

        def run(inputs):
            return attention(inputs, attention_mask=None)[0]

        return tensor_parallel.checkpoint(run, False, hidden) if outer_checkpoint else run(hidden)

    @pytest.mark.parametrize(
        "ratio,dtype,outer_checkpoint",
        [(ratio, torch.bfloat16, outer) for ratio in (0, 4, 128) for outer in (False, True)]
        + [(ratio, torch.float16, True) for ratio in (0, 128)],
    )
    def test_full_recompute_matches_core_after_parameter_updates(
        self, ratio, dtype, outer_checkpoint
    ):
        """Match all gradients for two updates, including full reentrant checkpoint replay."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        reference = self._build(ratio, dtype, native=False)
        native = self._build(ratio, dtype)
        native.load_state_dict(reference.state_dict())
        hidden = torch.randn(256, 1, reference.config.hidden_size, device=self.device, dtype=dtype)
        calls = {"reference": 0, "native": 0}

        def count(name):
            def hook(*unused):
                calls[name] += 1

            return hook

        handles = [
            reference.linear_q_up_proj.register_forward_hook(count("reference")),
            native.linear_q_up_proj.register_forward_hook(count("native")),
        ]
        try:
            for step in range(2):
                outputs, inputs = [], []
                for model in (reference, native):
                    model.zero_grad(set_to_none=True)
                    value = hidden.detach().clone().requires_grad_(True)
                    output = self._forward(model, value, outer_checkpoint)
                    output.float().square().mean().backward()
                    outputs.append(output.detach())
                    inputs.append(value.grad)
                    assert model.recompute_up_proj is False
                    assert model.qkv_up_checkpoint is None

                torch.testing.assert_close(
                    outputs[0], outputs[1], rtol=1e-2 if step else 0, atol=1e-5 if step else 0
                )
                torch.testing.assert_close(inputs[0], inputs[1], rtol=1e-2, atol=1e-5)
                for (name, expected), (actual_name, actual) in zip(
                    reference.named_parameters(), native.named_parameters()
                ):
                    assert name == actual_name
                    assert expected.grad is not None, name
                    assert actual.grad is not None, name
                    assert torch.isfinite(actual.grad).all(), name
                    torch.testing.assert_close(
                        actual.grad, expected.grad, rtol=1e-2, atol=1e-5, msg=name
                    )
                    with torch.no_grad():
                        expected.add_(expected.grad, alpha=-1e-3)
                        actual.add_(actual.grad, alpha=-1e-3)
        finally:
            for handle in handles:
                handle.remove()
        assert calls["reference"] == 2 * (2 if outer_checkpoint else 1)
        assert calls["native"] == 2 * (3 if outer_checkpoint else 2)

    @pytest.mark.parametrize("ratio", [0, 4, 128])
    def test_saved_query_teacher_and_duplicate_kv_storage(self, ratio, monkeypatch):
        """Omit saved raw Q while retaining output aliases through both backward branches."""
        from megatron.plugin.Ascend.transformer.experimental_attention_variant import csa

        # Record metadata only: keeping the projection tensor here would retain raw Q.
        for native in (False, True):
            attention = self._build(ratio, native=native)
            raw_storages, saved_storages = [], []

            def observe_projection(module, inputs, output):
                raw_storages.append(output[0].untyped_storage()._cdata)

            def observe_saved(tensor):
                saved_storages.append(tensor.untyped_storage()._cdata)
                return tensor

            handle = attention.linear_q_up_proj.register_forward_hook(observe_projection)
            q_compressed = torch.randn(
                256,
                1,
                attention.config.q_lora_rank,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            try:
                with torch.autograd.graph.saved_tensors_hooks(observe_saved, lambda tensor: tensor):
                    normalized = attention._q_up_proj_and_norm(q_compressed)
                assert len(raw_storages) == 1
                assert (raw_storages[0] in saved_storages) is not native
                del normalized
            finally:
                handle.remove()

        records = []
        get_qkv = attention.get_query_key_value_tensors
        indexer_forward = csa._SparseIndexerLoss.forward
        indexer_backward = csa._SparseIndexerLoss.backward
        attention_backward = csa._SparseAttention.backward

        def remember(record, name, tensor):
            storage = tensor.untyped_storage()
            assert storage.nbytes() > 0
            record[name] = tensor
            record["storage"][name] = (storage._cdata, storage.nbytes())
            record["values"][name] = tensor.detach().clone()

        def check_retained(record):
            for name, expected_storage in record["storage"].items():
                tensor = record[name]
                storage = tensor.untyped_storage()
                assert (storage._cdata, storage.nbytes()) == expected_storage
                torch.testing.assert_close(tensor, record["values"][name], rtol=0, atol=0)
            assert (
                record["key"].untyped_storage()._cdata == record["value"].untyped_storage()._cdata
            )
            if "teacher" in record:
                assert (
                    record["teacher"].untyped_storage()._cdata
                    == record["query"].untyped_storage()._cdata
                )

        def observe_qkv(*args, **kwargs):
            result = get_qkv(*args, **kwargs)
            if torch.is_grad_enabled():
                query, key, value = result[:3]
                assert key.untyped_storage()._cdata == value.untyped_storage()._cdata
                record = {"storage": {}, "values": {}}
                for name, tensor in (("query", query), ("key", key), ("value", value)):
                    remember(record, name, tensor)
                records.append(record)
            return result

        def observe_indexer_forward(ctx, *args):
            if records:
                teacher = args[3]
                assert (
                    teacher.untyped_storage()._cdata
                    == records[-1]["query"].untyped_storage()._cdata
                )
                remember(records[-1], "teacher", teacher)
            return indexer_forward(ctx, *args)

        def observe_indexer_backward(ctx, *args):
            check_retained(records[-1])
            assert (
                ctx.saved_tensors[3].untyped_storage()._cdata
                == records[-1]["query"].untyped_storage()._cdata
            )
            records[-1]["indexer_backward"] = True
            return indexer_backward(ctx, *args)

        def observe_attention_backward(ctx, *args):
            check_retained(records[-1])
            assert (
                ctx.saved_tensors[0].untyped_storage()._cdata
                == records[-1]["query"].untyped_storage()._cdata
            )
            records[-1]["attention_backward"] = True
            return attention_backward(ctx, *args)

        monkeypatch.setattr(attention, "get_query_key_value_tensors", observe_qkv)
        monkeypatch.setattr(
            csa._SparseIndexerLoss, "forward", staticmethod(observe_indexer_forward)
        )
        monkeypatch.setattr(
            csa._SparseIndexerLoss, "backward", staticmethod(observe_indexer_backward)
        )
        monkeypatch.setattr(
            csa._SparseAttention, "backward", staticmethod(observe_attention_backward)
        )
        hidden = torch.randn(
            256,
            1,
            attention.config.hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        self._forward(attention, hidden, outer_checkpoint=True).sum().backward()
        assert len(records) == 1
        check_retained(records[0])
        assert records[0]["attention_backward"]
        assert records[0].get("indexer_backward", False) == (ratio == 4)
        assert ("teacher" in records[0]) == (ratio == 4)
        assert torch.isfinite(hidden.grad).all()

    @pytest.mark.parametrize("ratio", [0, 4, 128])
    def test_existing_selective_up_projection_path(self, ratio):
        """Preserve the original selective path and its configured recompute flag."""
        attention = self._build(ratio, granularity="selective")
        hidden = torch.randn(
            256,
            1,
            attention.config.hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        calls = []
        handle = attention.linear_q_up_proj.register_forward_hook(lambda *unused: calls.append(1))
        try:
            self._forward(attention, hidden).sum().backward()
        finally:
            handle.remove()
        assert len(calls) == 2
        assert attention.recompute_up_proj is True
        assert attention.qkv_up_checkpoint is None
        assert torch.isfinite(hidden.grad).all()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in attention.parameters()
        )

    @pytest.mark.parametrize("case", ["fp32", "no_grad", "eval", "not_full"])
    def test_native_fallback_runs_original_attention(self, case, monkeypatch):
        """Unsupported runtime modes preserve the original forward without a local checkpoint."""
        from megatron.plugin.Ascend.transformer.experimental_attention_variant import (
            dsv4_hybrid_attention,
        )

        dtype = torch.float32 if case == "fp32" else torch.bfloat16
        granularity = None if case == "not_full" else "full"
        reference = self._build(0, dtype, granularity=granularity, native=False)
        attention = self._build(0, dtype, granularity=granularity)
        attention.load_state_dict(reference.state_dict())
        reference.train(case != "eval")
        attention.train(case != "eval")
        hidden = torch.randn(
            128,
            1,
            attention.config.hidden_size,
            device=self.device,
            dtype=dtype,
            requires_grad=True,
        )

        def unexpected_checkpoint(*args, **kwargs):
            pytest.fail("Unsupported attention mode entered the local Q checkpoint")

        monkeypatch.setattr(dsv4_hybrid_attention, "checkpoint", unexpected_checkpoint)
        calls = []
        handle = attention.linear_q_up_proj.register_forward_hook(lambda *unused: calls.append(1))
        try:
            with torch.set_grad_enabled(case != "no_grad"):
                expected = self._forward(reference, hidden)
                actual = self._forward(attention, hidden)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            handle.remove()
        assert len(calls) == 1
        assert attention.recompute_up_proj is False
        assert attention.qkv_up_checkpoint is None

    def test_frozen_compressed_query_preserves_projection_gradients(self):
        """Non-reentrant recomputation also trains Q-up weights with a frozen input."""
        reference = self._build(0, native=False)
        native = self._build(0)
        native.load_state_dict(reference.state_dict())
        q_compressed = torch.randn(
            128, 1, native.config.q_lora_rank, device=self.device, dtype=torch.bfloat16
        )
        calls = {"reference": 0, "native": 0}

        def count(name):
            def hook(*unused):
                calls[name] += 1

            return hook

        handles = [
            reference.linear_q_up_proj.register_forward_hook(count("reference")),
            native.linear_q_up_proj.register_forward_hook(count("native")),
        ]
        try:
            for step in range(2):
                outputs = []
                for model in (reference, native):
                    model.zero_grad(set_to_none=True)
                    output = model._q_up_proj_and_norm(q_compressed)
                    output.float().square().mean().backward()
                    outputs.append(output.detach())
                torch.testing.assert_close(
                    outputs[0], outputs[1], rtol=1e-2 if step else 0, atol=1e-5 if step else 0
                )
                for (name, expected), (actual_name, actual) in zip(
                    reference.linear_q_up_proj.named_parameters(),
                    native.linear_q_up_proj.named_parameters(),
                ):
                    assert name == actual_name
                    assert expected.grad is not None, name
                    assert actual.grad is not None, name
                    assert torch.isfinite(actual.grad).all(), name
                    torch.testing.assert_close(
                        actual.grad, expected.grad, rtol=1e-2, atol=1e-5, msg=name
                    )
                    with torch.no_grad():
                        expected.add_(expected.grad, alpha=-1e-3)
                        actual.add_(actual.grad, alpha=-1e-3)
        finally:
            for handle in handles:
                handle.remove()
        assert q_compressed.grad is None
        assert calls == {"reference": 2, "native": 4}

    def test_cpu_input_delegates_without_changing_recompute_flag(self, monkeypatch):
        """Check CPU delegation without requiring unsupported CPU execution of TE attention."""
        from types import SimpleNamespace

        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            DSv4HybridSelfAttention,
        )
        from megatron.plugin.Ascend.transformer.experimental_attention_variant.dsv4_hybrid_attention import (
            q_up_proj_and_norm,
        )

        attention = SimpleNamespace(config=SimpleNamespace(), recompute_up_proj=False)
        hidden = torch.randn(2, 1, 8, dtype=torch.bfloat16, requires_grad=True)
        observed = []

        def original(model, value):
            observed.append((model.recompute_up_proj, value.device.type))
            return value

        monkeypatch.setattr(DSv4HybridSelfAttention._q_up_proj_and_norm, "__wrapped__", original)
        assert q_up_proj_and_norm(attention, hidden) is hidden
        assert observed == [(False, "cpu")]
