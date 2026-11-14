# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
    _apply_rope,
    get_compress_topk_idxs,
    get_window_topk_idxs,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False
    _hadamard_transform = None


def mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Mock implementation of hadamard_transform for testing without the library installed."""
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Automatically patch hadamard_transform in both dsa and csa modules if not installed."""
    if not HAVE_HADAMARD:
        with (
            patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
                mock_hadamard_transform,
            ),
            patch(
                'megatron.core.transformer.experimental_attention_variant.csa.rotate_activation',
                lambda x: x * (x.size(-1) ** -0.5),
            ),
        ):
            yield
    else:
        yield


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestGetWindowTopkIdxs:
    """Test get_window_topk_idxs helper."""

    def test_basic_shape(self):
        batch_size, seqlen, window_size = 2, 16, 4
        idxs = get_window_topk_idxs(window_size, batch_size, seqlen, torch.device("cpu"))
        assert idxs.shape == (batch_size, seqlen, window_size)

    def test_causal_no_future(self):
        """Indices should never exceed the query position."""
        seqlen, window_size = 32, 8
        idxs = get_window_topk_idxs(window_size, 1, seqlen, torch.device("cpu"))
        for i in range(seqlen):
            valid = idxs[0, i][idxs[0, i] >= 0]
            assert torch.all(valid <= i), f"Position {i} has future indices"

    def test_invalid_marked_minus_one(self):
        """Early positions that cannot fill the window should use -1."""
        seqlen, window_size = 8, 4
        idxs = get_window_topk_idxs(window_size, 1, seqlen, torch.device("cpu"))
        assert idxs[0, 0, 0] == -1 or idxs[0, 0, 0] == 0
        for pos in range(window_size, seqlen):
            assert torch.all(idxs[0, pos] >= 0), f"Position {pos} has invalid -1"

    def test_window_larger_than_seqlen(self):
        """Window larger than sequence length should still work."""
        seqlen, window_size = 4, 16
        idxs = get_window_topk_idxs(window_size, 1, seqlen, torch.device("cpu"))
        assert idxs.shape == (1, seqlen, window_size)


class TestGetCompressTopkIdxs:
    """Test get_compress_topk_idxs helper."""

    def test_basic_shape(self):
        ratio, batch_size, seqlen, offset = 4, 2, 32, 32
        idxs = get_compress_topk_idxs(ratio, batch_size, seqlen, offset, torch.device("cpu"))
        n_compressed = seqlen // ratio
        assert idxs.shape == (batch_size, seqlen, n_compressed)

    def test_offset_applied(self):
        """Valid indices should be >= offset."""
        ratio, seqlen, offset = 4, 32, 100
        idxs = get_compress_topk_idxs(ratio, 1, seqlen, offset, torch.device("cpu"))
        valid = idxs[idxs >= 0]
        if valid.numel() > 0:
            assert torch.all(valid >= offset), "Valid indices should be offset"

    def test_causal_no_future(self):
        """Compressed indices should respect causality."""
        ratio, seqlen, offset = 4, 32, 32
        idxs = get_compress_topk_idxs(ratio, 1, seqlen, offset, torch.device("cpu"))
        for i in range(seqlen):
            n_valid = (i + 1) // ratio
            valid = idxs[0, i][idxs[0, i] >= 0]
            assert valid.numel() <= n_valid, f"Position {i} has too many valid compressed indices"

    def test_ratio_128(self):
        """Test with large compression ratio."""
        ratio, seqlen, offset = 128, 256, 256
        idxs = get_compress_topk_idxs(ratio, 1, seqlen, offset, torch.device("cpu"))
        assert idxs.shape == (1, seqlen, seqlen // ratio)


# ===========================================================================
# unfused_compressed_sparse_attn tests
# ===========================================================================


class TestUnfusedCompressedSparseAttn:
    """Test the unfused compressed sparse attention kernel."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_output_shape(self):
        """Test output shape of unfused compressed sparse attention."""
        sq, b, np_, hn = 16, 2, 4, 64
        n_kv = sq + sq // 4
        topk = 8

        query = torch.randn(sq, b, np_, hn, dtype=torch.bfloat16).cuda()
        kv_full = torch.randn(n_kv, b, hn, dtype=torch.bfloat16).cuda()
        attn_sink = torch.zeros(np_, dtype=torch.float32).cuda()
        topk_indices = torch.randint(0, n_kv, (b, sq, topk), dtype=torch.int32).cuda()
        softmax_scale = hn**-0.5

        output = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_indices, softmax_scale
        )

        assert output.shape == (sq, b, np_ * hn)
        assert output.dtype == query.dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_invalid_indices_masked(self):
        """Test that -1 indices are properly masked."""
        sq, b, np_, hn = 8, 1, 2, 32
        n_kv = sq
        topk = 4

        query = torch.randn(sq, b, np_, hn, dtype=torch.bfloat16).cuda()
        kv_full = torch.randn(n_kv, b, hn, dtype=torch.bfloat16).cuda()
        attn_sink = torch.zeros(np_, dtype=torch.float32).cuda()

        topk_indices = torch.full((b, sq, topk), -1, dtype=torch.int32).cuda()
        topk_indices[:, :, 0] = 0
        softmax_scale = hn**-0.5

        output = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_indices, softmax_scale
        )
        assert not torch.isnan(output).any(), "Output should not contain NaN"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradient_flow(self):
        """Test that gradients flow through sparse attention."""
        sq, b, np_, hn = 8, 1, 2, 32
        n_kv = sq
        topk = 4

        query = torch.randn(sq, b, np_, hn, dtype=torch.float32).cuda().requires_grad_(True)
        kv_full = torch.randn(n_kv, b, hn, dtype=torch.float32).cuda().requires_grad_(True)
        attn_sink = torch.nn.Parameter(torch.zeros(np_, dtype=torch.float32).cuda())

        topk_indices = torch.randint(0, n_kv, (b, sq, topk), dtype=torch.int32).cuda()
        softmax_scale = hn**-0.5

        output = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_indices, softmax_scale
        )
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert kv_full.grad is not None
        assert attn_sink.grad is not None


# ===========================================================================
# Compressor tests
# ===========================================================================


def _make_mla_config(
    num_layers=4,
    hidden_size=256,
    num_attention_heads=16,
    v_head_dim=64,
    qk_pos_emb_head_dim=32,
    csa_compress_ratios=None,
    csa_window_size=8,
    csa_dense_mode=False,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    dsa_indexer_n_heads=8,
    dsa_indexer_head_dim=64,
    dsa_indexer_topk=8,
    dsa_indexer_loss_coeff=0.0,
    dsa_indexer_use_sparse_loss=False,
):
    """Helper to create MLATransformerConfig for CSA tests."""
    if csa_compress_ratios is None:
        csa_compress_ratios = [0] * num_layers
    return MLATransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        csa_compress_ratios=csa_compress_ratios,
        csa_window_size=csa_window_size,
        csa_dense_mode=csa_dense_mode,
        dsa_indexer_n_heads=dsa_indexer_n_heads,
        dsa_indexer_head_dim=dsa_indexer_head_dim,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
    )


def _make_compressor_submodules():
    """Create Compressor submodules spec."""
    from megatron.core.extensions.transformer_engine import TELinear, TENorm
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CompressorSubmodules(
        linear_wkv=ModuleSpec(module=TELinear),
        linear_wgate=ModuleSpec(module=TELinear),
        norm=ModuleSpec(module=TENorm),
    )


def _make_csa_indexer_submodules():
    """Create CSAIndexer submodules spec."""
    from megatron.core.extensions.transformer_engine import TELinear, TENorm
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CSAIndexerSubmodules(
        linear_wq_b=ModuleSpec(module=TELinear),
        linear_weights_proj=ModuleSpec(module=TELinear),
        compressor=ModuleSpec(module=Compressor, submodules=_make_compressor_submodules()),
    )


def _make_csa_submodules():
    """Create CompressedSparseAttention submodules spec."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CompressedSparseAttentionSubmodules(
        compressor=ModuleSpec(module=Compressor, submodules=_make_compressor_submodules()),
        indexer=ModuleSpec(module=CSAIndexer, submodules=_make_csa_indexer_submodules()),
    )


# ===========================================================================
# Compressor tests
# ===========================================================================


@pytest.mark.parametrize("compress_ratio", [4, 128])
class TestCompressor:
    """Test Compressor module."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(csa_compress_ratios=[4, 128, 4, 128])
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compressor_output_shape(self, compress_ratio):
        """Test that compressor produces correct output shape."""
        seq_len = 256
        batch_size = 2
        head_dim = self.config.v_head_dim

        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        output = compressor(x)

        expected_len = seq_len // compress_ratio
        assert output is not None
        assert output.shape == (expected_len, batch_size, head_dim)
        assert output.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compressor_too_short_input(self, compress_ratio):
        """Test that compressor returns None when input is shorter than compress_ratio."""
        short_len = compress_ratio - 1
        batch_size = 2
        head_dim = self.config.v_head_dim

        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

        x = torch.randn(short_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        output = compressor(x)
        assert output is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compressor_gradient_flow(self, compress_ratio):
        """Test that gradients flow through the compressor."""
        seq_len = 256
        batch_size = 2
        head_dim = self.config.v_head_dim

        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

        x = (
            torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        output = compressor(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in compressor.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"


# ===========================================================================
# CSAIndexer tests
# ===========================================================================


@pytest.mark.parametrize("seqlen", [32, 128])
class TestCSAIndexer:
    """Test CSAIndexer module basic functionality."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.compress_ratio = 4
        cls.config = _make_mla_config(csa_compress_ratios=[4, 4, 4, 4], dsa_indexer_topk=8)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        cls.indexer = CSAIndexer(
            config=cls.config,
            submodules=_make_csa_indexer_submodules(),
            compress_ratio=cls.compress_ratio,
            rotary_pos_emb=cls.rotary_pos_emb,
            pg_collection=cls.pg_collection,
        )

        yield
        Utils.destroy_model_parallel()

    def test_csa_indexer_constructor(self, seqlen):
        """Test CSAIndexer initialization."""
        assert isinstance(self.indexer, CSAIndexer)
        assert self.indexer.compress_ratio == self.compress_ratio
        assert self.indexer.index_n_heads == self.config.dsa_indexer_n_heads
        assert self.indexer.index_head_dim == self.config.dsa_indexer_head_dim
        assert self.indexer.index_topk == self.config.dsa_indexer_topk

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_csa_indexer_forward(self, seqlen):
        """Test CSAIndexer forward pass."""
        batch_size = 2
        self.indexer.cuda()

        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        index_scores, topk_indices = self.indexer(x, qr)
        n_compressed = seqlen // self.compress_ratio
        effective_topk = min(self.config.dsa_indexer_topk, n_compressed)

        assert index_scores.shape == (batch_size, seqlen, n_compressed)
        assert topk_indices.shape == (batch_size, seqlen, effective_topk)
        assert index_scores.dtype == torch.float32
        assert topk_indices.dtype == torch.long

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_csa_indexer_forward_before_topk(self, seqlen):
        """Test CSAIndexer forward_before_topk."""
        batch_size = 2
        self.indexer.cuda()

        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        q, k, weights = self.indexer.forward_before_topk(x, qr)

        assert q.shape == (
            seqlen,
            batch_size,
            self.config.dsa_indexer_n_heads,
            self.config.dsa_indexer_head_dim,
        )
        n_compressed = seqlen // self.compress_ratio
        assert k.shape == (n_compressed, batch_size, self.config.dsa_indexer_head_dim)
        assert weights.shape == (seqlen, batch_size, self.config.dsa_indexer_n_heads)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_csa_indexer_with_mask(self, seqlen):
        """Test CSAIndexer with causal mask."""
        batch_size = 2
        self.indexer.cuda()

        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        n_compressed = seqlen // self.compress_ratio
        causal_mask = torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(seqlen, -1)
        positions = torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1)
        causal_mask = (
            torch.where(causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        index_scores, topk_indices = self.indexer(x, qr, mask=causal_mask)

        effective_topk = min(self.config.dsa_indexer_topk, n_compressed)
        assert index_scores.shape == (batch_size, seqlen, n_compressed)
        assert topk_indices.shape == (batch_size, seqlen, effective_topk)


# ===========================================================================
# CompressedSparseAttention tests
# ===========================================================================


class TestCompressedSparseAttentionRatio1:
    """Test CompressedSparseAttention with compress_ratio=1 (window-only)."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(csa_compress_ratios=[0, 0, 0, 0], csa_window_size=8)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        cls.csa = CompressedSparseAttention(
            config=cls.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=cls.pg_collection,
            rotary_pos_emb=rotary_pos_emb,
            compress_ratio=0,
        )

        yield
        Utils.destroy_model_parallel()

    def test_ratio1_no_compressor(self):
        """With ratio=1, compressor and indexer should not be built."""
        assert self.csa.compressor is None
        assert self.csa.indexer is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ratio1_forward(self):
        """Test forward pass with window-only attention."""
        seq_len = 32
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        self.csa.cuda()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = self.csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert output.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ratio1_backward(self):
        """Test backward pass with window-only attention."""
        seq_len = 32
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        self.csa.train()
        self.csa.cuda()

        query = (
            torch.randn(seq_len, batch_size, np_, hn, dtype=torch.float32)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, 1, hn, dtype=torch.float32).cuda().requires_grad_(True)
        )
        value = key.clone().detach().requires_grad_(True)
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = self.csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None


@pytest.mark.parametrize("compress_ratio", [4, 128])
class TestCompressedSparseAttentionCompressed:
    """Test CompressedSparseAttention with compress_ratio > 1."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(
            csa_compress_ratios=[4, 128, 4, 128],
            csa_window_size=8,
            dsa_indexer_topk=8,
            dsa_indexer_loss_coeff=1.0,
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    def _get_layer_number(self, compress_ratio):
        """Return a layer_number (1-indexed) whose compress_ratio matches."""
        for i, r in enumerate(self.config.csa_compress_ratios):
            if r == compress_ratio:
                return i + 1
        raise ValueError(f"No layer with compress_ratio={compress_ratio}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self, compress_ratio):
        """Test that compressor/indexer are conditionally built."""
        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()

        assert csa.compressor is not None
        if compress_ratio == 4:
            assert csa.indexer is not None
        elif compress_ratio == 128:
            assert csa.indexer is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward(self, compress_ratio):
        """Test forward pass with compressed attention."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward(self, compress_ratio):
        """Test backward pass with compressed attention."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()
        csa.train()

        query = (
            torch.randn(seq_len, batch_size, np_, hn, dtype=torch.float32)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, 1, hn, dtype=torch.float32).cuda().requires_grad_(True)
        )
        value = key.clone().detach().requires_grad_(True)
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None

        for name, param in csa.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_eval_mode(self, compress_ratio):
        """Test forward pass in eval mode."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()
        csa.eval()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        with torch.no_grad():
            output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert not torch.isnan(output).any()


# ===========================================================================
# csa_dense_mode tests
# ===========================================================================


class TestCompressedSparseAttentionDenseMode:
    """Test that csa_dense_mode=True disables the indexer for ratio=4 layers."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(
            csa_compress_ratios=[4, 128, 4, 128], csa_window_size=8, csa_dense_mode=True
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dense_mode_disables_indexer_for_ratio4(self):
        """With csa_dense_mode=True, ratio=4 layers should NOT build an indexer."""
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=4,
        ).cuda()

        assert csa.compress_ratio == 4
        assert csa.compressor is not None, "Compressor should still be built"
        assert csa.indexer is None, "Indexer should be disabled in dense mode"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dense_mode_forward_ratio4(self):
        """Forward pass should work for ratio=4 in dense mode (uses all compressed positions)."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=4,
        ).cuda()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert not torch.isnan(output).any()


# ===========================================================================
# _apply_rope tests
# ===========================================================================


class TestApplyRope:
    """Test ``_apply_rope`` — the layout-aware RoPE wrapper used by
    Compressor / CSAIndexer / DSv4HybridAttention.

    Behaviours covered:

    * 3-D ``[seq, batch, head_dim]`` and 4-D ``[seq, batch, heads, head_dim]``
      inputs both work (3-D gets a temporary head-dim unsqueeze).
    * Only the trailing ``pos_dim`` components are rotated; the leading
      ``nope_dim`` slice is bit-exact unchanged.
    * Both ``RotaryEmbedding`` (returns ``Tensor``) and
      ``YarnRotaryEmbedding`` (returns ``(emb, mscale)`` tuple) — DSv4
      hybrid silently swaps the class based on ``compress_ratio``.
    * Both unfused and fused (``config.apply_rope_fusion=True``) paths
      produce the same output (within bf16 precision).
    * For ``ratio > 1`` the rotary table is built at
      ``rotary_seq_len * ratio`` and strided by ``ratio``.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(0)
        model_parallel_cuda_manual_seed(0)
        cls = request.cls
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        # head_dim 32 = nope 24 + pos 8
        cls.config = _make_mla_config(v_head_dim=32, qk_pos_emb_head_dim=8)
        yield
        Utils.destroy_model_parallel()

    def _make_rotary(self, kind: str):
        from megatron.core.models.common.embeddings import RotaryEmbedding, YarnRotaryEmbedding

        pos_dim = self.config.qk_pos_emb_head_dim
        if kind == 'rope':
            return RotaryEmbedding(
                pos_dim, rotary_percent=1.0, rotary_base=10000, cp_group=self.pg_collection.cp
            )
        if kind == 'yarn':
            return YarnRotaryEmbedding(
                pos_dim,
                rotary_base=40000,
                scaling_factor=40,
                original_max_position_embeddings=4096,
                beta_fast=32,
                beta_slow=1,
                mscale=1.0,
                mscale_all_dim=0.0,
                cp_group=self.pg_collection.cp,
            )
        raise ValueError(kind)

    def _config_with(self, *, apply_rope_fusion: bool):
        # Reuse the class-level config; only flip the fusion flag.
        cfg = self.config
        cfg.apply_rope_fusion = apply_rope_fusion
        return cfg

    _ROTARY_FUSION_COMBOS = [
        pytest.param('rope', False, id='rope-unfused'),
        pytest.param('rope', True, id='rope-fused'),
        pytest.param('yarn', False, id='yarn-unfused'),
        pytest.param('yarn', True, id='yarn-fused'),
    ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(("rotary_kind", "apply_rope_fusion"), _ROTARY_FUSION_COMBOS)
    @pytest.mark.parametrize("input_ndim", [3, 4], ids=['3d', '4d'])
    @pytest.mark.parametrize("ratio", [1, 4], ids=['ratio_1', 'ratio_4'])
    def test_apply_rope(self, rotary_kind, apply_rope_fusion, input_ndim, ratio):
        """Output shape == input shape; no NaN; nope-dim slice is
        bit-exact unchanged. Sweeps the valid combinations of rotary
        class × apply_rope_fusion × input rank × ratio. Yarn's
        tuple-return is covered by the ``'yarn-*'`` combos.
        """
        rotary = self._make_rotary(rotary_kind).cuda()
        nope = self.config.v_head_dim - self.config.qk_pos_emb_head_dim
        pos = self.config.qk_pos_emb_head_dim
        head_dim = nope + pos
        seq, batch, heads = 8, 2, 4
        cfg = self._config_with(apply_rope_fusion=apply_rope_fusion)

        shape = (seq, batch, head_dim) if input_ndim == 3 else (seq, batch, heads, head_dim)
        x = torch.randn(*shape, dtype=torch.bfloat16, device='cuda')
        # ``fused_mla_rope_inplace`` mutates the input — give it a copy so
        # the nope-dim equality check below still has the original.
        out = _apply_rope(
            x.clone(),
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert not torch.isnan(out).any()
        # The leading nope_dim slice is the identity portion of RoPE.
        assert torch.equal(
            out[..., :nope], x[..., :nope]
        ), "RoPE must not touch the first nope_dim components"
        # Trailing pos_dim should rotate at non-zero positions.
        pe_changed = (out[..., nope:] != x[..., nope:]).any(dim=-1).flatten()
        assert pe_changed[
            1:
        ].any(), "RoPE should rotate the trailing pos_dim components for seq > 0"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("rotary_kind", ['rope', 'yarn'])
    def test_3d_input_matches_4d_with_single_head(self, rotary_kind):
        """For a single-head input, the 3-D ``(s, b, d)`` and 4-D
        ``(s, b, 1, d)`` invocations must produce numerically identical
        output (3-D path just inserts a temporary head dim).
        """
        rotary = self._make_rotary(rotary_kind).cuda()
        nope = self.config.v_head_dim - self.config.qk_pos_emb_head_dim
        pos = self.config.qk_pos_emb_head_dim
        head_dim = nope + pos
        seq, batch = 8, 2
        cfg = self._config_with(apply_rope_fusion=False)

        x_3d = torch.randn(seq, batch, head_dim, dtype=torch.bfloat16, device='cuda')
        x_4d = x_3d.unsqueeze(-2)

        out_3d = _apply_rope(
            x_3d,
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )
        out_4d = _apply_rope(
            x_4d,
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )

        assert out_3d.shape == x_3d.shape
        assert out_4d.shape == x_4d.shape
        assert torch.equal(out_3d, out_4d.squeeze(-2))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("rotary_kind", ['rope', 'yarn'])
    def test_ratio_strides_rotary_table(self, rotary_kind):
        """For ``ratio > 1``, the rotary table is built at
        ``rotary_seq_len * ratio`` and strided by ``ratio``. The result
        with ``ratio=k`` must equal an ``apply_rope`` call on the same
        positions of a length-``rotary_seq_len * k`` table.
        """
        rotary = self._make_rotary(rotary_kind).cuda()
        nope = self.config.v_head_dim - self.config.qk_pos_emb_head_dim
        pos = self.config.qk_pos_emb_head_dim
        head_dim = nope + pos
        seq, batch, heads, ratio = 4, 1, 2, 4
        cfg = self._config_with(apply_rope_fusion=False)

        x_comp = torch.randn(seq, batch, heads, head_dim, dtype=torch.bfloat16, device='cuda')
        out_comp = _apply_rope(
            x_comp.clone(),
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        x_full = torch.zeros(
            seq * ratio, batch, heads, head_dim, dtype=torch.bfloat16, device='cuda'
        )
        x_full[::ratio][:seq] = x_comp
        out_full = _apply_rope(
            x_full,
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq * ratio,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )
        out_ref = out_full[::ratio][:seq]

        assert torch.allclose(out_comp, out_ref, rtol=1e-3, atol=1e-3), (
            f"ratio={ratio} stride mismatch: "
            f"max abs diff = {(out_comp - out_ref).abs().max().item():.3e}"
        )


pytestmark = pytest.mark.experimental


@pytest.mark.skipif(not hasattr(torch, 'npu'), reason='Ascend NPU required')
class TestAscendBoundedCSA:
    """Check tiled attention, a full-denominator loss oracle, and recomputation."""

    @pytest.fixture(scope='class', autouse=True)
    def _parallel_group(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.fixture(autouse=True)
    def patch_hadamard_if_needed(self, monkeypatch):
        # Use the real Ascend rotation, overriding the module's CUDA-only mock fixture.
        from megatron.core import config as mcore_config
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            DSAIndexerLossAutoScaler,
            DSAIndexerLossLoggingHelper,
        )

        cpu_rng = torch.get_rng_state()
        npu_rng = torch.npu.get_rng_state()
        monkeypatch.setattr(mcore_config, 'ENABLE_EXPERIMENTAL', True)
        monkeypatch.setattr(
            DSAIndexerLossAutoScaler, 'main_loss_backward_scale', torch.ones((), device='npu')
        )
        monkeypatch.setattr(DSAIndexerLossLoggingHelper, 'tracker', {})
        torch.random.default_generator.manual_seed(123)
        torch.npu.manual_seed(123)
        try:
            yield
        finally:
            torch.set_rng_state(cpu_rng)
            torch.npu.set_rng_state(npu_rng)

    @pytest.mark.parametrize('ratio', [0, 4, 128])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
    def test_attention_output_and_gradients(self, ratio, dtype):
        from megatron.plugin.Ascend.transformer.experimental_attention_variant.csa import (
            _SparseAttention,
        )

        length = 257
        batch, heads, dim = (1, 64, 512) if dtype == torch.bfloat16 else (2, 4, 16)
        query = torch.randn(length, batch, heads, dim, device='npu', dtype=dtype)
        kv = torch.randn(
            length + (length // ratio if ratio else 0), batch, dim, device='npu', dtype=dtype
        )
        sink = torch.randn(heads, device='npu')
        indices = get_window_topk_idxs(17, batch, length, query.device).clone()
        if ratio:
            indices = torch.cat(
                (indices, get_compress_topk_idxs(ratio, batch, length, length, query.device)),
                dim=-1,
            )
        indices[:, :, 1] = indices[:, :, 0]
        indices[:, 0] = -1
        references = [t.detach().clone().requires_grad_() for t in (query, kv, sink)]
        candidates = [t.detach().clone().requires_grad_() for t in (query, kv, sink)]
        expected = unfused_compressed_sparse_attn(*references, indices.int(), dim**-0.5)
        actual = _SparseAttention.apply(*candidates, indices.int(), dim**-0.5)
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        actual.backward(upstream)
        tolerance = (
            dict(atol=0.012, rtol=0.02) if dtype == torch.bfloat16 else dict(atol=2e-5, rtol=3e-4)
        )
        torch.testing.assert_close(actual, expected, **tolerance)
        for candidate, reference in zip(candidates, references):
            torch.testing.assert_close(candidate.grad, reference.grad, **tolerance)
        assert torch.count_nonzero(actual[0]) == 0

    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize('per_token', [False, True])
    @pytest.mark.parametrize('production_dims', [False, True])
    @pytest.mark.parametrize('coeff', [0.0, 0.01])
    def test_sparse_indexer_loss_and_gradients(self, dtype, per_token, production_dims, coeff):
        from megatron.plugin.Ascend.transformer.experimental_attention_variant.csa import (
            _SparseIndexerLoss,
            _window_lse,
        )

        length, batch, heads, dim, ratio = 141, 2, 5, 16, 4
        keys = length // ratio
        teacher_heads, teacher_dim, topk = 3, 32, 9
        if production_dims:
            heads, dim, teacher_heads, teacher_dim, topk = 64, 128, 64, 512, keys
        q = torch.randn(length, batch, heads, dim, device='npu', dtype=dtype)
        k = torch.randn(keys, batch, dim, device='npu', dtype=dtype)
        weights = torch.randn(length, batch, heads, device='npu') * 0.1
        teacher_q = torch.randn(
            length, batch, teacher_heads, teacher_dim, device='npu', dtype=dtype
        )
        teacher_k = torch.randn(keys, batch, teacher_dim, device='npu', dtype=dtype)
        window_k = torch.randn(length, batch, teacher_dim, device='npu', dtype=dtype)
        window = get_window_topk_idxs(17, batch, length, q.device)
        sink = torch.randn(teacher_heads, device='npu')
        reference = [t.detach().cpu().double().requires_grad_() for t in (q, weights, k)]
        candidate = [t.detach().clone().requires_grad_() for t in (q, weights, k)]
        group = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        lse = _window_lse(teacher_q, window_k, sink, window, teacher_dim**-0.5)
        indices, actual = _SparseIndexerLoss.apply(
            *candidate,
            teacher_q,
            teacher_k,
            topk,
            ratio,
            teacher_dim**-0.5,
            coeff,
            group.tp,
            per_token,
            lse,
            True,
        )
        expected = _ascend_sparse_loss_oracle(
            *reference,
            teacher_q.cpu().double(),
            teacher_k.cpu().double(),
            window_k.cpu().double(),
            sink.cpu().double(),
            window.cpu(),
            indices.cpu(),
            ratio,
            teacher_dim**-0.5,
            coeff,
            per_token,
        )
        (expected * 2.75).backward()
        (actual * 2.75).backward()
        tolerance = (
            dict(atol=2e-5, rtol=0.015) if dtype == torch.bfloat16 else dict(atol=2e-7, rtol=4e-4)
        )
        torch.testing.assert_close(actual.cpu().double(), expected, **tolerance)
        for result, ref in zip(candidate, reference):
            torch.testing.assert_close(result.grad.cpu().double(), ref.grad, **tolerance)
        assert torch.count_nonzero(candidate[0].grad[: ratio - 1]) == 0

    def test_topk_across_key_tiles(self):
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            _compute_index_scores,
        )
        from megatron.plugin.Ascend.transformer.experimental_attention_variant.csa import (
            _indexer_topk,
        )

        length, ratio, count = 4113, 4, 17
        q = torch.randn(length, 1, 3, 8, device='npu')
        k = torch.randn(length // ratio, 1, 8, device='npu')
        weights = torch.rand(length, 1, 3, device='npu')
        scores = _compute_index_scores(q, weights, k)
        positions = torch.arange(1, length + 1, device='npu')
        scores.masked_fill_(
            torch.arange(k.shape[0], device='npu').view(1, 1, -1)
            >= (positions // ratio).view(1, -1, 1),
            float('-inf'),
        )
        indices = _indexer_topk(q, weights, k, count, ratio)
        torch.testing.assert_close(scores.gather(-1, indices), scores.topk(count, dim=-1).values)

    @pytest.mark.parametrize('ratio', [0, 4, 128])
    def test_real_csa_module_with_recompute(self, ratio):
        from types import MethodType

        from torch.utils.checkpoint import checkpoint

        from megatron.core.models.common.embeddings import RotaryEmbedding

        config = _make_mla_config(
            hidden_size=128,
            num_attention_heads=4,
            v_head_dim=64,
            csa_compress_ratios=[ratio] * 4,
            dsa_indexer_n_heads=4,
            dsa_indexer_head_dim=128,
            dsa_indexer_loss_coeff=0.01,
            dsa_indexer_use_sparse_loss=True,
        )
        config.gradient_accumulation_fusion = False
        config.apply_dsa_kernel_fusion = False
        config.deterministic_mode = True
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        rotary = RotaryEmbedding(
            config.qk_pos_emb_head_dim, rotary_percent=config.rotary_percent, cp_group=groups.cp
        )

        def make_module():
            return CompressedSparseAttention(
                config=config,
                submodules=_make_csa_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type='self',
                pg_collection=groups,
                rotary_pos_emb=rotary,
                compress_ratio=ratio,
            ).to('npu')

        reference, candidate = make_module(), make_module()
        candidate.load_state_dict(reference.state_dict())
        original = CompressedSparseAttention._forward_unfused_csa
        # Sparse CSA has corrected teacher semantics; compare eager and checkpointed
        # execution here. The independent full-denominator oracle is checked above.
        if ratio != 4:
            reference._forward_unfused_csa = MethodType(original.__wrapped__, reference)
        tensors = [
            torch.randn(257, 1, 4, 64, device='npu', dtype=torch.bfloat16),
            torch.randn(257, 1, 1, 64, device='npu', dtype=torch.bfloat16),
            torch.randn(257, 1, 128, device='npu', dtype=torch.bfloat16),
            torch.randn(257, 1, config.q_lora_rank, device='npu', dtype=torch.bfloat16),
        ]
        references = [t.clone().requires_grad_() for t in tensors]
        candidates = [t.clone().requires_grad_() for t in tensors]

        def run(module, q, kv, x, qr):
            return module(query=q, key=kv, value=kv, attention_mask=None, x=x, qr=qr)

        expected = run(reference, *references)
        actual = checkpoint(lambda *args: run(candidate, *args), *candidates, use_reentrant=True)
        upstream = torch.randn_like(expected) * 0.01
        expected.backward(upstream)
        actual.backward(upstream)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.025)
        for result, ref in zip(candidates, references):
            if ref.grad is None:
                assert result.grad is None
            else:
                torch.testing.assert_close(result.grad, ref.grad, atol=0.003, rtol=0.025)
        for (name, result), (_, ref) in zip(
            candidate.named_parameters(), reference.named_parameters()
        ):
            if ref.grad is None:
                assert result.grad is None, name
            else:
                torch.testing.assert_close(result.grad, ref.grad, atol=0.003, rtol=0.025, msg=name)

    @pytest.mark.parametrize('grad_enabled', [False, True])
    @pytest.mark.parametrize('unsupported', ['dense', 'tp', 'cp', 'packed'])
    def test_unsupported_npu_configuration(self, unsupported, grad_enabled):
        from types import SimpleNamespace

        from megatron.plugin.Ascend.transformer.experimental_attention_variant.csa import (
            forward_unfused_csa,
        )

        one = SimpleNamespace(size=lambda: 1)
        two = SimpleNamespace(size=lambda: 2)
        module = SimpleNamespace(
            training=True,
            indexer=object(),
            config=SimpleNamespace(dsa_indexer_use_sparse_loss=unsupported != 'dense'),
            pg_collection=SimpleNamespace(
                tp=two if unsupported == 'tp' else one, cp=two if unsupported == 'cp' else one
            ),
        )
        query = torch.zeros(1, 1, 1, 1, device='npu')
        packed = object() if unsupported == 'packed' else None
        message = {'dense': 'sparse loss', 'tp': 'TP=1', 'cp': 'CP=1', 'packed': 'packed sequences'}
        with (
            torch.set_grad_enabled(grad_enabled),
            pytest.raises(NotImplementedError, match=message[unsupported]),
        ):
            forward_unfused_csa(module, query, None, None, None, None, 0, 0, None, packed)

    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize('per_token', [False, True])
    @pytest.mark.parametrize('sink_logit', [0.0, 80.0, 90.0, 95.0, 120.0])
    def test_sparse_loss_extreme_sink(self, dtype, per_token, sink_logit):
        from megatron.plugin.Ascend.transformer.experimental_attention_variant.csa import (
            _indexer_loss_tile,
            _SparseIndexerLoss,
            _window_lse,
        )

        length, ratio, dim = 13, 4, 8
        q = torch.randn(length, 1, 2, 4, device='npu', dtype=dtype)
        weights = torch.randn(length, 1, 2, device='npu') * 0.15
        k = torch.randn(3, 1, 4, device='npu', dtype=dtype)
        teacher_q = torch.randn(length, 1, 3, dim, device='npu', dtype=dtype, requires_grad=True)
        teacher_k = torch.randn(3, 1, dim, device='npu', dtype=dtype, requires_grad=True)
        window_k = torch.randn(length, 1, dim, device='npu', dtype=dtype, requires_grad=True)
        sink = (torch.tensor([-1.0, 0.5, 2.0], device='npu') + sink_logit).requires_grad_()
        window = get_window_topk_idxs(5, 1, length, q.device)
        lse = _window_lse(teacher_q, window_k, sink, window, dim**-0.5)
        candidate = [t.detach().clone().requires_grad_() for t in (q, weights, k)]
        reference = [t.detach().cpu().double().requires_grad_() for t in (q, weights, k)]
        indices, actual = _SparseIndexerLoss.apply(
            *candidate,
            teacher_q.detach(),
            teacher_k.detach(),
            3,
            ratio,
            dim**-0.5,
            0.01,
            None,
            per_token,
            lse,
            True,
        )
        expected = _ascend_sparse_loss_oracle(
            *reference,
            teacher_q.detach().cpu().double(),
            teacher_k.detach().cpu().double(),
            window_k.detach().cpu().double(),
            sink.detach().cpu().double(),
            window.cpu(),
            indices.cpu(),
            ratio,
            dim**-0.5,
            0.01,
            per_token,
        )
        with torch.no_grad():
            target = _indexer_loss_tile(
                q[:, 0],
                weights[:, 0],
                k[:, 0],
                teacher_q[:, 0],
                teacher_k[:, 0],
                indices[0],
                torch.arange(1, length + 1, device='npu'),
                ratio,
                dim**-0.5,
                None,
                lse[0],
            )[3]
        target = target.cpu()
        assert torch.isfinite(target).all() and (target >= 0).all()
        valid_rows = torch.arange(1, length + 1) // ratio > 0
        torch.testing.assert_close(target.sum(dim=-1), valid_rows.float(), atol=2e-6, rtol=2e-6)
        assert actual.item() >= -2e-7
        (actual * 2.75).backward()
        (expected * 2.75).backward()
        tolerance = (
            dict(atol=2e-5, rtol=0.015) if dtype == torch.bfloat16 else dict(atol=2e-7, rtol=4e-4)
        )
        torch.testing.assert_close(actual.cpu().double(), expected, **tolerance)
        for result, ref in zip(candidate, reference):
            torch.testing.assert_close(result.grad.cpu().double(), ref.grad, **tolerance)
            assert torch.isfinite(result.grad).all()
        assert all(t.grad is None for t in (teacher_q, teacher_k, window_k, sink))

    def test_cpu_input_falls_back_to_core(self):
        from types import SimpleNamespace

        from megatron.plugin.Ascend.transformer.experimental_attention_variant.csa import (
            forward_unfused_csa,
        )

        length, heads, dim = 19, 3, 8
        module = SimpleNamespace(
            training=False,
            indexer=None,
            compress_ratio=0,
            attn_sink=torch.randn(heads),
            softmax_scale=dim**-0.5,
        )
        q = torch.randn(length, 1, heads, dim, requires_grad=True)
        kv = torch.randn(length, 1, dim, requires_grad=True)
        indices = get_window_topk_idxs(5, 1, length, q.device)
        args = (module, q, q, q, kv, None, 0, length, indices, None)
        expected, _ = CompressedSparseAttention._forward_unfused_csa.__wrapped__(*args)
        actual, loss = forward_unfused_csa(*args)
        assert loss is None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_grads = torch.autograd.grad(expected.sum(), (q, kv))
        actual_grads = torch.autograd.grad(actual.sum(), (q, kv))
        for result, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    @pytest.mark.parametrize('deterministic', [False, True])
    @pytest.mark.parametrize('pattern', ['zero', 'ties', 'near'])
    @pytest.mark.parametrize('count', [1, 17, 67, 70])
    def test_topk_ties_and_boundaries(self, monkeypatch, deterministic, pattern, count):
        from megatron.plugin.Ascend.transformer.experimental_attention_variant import csa

        monkeypatch.setattr(csa, '_QUERY_CHUNK', 7)
        monkeypatch.setattr(csa, '_KEY_CHUNK', 32)
        length, ratio, keys = 269, 4, 67
        q = torch.ones(length, 1, 1, 1, device='npu', dtype=torch.bfloat16)
        k = torch.zeros(keys, 1, 1, device='npu', dtype=torch.bfloat16)
        if pattern == 'ties':
            k[:, 0, 0] = (torch.arange(keys, device='npu') % 5).to(k.dtype)
        elif pattern == 'near':
            # Adjacent BF16 values have exact FP32 products; no artificial score epsilon.
            k[:, 0, 0] = 1 + torch.arange(keys, device='npu').to(k.dtype) / 128
        weights = torch.ones(length, 1, 1, device='npu')
        indices = csa._indexer_topk(q, weights, k, count, ratio, deterministic).cpu()[0]
        scores = k.cpu().float().view(1, -1).expand(length, -1).clone()
        positions = torch.arange(1, length + 1) // ratio
        scores.masked_fill_(torch.arange(keys).unsqueeze(0) >= positions.unsqueeze(1), -torch.inf)
        size = min(count, keys)
        torch.testing.assert_close(scores.gather(-1, indices), scores.topk(size).values)
        assert (indices.sort(dim=-1).values.diff(dim=-1) > 0).all()
        if deterministic:
            expected = scores.argsort(dim=-1, descending=True, stable=True)[:, :size]
            torch.testing.assert_close(indices, expected)
            monkeypatch.setattr(csa, '_QUERY_CHUNK', 19)
            monkeypatch.setattr(csa, '_KEY_CHUNK', 17)
            again = csa._indexer_topk(q, weights, k, count, ratio, True).cpu()[0]
            torch.testing.assert_close(again, expected)

    @pytest.mark.parametrize('change', ['window', 'sink', 'extreme'])
    def test_teacher_full_denominator(self, change):
        from megatron.plugin.Ascend.transformer.experimental_attention_variant.csa import (
            _indexer_loss_tile,
            _window_lse,
        )

        q = torch.ones(1, 1, 1, device='npu')
        k = torch.ones(2, 1, device='npu')
        weights = torch.ones(1, 1, device='npu')
        teacher_q = torch.eye(2, device='npu').unsqueeze(0)
        teacher_k = torch.eye(2, device='npu') * torch.tensor(3.0, device='npu').log()
        window_k = torch.zeros(1, 1, 2, device='npu', requires_grad=True)
        sink = torch.zeros(2, device='npu', requires_grad=True)
        window = torch.zeros(1, 1, 1, device='npu', dtype=torch.long)
        with torch.no_grad():
            if change == 'window':
                window_k[0, 0, 0] = torch.tensor(1000.0, device='npu').log()
            elif change == 'sink':
                sink[0] = torch.tensor(1000.0, device='npu').log()
            else:
                sink.fill_(80.0)
        lse = _window_lse(teacher_q.unsqueeze(1), window_k, sink, window, 1.0)
        assert not lse.requires_grad
        target = (
            _indexer_loss_tile(
                q,
                weights,
                k,
                teacher_q,
                teacher_k,
                torch.tensor([[0, 1]], device='npu'),
                torch.tensor([8], device='npu'),
                4,
                1.0,
                None,
                lse[0],
            )[3]
            .cpu()
            .double()
        )
        logits = torch.cat(
            (
                teacher_q.cpu().double() @ teacher_k.cpu().double().T,
                teacher_q.cpu().double() @ window_k.detach().cpu().double()[0].T,
                sink.detach().cpu().double().view(1, 2, 1),
            ),
            dim=-1,
        )
        expected = logits.softmax(dim=-1)[..., :2].sum(dim=1)
        expected /= expected.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(target, expected, atol=2e-6, rtol=2e-5)
        if change != 'extreme':
            assert target[0, 0] < 0.26
        torch.testing.assert_close(target.sum(-1), torch.ones(1, dtype=torch.float64))

    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
    def test_sparse_loss_recompute_with_ties(self, monkeypatch, dtype):
        from torch.utils.checkpoint import checkpoint

        from megatron.plugin.Ascend.transformer.experimental_attention_variant import csa

        monkeypatch.setattr(csa, '_QUERY_CHUNK', 13)
        monkeypatch.setattr(csa, '_KEY_CHUNK', 17)
        length, keys = 149, 37
        q = torch.ones(length, 1, 2, 4, device='npu', dtype=dtype)
        weights = torch.zeros(length, 1, 2, device='npu')
        k = torch.randn(keys, 1, 4, device='npu', dtype=dtype)
        tq = torch.randn(length, 1, 3, 8, device='npu', dtype=dtype, requires_grad=True)
        tk = torch.randn(keys, 1, 8, device='npu', dtype=dtype, requires_grad=True)
        wk = torch.randn(length, 1, 8, device='npu', dtype=dtype, requires_grad=True)
        sink = torch.randn(3, device='npu', requires_grad=True)
        window = get_window_topk_idxs(7, 1, length, q.device)
        lse = csa._window_lse(tq, wk, sink, window, 8**-0.5)
        eager = [t.clone().requires_grad_() for t in (q, weights, k)]
        recomputed = [t.clone().requires_grad_() for t in (q, weights, k)]

        def run(*args):
            return csa._SparseIndexerLoss.apply(
                *args, tq.detach(), tk.detach(), 9, 4, 8**-0.5, 0.01, None, False, lse, True
            )

        expected_indices, expected_loss = run(*eager)
        indices, loss = checkpoint(run, *recomputed, use_reentrant=True)
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        expected_loss.backward()
        loss.backward()
        for actual, expected in zip(recomputed, eager):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=1e-5)
        assert all(t.grad is None for t in (tq, tk, wk, sink))


def _ascend_sparse_loss_oracle(
    q,
    weights,
    k,
    teacher_q,
    teacher_k,
    window_k,
    sink,
    window,
    indices,
    ratio,
    scale,
    coeff,
    per_token,
):
    """Small dense autograd oracle; caller supplies the selected set to isolate KL math."""
    length, batch = q.shape[:2]
    rows = []
    for b in range(batch):
        for row in range(length):
            chosen = indices[b, row]
            chosen = chosen[chosen < (row + 1) // ratio].long()
            if chosen.numel() == 0:
                continue
            prediction = (
                ((q[row, b] @ k[chosen, b].T).relu() * weights[row, b].unsqueeze(-1))
                .sum(dim=0)
                .softmax(dim=-1)
            )
            with torch.no_grad():
                valid_window = window[b, row]
                valid_window = valid_window[valid_window >= 0].long()
                compressed = teacher_q[row, b] @ teacher_k[chosen, b].T * scale
                local = teacher_q[row, b] @ window_k[valid_window, b].T * scale
                logits = torch.cat((compressed, local, sink.unsqueeze(-1)), dim=-1)
                target = logits.softmax(dim=-1)[:, : chosen.numel()].sum(dim=0)
                target /= target.sum().clamp_min(torch.finfo(target.dtype).tiny)
            rows.append((target * ((target + 1e-10).log() - (prediction + 1e-10).log())).sum())
    return torch.stack(rows).sum() * (coeff if per_token else coeff / (batch * length))
