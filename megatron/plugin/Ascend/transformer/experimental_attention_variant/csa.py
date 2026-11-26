# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""Ascend operator bindings for DeepSeek V4 packed attention."""

import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_utils,
    packed_layout,
)


def _manager():
    from transformer_engine.plugin.core.manager import get_default_manager

    return get_default_manager()


def apply_dsv4_rope(
    operation,
    t,
    cos,
    sin,
    nope_dim,
    emb_dim,
    cu_seqlens_q=None,
    cp_rank=0,
    cp_size=1,
    rotary_interleaved=False,
    inverse=False,
    remove_interleaving=False,
    position_ids=None,
):
    """Rotate the DSv4 activation in private storage, including retained attention outputs."""
    if (
        t.device.type != "npu"
        or rotary_interleaved
        or not remove_interleaving
        or (cu_seqlens_q is not None and position_ids is None and cp_size != 1)
    ):
        return operation(
            t,
            cos,
            sin,
            nope_dim,
            emb_dim,
            cu_seqlens_q,
            cp_rank,
            cp_size,
            rotary_interleaved,
            inverse,
            remove_interleaving,
            position_ids,
        )

    if cu_seqlens_q is not None and position_ids is None:
        position_ids = cp_utils._thd_cp_position_ids(cu_seqlens_q, 0, t.size(0))
    if position_ids is None:
        cos, sin = cos[: t.size(0)], sin[: t.size(0)]
    else:
        positions = position_ids.long()
        cos, sin = cos.index_select(0, positions), sin.index_select(0, positions)
    content, rotary = t.split((nope_dim, emb_dim), dim=-1)
    rotated = _manager().call("mla_interleaved_rope", rotary, cos, sin, inverse=inverse)
    return torch.cat((content, rotated), dim=-1)


def validate_backend(config, attention):
    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
    )

    if not isinstance(attention, CompressedSparseAttention):
        return
    use_indexer = attention.indexer is not None
    use_loss = (
        use_indexer
        and attention.training
        and torch.is_grad_enabled()
        and (config.dsa_indexer_loss_coeff or 0.0) > 0
    )
    ops = ["packed_csa_attention_with_loss" if use_loss else "packed_csa_attention"]
    if use_indexer:
        ops.extend(("packed_csa_indexer_topk", "hadamard_transform"))
    if config.apply_rope_fusion:
        ops.append("mla_interleaved_rope")
    manager = _manager()
    for op in ops:
        try:
            manager.resolve(op)
        except RuntimeError as error:
            raise RuntimeError(f"Packed DSv4 requires TE-FL operator '{op}' on Ascend.") from error


def topk_alignment():
    return 1


def gathered_for_loss(tensor):
    # TE autograd reaches the local producer through the original CP gather.
    return tensor


def _key_ranges(cu_q, cu_k, offsets, rows, ratio):
    lengths = packed_layout.build_seq_lens(cu_q, cu_k, rows, ratio, offsets)
    documents = torch.bucketize(
        torch.arange(rows, device=cu_q.device, dtype=cu_q.dtype), cu_q[1:], right=True
    )
    return cu_k[documents], lengths


def indexer_topk(
    q,
    k,
    weights,
    topk,
    ratio=4,
    indexer_softmax_scale=1.0,
    *,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    q_causal_offsets=None,
):
    starts, lengths = _key_ranges(cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, q.size(0), ratio)
    indices = _manager().call(
        "packed_csa_indexer_topk",
        q,
        k,
        weights,
        starts,
        lengths,
        max_seqlen_kv,
        topk,
        indexer_softmax_scale,
    )
    return indices, (indices >= 0).sum(-1, dtype=torch.int32)


def csa_sparse_attn(
    query,
    kv,
    attn_sink,
    topk_idxs,
    softmax_scale,
    topk_length=None,
    kv_reconstruction_parts=None,
    q_padding_mask=None,
):
    return _manager().call("packed_csa_attention", query, kv, attn_sink, topk_idxs, softmax_scale)


def attention_with_loss(
    query,
    kv_full,
    attn_sink,
    topk_idxs,
    q_indexer,
    k_indexer,
    weights,
    indexer_topk_idxs,
    compressed_kv,
    softmax_scale,
    indexer_softmax_scale,
    loss_coeff,
    loss_divisor,
    sparse_loss,
    ratio,
    max_seqlen_q,
    indexer_layout,
    q_padding_mask,
    local_k_indexer,
    local_compressed_kv,
    cp_group,
    compressed_kv_start,
    indexer_rank_map,
    indexer_k_reduce_scatter_state,
    compressed_kv_reduce_scatter_state,
    logical_window_width,
    kv_reconstruction_parts,
):
    keys = indexer_topk_idxs
    if not sparse_loss:
        starts, lengths = _key_ranges(*indexer_layout, query.size(0), ratio)
        keys = (starts, lengths, max_seqlen_q // ratio)
    output, loss = _manager().call(
        "packed_csa_attention_with_loss",
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        q_indexer,
        k_indexer,
        weights,
        compressed_kv,
        keys,
        logical_window_width,
        indexer_softmax_scale,
        q_padding_mask,
    )
    return output, loss * loss_coeff / loss_divisor
