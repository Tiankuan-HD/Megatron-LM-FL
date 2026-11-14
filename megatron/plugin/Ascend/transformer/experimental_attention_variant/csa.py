# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""Bounded-memory Ascend CSA attention and sparse indexer training."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams

_QUERY_CHUNK = 128
_KEY_CHUNK = 1024
_EPS = 1e-10


def _gather_tokens(tokens, indices):
    return tokens.index_select(0, indices.clamp_min(0).reshape(-1).long()).reshape(
        *indices.shape, tokens.shape[-1]
    )


def _attention_tile(query, kv, sink, indices, scale):
    selected = _gather_tokens(kv, indices).float()
    scores = torch.bmm(query.float(), selected.transpose(1, 2)) * scale
    scores.masked_fill_(indices.unsqueeze(1) < 0, float('-inf'))
    sink = sink.float().view(1, -1, 1)
    maximum = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
    probability = (scores - maximum).exp()
    sink_probability = (sink - maximum).exp()
    denominator = probability.sum(dim=-1, keepdim=True) + sink_probability
    probability = probability / denominator
    sink_probability = sink_probability / denominator
    output = torch.bmm(probability, selected)
    return selected, probability, sink_probability, output


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, kv, sink, indices, scale):
        length, batch, heads, dim = query.shape
        output = torch.empty_like(query)
        for b in range(batch):
            for start in range(0, length, _QUERY_CHUNK):
                end = min(start + _QUERY_CHUNK, length)
                _, _, _, tile = _attention_tile(
                    query[start:end, b], kv[:, b], sink, indices[b, start:end], scale
                )
                output[start:end, b] = tile.to(query.dtype)
        ctx.save_for_backward(query, kv, sink, indices)
        ctx.scale = scale
        return output.reshape(length, batch, heads * dim)

    @staticmethod
    def backward(ctx, grad_output):
        query, kv, sink, indices = ctx.saved_tensors
        length, batch, heads, dim = query.shape
        grad_output = grad_output.reshape_as(query)
        grad_query = torch.empty_like(query)
        grad_kv = torch.zeros((batch, kv.shape[0], dim), device=kv.device, dtype=torch.float32)
        grad_sink = torch.zeros_like(sink, dtype=torch.float32)
        for b in range(batch):
            for start in range(0, length, _QUERY_CHUNK):
                end = min(start + _QUERY_CHUNK, length)
                idx = indices[b, start:end]
                q = query[start:end, b].float()
                selected, probability, sink_probability, output = _attention_tile(
                    q, kv[:, b], sink, idx, ctx.scale
                )
                grad = grad_output[start:end, b].float()
                delta = (grad * output).sum(dim=-1, keepdim=True)
                grad_score = probability * (torch.bmm(grad, selected.transpose(1, 2)) - delta)
                grad_query[start:end, b] = (torch.bmm(grad_score, selected) * ctx.scale).to(
                    query.dtype
                )
                grad_selected = torch.bmm(grad_score.transpose(1, 2), q) * ctx.scale + torch.bmm(
                    probability.transpose(1, 2), grad
                )
                # Core casts the selected-KV gradient before gather backward.
                grad_selected = grad_selected.to(kv.dtype).float().reshape(-1, dim)
                grad_kv[b].index_add_(0, idx.clamp_min(0).reshape(-1).long(), grad_selected)
                grad_sink -= (sink_probability * delta).sum(dim=(0, 2))
        return (
            grad_query,
            grad_kv.permute(1, 0, 2).to(kv.dtype),
            grad_sink.to(sink.dtype),
            None,
            None,
        )


@torch.no_grad()
def _window_lse(query, kv, sink, indices, scale):
    """Keep only the detached window-plus-sink mass needed by the CSA teacher."""
    length, batch, heads = query.shape[:3]
    lse = torch.empty((batch, length, heads), device=query.device, dtype=torch.float32)
    for b in range(batch):
        for start in range(0, length, _QUERY_CHUNK):
            end = min(start + _QUERY_CHUNK, length)
            idx = indices[b, start:end]
            selected = _gather_tokens(kv[:, b], idx).float()
            logits = torch.bmm(query[start:end, b].float(), selected.transpose(1, 2)) * scale
            logits.masked_fill_(idx.unsqueeze(1) < 0, float('-inf'))
            lse[b, start:end] = torch.logaddexp(logits.logsumexp(dim=-1), sink.float())
    return lse


@torch.no_grad()
def _indexer_topk(q, weights, k, topk, ratio, deterministic=False):
    length, batch, heads, dim = q.shape
    count = min(topk, k.shape[0])
    indices = torch.empty((batch, length, count), device=q.device, dtype=torch.long)
    for b in range(batch):
        for start in range(0, length, _QUERY_CHUNK):
            end = min(start + _QUERY_CHUNK, length)
            query = q[start:end, b].float()
            valid_count = torch.arange(start + 1, end + 1, device=q.device) // ratio
            best_scores = best_indices = None
            for key_start in range(0, k.shape[0], _KEY_CHUNK):
                key_end = min(key_start + _KEY_CHUNK, k.shape[0])
                scores = torch.matmul(
                    query.reshape(-1, dim), k[key_start:key_end, b].float().T
                ).reshape(end - start, heads, key_end - key_start)
                scores.relu_().mul_(weights[start:end, b].float().unsqueeze(-1))
                scores = scores.sum(dim=1)
                key_indices = torch.arange(key_start, key_end, device=q.device)
                scores.masked_fill_(
                    key_indices.unsqueeze(0) >= valid_count.unsqueeze(1), float('-inf')
                )
                block_indices = key_indices.expand(end - start, -1)
                if best_scores is not None:
                    scores = torch.cat((best_scores, scores), dim=-1)
                    block_indices = torch.cat((best_indices, block_indices), dim=-1)
                size = min(count, scores.shape[-1])
                if deterministic:
                    # Sort key ids first so stable score sorting breaks ties toward smaller ids.
                    order = block_indices.argsort(dim=-1)
                    selected = scores.gather(-1, order).argsort(
                        dim=-1, descending=True, stable=True
                    )[:, :size]
                    selected = order.gather(-1, selected)
                    best_scores = scores.gather(-1, selected)
                else:
                    best_scores, selected = scores.topk(size, dim=-1)
                best_indices = block_indices.gather(-1, selected)
            indices[b, start:end] = best_indices
    return indices


def _indexer_loss_tile(
    q, weights, k, teacher_q, teacher_k, indices, positions, ratio, scale, tp, window_lse
):
    selected = _gather_tokens(k, indices).float()
    dot = torch.bmm(q.float(), selected.transpose(1, 2))
    logits = (dot.relu() * weights.float().unsqueeze(-1)).sum(dim=1)
    valid = indices < (positions // ratio).unsqueeze(1)
    row_valid = valid.any(dim=-1, keepdim=True)
    logits = logits.masked_fill(~valid, float('-inf')).masked_fill(~row_valid, 0.0)
    prediction = logits.softmax(dim=-1) * row_valid

    teacher_selected = _gather_tokens(teacher_k, indices).float()
    teacher_logits = torch.bmm(teacher_q.float(), teacher_selected.transpose(1, 2)) * scale
    teacher_logits.masked_fill_(~valid.unsqueeze(1), float('-inf'))
    teacher_logits.masked_fill_(~row_valid.unsqueeze(1), 0.0)
    full_lse = torch.logaddexp(teacher_logits.logsumexp(dim=-1), window_lse).unsqueeze(-1)
    log_target = (teacher_logits - full_lse).logsumexp(dim=1)
    if tp is not None and tp.size() > 1:
        raise NotImplementedError('Ascend bounded CSA indexer loss requires TP=1')
    target = log_target.softmax(dim=-1) * row_valid
    return selected, dot, prediction, target, valid


class _SparseIndexerLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        weights,
        k,
        teacher_q,
        teacher_k,
        topk,
        ratio,
        scale,
        coeff,
        tp,
        per_token,
        window_lse,
        deterministic,
    ):
        indices = _indexer_topk(q, weights, k, topk, ratio, deterministic)
        length, batch = q.shape[:2]
        loss = torch.zeros((), device=q.device, dtype=torch.float32)
        for b in range(batch):
            for start in range(0, length, _QUERY_CHUNK):
                end = min(start + _QUERY_CHUNK, length)
                _, _, prediction, target, _ = _indexer_loss_tile(
                    q[start:end, b],
                    weights[start:end, b],
                    k[:, b],
                    teacher_q[start:end, b],
                    teacher_k[:, b],
                    indices[b, start:end],
                    torch.arange(start + 1, end + 1, device=q.device),
                    ratio,
                    scale,
                    tp,
                    window_lse[b, start:end],
                )
                loss += (target * ((target + _EPS).log() - (prediction + _EPS).log())).sum()
        loss *= coeff if per_token else coeff / (batch * length)
        ctx.save_for_backward(q, weights, k, teacher_q, teacher_k, indices, window_lse)
        ctx.ratio, ctx.scale, ctx.coeff, ctx.tp, ctx.per_token = ratio, scale, coeff, tp, per_token
        ctx.mark_non_differentiable(indices)
        return indices, loss

    @staticmethod
    def backward(ctx, grad_indices, grad_loss):
        q, weights, k, teacher_q, teacher_k, indices, window_lse = ctx.saved_tensors
        length, batch, _, dim = q.shape
        grad_q = torch.empty_like(q)
        grad_weights = torch.empty_like(weights)
        grad_k = torch.zeros((batch, k.shape[0], dim), device=k.device, dtype=torch.float32)
        loss_scale = grad_loss * ctx.coeff
        if not ctx.per_token:
            loss_scale = loss_scale / (batch * length)
        for b in range(batch):
            for start in range(0, length, _QUERY_CHUNK):
                end = min(start + _QUERY_CHUNK, length)
                idx = indices[b, start:end]
                selected, dot, prediction, target, valid = _indexer_loss_tile(
                    q[start:end, b],
                    weights[start:end, b],
                    k[:, b],
                    teacher_q[start:end, b],
                    teacher_k[:, b],
                    idx,
                    torch.arange(start + 1, end + 1, device=q.device),
                    ctx.ratio,
                    ctx.scale,
                    ctx.tp,
                    window_lse[b, start:end],
                )
                grad_probability = -target / (prediction + _EPS) * loss_scale
                grad_logits = prediction * (
                    grad_probability - (grad_probability * prediction).sum(dim=-1, keepdim=True)
                )
                grad_logits = grad_logits.masked_fill(~valid, 0.0)
                grad_weights[start:end, b] = (
                    (grad_logits.unsqueeze(1) * dot.relu()).sum(dim=-1).to(weights.dtype)
                )
                grad_dot = (
                    grad_logits.unsqueeze(1)
                    * weights[start:end, b].float().unsqueeze(-1)
                    * (dot > 0)
                )
                grad_q[start:end, b] = torch.bmm(grad_dot, selected).to(q.dtype)
                grad_selected = torch.bmm(grad_dot.transpose(1, 2), q[start:end, b].float())
                grad_k[b].index_add_(0, idx.reshape(-1).long(), grad_selected.reshape(-1, dim))
        return (
            grad_q,
            grad_weights,
            grad_k.permute(1, 0, 2).to(k.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def forward_unfused_csa(
    self,
    query: torch.Tensor,
    x: torch.Tensor,
    qr: torch.Tensor,
    kv_full: torch.Tensor,
    compressed_kv: torch.Tensor | None,
    n_compressed: int,
    offset: int,
    window_idxs: torch.Tensor,
    packed_seq_params: 'PackedSeqParams | None',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run tiled Ascend CSA for unpacked TP1/CP1 and sparse indexer training."""
    if query.device.type != 'npu':
        from megatron.core.transformer.experimental_attention_variant.csa import (
            CompressedSparseAttention,
        )

        return CompressedSparseAttention._forward_unfused_csa.__wrapped__(
            self,
            query,
            x,
            qr,
            kv_full,
            compressed_kv,
            n_compressed,
            offset,
            window_idxs,
            packed_seq_params,
        )

    if self.pg_collection.tp.size() != 1:
        raise NotImplementedError('Ascend bounded CSA requires TP=1')
    if self.pg_collection.cp.size() != 1:
        raise NotImplementedError('Ascend bounded CSA requires CP=1')
    if packed_seq_params is not None:
        raise NotImplementedError('Ascend bounded CSA does not support packed sequences')
    if self.training and self.indexer is not None and not self.config.dsa_indexer_use_sparse_loss:
        raise NotImplementedError('Ascend bounded CSA indexer training requires sparse loss')
    training_indexer = self.training and torch.is_grad_enabled() and self.indexer is not None

    from megatron.core.transformer.experimental_attention_variant.csa import get_compress_topk_idxs
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossLoggingHelper,
    )

    length, batch = query.shape[:2]
    indexer_loss = None
    indices = window_idxs
    if self.compress_ratio > 1 and n_compressed > 0:
        if self.indexer is None:
            compressed_indices = get_compress_topk_idxs(
                self.compress_ratio, batch, length, offset, query.device
            )
        else:
            q, k, weights = self.indexer.forward_before_topk(
                x.detach(), qr.detach(), packed_seq_params
            )
            weights = weights.float() * self.indexer.softmax_scale
            topk = min(self.indexer.index_topk, n_compressed)
            deterministic = self.config.deterministic_mode
            if training_indexer:
                coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)
                window_lse = _window_lse(
                    query.detach(),
                    kv_full.detach(),
                    self.attn_sink.detach(),
                    window_idxs,
                    self.softmax_scale,
                )
                compressed_indices, indexer_loss = _SparseIndexerLoss.apply(
                    q,
                    weights,
                    k,
                    query.detach(),
                    compressed_kv.detach(),
                    topk,
                    self.compress_ratio,
                    self.softmax_scale,
                    coeff,
                    self.indexer.pg_collection.tp,
                    self.config.calculate_per_token_loss,
                    window_lse,
                    deterministic,
                )
                if coeff > 0:
                    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                        loss=indexer_loss,
                        layer_number=self.layer_number,
                        num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
                    )
            else:
                compressed_indices = _indexer_topk(
                    q, weights, k, topk, self.compress_ratio, deterministic
                )
            valid_count = torch.arange(1, length + 1, device=query.device).view(1, -1, 1)
            valid = compressed_indices < valid_count // self.compress_ratio
            compressed_indices = torch.where(valid, compressed_indices + offset, -1)
        indices = torch.cat((window_idxs, compressed_indices), dim=-1)
    output = _SparseAttention.apply(
        query, kv_full, self.attn_sink.float(), indices.int(), self.softmax_scale
    )
    return output, indexer_loss
