# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM input replay using the existing fla_npu RecurrentKda operator.

Verify reads the checkpoint; commit reruns only accepted K/V/gate/beta inputs.
No per-candidate recurrent-state snapshots or Kimi-specific MegaKDA ABI are used.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.ops.kda import run_recurrent_kda
from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext

KDA_REPLAY_HEAD_DIM = 128
KDA_REPLAY_MAX_WIDTH = 8


@triton.jit
def _copy_replay_inputs_kernel(
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    values_ptr,
    kg_ptr,
    starts_ptr,
    states_ptr,
    lengths_ptr,
    packed_states_ptr,
    out_ptr,
    stride_out_token: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_values_block: tl.constexpr,
    stride_values_head: tl.constexpr,
    stride_values_pos: tl.constexpr,
    stride_kg_block: tl.constexpr,
    stride_kg_head: tl.constexpr,
    stride_kg_pos: tl.constexpr,
    stride_states: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    NULL_STATE: tl.constexpr,
    RECORD: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    state = tl.load(states_ptr + row * stride_states)
    start = tl.load(starts_ptr + row)
    if RECORD:
        length = tl.load(starts_ptr + row + 1) - start
    else:
        length = tl.load(lengths_ptr + row)
    dims = tl.arange(0, DIM)
    for pos in range(WIDTH):
        valid = (state != NULL_STATE) & (pos < length)
        token = start + pos
        values = values_ptr + state * stride_values_block + head * stride_values_head + pos * stride_values_pos
        kg = kg_ptr + state * stride_kg_block + head * stride_kg_head + pos * stride_kg_pos
        k = k_ptr + token * stride_k_token + head * DIM + dims
        v = v_ptr + token * stride_v_token + head * DIM + dims
        g = g_ptr + token * stride_g_token + head * DIM + dims
        beta = beta_ptr + token * stride_beta_token + head
        if RECORD:
            tl.store(out_ptr + token * stride_out_token + head * DIM + dims, 0, (state == NULL_STATE) & (pos < length))
            tl.store(kg + dims, tl.load(k, valid, 0), valid)
            tl.store(kg + DIM + dims, tl.load(g, valid, 0), valid)
            tl.store(values + dims, tl.load(v, valid, 0), valid)
            tl.store(values + DIM, tl.load(beta, valid, 0), valid)
        else:
            tl.store(k, tl.load(kg + dims, valid, 0), valid)
            tl.store(g, tl.load(kg + DIM + dims, valid, 0), valid)
            tl.store(v, tl.load(values + dims, valid, 0), valid)
            tl.store(beta, tl.load(values + DIM, valid, 0), valid)
            if head == 0:
                tl.store(packed_states_ptr + token, state, valid)


def _copy_inputs(k, v, gate, beta, values, kg, starts, states, width, *, lengths=None, packed_states=None, output=None):
    _copy_replay_inputs_kernel[(states.shape[0], k.shape[2])](
        k,
        v,
        gate,
        beta,
        values,
        kg,
        starts,
        states,
        lengths,
        packed_states,
        output,
        output.stride(1) if output is not None else 0,
        k.stride(1),
        v.stride(1),
        gate.stride(1),
        beta.stride(1),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        kg.stride(0),
        kg.stride(1),
        kg.stride(2),
        states.stride(0),
        HEADS=k.shape[2],
        DIM=k.shape[3],
        WIDTH=width,
        NULL_STATE=NULL_BLOCK_ID,
        RECORD=lengths is None,
        num_warps=4,
    )


def kda_recoverssm_verify_ascendc(
    q,
    k,
    v,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    lower_bound,
    checkpoint_state,
    value_beta_cache,
    kg_cache,
    query_start_loc,
    state_indices,
    spec_query_len,
):
    """Verify without writing persistent state; record raw inputs for replay."""
    if (
        q.ndim != 4
        or q.shape[0] != 1
        or q.shape[-1] != KDA_REPLAY_HEAD_DIM
        or not 1 <= spec_query_len <= KDA_REPLAY_MAX_WIDTH
    ):
        raise ValueError("AscendC RecoverSSM requires [1, tokens, heads, 128] and width in [1, 8]")
    if q.dtype != torch.bfloat16 or any(x.shape != q.shape or x.dtype != q.dtype for x in (k, v, raw_g)):
        raise ValueError("AscendC RecoverSSM Q/K/V/gate must have matching BF16 shapes")
    heads, dim = q.shape[2:]
    if raw_beta.shape != q.shape[:3] or raw_beta.dtype != q.dtype:
        raise ValueError("AscendC RecoverSSM beta must have BF16 shape [1, tokens, heads]")
    expected = (checkpoint_state.shape[0], heads, spec_query_len)
    if value_beta_cache.shape != (*expected, dim + 1) or value_beta_cache.dtype != torch.float32:
        raise ValueError("AscendC RecoverSSM value/beta cache must have FP32 shape [pages, heads, width, 129]")
    if kg_cache.shape != (*expected, 2 * dim) or kg_cache.dtype != q.dtype:
        raise ValueError("AscendC RecoverSSM key/gate cache layout is incompatible")
    if checkpoint_state.shape[1:] != (heads, dim, dim):
        raise ValueError("AscendC RecoverSSM checkpoint layout is incompatible")
    if checkpoint_state.dtype != torch.float32:
        raise ValueError("AscendC RecoverSSM currently requires FP32 checkpoints; use the triton backend for BF16")
    batch = state_indices.shape[0]
    if state_indices.ndim != 1 or query_start_loc.shape != (batch + 1,) or q.shape[1] > batch * spec_query_len:
        raise ValueError("AscendC RecoverSSM query metadata is incompatible")
    if any(x.stride()[2:] != (dim, 1) for x in (q, k, v, raw_g)) or raw_beta.stride(2) != 1:
        raise ValueError("AscendC RecoverSSM activation heads must be contiguous")
    if any(x.stride(-1) != 1 for x in (value_beta_cache, kg_cache)):
        raise ValueError("AscendC RecoverSSM record dimensions must be contiguous")
    if A_log.shape != (heads,) or dt_bias.numel() != heads * dim:
        raise ValueError("AscendC RecoverSSM gate parameters are incompatible")
    if any(
        x.device != q.device
        for x in (
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            checkpoint_state,
            value_beta_cache,
            kg_cache,
            query_start_loc,
            state_indices,
        )
    ):
        raise ValueError("AscendC RecoverSSM inputs must be on the same device")
    if q.shape[1] == 0:
        return torch.empty_like(v)
    # Every column selects the same checkpoint. With inplace_final_state=False,
    # RecurrentKda keeps the verified recurrence private to the kernel.
    indices = state_indices[:, None].expand(-1, spec_query_len).contiguous()
    output = run_recurrent_kda(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        checkpoint_state,
        query_start_loc,
        indices,
        A_log,
        dt_bias,
        lower_bound=lower_bound,
        beta_is_preprocessed=False,
        inplace_final_state=False,
    )
    _copy_inputs(
        k, v, raw_g, raw_beta, value_beta_cache, kg_cache, query_start_loc, state_indices, spec_query_len, output=output
    )
    valid = torch.arange(q.shape[1], device=q.device) < query_start_loc[-1]
    return output.masked_fill(~valid[None, :, None, None], 0)


@dataclass
class AscendCKDARecoverSSMCommitContext(KDARecoverSSMCommitContext):
    _extra_record_values: ClassVar[int] = 1
    replay_k: torch.Tensor | None = None
    replay_v: torch.Tensor | None = None
    replay_gate: torch.Tensor | None = None
    replay_beta: torch.Tensor | None = None
    replay_starts: torch.Tensor | None = None
    replay_states: torch.Tensor | None = None

    @classmethod
    def create(cls, layers, *, spec_query_len, max_num_reqs):
        context = super().create(layers, spec_query_len=spec_query_len, max_num_reqs=max_num_reqs)
        ref = context.kg_caches[0]
        if context.checkpoints[0].dtype != torch.float32:
            raise ValueError("AscendC RecoverSSM currently requires FP32 checkpoints; use the triton backend for BF16")
        heads, dim = context.checkpoints[0].shape[1:3]
        if dim != KDA_REPLAY_HEAD_DIM or not 1 <= spec_query_len <= KDA_REPLAY_MAX_WIDTH or ref.dtype != torch.bfloat16:
            raise ValueError("AscendC RecoverSSM requires BF16 records, head_dim=128 and width in [1, 8]")
        shape = (1, max_num_reqs * spec_query_len, heads, dim)
        context.replay_k = ref.new_empty(shape)
        context.replay_v = ref.new_empty(shape)
        context.replay_gate = ref.new_empty(shape)
        context.replay_beta = ref.new_empty(shape[:3])
        context.replay_starts = torch.zeros(max_num_reqs + 1, device=ref.device, dtype=torch.int32)
        context.replay_states = torch.empty(shape[1], device=ref.device, dtype=torch.int32)
        return context

    def _commit_kda(self, state_indices: torch.Tensor, align_mode: bool) -> None:
        if align_mode:
            raise ValueError("AscendC RecoverSSM requires Mamba cache mode none")
        batch = state_indices.shape[0]
        starts = self.replay_starts[: batch + 1]
        torch.cumsum(self.commit_lens[:batch], dim=0, dtype=torch.int32, out=starts[1:])
        capacity = batch * self.spec_query_len
        k, v, gate = (x[:, :capacity] for x in (self.replay_k, self.replay_v, self.replay_gate))
        beta = self.replay_beta[:, :capacity]
        packed_states = self.replay_states[:capacity]
        # Scratch is shared across layers, and all launches use the caller's
        # stream. RecurrentKda consumes each layer before the next gather.
        for layer, checkpoint in enumerate(self.checkpoints):
            _copy_inputs(
                k,
                v,
                gate,
                beta,
                self.correction_caches[layer],
                self.kg_caches[layer],
                starts,
                state_indices,
                self.spec_query_len,
                lengths=self.commit_lens,
                packed_states=packed_states,
            )
            run_recurrent_kda(
                k,
                k,
                v,
                gate,
                beta,
                checkpoint,
                starts,
                packed_states,
                self.A_log[layer],
                self.dt_bias[layer],
                lower_bound=self.lower_bound,
                beta_is_preprocessed=False,
            )
