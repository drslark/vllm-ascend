# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.dspark_attention import (
    _dspark_sas_lens_match_scheduling,
    _dspark_sas_window,
    _dspark_sparse_sas_window,
    build_dspark_swa_indices,
    dspark_attention_from_standard_cache,
)


def _dspark_attention_loop(
    q: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    rows = []
    for token_idx in range(q.shape[0]):
        scores = torch.einsum("hd,khd->hk", q[token_idx].float(), k_ctx.float()) * scale
        sink = attn_sink[: q.shape[1]].float().unsqueeze(-1)
        scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
        exp_scores = torch.exp(scores - scores_max)
        probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
        rows.append(torch.einsum("hk,khd->hd", probs, v_ctx.float()).to(q.dtype))
    return torch.stack(rows, dim=0)


def _dspark_attention_reference(
    q: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores = torch.einsum("qhd,khd->qhk", q.float(), k_ctx.float()) * scale
    sink = attn_sink[: q.shape[1]].float().view(1, q.shape[1], 1)
    scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
    exp_scores = torch.exp(scores - scores_max)
    probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
    return torch.einsum("qhk,khd->qhd", probs, v_ctx.float()).to(q.dtype)


def _band_visible_indices(
    query_idx: int,
    s1_size: int,
    s2_size: int,
    ori_win_left: int,
    ori_win_right: int,
) -> list[int]:
    left = max(s2_size - s1_size + query_idx - ori_win_left, 0)
    right = min(s2_size - s1_size + query_idx + ori_win_right, s2_size - 1)
    if right < left:
        return []
    return list(range(left, right + 1))


def test_dspark_attention_loop_matches_vectorized_reference():
    torch.manual_seed(0)
    q = torch.randn(4, 3, 8, dtype=torch.float32)
    k_ctx = torch.randn(9, 3, 8, dtype=torch.float32)
    v_ctx = torch.randn(9, 3, 8, dtype=torch.float32)
    attn_sink = torch.randn(3, dtype=torch.float32)

    actual = _dspark_attention_loop(q, k_ctx, v_ctx, attn_sink, scale=0.125)
    expected = _dspark_attention_reference(q, k_ctx, v_ctx, attn_sink, scale=0.125)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dspark_sas_window_covers_context_and_full_draft_block():
    block_size = 5
    window_size = 7
    context_len = window_size
    s2_size = context_len + block_size

    mask_mode, ori_win_left, ori_win_right = _dspark_sas_window(block_size, window_size)

    assert mask_mode == 4
    assert ori_win_left == window_size + block_size - 1
    assert ori_win_right == block_size - 1
    for query_idx in range(block_size):
        assert _band_visible_indices(
            query_idx,
            block_size,
            s2_size,
            ori_win_left,
            ori_win_right,
        ) == list(range(s2_size))


def test_dspark_sparse_sas_window_is_fixed_upper_bound_only():
    block_size = 5
    window_size = 7
    s2_size = window_size + block_size

    mask_mode, ori_win_left, ori_win_right = _dspark_sparse_sas_window(block_size, window_size)

    assert mask_mode == 4
    assert ori_win_left == window_size + block_size - 1
    assert ori_win_right == 0
    assert _band_visible_indices(0, block_size, s2_size, ori_win_left, ori_win_right) != list(range(s2_size))


def test_dspark_sas_window_rejects_plain_sliding_window_formula():
    block_size = 5
    window_size = 7
    context_len = window_size
    s2_size = context_len + block_size

    bad_left = window_size - 1
    right = block_size - 1

    assert _band_visible_indices(
        block_size - 1,
        block_size,
        s2_size,
        bad_left,
        right,
    ) != list(range(s2_size))


def test_dspark_swa_indices_match_upstream_noncausal_formula():
    block_size = 3
    cache_block_size = 4
    window_size = 4
    positions = torch.tensor([10, 11, 12, 20, 21, 0], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [3, 1, 6, 0, 4, 5],
            [9, 11, 8, 7, 10, 2],
        ],
        dtype=torch.int32,
    )
    slot_mapping = torch.tensor(
        [
            [6, 2],
            [6, 3],
            [0, 0],
            [10, 0],
            [10, 1],
            [-1, -1],
        ],
        dtype=torch.int32,
    )

    indices, lens = build_dspark_swa_indices(
        block_table,
        positions,
        slot_mapping,
        block_size,
        window_size,
        cache_block_size,
        index_width=8,
    )

    def slot_ids(req_idx: int, start_pos: int, end_pos: int) -> torch.Tensor:
        return torch.tensor(
            [
                int(block_table[req_idx, pos // cache_block_size]) * cache_block_size + pos % cache_block_size
                for pos in range(start_pos, end_pos)
            ],
            dtype=torch.int32,
        )

    expected_req0 = slot_ids(0, 6, 13)
    expected_req1 = slot_ids(1, 16, 22)

    torch.testing.assert_close(
        lens,
        torch.tensor([7, 7, 7, 6, 6, 0], dtype=torch.int32),
    )
    for row in range(3):
        torch.testing.assert_close(indices[row, 0, :7], expected_req0)
        assert indices[row, 0, 7].item() == -1
    for row in range(3, 5):
        torch.testing.assert_close(indices[row, 0, :6], expected_req1)
        assert torch.all(indices[row, 0, 6:] == -1)
    assert torch.all(indices[5] == -1)

    # If the old packed-KV band window is applied directly to the full paged
    # cache, the first query would start at position 4 instead of 6.
    _, ori_win_left, ori_win_right = _dspark_sas_window(block_size, window_size)
    assert _band_visible_indices(0, block_size, 13, ori_win_left, ori_win_right)[0] == 4
    assert expected_req0[0].item() == (
        int(block_table[0, 6 // cache_block_size]) * cache_block_size + 6 % cache_block_size
    )


def test_dspark_swa_indices_default_width_is_operator_aligned():
    indices, lens = build_dspark_swa_indices(
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        None,
        block_size=1,
        window_size=1,
        cache_block_size=4,
    )

    assert indices.shape == (1, 1, 128)
    torch.testing.assert_close(lens, torch.tensor([1], dtype=torch.int32))


def test_dspark_sas_lens_guard_accepts_full_block_metadata():
    block_size = 3
    window_size = 4
    positions = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)
    seq_lens = torch.tensor([13, 23], dtype=torch.int32)
    block_table = torch.arange(16, dtype=torch.int32).view(2, 8)

    _, lens = build_dspark_swa_indices(
        block_table,
        positions,
        torch.arange(positions.numel(), dtype=torch.int32),
        block_size,
        window_size,
        cache_block_size=4,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
    )

    assert _dspark_sas_lens_match_scheduling(
        lens,
        query_start_loc,
        seq_lens,
        token_to_req_indices=None,
        block_size=block_size,
        window_size=window_size,
        num_query_tokens=positions.numel(),
    )


def test_dspark_sas_lens_guard_accepts_partial_block_padding():
    block_size = 5
    window_size = 7
    positions = torch.tensor([5, 6, 9, 10, 11], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    seq_lens = torch.tensor([7, 12], dtype=torch.int32)
    block_table = torch.arange(16, dtype=torch.int32).view(2, 8)

    _, lens = build_dspark_swa_indices(
        block_table,
        positions,
        torch.arange(positions.numel(), dtype=torch.int32),
        block_size,
        window_size,
        cache_block_size=4,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
    )

    assert _dspark_sas_lens_match_scheduling(
        lens,
        query_start_loc,
        seq_lens,
        token_to_req_indices=None,
        block_size=block_size,
        window_size=window_size,
        num_query_tokens=positions.numel(),
    )


def test_dspark_sas_lens_guard_rejects_interleaved_token_to_req():
    block_size = 2
    window_size = 4
    positions = torch.tensor([10, 20, 11, 21], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    seq_lens = torch.tensor([12, 22], dtype=torch.int32)
    token_to_req_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    block_table = torch.arange(16, dtype=torch.int32).view(2, 8)

    _, lens = build_dspark_swa_indices(
        block_table,
        positions,
        torch.arange(positions.numel(), dtype=torch.int32),
        block_size,
        window_size,
        cache_block_size=4,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        token_to_req_indices=token_to_req_indices,
    )

    assert not _dspark_sas_lens_match_scheduling(
        lens,
        query_start_loc,
        seq_lens,
        token_to_req_indices=token_to_req_indices,
        block_size=block_size,
        window_size=window_size,
        num_query_tokens=positions.numel(),
    )


def test_dspark_sas_lens_guard_accepts_contiguous_token_to_req():
    block_size = 2
    window_size = 4
    positions = torch.tensor([10, 11, 20, 21], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    seq_lens = torch.tensor([12, 22], dtype=torch.int32)
    token_to_req_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    block_table = torch.arange(16, dtype=torch.int32).view(2, 8)

    _, lens = build_dspark_swa_indices(
        block_table,
        positions,
        torch.arange(positions.numel(), dtype=torch.int32),
        block_size,
        window_size,
        cache_block_size=4,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        token_to_req_indices=token_to_req_indices,
    )

    assert _dspark_sas_lens_match_scheduling(
        lens,
        query_start_loc,
        seq_lens,
        token_to_req_indices=token_to_req_indices,
        block_size=block_size,
        window_size=window_size,
        num_query_tokens=positions.numel(),
    )


def test_dspark_sas_lens_guard_rejects_mismatched_lens_within_request():
    assert not _dspark_sas_lens_match_scheduling(
        torch.tensor([6, 5], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([12], dtype=torch.int32),
        token_to_req_indices=None,
        block_size=2,
        window_size=4,
        num_query_tokens=2,
    )


def test_dspark_swa_indices_use_explicit_metadata_not_block_offsets():
    block_size = 3
    cache_block_size = 4
    window_size = 4
    positions = torch.tensor([10, 11, 20, 21, 22, 30], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    seq_lens = torch.tensor([12, 23, 31], dtype=torch.int32)
    block_table = torch.arange(24, dtype=torch.int32).view(3, 8)

    indices, lens = build_dspark_swa_indices(
        block_table,
        positions,
        torch.arange(positions.numel(), dtype=torch.int32),
        block_size,
        window_size,
        cache_block_size,
        index_width=8,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
    )

    def slot_ids(req_idx: int, start_pos: int, end_pos: int) -> torch.Tensor:
        return torch.tensor(
            [
                int(block_table[req_idx, pos // cache_block_size]) * cache_block_size + pos % cache_block_size
                for pos in range(start_pos, end_pos)
            ],
            dtype=torch.int32,
        )

    expected_req0 = slot_ids(0, 6, 12)
    expected_req1 = slot_ids(1, 16, 23)
    expected_req2 = slot_ids(2, 26, 31)

    torch.testing.assert_close(lens, torch.tensor([6, 6, 7, 7, 7, 5], dtype=torch.int32))
    torch.testing.assert_close(indices[0, 0, :6], expected_req0)
    torch.testing.assert_close(indices[2, 0, :7], expected_req1)
    torch.testing.assert_close(indices[5, 0, :5], expected_req2)


def test_dspark_attention_from_standard_cache_matches_paged_swa_reference():
    torch.manual_seed(2)
    block_size = 3
    cache_block_size = 4
    window_size = 4
    heads = 2
    dim = 5
    scale = 0.125

    q = torch.randn(6, heads, dim, dtype=torch.float32)
    positions = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [3, 1, 6, 0, 4, 5],
            [9, 11, 8, 7, 10, 2],
        ],
        dtype=torch.int32,
    )
    cache = torch.zeros(12, cache_block_size, 1, dim, dtype=torch.float32)

    def write_req(req_idx: int, start: int, end: int) -> None:
        for pos in range(start, end):
            block_id = int(block_table[req_idx, pos // cache_block_size].item())
            offset = pos % cache_block_size
            cache[block_id, offset, 0] = torch.randn(dim)

    write_req(0, 6, 13)
    write_req(1, 16, 23)
    attn_sink = torch.tensor([0.1, -0.2], dtype=torch.float32)

    actual = dspark_attention_from_standard_cache(
        q,
        cache,
        block_table,
        positions,
        slot_mapping=torch.arange(6, dtype=torch.int32),
        attn_sink=attn_sink,
        block_size=block_size,
        window_size=window_size,
        cache_block_size=cache_block_size,
        softmax_scale=scale,
    )

    expected_blocks = []
    for req_idx, (start_pos, end_pos) in enumerate([(6, 13), (16, 23)]):
        k_ctx = torch.stack(
            [
                cache[
                    block_table[req_idx, pos // cache_block_size],
                    pos % cache_block_size,
                ]
                for pos in range(start_pos, end_pos)
            ]
        )
        k_ctx = k_ctx.expand(-1, heads, -1)
        expected_blocks.append(
            _dspark_attention_reference(
                q[req_idx * block_size : (req_idx + 1) * block_size],
                k_ctx,
                k_ctx,
                attn_sink,
                scale,
            )
        )
    torch.testing.assert_close(actual, torch.cat(expected_blocks), rtol=1e-6, atol=1e-6)


def test_dspark_attention_from_standard_cache_reads_current_draft_kv_from_cache():
    torch.manual_seed(7)
    block_size = 3
    cache_block_size = 4
    window_size = 4
    heads = 2
    dim = 5
    scale = 0.125

    q = torch.randn(block_size, heads, dim, dtype=torch.float32)
    positions = torch.tensor([10, 11, 12], dtype=torch.int32)
    block_table = torch.tensor([[3, 1, 6, 0]], dtype=torch.int32)
    cache = torch.zeros(8, cache_block_size, 1, dim, dtype=torch.float32)
    for pos in range(6, 10):
        cache[block_table[0, pos // cache_block_size], pos % cache_block_size, 0] = torch.randn(dim)
    current_draft_kv = torch.randn(block_size, 1, dim, dtype=torch.float32)
    for pos in range(10, 13):
        cache[block_table[0, pos // cache_block_size], pos % cache_block_size, 0] = current_draft_kv[pos - 10]
    attn_sink = torch.tensor([0.1, -0.2], dtype=torch.float32)

    actual = dspark_attention_from_standard_cache(
        q,
        cache,
        block_table,
        positions,
        slot_mapping=torch.arange(block_size, dtype=torch.int32),
        attn_sink=attn_sink,
        block_size=block_size,
        window_size=window_size,
        cache_block_size=cache_block_size,
        softmax_scale=scale,
    )

    context_kv = torch.stack(
        [
            cache[
                block_table[0, pos // cache_block_size],
                pos % cache_block_size,
            ]
            for pos in range(6, 10)
        ]
    )
    k_ctx = torch.cat([context_kv, current_draft_kv], dim=0).expand(-1, heads, -1)
    expected = _dspark_attention_reference(q, k_ctx, k_ctx, attn_sink, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dspark_attention_from_standard_cache_ignores_private_cache_validity():
    torch.manual_seed(11)
    block_size = 3
    cache_block_size = 4
    window_size = 4
    heads = 2
    dim = 5
    scale = 0.125

    q = torch.randn(block_size, heads, dim, dtype=torch.float32)
    positions = torch.tensor([10, 11, 12], dtype=torch.int32)
    block_table = torch.tensor([[3, 1, 6, 0]], dtype=torch.int32)
    cache = torch.zeros(8, cache_block_size, 1, dim, dtype=torch.float32)
    for pos in range(6, 13):
        cache[block_table[0, pos // cache_block_size], pos % cache_block_size, 0] = torch.randn(dim)
    cache[block_table[0, 8 // cache_block_size], 8 % cache_block_size, 0] = 1000.0
    attn_sink = torch.tensor([0.1, -0.2], dtype=torch.float32)

    actual = dspark_attention_from_standard_cache(
        q,
        cache,
        block_table,
        positions,
        slot_mapping=torch.arange(block_size, dtype=torch.int32),
        attn_sink=attn_sink,
        block_size=block_size,
        window_size=window_size,
        cache_block_size=cache_block_size,
        softmax_scale=scale,
    )

    context_kv = torch.stack(
        [
            cache[
                block_table[0, pos // cache_block_size],
                pos % cache_block_size,
            ]
            for pos in range(6, 13)
        ]
    )
    k_ctx = context_kv.expand(-1, heads, -1)
    expected = _dspark_attention_reference(q, k_ctx, k_ctx, attn_sink, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dspark_attention_from_standard_cache_zeros_padded_rows_with_2d_slots():
    torch.manual_seed(3)
    block_size = 3
    cache_block_size = 4
    heads = 2
    dim = 4
    q = torch.randn(6, heads, dim, dtype=torch.float32)
    positions = torch.tensor([10, 11, 12, 20, 21, 0], dtype=torch.int32)
    block_table = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=torch.int32)
    cache = torch.randn(12, cache_block_size, 1, dim, dtype=torch.float32)
    slot_mapping = torch.tensor(
        [
            [2, 2],
            [2, 3],
            [3, 0],
            [7, 0],
            [7, 1],
            [-1, -1],
        ],
        dtype=torch.int32,
    )

    actual = dspark_attention_from_standard_cache(
        q,
        cache,
        block_table,
        positions,
        slot_mapping=slot_mapping,
        attn_sink=torch.zeros(heads, dtype=torch.float32),
        block_size=block_size,
        window_size=4,
        cache_block_size=cache_block_size,
        softmax_scale=0.25,
    )

    assert actual[-1].abs().max().item() == 0.0
    assert actual[:-1].abs().max().item() > 0.0


def test_dspark_attention_from_standard_cache_uses_query_start_loc_ranges():
    torch.manual_seed(13)
    block_size = 3
    cache_block_size = 4
    window_size = 4
    heads = 2
    dim = 5
    scale = 0.125

    q = torch.randn(7, heads, dim, dtype=torch.float32)
    positions = torch.tensor([10, 11, 20, 21, 22, 30, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    seq_lens = torch.tensor([12, 23, 31], dtype=torch.int32)
    block_table = torch.arange(24, dtype=torch.int32).view(3, 8)
    cache = torch.randn(24, cache_block_size, 1, dim, dtype=torch.float32)
    slot_mapping = torch.tensor([1, 2, 3, 4, 5, 6, -1], dtype=torch.int32)
    attn_sink = torch.tensor([0.1, -0.2], dtype=torch.float32)

    actual = dspark_attention_from_standard_cache(
        q,
        cache,
        block_table,
        positions,
        slot_mapping=slot_mapping,
        attn_sink=attn_sink,
        block_size=block_size,
        window_size=window_size,
        cache_block_size=cache_block_size,
        softmax_scale=scale,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
    )

    expected = torch.zeros_like(q)
    for req_idx, (row_start, row_end, start_pos, end_pos) in enumerate([(0, 2, 6, 12), (2, 5, 16, 23), (5, 6, 26, 31)]):
        k_ctx = torch.stack(
            [
                cache[
                    block_table[req_idx, pos // cache_block_size],
                    pos % cache_block_size,
                ]
                for pos in range(start_pos, end_pos)
            ]
        ).expand(-1, heads, -1)
        expected[row_start:row_end] = _dspark_attention_reference(
            q[row_start:row_end],
            k_ctx,
            k_ctx,
            attn_sink,
            scale,
        )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dspark_attention_from_standard_cache_uses_token_to_req_indices():
    torch.manual_seed(17)
    block_size = 2
    cache_block_size = 4
    window_size = 4
    heads = 2
    dim = 5
    scale = 0.125

    q = torch.randn(4, heads, dim, dtype=torch.float32)
    positions = torch.tensor([10, 20, 11, 21], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    seq_lens = torch.tensor([12, 22], dtype=torch.int32)
    token_to_req_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    block_table = torch.arange(16, dtype=torch.int32).view(2, 8)
    cache = torch.randn(16, cache_block_size, 1, dim, dtype=torch.float32)
    slot_mapping = torch.arange(4, dtype=torch.int32)
    attn_sink = torch.tensor([0.1, -0.2], dtype=torch.float32)

    actual = dspark_attention_from_standard_cache(
        q,
        cache,
        block_table,
        positions,
        slot_mapping=slot_mapping,
        attn_sink=attn_sink,
        block_size=block_size,
        window_size=window_size,
        cache_block_size=cache_block_size,
        softmax_scale=scale,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        token_to_req_indices=token_to_req_indices,
    )

    expected = torch.zeros_like(q)
    for req_idx, row_indices, start_pos, end_pos in [
        (0, torch.tensor([0, 2]), 6, 12),
        (1, torch.tensor([1, 3]), 16, 22),
    ]:
        k_ctx = torch.stack(
            [
                cache[
                    block_table[req_idx, pos // cache_block_size],
                    pos % cache_block_size,
                ]
                for pos in range(start_pos, end_pos)
            ]
        ).expand(-1, heads, -1)
        expected[row_indices] = _dspark_attention_reference(
            q[row_indices],
            k_ctx,
            k_ctx,
            attn_sink,
            scale,
        )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dspark_attention_from_standard_cache_rejects_unsupported_cache_layout():
    q = torch.zeros(1, 1, 4, dtype=torch.float32)
    cache = torch.zeros(1, 4, 1, 5, dtype=torch.float32)

    with pytest.raises(ValueError, match="standard SWA cache PTA path"):
        dspark_attention_from_standard_cache(
            q,
            cache,
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            slot_mapping=None,
            attn_sink=torch.zeros(1, dtype=torch.float32),
            block_size=1,
            window_size=1,
            cache_block_size=4,
            softmax_scale=1.0,
        )
