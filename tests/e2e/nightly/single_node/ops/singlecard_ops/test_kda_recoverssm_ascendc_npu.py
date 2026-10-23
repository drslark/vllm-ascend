# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native verify and accepted-prefix replay against independent recurrence."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first  # noqa: E402

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.test_kda_recoverssm_npu import (  # noqa: E402
    page_view,
    reference,
)
from vllm_ascend.ops.kda_recoverssm import (  # noqa: E402
    AscendCKDARecoverSSMCommitContext,
    kda_recoverssm_verify_ascendc,
)


@pytest.mark.parametrize("width", [1, 6, 8])
@pytest.mark.parametrize("padding", [0, 128])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_native_verify_and_replay_preserve_paged_state(width, padding):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(731)
    heads, dim, pages = 2, 128, 8
    blocks, lengths = [3, 1, 5, 2, 0, 4], [width, 1, width - 1, width, 1, 0]
    accepted = [0, 1, min(3, width - 1), width, 1, 0]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    indices = torch.tensor(blocks, device="npu", dtype=torch.int32)
    offsets = torch.tensor(starts, device="npu", dtype=torch.int32)
    # Reorder sampler rows and include a non-spec request.
    request_rows = torch.tensor([4, 1, 5, 0, 2, 3], device="npu", dtype=torch.int32)
    counts = torch.tensor(
        [accepted[3], accepted[1], accepted[4], accepted[5], accepted[0], accepted[2], 1],
        device="npu",
        dtype=torch.int32,
    )
    layers, backings = [], []
    for _ in range(2):
        sb, state = page_view((pages, heads, dim, dim), torch.float32, padding)
        vb, values = page_view((pages, heads, width, dim + 1), torch.float32, padding)
        kb, kg = page_view((pages, heads, width, 2 * dim), torch.bfloat16, padding)
        conv_shape = (pages, 3 * heads * dim, width + 2)
        if not is_conv_state_dim_first():
            conv_shape = (pages, width + 2, 3 * heads * dim)
        cb, conv = page_view(conv_shape, torch.bfloat16, padding)
        state.copy_(torch.randn_like(state) * 0.05)
        a_log = torch.randn(heads, device="npu") * 0.2
        bias = torch.randn(heads * dim, device="npu") * 0.1
        layers.append(
            SimpleNamespace(
                kv_cache=(conv, state, values, kg),
                A_log=a_log,
                dt_bias=bias,
                gate_lower_bound=-5.0,
                local_num_heads=heads,
                head_dim=dim,
            )
        )
        backings.extend((sb, vb, kb, cb))
    context = AscendCKDARecoverSSMCommitContext.create(layers, spec_query_len=width, max_num_reqs=len(blocks))
    for _ in range(2):
        expected_states = []
        expected_convs = []
        for layer in layers:
            conv, state, values, kg = layer.kv_cache
            initial = state.clone()
            original_conv = conv.clone()
            # Includes a padded output token and non-contiguous Q/K/V views.
            total = starts[-1] + 1
            projections = torch.randn(1, total, heads * dim * 4, device="npu", dtype=torch.bfloat16)
            q, k, v, gate = (x.view(1, total, heads, dim) for x in projections.split(heads * dim, dim=-1))
            beta = torch.randn(1, total, heads, device="npu", dtype=torch.bfloat16)
            expected_output, expected_state = reference(
                q, k, v, gate, beta, layer.A_log, layer.dt_bias, -5.0, initial, starts, blocks, accepted
            )
            output = kda_recoverssm_verify_ascendc(
                q, k, v, gate, beta, layer.A_log, layer.dt_bias, -5.0, state, values, kg, offsets, indices, width
            )
            torch.testing.assert_close(state, initial, atol=0, rtol=0)
            torch.testing.assert_close(output.float(), expected_output, atol=2e-2, rtol=2e-2)
            for row, block in enumerate(blocks):
                if block:
                    values[block, :, accepted[row] :] = float("nan")
                    kg[block, :, accepted[row] :] = float("nan")
            expected_states.append(expected_state)
            destination = original_conv if is_conv_state_dim_first() else original_conv.transpose(-1, -2)
            source = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
            for block, count in zip(blocks, accepted):
                if block and count:
                    destination[block, :, :3] = source[block, :, count - 1 : count + 2]
            expected_convs.append(original_conv)
        context.commit(counts, indices, offsets, request_indices=request_rows)
        for layer, expected, conv in zip(layers, expected_states, expected_convs):
            torch.testing.assert_close(layer.kv_cache[1], expected, atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(layer.kv_cache[0], conv, atol=0, rtol=0)
    if padding:
        for backing in backings:
            torch.testing.assert_close(
                backing[:, -padding:], torch.full_like(backing[:, -padding:], 11.0), atol=0, rtol=0
            )


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_native_graph_replay_refreshes_pages_lengths_and_zero_acceptance():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(92)
    width, heads, dim = 6, 2, 128
    q, k, v, gate = (torch.randn(1, 12, heads, dim, device="npu", dtype=torch.bfloat16) for _ in range(4))
    beta = torch.randn(1, 12, heads, device="npu", dtype=torch.bfloat16)
    state = torch.randn(8, heads, dim, dim, device="npu") * 0.05
    values = torch.empty(8, heads, width, dim + 1, device="npu")
    kg = torch.empty(8, heads, width, 2 * dim, device="npu", dtype=torch.bfloat16)
    conv = torch.zeros(8, 3 * heads * dim, width + 2, device="npu", dtype=torch.bfloat16)
    if not is_conv_state_dim_first():
        conv = conv.transpose(-1, -2).contiguous()
    a_log, bias = torch.zeros(heads, device="npu"), torch.zeros(heads * dim, device="npu")
    starts = torch.tensor([0, 6, 12, 12], device="npu", dtype=torch.int32)
    indices = torch.tensor([3, 1, 0], device="npu", dtype=torch.int32)
    counts = torch.tensor([2, 6, 0], device="npu", dtype=torch.int32)
    layer = SimpleNamespace(
        kv_cache=(conv, state, values, kg),
        A_log=a_log,
        dt_bias=bias,
        gate_lower_bound=-5.0,
        local_num_heads=heads,
        head_dim=dim,
    )
    context = AscendCKDARecoverSSMCommitContext.create([layer], spec_query_len=width, max_num_reqs=3)

    def verify():
        return kda_recoverssm_verify_ascendc(
            q, k, v, gate, beta, a_log, bias, -5.0, state, values, kg, starts, indices, width
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        verify()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = verify()
    torch.npu.current_stream().wait_stream(stream)
    for blocks, offsets, accepted in [
        ([3, 1, 0], [0, 6, 12, 12], [2, 6, 0]),
        ([2, 5, 0], [0, 6, 8, 8], [6, 1, 0]),
        ([2, 5, 0], [0, 6, 8, 8], [0, 0, 0]),
    ]:
        indices.copy_(torch.tensor(blocks, device="npu", dtype=torch.int32))
        starts.copy_(torch.tensor(offsets, device="npu", dtype=torch.int32))
        counts.copy_(torch.tensor(accepted, device="npu", dtype=torch.int32))
        k.add_(0.125)
        initial = state.clone()
        expected_output, expected_state = reference(
            q, k, v, gate, beta, a_log, bias, -5.0, initial, offsets, blocks, accepted
        )
        graph.replay()
        torch.testing.assert_close(output.float(), expected_output, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(state, initial, atol=0, rtol=0)
        context.commit(counts, indices, starts)
        torch.testing.assert_close(state, expected_state, atol=2e-3, rtol=2e-3)
