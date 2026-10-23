# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_ascend.ops.kda_recoverssm as ops


def test_verify_uses_readonly_native_state_and_clears_padding(monkeypatch):
    q, k, v, gate = (torch.ones(1, 4, 2, 128, dtype=torch.bfloat16) for _ in range(4))
    beta = torch.zeros(1, 4, 2, dtype=torch.bfloat16)
    state = torch.zeros(8, 2, 128, 128)
    values = torch.zeros(8, 2, 3, 129)
    kg = torch.zeros(8, 2, 3, 256, dtype=torch.bfloat16)
    starts, indices = torch.tensor([0, 2, 2]), torch.tensor([3, 0])
    events = []

    def recurrent(q_arg, k_arg, v_arg, gate_arg, beta_arg, state_arg, starts_arg, slots, *args, **kwargs):
        assert q_arg is q and k_arg is k and v_arg is v and gate_arg is gate and beta_arg is beta
        assert state_arg is state and starts_arg is starts
        assert kwargs == dict(lower_bound=-5.0, beta_is_preprocessed=False, inplace_final_state=False)
        torch.testing.assert_close(slots, torch.tensor([[3, 3, 3], [0, 0, 0]]))
        events.append("native")
        return torch.full_like(v, 9)

    def record(k_arg, v_arg, gate_arg, beta_arg, values_arg, kg_arg, starts_arg, indices_arg, width, **kwargs):
        assert k_arg is k and v_arg is v and gate_arg is gate and beta_arg is beta
        assert values_arg is values and kg_arg is kg and starts_arg is starts and indices_arg is indices
        assert width == 3 and kwargs["output"].shape == v.shape
        events.append("record")

    monkeypatch.setattr(ops, "run_recurrent_kda", recurrent)
    monkeypatch.setattr(ops, "_copy_inputs", record)
    output = ops.kda_recoverssm_verify_ascendc(
        q, k, v, gate, beta, torch.zeros(2), torch.zeros(256), -5.0, state, values, kg, starts, indices, 3
    )
    assert events == ["native", "record"]
    assert (output[:, :2] == 9).all() and (output[:, 2:] == 0).all()
    assert not state.any()


@pytest.mark.parametrize("width", [0, 9])
def test_native_width_limit_fails_before_kernel_launch(width):
    q = torch.empty(1, 0, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="width in"):
        ops.kda_recoverssm_verify_ascendc(q, q, q, q, None, None, None, -5.0, None, None, None, None, None, width)


def test_native_bf16_checkpoint_is_rejected_before_execution():
    q = torch.empty(1, 0, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires FP32 checkpoints"):
        ops.kda_recoverssm_verify_ascendc(
            q,
            q,
            q,
            q,
            torch.empty(1, 0, 1, dtype=q.dtype),
            None,
            None,
            -5.0,
            torch.zeros(8, 1, 128, 128, dtype=q.dtype),
            torch.empty(8, 1, 1, 129),
            torch.empty(8, 1, 1, 256, dtype=q.dtype),
            torch.tensor([0]),
            torch.empty(0, dtype=torch.int32),
            1,
        )
