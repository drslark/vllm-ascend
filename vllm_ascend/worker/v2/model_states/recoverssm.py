# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


class AscendRecoverSSMState(RecoverSSMState):
    """Submit target state commits alongside the GLM MLA MTP forward."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.commit_stream = torch.npu.Stream(device=device)
        self.commit_done = torch.npu.Event()
        self._pending: tuple[object, ...] | None = None

    def commit_step(
        self,
        num_sampled: torch.Tensor | int,
        idx_mapping: torch.Tensor,
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        # Retain the metadata and borrowed sampling tensors until the main
        # stream has joined the commit, before their storage can be reused.
        self._pending = (self._step, num_sampled, idx_mapping, state_indices, num_accepted_tokens)
        self.commit_stream.wait_stream(torch.npu.current_stream(self.device))
        with torch.npu.stream(self.commit_stream):
            try:
                super().commit_step(
                    num_sampled,
                    idx_mapping,
                    state_indices=state_indices,
                    num_accepted_tokens=num_accepted_tokens,
                )
            finally:
                self.commit_done.record()

    def wait_for_commit(self) -> None:
        if self._pending is None:
            return
        # Device-side dependency only: do not block CPU submission or make
        # MTP wait before its forward has been enqueued.
        torch.npu.current_stream(self.device).wait_event(self.commit_done)
        self._pending = None
