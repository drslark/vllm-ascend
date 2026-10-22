# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# AscendMTPSpeculator layers the shared Ascend draft loop
# (AscendAutoRegressiveSpeculator) on upstream MTPSpeculator. All MLA-specific
# handling lives in the base, gated by is_mla.
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.model_states.recoverssm import AscendRecoverSSMState
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator


class AscendMTPSpeculator(AscendAutoRegressiveSpeculator, MTPSpeculator):
    """Ascend MTP speculator (MLA draft). All MLA handling is in the base
    (AscendAutoRegressiveSpeculator)"""

    recoverssm_state: AscendRecoverSSMState | None = None

    def propose(self, *args, **kwargs):
        draft_tokens = super().propose(*args, **kwargs)
        # Join target state commits after MTP and before KV transfer.
        if self.recoverssm_state is not None:
            self.recoverssm_state.wait_for_commit()
        return draft_tokens
