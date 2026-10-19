# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, default-off GLM KDA RecoverSSM configuration."""

from vllm.config import VllmConfig


def use_glm_recoverssm(vllm_config: VllmConfig) -> bool:
    enabled = vllm_config.additional_config.get("kda_recoverssm", False)
    if enabled:
        # Activate the upstream worker/cache contracts for this GLM extension.
        vllm_config.cache_config.use_kda_recoverssm = True
    return enabled
