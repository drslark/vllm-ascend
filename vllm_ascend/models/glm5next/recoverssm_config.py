# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, default-off GLM KDA RecoverSSM configuration."""

from vllm.config import VllmConfig


def use_glm_recoverssm(vllm_config: VllmConfig) -> bool:
    return vllm_config.additional_config.get("kda_recoverssm", False)


def init_glm_recoverssm(vllm_config: VllmConfig) -> None:
    if not use_glm_recoverssm(vllm_config):
        return
    if vllm_config.speculative_config is None:
        raise ValueError("GLM RecoverSSM requires speculative decoding")
    # Model state is initialized after target/draft configuration construction.
    vllm_config.cache_config.use_kda_recoverssm = True
