# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, default-off GLM RecoverSSM configuration for MRV2."""

from vllm.config import VllmConfig


def use_glm_recoverssm(vllm_config: VllmConfig) -> bool:
    enabled = vllm_config.additional_config.get("kda_recoverssm", False)
    if type(enabled) is not bool:
        raise ValueError("additional_config.kda_recoverssm must be a boolean")
    if not enabled:
        return False
    if not vllm_config.use_v2_model_runner:
        raise ValueError("GLM RecoverSSM requires Model Runner V2")
    if vllm_config.num_speculative_tokens <= 0:
        raise ValueError("GLM RecoverSSM requires speculative decoding")
    if vllm_config.speculative_config.method not in ("mtp", "deepseek_mtp"):
        raise ValueError("GLM RecoverSSM currently requires MTP")
    cache = vllm_config.cache_config
    if cache.mamba_cache_mode != "none" or cache.enable_prefix_caching:
        raise ValueError("GLM RecoverSSM currently requires Mamba cache mode none and prefix caching disabled")
    parallel = vllm_config.parallel_config
    if parallel.pipeline_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        raise ValueError("GLM RecoverSSM currently requires PP=1 and PCP=1")
    if parallel.decode_context_parallel_size != 1:
        raise ValueError("GLM RecoverSSM currently requires DCP=1")
    if vllm_config.kv_transfer_config is not None:
        raise ValueError("GLM RecoverSSM does not yet support KV transfer")
    # The verified vLLM only accepts --use-replayssm for Kimi. Activate its
    # shared worker/cache contracts from this GLM-specific extension instead.
    cache.use_kda_recoverssm = True
    return True
