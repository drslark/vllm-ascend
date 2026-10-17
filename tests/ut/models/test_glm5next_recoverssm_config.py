# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm


def make_config():
    return SimpleNamespace(
        additional_config={"kda_recoverssm": True},
        use_v2_model_runner=True,
        num_speculative_tokens=5,
        speculative_config=SimpleNamespace(method="mtp"),
        cache_config=SimpleNamespace(mamba_cache_mode="none", enable_prefix_caching=False, use_kda_recoverssm=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        kv_transfer_config=None,
    )


def test_default_does_not_change_native_cache_contract():
    config = make_config()
    config.additional_config = {}
    assert not use_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize("method", ["mtp", "deepseek_mtp"])
def test_enabled_activates_upstream_cache_and_worker_contract(method):
    config = make_config()
    config.speculative_config.method = method
    assert use_glm_recoverssm(config)
    assert config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize(
    "path,value",
    [
        ("use_v2_model_runner", False),
        ("num_speculative_tokens", 0),
        ("speculative_config.method", "eagle3"),
        ("cache_config.mamba_cache_mode", "align"),
        ("cache_config.enable_prefix_caching", True),
        ("parallel_config.pipeline_parallel_size", 2),
        ("parallel_config.prefill_context_parallel_size", 2),
        ("parallel_config.decode_context_parallel_size", 2),
        ("kv_transfer_config", object()),
    ],
)
def test_unsupported_execution_fails_before_cache_allocation(path, value):
    config = make_config()
    owner = config
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)
    with pytest.raises(ValueError, match="GLM RecoverSSM"):
        use_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize("value", [0, 1, "1", None])
def test_option_requires_boolean(value):
    config = make_config()
    config.additional_config["kda_recoverssm"] = value
    with pytest.raises(ValueError, match="must be a boolean"):
        use_glm_recoverssm(config)
