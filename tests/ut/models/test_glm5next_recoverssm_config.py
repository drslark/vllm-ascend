# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm


def make_config():
    return SimpleNamespace(
        additional_config={"kda_recoverssm": True},
        cache_config=SimpleNamespace(use_kda_recoverssm=False),
    )


def test_ascend_configuration_registers_default_off_recoverssm_switch():
    assert AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig()).kda_recoverssm is False
    config = AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig(), kda_recoverssm=True)
    assert config.kda_recoverssm is True


def test_default_does_not_change_native_cache_contract():
    config = make_config()
    config.additional_config = {}
    assert not use_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm


def test_enabled_activates_upstream_cache_and_worker_contract():
    config = make_config()
    assert use_glm_recoverssm(config)
    assert config.cache_config.use_kda_recoverssm
