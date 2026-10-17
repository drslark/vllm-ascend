# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from inspect import signature

from vllm.models.kimi_k3.nvidia.ops import recoverssm as upstream

from vllm_ascend.ops.triton.kda import recoverssm as ascend


def test_verify_and_commit_keep_upstream_parameter_contracts():
    for actual, expected in (
        (ascend.kda_recoverssm_verify, upstream.kda_recoverssm_verify),
        (ascend.KDARecoverSSMCommitContext.create, upstream.KDARecoverSSMCommitContext.create),
        (ascend.KDARecoverSSMCommitContext.commit, upstream.KDARecoverSSMCommitContext.commit),
    ):
        left, right = signature(actual).parameters, signature(expected).parameters
        assert list(left) == list(right)
        for name in left:
            assert left[name].kind == right[name].kind
            assert left[name].default == right[name].default
