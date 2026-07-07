# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft model for Ascend.

DSpark weights are stored under the target checkpoint's ``mtp.*`` namespace,
but the draft path is a block drafter rather than the ordinary serial MTP
module. The target model provides selected layer hidden states; this model
projects them into the draft attention context and emits a full draft block.
"""

import typing
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV4Attention,
    _apply_dsv4_rope,
    _apply_dsv4_rope_tail,
    _grouped_wo_a_projection,
    _hc_head_torch,
    _linear_output,
    _make_deepseek_v4_expert_params_mapping,
    _wo_a_weight_for_eager_projection,
)
from vllm_ascend.ops.dspark_attention import (
    dspark_attention_from_standard_cache,
    dspark_attention_from_standard_cache_sas,
)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.scale$")
_LAYER_ID_RE = re.compile(r"model\.layers\.(\d+)\.")


def _draft_quant_config(vllm_config: VllmConfig):
    assert vllm_config.speculative_config is not None
    draft_config = vllm_config.speculative_config.draft_model_config.hf_config
    if getattr(draft_config, "dspark_mtp_dequantized_to_bf16", False):
        return None
    return vllm_config.quant_config


def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", 3)
    return int(num_layers or 3)


def _sync_npu_device_for_standard_pta(tensor: torch.Tensor) -> None:
    if tensor.device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _select_layer_value(
    value: typing.Any,
    layer_idx: int,
    layer_key: str,
    layer_prefix: str,
):
    if isinstance(value, dict):
        if layer_prefix in value:
            return value[layer_prefix]
        if layer_key in value:
            return value[layer_key]
        if layer_idx in value:
            return value[layer_idx]
        return None
    if isinstance(value, (list, tuple)):
        return value[layer_idx]
    return value


def _get_layer_prefix(layer: nn.Module, layer_key: str) -> str:
    return getattr(
        getattr(getattr(getattr(layer, "self_attn", None), "dsa_attn", None), "swa_cache_layer", None),
        "prefix",
        layer_key,
    )


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    """DSpark sliding-window attention with an internal eager context cache."""

    def __init__(self, *args, **kwargs) -> None:
        config = kwargs["config"]
        super().__init__(*args, **kwargs)
        self.compress_ratio = 1
        self.dsa_attn.compress_ratio = 1
        self.block_size = int(config.dspark_block_size)

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, self.head_dim).contiguous()

    def _store_standard_swa_kv(
        self,
        shared_kv: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        if slot_mapping is None or slot_mapping.numel() == 0:
            return

        swa_cache_layer = self.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        if swa_kv_cache is None:
            return
        while isinstance(swa_kv_cache, (list, tuple)) and len(swa_kv_cache) == 1:
            swa_kv_cache = swa_kv_cache[0]

        from vllm_ascend.device.device_op import DeviceOperator

        slot_mapping = slot_mapping.to(device=shared_kv.device, dtype=torch.int32)
        valid = slot_mapping >= 0 if slot_mapping.ndim == 1 else torch.all(slot_mapping >= 0, dim=-1)
        if not bool(torch.any(valid).item()):
            return
        if not bool(torch.all(valid).item()):
            shared_kv = shared_kv[valid]
            slot_mapping = slot_mapping[valid]
        if slot_mapping.ndim == 1:
            slot_mapping = DeviceOperator.format_dsa_slot_mapping(slot_mapping, swa_cache_layer.block_size)
        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)
        # The PTA reference reads the raw SWA cache immediately after scatter,
        # outside the normal DSA attention op stream choreography.
        _sync_npu_device_for_standard_pta(shared_kv)

    def _standard_query_slot_mapping_from_block_table(
        self,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if block_table is None:
            return slot_mapping

        swa_cache_layer = self.dsa_attn.swa_cache_layer
        cache_block_size = int(swa_cache_layer.block_size)
        out = torch.full_like(positions, -1, dtype=torch.int32)
        valid = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
        if slot_mapping is not None:
            slot_mapping = slot_mapping.to(device=positions.device)
            valid = slot_mapping >= 0 if slot_mapping.ndim == 1 else torch.all(slot_mapping >= 0, dim=-1)

        pos_long = positions.to(torch.long)
        if token_to_req_indices is not None:
            if token_to_req_indices.numel() < positions.numel():
                raise ValueError(
                    "DSpark token_to_req_indices must cover query tokens: "
                    f"token_to_req_indices={token_to_req_indices.numel()}, positions={positions.numel()}"
                )
            token_to_req = token_to_req_indices[: positions.numel()].to(
                device=positions.device,
                dtype=torch.long,
            )
            for req_idx in range(block_table.shape[0]):
                row_indices = torch.nonzero(token_to_req == req_idx, as_tuple=False).flatten()
                row_indices = row_indices[row_indices < positions.numel()]
                if row_indices.numel() == 0:
                    continue
                block_pos = pos_long.index_select(0, row_indices)
                block_nums = block_pos // cache_block_size
                block_offsets = block_pos % cache_block_size
                block_ids = (
                    block_table[req_idx]
                    .to(device=positions.device, dtype=torch.long)
                    .index_select(
                        0,
                        block_nums,
                    )
                )
                out[row_indices] = (block_ids * cache_block_size + block_offsets).to(torch.int32)
        else:
            for block_offset in range(0, positions.numel(), self.block_size):
                block_end = min(block_offset + self.block_size, positions.numel())
                req_idx = block_offset // self.block_size
                if req_idx >= block_table.shape[0]:
                    continue
                block_pos = pos_long[block_offset:block_end]
                block_nums = block_pos // cache_block_size
                block_offsets = block_pos % cache_block_size
                block_ids = (
                    block_table[req_idx].to(device=positions.device, dtype=torch.long).index_select(0, block_nums)
                )
                out[block_offset:block_end] = (block_ids * cache_block_size + block_offsets).to(torch.int32)
        out.masked_fill_(~valid, -1)
        return out

    def _run_standard_dspark_attention(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        swa_cache_layer = self.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        # determine_available_memory runs before the paged swa_kv_cache /
        # block_table are wired up; in profile/dummy we return a zero tensor so
        # standard-DSA covers that path. In production these must exist -- a
        # missing block_table/swa_kv_cache outside profile is a bug.
        if block_table is None or swa_kv_cache is None:
            fc = get_forward_context()
            if fc is not None and getattr(fc, "in_profile_run", False):
                return torch.zeros_like(q)
            raise RuntimeError(
                "DSpark standard-DSA missing block_table or swa_kv_cache in "
                "production; private cache fallback is disabled."
            )

        sas_out = dspark_attention_from_standard_cache_sas(
            q,
            swa_kv_cache,
            block_table,
            positions,
            slot_mapping,
            self.attn_sink[: self.n_local_heads],
            self.block_size,
            int(self.window_size),
            int(swa_cache_layer.block_size),
            float(self.scale),
            query_start_loc=dspark_query_start_loc,
            seq_lens=dspark_seq_lens,
            token_to_req_indices=dspark_token_to_req_indices,
        )
        if sas_out is not None:
            return sas_out

        return dspark_attention_from_standard_cache(
            q,
            swa_kv_cache,
            block_table,
            positions,
            slot_mapping,
            self.attn_sink[: self.n_local_heads],
            self.block_size,
            int(self.window_size),
            int(swa_cache_layer.block_size),
            float(self.scale),
            query_start_loc=dspark_query_start_loc,
            seq_lens=dspark_seq_lens,
            token_to_req_indices=dspark_token_to_req_indices,
        )

    def precompute_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        if positions.numel() == 0:
            return
        shared_kv = self._project_shared_kv(main_x, positions)
        self._store_standard_swa_kv(shared_kv, context_slot_mapping)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shared_kv = self._project_shared_kv(hidden_states, positions)
        qr = self.q_norm(_linear_output(self.wq_a, hidden_states))
        q = _linear_output(self.wq_b, qr).view(-1, self.n_local_heads, self.head_dim)
        q = self.q_norm_without_weight(q)
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = _apply_dsv4_rope(self.rotary_emb, positions, q_pe)
        q = torch.cat([q_nope, q_pe], dim=-1)
        standard_slot_mapping = self._standard_query_slot_mapping_from_block_table(
            positions,
            slot_mapping,
            block_table,
            dspark_token_to_req_indices,
        )
        self._store_standard_swa_kv(shared_kv, standard_slot_mapping)
        standard_attn_out = self._run_standard_dspark_attention(
            q,
            positions,
            standard_slot_mapping,
            block_table,
            dspark_query_start_loc,
            dspark_seq_lens,
            dspark_token_to_req_indices,
        )
        if standard_attn_out is None:
            raise RuntimeError(
                "DSpark standard-DSA attention returned None; private cache "
                "fallback is disabled. standard-DSA has a coverage gap "
                "(missing block_table or swa_kv_cache)."
            )
        attn_out = standard_attn_out

        attn_out = _apply_dsv4_rope_tail(
            self.rotary_emb,
            positions,
            attn_out,
            inverse=True,
        )
        group_dim = self.n_local_heads * self.head_dim // self.n_local_groups
        attn_out = attn_out.reshape(-1, self.n_local_groups, group_dim)
        wo_a = _wo_a_weight_for_eager_projection(
            self.wo_a.weight,
            self.n_local_groups,
            self.o_lora_rank,
            group_dim,
        )
        z = _grouped_wo_a_projection(attn_out, wo_a).flatten(1)
        projected = _linear_output(self.wo_b, z)
        return projected


class DeepseekV4DSparkDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            config=config,
            topk_indices_buffer=None,
            is_draft_layer=True,
            attn_cls=DeepseekV4DSparkAttention,
            quant_config_override=_draft_quant_config(vllm_config),
            use_quant_config_override=True,
        )
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # MHC pre/post reuse the upstream NPU fused ops ``npu_hc_pre_v2`` /
        # ``npu_hc_post`` inherited from DeepseekV2DecoderLayer instead of the
        # torch reference. The torch reference fused the previous step's post
        # with the current step's pre into one call; here they are two NPU ops,
        # numerically equivalent at the bf16 ulp level (the NPU op hardcodes the
        # post alpha of 2.0; equivalence was verified against a torch reference
        # before that reference was removed).
        if residual is None:
            residual = hidden_states
            hidden_states, post_mix, res_mix = self.hc_pre(
                hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        else:
            assert post_mix is not None and res_mix is not None
            residual = self.hc_post(hidden_states, residual, post_mix, res_mix)
            hidden_states, post_mix, res_mix = self.hc_pre(
                residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        hidden_states = self.input_layernorm(hidden_states)
        attn_kwargs = {
            "slot_mapping": slot_mapping,
            "block_table": block_table,
        }
        if dspark_query_start_loc is not None or dspark_seq_lens is not None or dspark_token_to_req_indices is not None:
            attn_kwargs.update(
                dspark_query_start_loc=dspark_query_start_loc,
                dspark_seq_lens=dspark_seq_lens,
                dspark_token_to_req_indices=dspark_token_to_req_indices,
            )
        hidden_states = self.self_attn(
            positions,
            hidden_states,
            **attn_kwargs,
        )

        residual = self.hc_post(hidden_states, residual, post_mix, res_mix)
        hidden_states, post_mix, res_mix = self.hc_pre(
            residual, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, input_ids)
        return hidden_states, residual, post_mix, res_mix


class DSparkMarkovHead(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=f"{prefix}.markov_w1",
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            config.dspark_markov_rank,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.markov_w2, markov_embed)


class DeepseekV4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.block_size = int(config.dspark_block_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): DeepseekV4DSparkDecoderLayer(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx + idx}"),
                )
                for idx in range(self.num_dspark_layers)
            }
        )

        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        self.markov_head = DSparkMarkovHead(
            config,
            maybe_prefix(prefix, f"layers.{last_layer_idx}.markov_head"),
        )
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(_linear_output(self.main_proj, aux_hidden_states))

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor
        | list[torch.Tensor | None]
        | tuple[torch.Tensor | None, ...]
        | dict[str, torch.Tensor | None]
        | dict[int, torch.Tensor | None]
        | None = None,
    ) -> None:
        if context_states.numel() == 0:
            return
        for layer_idx, (layer_key, layer) in enumerate(self.layers.items()):
            layer_prefix = _get_layer_prefix(layer, layer_key)
            layer_context_slot_mapping = _select_layer_value(
                context_slot_mapping,
                layer_idx,
                layer_key,
                layer_prefix,
            )
            layer.self_attn.precompute_context_kv(
                context_states,
                context_positions,
                context_slot_mapping=layer_context_slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | dict[str, torch.Tensor] | dict[int, torch.Tensor] | None = None,
        block_table: torch.Tensor | dict[str, torch.Tensor] | dict[int, torch.Tensor] | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids).unsqueeze(-2).repeat(1, self.hc_mult, 1)
        residual = post_mix = res_mix = None
        for layer_idx, (layer_key, layer) in enumerate(self.layers.items()):
            layer_prefix = _get_layer_prefix(layer, layer_key)
            layer_kwargs = {
                "positions": positions,
                "hidden_states": hidden_states,
                "residual": residual,
                "post_mix": post_mix,
                "res_mix": res_mix,
                "input_ids": input_ids,
                "slot_mapping": _select_layer_value(slot_mapping, layer_idx, layer_key, layer_prefix),
                "block_table": _select_layer_value(block_table, layer_idx, layer_key, layer_prefix),
            }
            if (
                dspark_query_start_loc is not None
                or dspark_seq_lens is not None
                or dspark_token_to_req_indices is not None
            ):
                layer_kwargs.update(
                    dspark_query_start_loc=dspark_query_start_loc,
                    dspark_seq_lens=dspark_seq_lens,
                    dspark_token_to_req_indices=dspark_token_to_req_indices,
                )
            layer_output = layer(**layer_kwargs)
            if isinstance(layer_output, tuple) and len(layer_output) == 4:
                hidden_states, residual, post_mix, res_mix = layer_output
            else:
                hidden_states = layer_output
        head_hidden = self.compute_head_hidden(hidden_states, residual, post_mix, res_mix)
        return head_hidden

    def compute_head_hidden(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual is not None and post_mix is not None and res_mix is not None:
            # Final MHC post of the last decoder layer's output, reusing the
            # inherited hc_post (DeepseekV2DecoderLayer.hc_post -> npu_hc_post).
            last_layer = self.layers[str(self.mtp_start_layer_idx + self.num_dspark_layers - 1)]
            hidden_states = last_layer.hc_post(hidden_states, residual, post_mix, res_mix)
        if hidden_states.dim() == 2:
            return hidden_states
        return _hc_head_torch(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(markov_embed)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        head_hidden = self.compute_head_hidden(hidden_states)
        return logits_processor(lm_head, self.norm(head_hidden))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return _make_deepseek_v4_expert_params_mapping(
            self,
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            finalize = getattr(layer.mlp, "finalize_mega_moe_weights", None)
            if finalize is not None:
                finalize()


@support_torch_compile
class DeepSeekV4DSparkMTP(nn.Module, DeepseekV2MixtureOfExperts):
    # DSpark draft embed/head are aliases of the target model, matching
    # upstream vLLM's DSparkDeepseekV4ForCausalLM contract.
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = _draft_quant_config(vllm_config)
        self.model = DeepseekV4DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.set_moe_parameters_from_layers(self.config, self.model.layers.values())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            dspark_query_start_loc=dspark_query_start_loc,
            dspark_seq_lens=dspark_seq_lens,
            dspark_token_to_req_indices=dspark_token_to_req_indices,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        del spec_step_idx
        return self.model.compute_logits(
            hidden_states,
            self.lm_head,
            self.logits_processor,
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_bias(markov_embed)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor
        | list[torch.Tensor | None]
        | tuple[torch.Tensor | None, ...]
        | dict[str, torch.Tensor | None]
        | dict[int, torch.Tensor | None]
        | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
            ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
            ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        missing_mtp_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank
        head_end = head_start + heads_per_rank
        expert_mapping = self.model.get_expert_mapping()
        expert_scale_suffix = (
            ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4" else ".weight_scale_inv"
        )
        start_layer_idx = self.config.num_hidden_layers
        last_layer_idx = start_layer_idx + self.model.num_dspark_layers - 1

        for name, loaded_weight in weights:
            if name == "embed.weight":
                embed_name = "model.embed_tokens.weight"
                param = params_dict[embed_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(embed_name)
                continue
            if name == "head.weight":
                head_name = "lm_head.weight"
                param = params_dict[head_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(head_name)
                continue
            mapped_name = self._remap_dspark_name(name)
            if mapped_name is None:
                continue
            name = mapped_name
            if name.startswith(f"model.layers.{last_layer_idx}.hc_head_"):
                canonical_name = name.replace(f"model.layers.{last_layer_idx}.", "model.", 1)
                if canonical_name in params_dict:
                    name = canonical_name
            if name.endswith(".scale"):
                suffix = expert_scale_suffix if _EXPERT_SCALE_RE.search(name) else ".weight_scale"
                name = name.removesuffix(".scale") + suffix
                if name not in params_dict:
                    continue
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if ".experts." in name or f".{weight_name}." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    missing_mtp_params.add(mapped)
                    break
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(mapped)
                break
            else:
                if ".experts." in name:
                    matched_expert_mapping = False
                    if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for param_name, weight_name, expert_id, expert_shard_id in expert_mapping:
                        if weight_name not in name:
                            continue
                        matched_expert_mapping = True
                        mapped = name.replace(weight_name, param_name)
                        if mapped not in params_dict:
                            continue
                        param = params_dict[mapped]
                        weight_loader = typing.cast(typing.Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            loaded_params.add(mapped)
                            break
                    if not matched_expert_mapping:
                        missing_mtp_params.add(name)
                    continue
                if "attn_sink" in name:
                    if name not in params_dict:
                        missing_mtp_params.add(name)
                        continue
                    narrow = loaded_weight[head_start:head_end]
                    with torch.no_grad():
                        params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if name not in params_dict:
                    missing_mtp_params.add(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if missing_mtp_params:
            raise ValueError(
                "DSpark speculative decoding checkpoint weights did not match model parameters: "
                f"{sorted(missing_mtp_params)}"
            )

        loaded_layer_ids: set[int] = set()
        for param_name in loaded_params:
            match = _LAYER_ID_RE.search(param_name)
            if match:
                loaded_layer_ids.add(int(match.group(1)))
        for layer_idx in range(start_layer_idx, start_layer_idx + self.model.num_dspark_layers):
            if layer_idx not in loaded_layer_ids:
                raise ValueError(f"DSpark speculative decoding layer {layer_idx} weights missing from checkpoint.")
        required_params = {
            f"model.layers.{start_layer_idx}.main_proj.weight",
            f"model.layers.{start_layer_idx}.main_norm.weight",
            f"model.layers.{last_layer_idx}.norm.weight",
            "model.hc_head_fn",
            "model.hc_head_base",
            "model.hc_head_scale",
            f"model.layers.{last_layer_idx}.markov_head.markov_w1.weight",
            f"model.layers.{last_layer_idx}.markov_head.markov_w2.weight",
        }
        missing_required = sorted(required_params - loaded_params)
        if missing_required:
            raise ValueError(
                f"DSpark speculative decoding required weights missing from checkpoint load: {missing_required}"
            )
        self.model.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _remap_dspark_name(self, name: str) -> str | None:
        match = re.match(r"mtp\.(\d+)\.(.*)", name)
        if match is None:
            return None
        stage_idx = int(match.group(1))
        layer_idx = self.config.num_hidden_layers + stage_idx
        rest = match.group(2)
        if rest.startswith("confidence_head."):
            return None
        name = f"model.layers.{layer_idx}.{rest}"
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        name = name.replace(".mlp.gate.bias", ".mlp.gate.e_score_correction_bias")
        return name


DSparkDeepseekV4ForCausalLM = DeepSeekV4DSparkMTP
