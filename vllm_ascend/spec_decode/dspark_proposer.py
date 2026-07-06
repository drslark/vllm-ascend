# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from copy import copy
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.llm_base_proposer import greedy_sample


def _dspark_reduce_sample_enabled() -> bool:
    try:
        return bool(get_ascend_config().enable_reduce_sample)
    except RuntimeError:
        return False


def _dspark_greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    if _dspark_reduce_sample_enabled():
        return greedy_sample(logits)
    return logits.argmax(dim=-1)


class AscendDSparkProposer(AscendDflashProposer):
    """DSpark block proposer.

    DSpark uses vLLM's ``mtp`` method in user config, but its execution shape is
    closer to DFlash: target hidden states prepopulate draft K/V, then one
    anchor-first query block emits all speculative tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        draft_hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        if vllm_config.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "DSpark probabilistic draft sampling is not supported on the v1 "
                "model runner; use greedy (the default) instead."
            )
        dspark_target_layer_ids = getattr(draft_hf_config, "dspark_target_layer_ids", None)
        if dspark_target_layer_ids:
            self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size() * len(
                dspark_target_layer_ids
            )
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            self._dflash_hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
        self.method = "dflash"
        self.parallel_drafting = True
        self.block_size = self.num_speculative_tokens
        self.extra_slots_per_request = self.num_speculative_tokens
        self.net_num_new_slots_per_request = self.num_speculative_tokens
        self.needs_extra_input_slots = True
        self.is_rejected_token_mask: torch.Tensor | None = getattr(self, "is_rejected_token_mask", None)
        if self.is_rejected_token_mask is None:
            self.is_rejected_token_mask = torch.zeros(
                (self.max_num_tokens,),
                dtype=torch.bool,
                device=device,
            )
        self.is_masked_token_mask: torch.Tensor | None = getattr(self, "is_masked_token_mask", None)
        if self.is_masked_token_mask is None:
            self.is_masked_token_mask = torch.zeros(
                (self.max_num_tokens,),
                dtype=torch.bool,
                device=device,
            )
        self.parallel_drafting_token_id = getattr(
            draft_hf_config,
            "ptd_token_id",
            getattr(draft_hf_config, "dspark_noise_token_id", 0),
        )
        self.use_cuda_graph = False
        self.max_query_tokens = self.max_batch_size * self.num_speculative_tokens
        self.max_positions = self.max_num_tokens + self.max_query_tokens
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        # Markov block drafting: ``_markov_anchor_tokens`` is the target model's
        # next token (the seed that starts the Markov chain); ``_markov_draft_tokens``
        # holds each step's sampled token and feeds it back as the next step's
        # input. DSpark-only -- DFlash/Eagle sample in parallel without Markov.
        self._markov_anchor_tokens = torch.zeros(
            self.max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        # Per-token -> request index map consumed by the SAS attention op. Sliced
        # to num_query_total for real query tokens; padding slots in
        # [num_actual_tokens, num_input_tokens) are filled with -1.
        self._dspark_token_to_req_indices_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._dspark_token_to_req_indices: torch.Tensor | None = None
        # Cached slices of the runner's common_attn_metadata (cad): in
        # set_inputs_first_pass they alias cad.query_start_loc_cpu / cad.seq_lens;
        # in dummy_run/profile_run (no cad available) they are synthesized locally.
        # Kept under the _dspark_ prefix because llm_base already defines
        # self.query_start_loc for Eagle.
        self._dspark_query_start_loc: torch.Tensor | None = None
        self._dspark_seq_lens: torch.Tensor | None = None
        self._markov_draft_tokens = torch.zeros(
            (self.max_batch_size, self.num_speculative_tokens),
            dtype=torch.int64,
            device=device,
        )
        self._dspark_block_tables_by_gid: dict[int, torch.Tensor] = {}
        self._dspark_block_tables_by_layer: dict[str, torch.Tensor] = {}
        self._dspark_per_group_block_tables: dict[int, torch.Tensor] = {}
        self._dspark_per_group_slot_mappings: dict[int, torch.Tensor] = {}
        self._dspark_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._dspark_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._dspark_query_slot_mappings_by_gid: dict[int, torch.Tensor] = {}
        self._dspark_context_slot_mappings_by_gid: dict[int, torch.Tensor] = {}
        self._dspark_query_slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        self._dspark_context_slot_mappings_by_layer: dict[str, torch.Tensor] = {}

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        self._draft_attn_layer_names: set[str] = set()
        self.attn_layer_names: list[str] = []
        self.piece_all_attn_layer_name: list[list[str]] = [[] for _ in range(self.num_speculative_tokens)]
        self.draft_attn_groups: list[Any] = []
        self.kv_cache_gid = 0

        if hasattr(self.model, "get_draft_kv_cache_layer_names"):
            draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())
            self._draft_attn_layer_names = draft_attn_layer_names
            self.attn_layer_names = sorted(draft_attn_layer_names)
            self.piece_all_attn_layer_name = [
                [name for name in self.attn_layer_names] for _ in range(self.num_speculative_tokens)
            ]

            layers = get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            )

            for kv_cache_gid, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
                layer_names = [name for name in kv_cache_group_spec.layer_names if name in draft_attn_layer_names]
                if not layer_names:
                    continue

                attn_backend_layers: dict[tuple[str, Any], list[str]] = defaultdict(list)
                attn_backends: dict[tuple[str, Any], tuple[type[Any], Any]] = {}
                for layer_name in layer_names:
                    attn_backend = layers[layer_name].get_attn_backend()
                    kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                        kv_cache_spec = kv_cache_spec.kv_cache_specs[layer_name]
                    key = (attn_backend.full_cls_name(), kv_cache_spec)
                    attn_backends[key] = (attn_backend, kv_cache_spec)
                    attn_backend_layers[key].append(layer_name)

                for key, grouped_layer_names in attn_backend_layers.items():
                    attn_backend, kv_cache_spec = attn_backends[key]
                    metadata_builder = attn_backend.get_builder_cls()(
                        kv_cache_spec,
                        grouped_layer_names,
                        self.vllm_config,
                        self.device,
                    )
                    self.draft_attn_groups.append(
                        AttentionGroup(
                            attn_backend,
                            grouped_layer_names,
                            kv_cache_spec,
                            kv_cache_gid,
                            [metadata_builder],
                        )
                    )

            if self.draft_attn_groups:
                self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
                self.kernel_block_size = int(self.draft_attn_groups[0].kv_cache_spec.block_size)
                return
            raise RuntimeError(
                "DSpark standard-cache path requires registered draft attention "
                f"groups. Missing layers: {sorted(draft_attn_layer_names)}"
            )

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._dspark_per_group_block_tables[gid] = block_table
        self._dspark_per_group_slot_mappings[gid] = slot_mapping

    def _slot_mapping_buffer_for_gid(self, gid: int, *, context: bool) -> torch.Tensor:
        if gid == getattr(self, "kv_cache_gid", 0):
            return self._context_slot_mapping_buffer if context else self._slot_mapping_buffer
        buffers = self._dspark_context_slot_mapping_buffers if context else self._dspark_query_slot_mapping_buffers
        buf = buffers.get(gid)
        if buf is None:
            size = self.max_num_tokens if context else self.max_query_tokens
            buf = torch.zeros(size, dtype=torch.int32, device=self.device)
            buffers[gid] = buf
        return buf

    def _layer_map_from_gid_map(self, gid_map: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        per_layer: dict[str, torch.Tensor] = {}
        for attn_group in getattr(self, "draft_attn_groups", []):
            value = gid_map.get(attn_group.kv_cache_group_id)
            if value is None:
                continue
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = value
        return per_layer

    @staticmethod
    def _slice_tensor_map(tensors: dict[str, torch.Tensor], num_tokens: int) -> dict[str, torch.Tensor]:
        return {name: tensor[:num_tokens] for name, tensor in tensors.items()}

    @staticmethod
    def _get_block_table_device_tensor(block_table, batch_size: int) -> torch.Tensor:
        try:
            return block_table.get_device_tensor(batch_size)
        except TypeError:
            return block_table.get_device_tensor()

    def _build_standard_dsa_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_input_tokens: int,
        num_actual_tokens: int,
    ) -> list[dict[str, Any]]:
        if not self.draft_attn_groups:
            return []

        if num_input_tokens > num_actual_tokens:
            self.positions[num_actual_tokens:num_input_tokens].fill_(0)
            self._slot_mapping_buffer[num_actual_tokens:num_input_tokens].fill_(-1)

        base_cm = common_attn_metadata
        base_cm.positions = self.positions[:num_input_tokens]
        base_cm.slot_mapping = self._slot_mapping_buffer[:num_input_tokens]
        base_cm.num_input_tokens = num_input_tokens
        base_cm.num_actual_tokens = num_actual_tokens
        base_cm.causal = False
        base_cm.attn_state = AscendAttentionState.ChunkedPrefill
        token_to_req_indices = getattr(self, "_dspark_token_to_req_indices_buffer", None)
        if isinstance(token_to_req_indices, torch.Tensor):
            base_cm.token_to_req_indices = token_to_req_indices[:num_input_tokens]

        per_layer_attn_metadata: dict[str, Any] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            common_attn_metadata = copy(base_cm)
            block_table = getattr(self, "_dspark_block_tables_by_gid", {}).get(gid)
            if block_table is not None:
                common_attn_metadata.block_table_tensor = block_table[: common_attn_metadata.num_reqs]
            slot_mapping = getattr(self, "_dspark_query_slot_mappings_by_gid", {}).get(gid)
            if slot_mapping is not None:
                common_attn_metadata.slot_mapping = slot_mapping[:num_input_tokens]
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata,
                draft_index=1,
                block_size=attn_group.kv_cache_spec.block_size,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return [per_layer_attn_metadata]

    def _pad_draft_query_buffers(
        self,
        num_actual_tokens: int,
        num_input_tokens: int,
    ) -> None:
        if num_input_tokens <= num_actual_tokens:
            return

        self.input_ids[num_actual_tokens:num_input_tokens].fill_(self.parallel_drafting_token_id)
        self.positions[num_actual_tokens:num_input_tokens].fill_(0)
        self._slot_mapping_buffer[num_actual_tokens:num_input_tokens].fill_(-1)
        token_to_req_indices = getattr(self, "_dspark_token_to_req_indices_buffer", None)
        if isinstance(token_to_req_indices, torch.Tensor):
            token_to_req_indices[num_actual_tokens:num_input_tokens].fill_(-1)
        for buf in getattr(self, "_dspark_query_slot_mapping_buffers", {}).values():
            buf[num_actual_tokens:num_input_tokens].fill_(-1)

    def _get_draft_block_table_for_gid(
        self,
        cad: CommonAttentionMetadata,
        batch_size: int,
        gid: int,
    ) -> torch.Tensor | None:
        block_table = getattr(self, "_dspark_per_group_block_tables", {}).get(gid)
        input_batch = getattr(getattr(self, "runner", None), "input_batch", None)
        block_tables = getattr(input_batch, "block_table", None)
        if block_table is None and block_tables is not None:
            try:
                draft_block_table = block_tables[gid]
            except (IndexError, KeyError, TypeError):
                draft_block_table = None
            if draft_block_table is not None:
                block_table = AscendDSparkProposer._get_block_table_device_tensor(
                    draft_block_table,
                    batch_size,
                )
        if block_table is None and gid == getattr(self, "kv_cache_gid", 0):
            block_table = getattr(cad, "block_table_tensor", None)
        if block_table is None:
            return None
        block_table = block_table[:batch_size]
        # Ascend block-table tensors are reused by the runner; DSpark consumes
        # them after query slot mappings have been built.
        block_table = block_table.clone()
        return block_table

    def _get_draft_block_tables(
        self,
        cad: CommonAttentionMetadata,
        batch_size: int,
    ) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
        if not getattr(self, "draft_attn_groups", []):
            return {}, {}
        by_gid: dict[int, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in by_gid:
                continue
            block_table = self._get_draft_block_table_for_gid(cad, batch_size, gid)
            if block_table is not None:
                by_gid[gid] = block_table
        return by_gid, self._layer_map_from_gid_map(by_gid)

    def _slot_mapping_from_block_table(
        self,
        positions: torch.Tensor,
        req_idx: int,
        block_table: torch.Tensor,
        block_size: int | None = None,
    ) -> torch.Tensor:
        if block_size is None:
            block_size = self.kernel_block_size
        block_nums = positions // block_size
        block_offsets = positions % block_size
        block_ids = block_table[req_idx].index_select(0, block_nums.long())
        return block_ids.to(torch.int32) * block_size + block_offsets

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        del (
            target_token_ids,
            token_indices_to_sample,
            req_scheduled_tokens,
            long_seq_metadata,
            num_prefill_reqs,
            num_decode_reqs,
        )
        batch_size = cad.num_reqs
        block_size = self.num_speculative_tokens
        num_query_total = batch_size * block_size
        has_num_rejected = num_rejected_tokens_gpu is not None
        token_to_req_capacity = max(int(self.positions.numel()), num_query_total)
        token_to_req_indices = getattr(self, "_dspark_token_to_req_indices_buffer", None)
        if not isinstance(token_to_req_indices, torch.Tensor) or token_to_req_indices.numel() < token_to_req_capacity:
            token_to_req_indices = torch.empty(
                token_to_req_capacity,
                dtype=torch.int32,
                device=self.device,
            )
            self._dspark_token_to_req_indices_buffer = token_to_req_indices
        primary_gid = getattr(self, "kv_cache_gid", 0)
        block_tables_by_gid, block_tables_by_layer = self._get_draft_block_tables(cad, batch_size)
        self._dspark_block_tables_by_gid = block_tables_by_gid
        self._dspark_block_tables_by_layer = block_tables_by_layer
        self._dspark_query_slot_mappings_by_gid = {}
        self._dspark_context_slot_mappings_by_gid = {}
        self._dspark_query_slot_mappings_by_layer = {}
        self._dspark_context_slot_mappings_by_layer = {}
        self._markov_anchor_tokens[:batch_size].copy_(next_token_ids)
        if batch_size < self._markov_anchor_tokens.shape[0]:
            self._markov_anchor_tokens[batch_size:].fill_(0)

        context_cursor = 0
        for req_idx in range(batch_size):
            ctx_start = int(cad.query_start_loc[req_idx].item())
            ctx_end = int(cad.query_start_loc[req_idx + 1].item())
            ctx_len = ctx_end - ctx_start
            if ctx_len == 0:
                continue
            out_end = context_cursor + ctx_len
            self._dflash_hidden_states[context_cursor:out_end] = target_hidden_states[ctx_start:ctx_end]
            self._context_positions_buffer[context_cursor:out_end] = target_positions[ctx_start:ctx_end]
            draft_attn_groups = getattr(self, "draft_attn_groups", [])
            if block_tables_by_gid and draft_attn_groups:
                for attn_group in draft_attn_groups:
                    gid = attn_group.kv_cache_group_id
                    gid_block_table = block_tables_by_gid.get(gid)
                    if gid_block_table is None:
                        continue
                    self._slot_mapping_buffer_for_gid(gid, context=True)[context_cursor:out_end] = (
                        self._slot_mapping_from_block_table(
                            target_positions[ctx_start:ctx_end],
                            req_idx,
                            gid_block_table,
                            int(attn_group.kv_cache_spec.block_size),
                        )
                    )
            context_cursor = out_end
        self._dflash_num_context = context_cursor
        if block_tables_by_gid:
            self._dspark_context_slot_mappings_by_gid = {
                gid: self._slot_mapping_buffer_for_gid(gid, context=True)[:context_cursor]
                for gid in block_tables_by_gid
            }
            self._dspark_context_slot_mappings_by_layer = self._layer_map_from_gid_map(
                self._dspark_context_slot_mappings_by_gid
            )

        token_indices_to_sample = torch.arange(
            num_query_total,
            dtype=torch.int32,
            device=self.device,
        )

        for req_idx in range(batch_size):
            ctx_start = int(cad.query_start_loc[req_idx].item())
            ctx_end = int(cad.query_start_loc[req_idx + 1].item())
            valid_ctx_end = ctx_end
            if has_num_rejected:
                assert num_rejected_tokens_gpu is not None
                valid_ctx_end -= int(num_rejected_tokens_gpu[req_idx].item())
            last_pos = target_positions[valid_ctx_end - 1]
            out_start = req_idx * block_size
            out_end = out_start + block_size
            self.positions[out_start:out_end] = last_pos + 1 + self.arange_dflash[:block_size]
            self.input_ids[out_start] = next_token_ids[req_idx]
            if block_size > 1:
                self.input_ids[out_start + 1 : out_end] = self.parallel_drafting_token_id
            token_to_req_indices[out_start:out_end] = req_idx

            draft_attn_groups = getattr(self, "draft_attn_groups", [])
            if block_tables_by_gid and draft_attn_groups:
                for attn_group in draft_attn_groups:
                    gid = attn_group.kv_cache_group_id
                    gid_block_table = block_tables_by_gid.get(gid)
                    if gid_block_table is None:
                        continue
                    self._slot_mapping_buffer_for_gid(gid, context=False)[out_start:out_end] = (
                        self._slot_mapping_from_block_table(
                            self.positions[out_start:out_end],
                            req_idx,
                            gid_block_table,
                            int(attn_group.kv_cache_spec.block_size),
                        )
                    )

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dflash[: batch_size + 1] * block_size
        cad.seq_lens = effective_seq_lens + block_size
        cad.query_start_loc_cpu = (torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * block_size).to(
            torch.int32
        )

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [block_size] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = block_size

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = block_size
        cad.max_seq_len = cad.max_seq_len + block_size
        cad.slot_mapping = self._slot_mapping_buffer[:num_query_total]
        if block_tables_by_gid:
            self._dspark_query_slot_mappings_by_gid = {
                gid: self._slot_mapping_buffer_for_gid(gid, context=False) for gid in block_tables_by_gid
            }
            self._dspark_query_slot_mappings_by_layer = self._layer_map_from_gid_map(
                self._dspark_query_slot_mappings_by_gid
            )
            if primary_gid in self._dspark_query_slot_mappings_by_gid:
                cad.slot_mapping = self._dspark_query_slot_mappings_by_gid[primary_gid][:num_query_total]
        cad.positions = self.positions[:num_query_total]
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill
        self._dspark_query_start_loc = cad.query_start_loc_cpu[: batch_size + 1]
        self._dspark_seq_lens = cad.seq_lens[:batch_size]
        self._dspark_token_to_req_indices = token_to_req_indices[:num_query_total]

        return num_query_total, token_indices_to_sample, cad, None

    def _prepare_dspark_dummy_standard_inputs(
        self,
        num_reqs: int,
        num_input_tokens: int,
        model_num_query_tokens: int,
    ) -> None:
        """Build dummy paged SWA inputs so dummy_run/profile_run exercises the
        standard-DSA path (which needs block_table/slot_mapping/indices, unlike
        the private ring-buffer path). All-zero block tables / slot mappings
        are fine: profile_run only needs correct shapes, not correct values.
        """
        batch_size = max(num_reqs, 1)
        block_size = self.num_speculative_tokens
        cache_block_size = int(self.draft_attn_groups[0].kv_cache_spec.block_size)
        num_blocks = (self.max_positions + cache_block_size) // cache_block_size + 1
        block_tables_by_gid: dict[int, torch.Tensor] = {}
        query_slot_mappings_by_gid: dict[int, torch.Tensor] = {}
        context_slot_mappings_by_gid: dict[int, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            block_tables_by_gid[gid] = torch.zeros(
                (batch_size, num_blocks), dtype=torch.int32, device=self.device
            )
            query_slot_mappings_by_gid[gid] = torch.zeros(
                model_num_query_tokens, dtype=torch.int32, device=self.device
            )
            context_slot_mappings_by_gid[gid] = torch.zeros(
                num_input_tokens, dtype=torch.int32, device=self.device
            )
        self._dspark_block_tables_by_gid = block_tables_by_gid
        self._dspark_block_tables_by_layer = self._layer_map_from_gid_map(block_tables_by_gid)
        self._dspark_query_slot_mappings_by_gid = query_slot_mappings_by_gid
        self._dspark_context_slot_mappings_by_gid = context_slot_mappings_by_gid
        self._dspark_query_slot_mappings_by_layer = self._layer_map_from_gid_map(query_slot_mappings_by_gid)
        self._dspark_context_slot_mappings_by_layer = self._layer_map_from_gid_map(context_slot_mappings_by_gid)
        self._dspark_query_start_loc = (
            self.arange_dflash[: batch_size + 1] * block_size
        ).to(torch.int32)
        self._dspark_seq_lens = torch.full(
            (batch_size,), block_size, dtype=torch.int32, device=self.device
        )
        token_to_req = self._dspark_token_to_req_indices_buffer[:model_num_query_tokens]
        req_ids = (
            torch.arange(model_num_query_tokens, device=self.device, dtype=torch.int32) // block_size % batch_size
        )
        token_to_req.copy_(req_ids)
        self._dspark_token_to_req_indices = token_to_req

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        del dummy_compute_logits, kwargs
        block_size = self.num_speculative_tokens
        num_query_tokens = min(num_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(
            num_query_tokens,
            is_draft_model=True,
        )
        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        num_query_total = min(num_reqs * block_size, num_query_tokens)
        model_num_query_tokens = num_input_tokens
        self._pad_draft_query_buffers(num_query_total, num_input_tokens)

        standard_ready = (
            bool(getattr(self, "draft_attn_groups", []))
            and model_num_query_tokens > 0
        )
        if standard_ready:
            self._prepare_dspark_dummy_standard_inputs(
                num_reqs, num_input_tokens, model_num_query_tokens
            )

        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=[],
        ):
            self._dflash_num_context = num_input_tokens
            context_slot_mapping = (
                getattr(self, "_dspark_context_slot_mappings_by_layer", {})
                if standard_ready
                else None
            )
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_input_tokens],
                self._context_positions_buffer[:num_input_tokens],
                context_slot_mapping,
            )
            if model_num_query_tokens:
                self.model(
                    input_ids=self.input_ids[:model_num_query_tokens],
                    positions=self.positions[:model_num_query_tokens],
                    inputs_embeds=None,
                    slot_mapping=(
                        AscendDSparkProposer._slice_tensor_map(
                            getattr(self, "_dspark_query_slot_mappings_by_layer", {}),
                            model_num_query_tokens,
                        )
                        if standard_ready
                        else self._slot_mapping_buffer[:model_num_query_tokens]
                    ),
                    block_table=(
                        getattr(self, "_dspark_block_tables_by_layer", {})
                        if standard_ready
                        else None
                    ),
                    dspark_query_start_loc=getattr(self, "_dspark_query_start_loc", None),
                    dspark_seq_lens=getattr(self, "_dspark_seq_lens", None),
                    dspark_token_to_req_indices=getattr(self, "_dspark_token_to_req_indices", None),
                )
            forward_context = get_forward_context()
            if (
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not _EXTRA_CTX.capturing
                and self.draft_attn_groups
            ):
                self._update_full_graph_params(forward_context, num_tokens, [])

    def build_model_inputs_first_pass(self, num_input_tokens: int) -> dict[str, Any]:
        num_context = self._dflash_num_context
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            getattr(self, "_dspark_context_slot_mappings_by_layer", {}),
        )
        return dict(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self.positions[:num_input_tokens],
            inputs_embeds=None,
            slot_mapping=AscendDSparkProposer._slice_tensor_map(
                getattr(self, "_dspark_query_slot_mappings_by_layer", {}),
                num_input_tokens,
            ),
            block_table=getattr(self, "_dspark_block_tables_by_layer", {}),
            dspark_query_start_loc=getattr(self, "_dspark_query_start_loc", None),
            dspark_seq_lens=getattr(self, "_dspark_seq_lens", None),
            dspark_token_to_req_indices=getattr(self, "_dspark_token_to_req_indices_buffer", None)[:num_input_tokens]
            if isinstance(getattr(self, "_dspark_token_to_req_indices_buffer", None), torch.Tensor)
            else None,
        )

    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        block_size = self.num_speculative_tokens
        num_sample = num_reqs * block_size
        sample_hidden_states = head_hidden[token_indices_to_sample[:num_sample]]
        base_logits = self.model.compute_logits(sample_hidden_states)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, block_size, vocab_size)

        prev_ids = self._markov_anchor_tokens[:num_reqs]
        for idx in range(block_size):
            markov_embed = self.model.markov_embed(prev_ids)
            markov_bias = self.model.markov_bias(markov_embed)
            logits = base_logits[:, idx, :] + markov_bias
            draft_ids = _dspark_greedy_sample(logits)
            self._markov_draft_tokens[:num_reqs, idx].copy_(draft_ids)
            prev_ids = self._markov_draft_tokens[:num_reqs, idx]
        return self._markov_draft_tokens[:num_reqs, :block_size]

    def _propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        target_model_batch_desc: BatchDescriptor,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
        scheduler_output: SchedulerOutput | None = None,
        num_scheduled_tokens: int = 0,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del (
            target_model_batch_desc,
            mm_embed_inputs,
            scheduler_output,
            num_scheduled_tokens,
        )

        num_tokens, token_indices_to_sample, _, _ = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )
        assert self.runner is not None

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_tokens, is_draft_model=True)
        multi_steps_attn_metadata = self._build_standard_dsa_attn_metadata(
            common_attn_metadata, num_input_tokens, num_tokens
        )
        model_num_tokens = num_input_tokens
        self._pad_draft_query_buffers(num_tokens, num_input_tokens)
        if isinstance(getattr(self, "_dspark_token_to_req_indices_buffer", None), torch.Tensor):
            self._dspark_token_to_req_indices = self._dspark_token_to_req_indices_buffer[:model_num_tokens]

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_tokens,
            batch_descriptor=None,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            num_context = self._dflash_num_context
            self.model.precompute_and_store_context_kv(
                self._dflash_hidden_states[:num_context],
                self._context_positions_buffer[:num_context],
                getattr(self, "_dspark_context_slot_mappings_by_layer", {}),
            )
            hidden_states = self.model(
                input_ids=self.input_ids[:model_num_tokens],
                positions=self.positions[:model_num_tokens],
                inputs_embeds=None,
                slot_mapping=AscendDSparkProposer._slice_tensor_map(
                    getattr(self, "_dspark_query_slot_mappings_by_layer", {}),
                    model_num_tokens,
                ),
                block_table=getattr(self, "_dspark_block_tables_by_layer", {}),
                dspark_query_start_loc=common_attn_metadata.query_start_loc_cpu[: common_attn_metadata.num_reqs + 1],
                dspark_seq_lens=common_attn_metadata.seq_lens[: common_attn_metadata.num_reqs],
                dspark_token_to_req_indices=getattr(self, "_dspark_token_to_req_indices", None),
            )
            draft_token_ids = self._sample_sequential(
                common_attn_metadata.num_reqs,
                hidden_states,
                token_indices_to_sample,
                sampling_metadata,
            )
        return draft_token_ids
