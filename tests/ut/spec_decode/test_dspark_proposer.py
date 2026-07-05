# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.worker.utils import AttentionGroup

import vllm_ascend.models.deepseek_v4_dspark as dspark_model_module
import vllm_ascend.spec_decode.dspark_proposer as dspark_proposer_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer


class _FakeDSparkModel:
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.to(torch.long)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        vocab_size = 5
        bias = torch.zeros(
            (markov_embed.numel(), vocab_size),
            dtype=torch.float32,
            device=markov_embed.device,
        )
        next_ids = (markov_embed.to(torch.long) + 1) % vocab_size
        bias.scatter_(1, next_ids.view(-1, 1), 10.0)
        return bias


def test_dspark_sample_sequential_uses_previous_draft_token(monkeypatch):
    monkeypatch.setattr(
        dspark_proposer_module,
        "greedy_sample",
        lambda logits: logits.argmax(dim=-1),
    )
    proposer = SimpleNamespace(
        num_speculative_tokens=3,
        model=_FakeDSparkModel(),
        _dspark_seed_buffer=torch.tensor([1, 3], dtype=torch.int64),
        _dspark_draft_buffer=torch.zeros((2, 3), dtype=torch.int64),
    )
    head_hidden = torch.zeros((6, 5), dtype=torch.float32)
    token_indices = torch.arange(6, dtype=torch.int32)

    draft_tokens = AscendDSparkProposer._sample_sequential(
        proposer,
        num_reqs=2,
        head_hidden=head_hidden,
        token_indices_to_sample=token_indices,
    )

    assert draft_tokens.data_ptr() == proposer._dspark_draft_buffer.data_ptr()
    torch.testing.assert_close(
        draft_tokens,
        torch.tensor([[2, 3, 4], [4, 0, 1]], dtype=torch.int64),
    )


def test_dspark_sample_sequential_full_vocab_greedy_uses_direct_argmax(monkeypatch):
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_reduce_sample=False),
    )

    def unexpected_tp_greedy(logits):
        raise AssertionError("full-vocab DSpark logits should use direct argmax")

    monkeypatch.setattr(dspark_proposer_module, "greedy_sample", unexpected_tp_greedy)
    model = SimpleNamespace(
        compute_logits=lambda hidden_states: hidden_states,
        markov_embed=lambda token_ids: token_ids,
        markov_bias=lambda markov_embed: torch.zeros((markov_embed.numel(), 4), dtype=torch.float32),
    )
    proposer = SimpleNamespace(
        num_speculative_tokens=1,
        model=model,
        _dspark_seed_buffer=torch.tensor([0], dtype=torch.int64),
        _dspark_draft_buffer=torch.zeros((1, 1), dtype=torch.int64),
    )
    head_hidden = torch.tensor([[0.0, 1.0, 8.0, 3.0]], dtype=torch.float32)

    draft_tokens = AscendDSparkProposer._sample_sequential(
        proposer,
        num_reqs=1,
        head_hidden=head_hidden,
        token_indices_to_sample=torch.tensor([0], dtype=torch.int32),
    )

    torch.testing.assert_close(draft_tokens, torch.tensor([[2]], dtype=torch.int64))


def test_dspark_sample_sequential_reduce_sample_uses_tp_greedy(monkeypatch):
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_reduce_sample=True),
    )
    calls = []

    def fake_tp_greedy(logits):
        calls.append(logits.clone())
        return torch.tensor([3], dtype=torch.int64)

    monkeypatch.setattr(dspark_proposer_module, "greedy_sample", fake_tp_greedy)
    model = SimpleNamespace(
        compute_logits=lambda hidden_states: hidden_states,
        markov_embed=lambda token_ids: token_ids,
        markov_bias=lambda markov_embed: torch.zeros((markov_embed.numel(), 4), dtype=torch.float32),
    )
    proposer = SimpleNamespace(
        num_speculative_tokens=1,
        model=model,
        _dspark_seed_buffer=torch.tensor([0], dtype=torch.int64),
        _dspark_draft_buffer=torch.zeros((1, 1), dtype=torch.int64),
    )

    draft_tokens = AscendDSparkProposer._sample_sequential(
        proposer,
        num_reqs=1,
        head_hidden=torch.tensor([[0.0, 8.0, 1.0, 3.0]], dtype=torch.float32),
        token_indices_to_sample=torch.tensor([0], dtype=torch.int32),
    )

    assert len(calls) == 1
    torch.testing.assert_close(draft_tokens, torch.tensor([[3]], dtype=torch.int64))


def test_dspark_standard_dsa_uses_draft_group_block_table(monkeypatch):

    class FakeBlockTable:
        def __init__(self, table):
            self._table = table

        def get_device_tensor(self):
            return self._table

    class FakeMultiGroupBlockTable:
        def __init__(self, tables):
            self._tables = tables

        def __getitem__(self, idx):
            return self._tables[idx]

    device = torch.device("cpu")
    block_size = 3
    batch_size = 2
    draft_block_table = torch.tensor(
        [
            [30, 40, 50],
            [31, 41, 51],
        ],
        dtype=torch.int32,
    )
    proposer = SimpleNamespace(
        device=device,
        num_speculative_tokens=block_size,
        parallel_drafting_token_id=99,
        kernel_block_size=8,
        kv_cache_gid=1,
        runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                block_table=FakeMultiGroupBlockTable(
                    [
                        FakeBlockTable(torch.full((batch_size, 3), 9, dtype=torch.int32)),
                        FakeBlockTable(draft_block_table),
                    ]
                )
            )
        ),
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(32, dtype=torch.int32),
        input_ids=torch.zeros(batch_size * block_size, dtype=torch.int64),
        positions=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _dspark_seed_buffer=torch.full((4,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(8, 2, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(8, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(8, dtype=torch.int32),
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                SimpleNamespace(block_size=8),
                1,
                [_FakeMetadataBuilder(SimpleNamespace(block_size=8), ["draft.swa"], SimpleNamespace(), device)],
            )
        ],
    )
    proposer._get_draft_block_table_for_gid = AscendDSparkProposer._get_draft_block_table_for_gid.__get__(proposer)
    proposer._get_draft_block_tables = AscendDSparkProposer._get_draft_block_tables.__get__(proposer)
    proposer._layer_map_from_gid_map = AscendDSparkProposer._layer_map_from_gid_map.__get__(proposer)
    proposer._slot_mapping_buffer_for_gid = AscendDSparkProposer._slot_mapping_buffer_for_gid.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)

    target_positions = torch.tensor([5, 6, 7, 15, 16, 17], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([8, 18], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([8, 18], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=batch_size,
        num_actual_tokens=6,
        num_input_tokens=6,
        max_query_len=3,
        actual_seq_lengths_q=[3, 3],
        block_table_tensor=torch.full((batch_size, 3), 99, dtype=torch.int32),
        slot_mapping=torch.arange(100, 106, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=18,
    )

    AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(6, dtype=torch.int64),
        next_token_ids=torch.tensor([101, 202], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=torch.arange(12, dtype=torch.float32).view(6, 2),
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    torch.testing.assert_close(proposer._dspark_block_tables_by_gid[1], draft_block_table)
    torch.testing.assert_close(
        proposer._context_slot_mapping_buffer[:6],
        torch.tensor([245, 246, 247, 335, 408, 409], dtype=torch.int32),
    )
    torch.testing.assert_close(
        cad.slot_mapping,
        torch.tensor([320, 321, 322, 410, 411, 412], dtype=torch.int32),
    )
    assert proposer._dspark_block_tables_by_layer["draft.swa"] is proposer._dspark_block_tables_by_gid[1]


def test_dspark_standard_dsa_keeps_compact_block_table_order(monkeypatch):

    class FakeBlockTable:
        def __init__(self, table):
            self._table = table

        def get_device_tensor(self):
            return self._table

    class FakeMultiGroupBlockTable:
        def __init__(self, tables):
            self._tables = tables

        def __getitem__(self, idx):
            return self._tables[idx]

    device = torch.device("cpu")
    block_size = 3
    draft_block_table = torch.tensor(
        [
            [1, 2, 3],
            [10, 11, 12],
            [30, 40, 50],
        ],
        dtype=torch.int32,
    )
    proposer = SimpleNamespace(
        device=device,
        num_speculative_tokens=block_size,
        parallel_drafting_token_id=99,
        kernel_block_size=8,
        kv_cache_gid=1,
        runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=["live"],
                req_id_to_index={"live": 2},
                block_table=FakeMultiGroupBlockTable(
                    [
                        FakeBlockTable(torch.full((3, 3), 9, dtype=torch.int32)),
                        FakeBlockTable(draft_block_table),
                    ]
                ),
            )
        ),
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(32, dtype=torch.int32),
        input_ids=torch.zeros(block_size, dtype=torch.int64),
        positions=torch.zeros(block_size, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(block_size, dtype=torch.int32),
        _dspark_seed_buffer=torch.full((2,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(4, 2, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(4, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(4, dtype=torch.int32),
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                SimpleNamespace(block_size=8),
                1,
                [_FakeMetadataBuilder(SimpleNamespace(block_size=8), ["draft.swa"], SimpleNamespace(), device)],
            )
        ],
    )
    proposer._get_draft_block_table_for_gid = AscendDSparkProposer._get_draft_block_table_for_gid.__get__(proposer)
    proposer._get_draft_block_tables = AscendDSparkProposer._get_draft_block_tables.__get__(proposer)
    proposer._layer_map_from_gid_map = AscendDSparkProposer._layer_map_from_gid_map.__get__(proposer)
    proposer._slot_mapping_buffer_for_gid = AscendDSparkProposer._slot_mapping_buffer_for_gid.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)

    target_positions = torch.tensor([5, 6], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([7], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        actual_seq_lengths_q=[2],
        block_table_tensor=torch.full((1, 3), 99, dtype=torch.int32),
        slot_mapping=torch.arange(100, 102, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=7,
    )

    AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(2, dtype=torch.int64),
        next_token_ids=torch.tensor([101], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=torch.arange(4, dtype=torch.float32).view(2, 2),
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    torch.testing.assert_close(proposer._dspark_block_tables_by_gid[1], draft_block_table[:1])
    torch.testing.assert_close(
        proposer._context_slot_mapping_buffer[:2],
        torch.tensor([13, 14], dtype=torch.int32),
    )
    torch.testing.assert_close(cad.slot_mapping, torch.tensor([15, 16, 17], dtype=torch.int32))


def test_dspark_standard_dsa_prefers_runner_per_group_metadata(monkeypatch):

    class FakeBlockTable:
        def __init__(self, table):
            self._table = table

        def get_device_tensor(self):
            return self._table

    class FakeMultiGroupBlockTable:
        def __init__(self, tables):
            self._tables = tables

        def __getitem__(self, idx):
            return self._tables[idx]

    device = torch.device("cpu")
    proposer = SimpleNamespace(
        device=device,
        num_speculative_tokens=2,
        parallel_drafting_token_id=99,
        kernel_block_size=4,
        kv_cache_gid=1,
        runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                block_table=FakeMultiGroupBlockTable(
                    [
                        FakeBlockTable(torch.full((1, 3), 7, dtype=torch.int32)),
                        FakeBlockTable(torch.zeros((1, 3), dtype=torch.int32)),
                    ]
                )
            )
        ),
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(32, dtype=torch.int32),
        input_ids=torch.zeros(2, dtype=torch.int64),
        positions=torch.zeros(2, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(2, dtype=torch.int32),
        _dspark_seed_buffer=torch.full((2,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(4, 2, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(4, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(4, dtype=torch.int32),
        _dspark_per_group_block_tables={},
        _dspark_per_group_slot_mappings={},
        _dspark_query_slot_mapping_buffers={},
        _dspark_context_slot_mapping_buffers={},
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                SimpleNamespace(block_size=4),
                1,
                [_FakeMetadataBuilder(SimpleNamespace(block_size=4), ["draft.swa"], SimpleNamespace(), device)],
            )
        ],
    )
    proposer.set_per_group_attn_metadata = AscendDSparkProposer.set_per_group_attn_metadata.__get__(proposer)
    proposer._get_draft_block_table_for_gid = AscendDSparkProposer._get_draft_block_table_for_gid.__get__(proposer)
    proposer._get_draft_block_tables = AscendDSparkProposer._get_draft_block_tables.__get__(proposer)
    proposer._layer_map_from_gid_map = AscendDSparkProposer._layer_map_from_gid_map.__get__(proposer)
    proposer._slot_mapping_buffer_for_gid = AscendDSparkProposer._slot_mapping_buffer_for_gid.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)

    runner_block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)
    runner_slot_mapping = torch.tensor([500, 501], dtype=torch.int32)
    proposer.set_per_group_attn_metadata(1, runner_block_table, runner_slot_mapping)

    target_positions = torch.tensor([5, 6], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([7], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        actual_seq_lengths_q=[2],
        block_table_tensor=torch.full((1, 3), 99, dtype=torch.int32),
        slot_mapping=torch.arange(2, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=7,
    )

    AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(2, dtype=torch.int64),
        next_token_ids=torch.tensor([101], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=torch.arange(4, dtype=torch.float32).view(2, 2),
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    torch.testing.assert_close(proposer._dspark_block_tables_by_gid[1], runner_block_table)
    assert proposer._dspark_block_tables_by_gid[1].data_ptr() != runner_block_table.data_ptr()
    torch.testing.assert_close(proposer._context_slot_mapping_buffer[:2], torch.tensor([45, 46], dtype=torch.int32))
    torch.testing.assert_close(cad.slot_mapping, torch.tensor([47, 48], dtype=torch.int32))
    assert proposer._dspark_block_tables_by_layer["draft.swa"] is proposer._dspark_block_tables_by_gid[1]


def test_dspark_standard_dsa_keeps_per_layer_block_tables(monkeypatch):

    class FakeBlockTable:
        def __init__(self, table):
            self._table = table

        def get_device_tensor(self):
            return self._table

    class FakeMultiGroupBlockTable:
        def __init__(self, tables):
            self._tables = tables

        def __getitem__(self, idx):
            return self._tables[idx]

    device = torch.device("cpu")
    proposer = object.__new__(AscendDSparkProposer)
    proposer.device = device
    proposer.num_speculative_tokens = 2
    proposer.parallel_drafting_token_id = 99
    proposer.kernel_block_size = 4
    proposer.kv_cache_gid = 1
    proposer.max_num_tokens = 8
    proposer.max_query_tokens = 4
    proposer.token_arange_np = np.arange(16, dtype=np.int32)
    proposer.arange_dspark = torch.arange(32, dtype=torch.int32)
    proposer.input_ids = torch.zeros(4, dtype=torch.int64)
    proposer.positions = torch.zeros(4, dtype=torch.int32)
    proposer._slot_mapping_buffer = torch.zeros(4, dtype=torch.int32)
    proposer._dspark_seed_buffer = torch.full((2,), -1, dtype=torch.int64)
    proposer._dflash_hidden_states = torch.zeros(8, 2, dtype=torch.float32)
    proposer._context_positions_buffer = torch.zeros(8, dtype=torch.int32)
    proposer._context_slot_mapping_buffer = torch.zeros(8, dtype=torch.int32)
    proposer._dspark_query_slot_mapping_buffers = {}
    proposer._dspark_context_slot_mapping_buffers = {}

    group1_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)
    group2_table = torch.tensor([[30, 31, 32, 33, 34]], dtype=torch.int32)
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            block_table=FakeMultiGroupBlockTable(
                [
                    FakeBlockTable(torch.full((1, 3), 7, dtype=torch.int32)),
                    FakeBlockTable(group1_table),
                    FakeBlockTable(group2_table),
                ]
            )
        )
    )
    proposer.draft_attn_groups = [
        AttentionGroup(
            _FakeBackend,
            ["layer.a"],
            SimpleNamespace(block_size=4),
            1,
            [_FakeMetadataBuilder(SimpleNamespace(block_size=4), ["layer.a"], SimpleNamespace(), device)],
        ),
        AttentionGroup(
            _FakeBackend,
            ["layer.b"],
            SimpleNamespace(block_size=2),
            2,
            [_FakeMetadataBuilder(SimpleNamespace(block_size=2), ["layer.b"], SimpleNamespace(), device)],
        ),
    ]

    target_positions = torch.tensor([5, 6], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([7], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        actual_seq_lengths_q=[2],
        block_table_tensor=torch.full((1, 3), 99, dtype=torch.int32),
        slot_mapping=torch.arange(2, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=7,
    )

    AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(2, dtype=torch.int64),
        next_token_ids=torch.tensor([101], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=torch.arange(4, dtype=torch.float32).view(2, 2),
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    torch.testing.assert_close(proposer._dspark_block_tables_by_layer["layer.a"], group1_table)
    torch.testing.assert_close(proposer._dspark_block_tables_by_layer["layer.b"], group2_table)
    torch.testing.assert_close(
        proposer._dspark_context_slot_mappings_by_layer["layer.a"],
        torch.tensor([45, 46], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._dspark_context_slot_mappings_by_layer["layer.b"],
        torch.tensor([65, 66], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._dspark_query_slot_mappings_by_layer["layer.a"][:2],
        torch.tensor([47, 48], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._dspark_query_slot_mappings_by_layer["layer.b"][:2],
        torch.tensor([67, 68], dtype=torch.int32),
    )
    torch.testing.assert_close(cad.slot_mapping, torch.tensor([47, 48], dtype=torch.int32))


def test_dspark_build_model_inputs_first_pass_returns_query_slot_mapping():
    calls = []

    class FakeModel:
        def precompute_and_store_context_kv(
            self,
            context_states,
            context_positions,
            context_slot_mapping,
        ):
            calls.append(
                (
                    context_states,
                    context_positions,
                    context_slot_mapping,
                )
            )

    context_slot_mapping_by_layer = {"draft.swa": torch.tensor([50, 51, 52], dtype=torch.int32)}
    query_slot_mapping_by_layer = {"draft.swa": torch.tensor([160, 161, 162], dtype=torch.int32)}
    block_tables_by_layer = {"draft.swa": torch.tensor([[0]], dtype=torch.int32)}
    proposer = SimpleNamespace(
        _dflash_num_context=2,
        model=FakeModel(),
        _dflash_hidden_states=torch.arange(6, dtype=torch.float32).view(3, 2),
        _context_positions_buffer=torch.tensor([5, 6, 7], dtype=torch.int32),
        _dspark_context_slot_mappings_by_layer=context_slot_mapping_by_layer,
        _dspark_query_slot_mappings_by_layer=query_slot_mapping_by_layer,
        _dspark_block_tables_by_layer=block_tables_by_layer,
        input_ids=torch.tensor([101, 99, 99], dtype=torch.int64),
        positions=torch.tensor([8, 9, 10], dtype=torch.int32),
        _dspark_token_to_req_indices_buffer=torch.tensor([0, 0, 0], dtype=torch.int32),
    )

    model_inputs = AscendDSparkProposer.build_model_inputs_first_pass(proposer, 3)

    assert len(calls) == 1
    context_states, context_positions, context_slot_mapping = calls[0]
    torch.testing.assert_close(context_states, proposer._dflash_hidden_states[:2])
    torch.testing.assert_close(context_positions, proposer._context_positions_buffer[:2])
    assert context_slot_mapping is context_slot_mapping_by_layer
    assert model_inputs["input_ids"].data_ptr() == proposer.input_ids.data_ptr()
    torch.testing.assert_close(model_inputs["positions"], proposer.positions)
    assert set(model_inputs["slot_mapping"]) == {"draft.swa"}
    torch.testing.assert_close(
        model_inputs["slot_mapping"]["draft.swa"],
        torch.tensor([160, 161, 162], dtype=torch.int32),
    )
    assert model_inputs["block_table"] is block_tables_by_layer


class _FakeKVSpec:
    block_size = 64


class _FakeMetadataBuilder:
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.calls = []

    def build_for_drafting(self, common_attn_metadata, draft_index, **kwargs):
        self.calls.append(
            {
                "positions": common_attn_metadata.positions.clone(),
                "slot_mapping": common_attn_metadata.slot_mapping.clone(),
                "block_table": getattr(common_attn_metadata, "block_table_tensor", None),
                "num_input_tokens": common_attn_metadata.num_input_tokens,
                "num_actual_tokens": common_attn_metadata.num_actual_tokens,
                "causal": common_attn_metadata.causal,
                "attn_state": common_attn_metadata.attn_state,
                "draft_index": draft_index,
                "block_size": kwargs.get("block_size"),
            }
        )
        return SimpleNamespace(tag="metadata")


class _FakeBackend:
    @classmethod
    def full_cls_name(cls):
        return "fake.Backend"

    @staticmethod
    def get_builder_cls():
        return _FakeMetadataBuilder


class _FakeLayer:
    def get_attn_backend(self):
        return _FakeBackend


def test_dspark_initialize_attn_backend_standard_dsa(monkeypatch):
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {
            "model.layers.61.self_attn.swa_cache": _FakeLayer(),
            "model.layers.62.self_attn.swa_cache": _FakeLayer(),
        },
    )

    kv_spec = _FakeKVSpec()
    proposer = SimpleNamespace(
        model=SimpleNamespace(
            get_draft_kv_cache_layer_names=lambda: [
                "model.layers.61.self_attn.swa_cache",
                "model.layers.62.self_attn.swa_cache",
            ]
        ),
        vllm_config=SimpleNamespace(),
        device=torch.device("cpu"),
        num_speculative_tokens=5,
        block_size=5,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    "model.layers.0.self_attn.swa_cache",
                    "model.layers.61.self_attn.swa_cache",
                    "model.layers.62.self_attn.swa_cache",
                ],
                kv_cache_spec=kv_spec,
            )
        ]
    )

    AscendDSparkProposer.initialize_attn_backend(proposer, kv_cache_config)

    assert proposer.attn_layer_names == [
        "model.layers.61.self_attn.swa_cache",
        "model.layers.62.self_attn.swa_cache",
    ]
    assert len(proposer.draft_attn_groups) == 1
    group = proposer.draft_attn_groups[0]
    assert group.backend is _FakeBackend
    assert group.kv_cache_spec is kv_spec
    assert group.kv_cache_group_id == 0
    assert proposer.kernel_block_size == 64
    assert proposer.block_size == 5


def test_dspark_build_standard_dsa_metadata_sets_query_buffers():
    builder = _FakeMetadataBuilder(_FakeKVSpec(), ["draft.swa"], SimpleNamespace(), torch.device("cpu"))
    proposer = SimpleNamespace(
        positions=torch.tensor([10, 11, 12, 99], dtype=torch.int32),
        _slot_mapping_buffer=torch.tensor([100, 101, 102, 999], dtype=torch.int32),
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                builder.kv_cache_spec,
                0,
                [builder],
            )
        ],
    )
    common_metadata = SimpleNamespace(
        positions=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int32),
        num_input_tokens=0,
        num_actual_tokens=0,
        causal=True,
        attn_state=None,
    )

    result = AscendDSparkProposer._build_standard_dsa_attn_metadata(
        proposer,
        common_metadata,
        num_input_tokens=4,
        num_actual_tokens=3,
    )

    assert len(result) == 1
    assert result[0]["draft.swa"].tag == "metadata"
    call = builder.calls[0]
    torch.testing.assert_close(call["positions"], torch.tensor([10, 11, 12, 0], dtype=torch.int32))
    torch.testing.assert_close(call["slot_mapping"], torch.tensor([100, 101, 102, -1], dtype=torch.int32))
    assert call["num_input_tokens"] == 4
    assert call["num_actual_tokens"] == 3
    assert call["causal"] is False
    assert call["attn_state"] == AscendAttentionState.ChunkedPrefill
    assert call["draft_index"] == 1
    assert call["block_size"] == 64


def test_dspark_standard_dsa_propose_pads_model_inputs(monkeypatch):

    context_calls = []

    class FakeForwardContextManager:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_set_ascend_forward_context(attn_metadata, vllm_config, **kwargs):
        context_calls.append((attn_metadata, vllm_config, kwargs))
        return FakeForwardContextManager()

    monkeypatch.setattr(
        dspark_proposer_module,
        "set_ascend_forward_context",
        fake_set_ascend_forward_context,
    )
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_forward_context",
        lambda: SimpleNamespace(moe_layer_index=None),
    )

    model_calls = []
    precompute_calls = []

    def clone_value(value):
        if isinstance(value, dict):
            return {key: val.clone() if isinstance(val, torch.Tensor) else val for key, val in value.items()}
        if isinstance(value, torch.Tensor):
            return value.clone()
        return value

    class FakeModel:
        def precompute_and_store_context_kv(
            self,
            context_states,
            context_positions,
            context_slot_mapping,
        ):
            precompute_calls.append(
                (
                    context_states.clone(),
                    context_positions.clone(),
                    clone_value(context_slot_mapping),
                )
            )

        def __call__(
            self,
            *,
            input_ids,
            positions,
            inputs_embeds,
            slot_mapping,
            block_table,
            dspark_query_start_loc=None,
            dspark_seq_lens=None,
            dspark_token_to_req_indices=None,
        ):
            del inputs_embeds
            model_calls.append(
                {
                    "input_ids": input_ids.clone(),
                    "positions": positions.clone(),
                    "slot_mapping": clone_value(slot_mapping),
                    "block_table": clone_value(block_table),
                    "dspark_query_start_loc": clone_value(dspark_query_start_loc),
                    "dspark_seq_lens": clone_value(dspark_seq_lens),
                    "dspark_token_to_req_indices": clone_value(dspark_token_to_req_indices),
                }
            )
            return torch.arange(input_ids.numel() * 4, dtype=torch.float32).view(input_ids.numel(), 4)

    builder = _FakeMetadataBuilder(_FakeKVSpec(), ["draft.swa"], SimpleNamespace(), torch.device("cpu"))
    proposer = SimpleNamespace(
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(),
        runner=SimpleNamespace(
            _sync_metadata_across_dp=lambda num_tokens, is_draft_model: (
                6,
                torch.tensor([6], dtype=torch.int32),
                None,
            )
        ),
        model=FakeModel(),
        num_speculative_tokens=5,
        parallel_drafting_token_id=99,
        kernel_block_size=64,
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(32, dtype=torch.int32),
        input_ids=torch.zeros(6, dtype=torch.int64),
        positions=torch.zeros(6, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(6, dtype=torch.int32),
        _dspark_token_to_req_indices_buffer=torch.zeros(6, dtype=torch.int32),
        _dspark_token_to_req_indices=None,
        _dspark_query_start_loc=None,
        _dspark_seq_lens=None,
        _dspark_seed_buffer=torch.full((2,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(4, 4, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(4, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(4, dtype=torch.int32),
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                builder.kv_cache_spec,
                0,
                [builder],
            )
        ],
    )

    sample_calls = []

    def fake_sample_sequential(num_reqs, head_hidden, token_indices_to_sample, sampling_metadata):
        sample_calls.append((num_reqs, head_hidden.clone(), token_indices_to_sample.clone(), sampling_metadata))
        return torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.int64)

    proposer._sample_sequential = fake_sample_sequential
    proposer._pad_draft_query_buffers = AscendDSparkProposer._pad_draft_query_buffers.__get__(proposer)
    proposer._get_draft_block_table_for_gid = AscendDSparkProposer._get_draft_block_table_for_gid.__get__(proposer)
    proposer._get_draft_block_tables = AscendDSparkProposer._get_draft_block_tables.__get__(proposer)
    proposer._layer_map_from_gid_map = AscendDSparkProposer._layer_map_from_gid_map.__get__(proposer)
    proposer._slot_mapping_buffer_for_gid = AscendDSparkProposer._slot_mapping_buffer_for_gid.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)
    proposer._build_standard_dsa_attn_metadata = AscendDSparkProposer._build_standard_dsa_attn_metadata.__get__(
        proposer
    )
    proposer.set_inputs_first_pass = AscendDSparkProposer.set_inputs_first_pass.__get__(proposer)

    target_positions = torch.tensor([3, 4], dtype=torch.int32)
    target_hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        actual_seq_lengths_q=[2],
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([30, 31], dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=5,
    )

    draft_tokens = AscendDSparkProposer._propose(
        proposer,
        target_token_ids=torch.tensor([1, 2], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=torch.tensor([111], dtype=torch.int64),
        token_indices_to_sample=None,
        common_attn_metadata=cad,
        target_model_batch_desc=SimpleNamespace(),
        sampling_metadata=SimpleNamespace(),
    )

    torch.testing.assert_close(draft_tokens, torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.int64))
    assert len(context_calls) == 1
    _, _, context_kwargs = context_calls[0]
    assert context_kwargs["num_tokens"] == 6
    assert context_kwargs["num_actual_tokens"] == 5
    assert len(precompute_calls) == 1
    assert len(model_calls) == 1
    torch.testing.assert_close(
        model_calls[0]["input_ids"],
        torch.tensor([111, 99, 99, 99, 99, 99], dtype=torch.int64),
    )
    torch.testing.assert_close(
        model_calls[0]["positions"],
        torch.tensor([5, 6, 7, 8, 9, 0], dtype=torch.int32),
    )
    assert set(precompute_calls[0][2]) == {"draft.swa"}
    torch.testing.assert_close(precompute_calls[0][2]["draft.swa"], torch.tensor([3, 4], dtype=torch.int32))
    assert set(model_calls[0]["slot_mapping"]) == {"draft.swa"}
    torch.testing.assert_close(
        model_calls[0]["slot_mapping"]["draft.swa"],
        torch.tensor([5, 6, 7, 8, 9, -1], dtype=torch.int32),
    )
    assert set(model_calls[0]["block_table"]) == {"draft.swa"}
    torch.testing.assert_close(model_calls[0]["block_table"]["draft.swa"], torch.tensor([[0]], dtype=torch.int32))
    torch.testing.assert_close(
        model_calls[0]["dspark_query_start_loc"],
        torch.tensor([0, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(model_calls[0]["dspark_seq_lens"], torch.tensor([10], dtype=torch.int32))
    torch.testing.assert_close(
        model_calls[0]["dspark_token_to_req_indices"],
        torch.tensor([0, 0, 0, 0, 0, -1], dtype=torch.int32),
    )
    assert len(sample_calls) == 1
    assert sample_calls[0][1].shape[0] == 6
    torch.testing.assert_close(sample_calls[0][2], torch.arange(5, dtype=torch.int32))
    metadata_call = builder.calls[0]
    assert metadata_call["num_input_tokens"] == 6
    assert metadata_call["num_actual_tokens"] == 5
    torch.testing.assert_close(metadata_call["positions"], model_calls[0]["positions"])
    torch.testing.assert_close(metadata_call["slot_mapping"], model_calls[0]["slot_mapping"]["draft.swa"])


def test_dspark_dummy_run_keeps_drafter_eager_when_graph_disabled(monkeypatch):

    context_calls = []

    class FakeForwardContextManager:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_set_ascend_forward_context(attn_metadata, vllm_config, **kwargs):
        context_calls.append((attn_metadata, vllm_config, kwargs))
        return FakeForwardContextManager()

    def fake_get_forward_context():
        runtime_mode = context_calls[-1][2]["aclgraph_runtime_mode"]
        return SimpleNamespace(cudagraph_runtime_mode=runtime_mode)

    monkeypatch.setattr(
        dspark_proposer_module,
        "set_ascend_forward_context",
        fake_set_ascend_forward_context,
    )
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_forward_context",
        fake_get_forward_context,
    )

    class FakeModel:
        def precompute_and_store_context_kv(
            self,
            context_states,
            context_positions,
            context_slot_mapping,
        ):
            del context_states, context_positions, context_slot_mapping

        def __call__(
            self,
            *,
            input_ids,
            positions,
            inputs_embeds,
            slot_mapping,
            block_table,
            dspark_query_start_loc=None,
            dspark_seq_lens=None,
            dspark_token_to_req_indices=None,
        ):
            del positions, inputs_embeds, slot_mapping, block_table
            del dspark_query_start_loc, dspark_seq_lens, dspark_token_to_req_indices
            return torch.zeros((input_ids.numel(), 4), dtype=torch.float32)

    proposer = SimpleNamespace(
        use_cuda_graph=False,
        runner=SimpleNamespace(
            _sync_metadata_across_dp=lambda num_tokens, is_draft_model: (
                num_tokens,
                torch.tensor([num_tokens], dtype=torch.int32),
                None,
            )
        ),
        vllm_config=SimpleNamespace(),
        model=FakeModel(),
        num_speculative_tokens=5,
        max_query_tokens=8,
        input_ids=torch.zeros(8, dtype=torch.int64),
        positions=torch.zeros(8, dtype=torch.int32),
        hidden_states=torch.zeros(8, 4, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(8, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(8, dtype=torch.int32),
        parallel_drafting_token_id=99,
        draft_attn_groups=[],
    )
    proposer._pad_draft_query_buffers = AscendDSparkProposer._pad_draft_query_buffers.__get__(proposer)
    proposer._update_full_graph_params = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("draft dummy_run must not update full-graph params when use_cuda_graph is false")
    )

    AscendDSparkProposer.dummy_run(
        proposer,
        num_tokens=5,
        num_reqs=1,
        aclgraph_runtime_mode=dspark_proposer_module.CUDAGraphMode.FULL,
    )

    assert len(context_calls) == 1
    assert context_calls[0][2]["aclgraph_runtime_mode"] == dspark_proposer_module.CUDAGraphMode.NONE


def test_dspark_attention_uses_standard_cache_pta_when_enabled(monkeypatch):
    calls = []

    def fake_standard_cache_attention(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(args[0], 3.0)

    monkeypatch.setattr(
        dspark_model_module,
        "dspark_attention_from_standard_cache",
        fake_standard_cache_attention,
    )
    cache = torch.zeros(4, 8, 1, 4)
    attention = SimpleNamespace(
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=8),
        ),
        attn_sink=torch.tensor([0.5], dtype=torch.float32),
        n_local_heads=1,
        block_size=2,
        window_size=4,
        scale=0.25,
    )
    q = torch.zeros(2, 1, 4)
    positions = torch.tensor([10, 11], dtype=torch.int32)
    slot_mapping = torch.tensor([10, 11], dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    output = dspark_model_module.DeepseekV4DSparkAttention._run_standard_dspark_attention(
        attention,
        q,
        positions,
        slot_mapping,
        block_table,
    )

    assert len(calls) == 1
    assert calls[0][0][1] is cache
    assert calls[0][0][2] is block_table
    assert calls[0][0][3] is positions
    assert calls[0][0][4] is slot_mapping
    torch.testing.assert_close(calls[0][0][5], torch.tensor([0.5], dtype=torch.float32))
    assert calls[0][0][6] == 2
    assert calls[0][0][7] == 4
    assert calls[0][0][8] == 8
    assert calls[0][0][9] == 0.25
    torch.testing.assert_close(output, torch.full_like(q, 3.0))


def test_dspark_standard_swa_store_unwraps_singleton_cache(monkeypatch):
    from vllm_ascend.device import device_op as device_op_module

    calls = []

    def fake_scatter(cache, shared_kv, slot_mapping):
        calls.append((cache, shared_kv, slot_mapping))

    monkeypatch.setattr(
        device_op_module.DeviceOperator,
        "dsa_kv_compress_scatter",
        staticmethod(fake_scatter),
    )

    cache = torch.zeros(1, 64, 1, 4)
    shared_kv = torch.ones(1, 1, 4)
    slot_mapping = torch.tensor([[0, 0]], dtype=torch.int32)
    attention = SimpleNamespace(
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(kv_cache=[[cache]], block_size=64),
        ),
    )

    dspark_model_module.DeepseekV4DSparkAttention._store_standard_swa_kv(
        attention,
        shared_kv,
        slot_mapping,
    )

    assert len(calls) == 1
    assert calls[0][0] is cache
    assert calls[0][1] is shared_kv
    torch.testing.assert_close(calls[0][2], slot_mapping)
