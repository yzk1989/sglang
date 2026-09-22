import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata,
    build_qsa_prefill_compressed_locs,
    build_qsa_prefill_kv_locs,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_QSA_COMPRESS_RATIO = 4
_QSA_FULL_PAGE_SIZE = 64


def _page_aligned_table(num_rows: int, width: int) -> torch.Tensor:
    rows = torch.arange(num_rows, dtype=torch.long)[:, None]
    cols = torch.arange(width, dtype=torch.long)[None, :]
    return (
        (rows * 8 + cols // _QSA_FULL_PAGE_SIZE) * _QSA_FULL_PAGE_SIZE
        + cols % _QSA_FULL_PAGE_SIZE
    ).to(torch.int32)


def test_qsa_prefill_compressed_locations_are_packed_once():
    table = _page_aligned_table(3, 64)
    locations = build_qsa_prefill_compressed_locs(
        token_slot_table=table,
        sequence_lengths_cpu=torch.tensor([3, 8, 5], dtype=torch.int32),
        compress_ratio=_QSA_COMPRESS_RATIO,
    )
    assert locations.tolist() == [
        int(table[1, 0]) // _QSA_COMPRESS_RATIO,
        int(table[1, 4]) // _QSA_COMPRESS_RATIO,
        int(table[2, 0]) // _QSA_COMPRESS_RATIO,
    ]


def test_qsa_prefill_compressed_locations_allow_no_complete_blocks():
    locations = build_qsa_prefill_compressed_locs(
        token_slot_table=_page_aligned_table(2, 64),
        sequence_lengths_cpu=torch.tensor([1, 3], dtype=torch.int32),
        compress_ratio=_QSA_COMPRESS_RATIO,
    )
    assert locations.dtype == torch.long
    assert locations.numel() == 0


def test_qsa_prefill_full_kv_locations_are_packed_once():
    table = _page_aligned_table(3, 64)
    locations = build_qsa_prefill_kv_locs(
        token_slot_table=table,
        sequence_lengths_cpu=torch.tensor([2, 0, 3], dtype=torch.int32),
    )
    assert torch.equal(locations, torch.cat([table[0, :2], table[2, :3]]).long())


def test_qsa_prefill_mqa_reuses_locations_across_layers():
    buffers = [
        torch.arange(512, dtype=torch.float32).reshape(512, 1, 1),
        torch.arange(512, dtype=torch.float32).reshape(512, 1, 1) + 1000,
    ]

    class Pool:
        qsa_index_kv_heads = 1
        qsa_index_head_dim = 1

        @staticmethod
        def get_qsa_compressed_k_buffer(layer_id):
            return buffers[layer_id]

    sequence_lengths = torch.tensor([3, 8, 5], dtype=torch.int32)
    token_to_batch_idx = torch.repeat_interleave(
        torch.arange(3, dtype=torch.int32), sequence_lengths.long()
    )
    table = _page_aligned_table(3, 64)
    locations = build_qsa_prefill_compressed_locs(
        token_slot_table=table,
        sequence_lengths_cpu=sequence_lengths,
        compress_ratio=_QSA_COMPRESS_RATIO,
    )
    metadata = QSAIndexerMetadata(
        sequence_lengths=sequence_lengths,
        token_to_batch_idx=token_to_batch_idx,
        token_slot_table=table,
        out_cache_loc=torch.empty(0, dtype=torch.long),
        token_to_kv_pool=Pool(),
        compress_ratio=_QSA_COMPRESS_RATIO,
        block_topk=512,
        prefill_compressed_locs=locations,
    )
    positions = torch.cat(
        [torch.arange(int(length), dtype=torch.int32) for length in sequence_lengths]
    )

    layer0, _, _, _ = metadata.get_prefill_mqa_inputs(0, positions)
    layer1, _, _, _ = metadata.get_prefill_mqa_inputs(1, positions)
    assert torch.equal(layer0, buffers[0].index_select(0, locations))
    assert torch.equal(layer1, buffers[1].index_select(0, locations))


def test_qsa_allocations_follow_parent_mooncake_scope(monkeypatch):
    active_scopes = set()
    allocations = 0
    original_zeros = torch.zeros

    @contextmanager
    def scope(name):
        active_scopes.add(name)
        try:
            yield
        finally:
            active_scopes.remove(name)

    def init_parent(pool, **_):
        pool.full_kv_pool = SimpleNamespace(
            memory_saver_adapter=SimpleNamespace(
                region=lambda _: scope("memory_saver")
            ),
            enable_custom_mem_pool=True,
            custom_mem_pool=object(),
        )

    def allocate(*args, **kwargs):
        nonlocal allocations
        assert active_scopes == {"memory_saver", "custom_pool"}
        allocations += 1
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(HybridLinearKVPool, "__init__", init_parent)
    monkeypatch.setattr(QSATokenToKVPool, "get_kv_size_bytes", lambda _: (0, 0))
    monkeypatch.setattr(torch.cuda, "use_mem_pool", lambda _: scope("custom_pool"))
    monkeypatch.setattr("sglang.srt.mem_cache.qsa_kv_pool.torch.zeros", allocate)

    QSATokenToKVPool(
        size=8,
        dtype=torch.bfloat16,
        page_size=4,
        head_num=1,
        head_dim=8,
        full_attention_layer_ids=[1, 3],
        device="cpu",
        mamba_pool=object(),
        qsa_index_kv_heads=1,
        qsa_index_head_dim=8,
        qsa_compress_ratio=2,
        qsa_token_topk=4,
        num_request_slots=3,
    )

    assert allocations == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
