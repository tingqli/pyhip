"""Opt-in M32/P-exchange regression; the default attention backend is unchanged."""

import pytest
import torch

if __package__:
    from . import mha_pa_bf16_256_pexchange_942 as candidate
    from . import test_mha_pa as common
else:
    import mha_pa_bf16_256_pexchange_942 as candidate
    import test_mha_pa as common


PERSISTENT = common.Backend("bf16_pexchange", "mha_pa_bf16_256_pexchange_942", "gfx942", torch.bfloat16,
                            empty_kv=False, causal_short_kv=False)
GRID = common.Backend("bf16_pexchange_grid", "mha_pa_bf16_256_pexchange_942", "gfx942", torch.bfloat16,
                     persistent=False, empty_kv=False, causal_short_kv=False)


@pytest.fixture
def gfx942():
    if common.gpu_arch() != "gfx942":
        pytest.skip("requires gfx942")


def test_factory_isolation():
    for persistent in (None, False):
        default = common.BF16_942.load().PagedAttention(24, 2, 256, 256, 64, False, persistent=persistent)
        alternate = candidate.PagedAttention(24, 2, 256, 256, 64, False, persistent=persistent)
        assert alternate is not default
        assert alternate._compiled is not default._compiled
        assert alternate._launch is candidate._launch_attention_256_pexchange
        assert default._launch is not alternate._launch
        assert alternate.persistent is (persistent is not False)
    with pytest.raises(NotImplementedError):
        candidate.PagedAttention(24, 2, 128, 128, 64, False)
    with pytest.raises(ValueError):
        candidate.PagedAttention(23, 2, 256, 256, 64, False)


def test_lds_and_p_exchange_layout():
    """Tagged BF16 payloads prove K/P/V coverage and peer P key ordering."""
    stored = {}
    for tid in range(512):
        for packet in range(4):
            address = candidate._k_write_offset(tid) + packet * 8192
            chunk, token = tid // 64 + packet * 8, tid % 64
            for element in range(8):
                location = address + element * 2
                assert location not in stored and 0 <= location < 32768
                stored[location] = (token, chunk * 8 + element)
    assert len(stored) == 64 * 256
    for half in range(2):
        consumed = []
        for lane in range(64):
            for k in range(16):
                address = candidate._k_read_offset(lane, half) + k * 2048
                consumed += [stored[address + element * 2] for element in range(8)]
        assert len(consumed) == len(set(consumed)) == 32 * 256
        assert set(consumed) == {(key, dim) for key in range(half * 32, (half + 1) * 32) for dim in range(256)}

    stored = {}
    for tid in range(512):
        half, pair, lane = tid // 256, (tid // 64) % 4, tid % 64
        for index in range(16):
            packet, within = divmod(index, 8)
            address = candidate._p_write_offset(tid) + packet * 8192 + within * 2
            row = pair * 32 + lane % 32
            key = half * 32 + (lane // 32) * 8 + (index // 8) * 16 + index % 8
            assert address not in stored and 0 <= address < candidate.P_BYTES
            stored[address] = (row, key)
    assert len(set(stored.values())) == 128 * 64
    for tid in range(512):
        half, pair, lane = tid // 256, (tid // 64) % 4, tid % 64
        own = [stored[candidate._p_write_offset(tid) + (i // 8) * 8192 + (i % 8) * 2] for i in range(16)]
        peer = [stored[candidate._p_peer_offset(tid) + (i // 8) * 8192 + (i % 8) * 2] for i in range(16)]
        merged = own + peer if half == 0 else peer + own
        expected = [(pair * 32 + lane % 32, (lane // 32) * 8 + (i // 8) * 16 + i % 8) for i in range(32)]
        assert merged == expected
    assert candidate.P_BYTES <= candidate.MAX_BASE < candidate.SUM_BASE
    assert candidate.SUM_BASE + 512 * 4 <= candidate.K_BYTES

    for half in range(2):
        addresses = set()
        for quarter in range(2):
            for lane in range(64):
                for n in range(2):
                    for k in range(4):
                        address = candidate._v_read_offset(lane, half, quarter) + n * 512 + k * 8192
                        assert address % 16 == 0
                        for element in range(8):
                            offset = address + element * 2 - candidate.K_BYTES
                            token_group, rest = divmod(offset // 2, 256 * 8)
                            channel, token_in_group = divmod(rest, 8)
                            assert half * 128 <= channel < (half + 1) * 128
                            addresses.add((token_group * 8 + token_in_group, channel))
        assert len(addresses) == 64 * 128


@pytest.mark.parametrize("kv_len", (1, 31, 32, 33, 63, 64, 65, 127, 128, 129))
def test_stage_boundaries(gfx942, kv_len):
    case = common.make_case((65,), (kv_len,), dq=256, dv=256, heads=24, kv_heads=2,
                            poison_tail=True, source_dtype=torch.bfloat16)
    reference = common.torch_reference(case, False)
    for backend in (PERSISTENT, GRID):
        call, out, _ = common.make_call(case, backend, False)
        common.accuracy(call(), reference, f"{backend.name}/kv{kv_len}")
        saved = out.clone()
        for _ in range(2):
            torch.testing.assert_close(call(), saved, atol=0, rtol=0)


@pytest.mark.parametrize("page", (32, 64, 128))
@pytest.mark.parametrize("causal", (False, True))
def test_layout_contract(gfx942, page, causal):
    case = common.make_case((0, 7, 33, 257), (31, 33, 97, 321), dq=256, dv=256, page=page,
                            heads=24, kv_heads=2, mode="per-tensor" if page == 32 else "per-token",
                            q_offset=5, table_offset=3, nonunit_scales=True, poison_tail=True,
                            source_dtype=torch.bfloat16)
    valid = slice(case.q_offset, case.q_offset + sum(case.q_lens))
    reference, lse_reference = common.torch_reference(case, causal, 0.0625, return_lse=True)
    for backend in (PERSISTENT, GRID):
        call, out, _ = common.make_call(case, backend, causal)
        common.accuracy(call(softmax_scale=0.0625)[valid], reference[valid], f"{backend.name}/no_lse/p{page}")
        lse = torch.full(case.q.shape[:2], -123.0, device="cuda")
        actual, actual_lse = call(return_lse=True, lse=lse, softmax_scale=0.0625)
        assert actual.data_ptr() == out.data_ptr() and actual_lse.data_ptr() == lse.data_ptr()
        common.accuracy(actual[valid], reference[valid], f"{backend.name}/lse/p{page}")
        torch.testing.assert_close(lse[valid], lse_reference[valid], atol=0.002, rtol=0.002)
        first, first_lse = out.clone(), lse.clone()
        for _ in range(2):
            call(lse=lse, softmax_scale=0.0625)
            torch.testing.assert_close(out, first, atol=0, rtol=0)
            torch.testing.assert_close(lse, first_lse, atol=0, rtol=0)
        for value in (out, lse):
            assert bool((value[:valid.start] == -123).all())
            assert bool((value[valid.stop:] == -123).all())


@pytest.mark.parametrize("backend", (PERSISTENT, GRID), ids=lambda backend: backend.name)
def test_streams_and_graph(gfx942, backend):
    common.test_bf16_streams_and_graph(backend, 256)


def test_cross_half_softmax_and_rescale(gfx942):
    case = common.make_case((129,), (193,), dq=256, dv=256, heads=24, kv_heads=2,
                            source_dtype=torch.bfloat16, poison_tail=True)
    case.q.fill_(0.25)
    # Later key tiles and one key half have larger logits. Both output halves
    # must use the same denominator/correction, including an all-masked half.
    for tile, physical_page in enumerate(case.page_order):
        for key in range(64):
            logical = tile * 64 + key
            if logical < 193:
                case.k_pages[physical_page, key].fill_(tile * 2 + (key >= 32))
                case.v_pages[physical_page, key, :, :128].fill_(logical / 193)
                case.v_pages[physical_page, key, :, 128:].fill_(-logical / 97)
    case.k, case.v = common.vectorize_kv(case.k_pages, case.v_pages)
    reference, lse_reference = common.torch_reference(case, True, return_lse=True)
    for backend in (PERSISTENT, GRID):
        call, _, _ = common.make_call(case, backend, True)
        out, lse = call(return_lse=True)
        common.accuracy(out, reference, f"{backend.name}/rescale")
        torch.testing.assert_close(lse, lse_reference, atol=0.002, rtol=0.002)