"""Long multi-stream graph replay with live metadata, never a performance gate."""

import itertools

import pytest
import torch

from ._testing import BACKENDS, BF16_942, SWA, assert_close, make_call, make_case


@pytest.mark.parametrize("backend", (*BACKENDS, SWA), ids=lambda backend: backend.name)
@pytest.mark.parametrize("dq", (128, 192))
def test_graph_stress_live_metadata_and_independent_outputs(backend, dq):
    if not backend.available:
        pytest.skip(f"{backend.name} requires native {backend.arch}; no cross-target execution")
    causal = backend == SWA
    case = make_case((33, 65), (193, 321), dtype=backend.dtype, dq=dq, heads=4, kv_heads=2,
                     window_left=128 if causal else -1, has_sink=causal,
                     nonunit_scales=True, poison_tail=False)
    call, _, kernel = make_call(case, backend, causal)
    main = torch.cuda.current_stream()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outputs, lses = [], [], []
    # Reserve enough query tiles for later redistribution without recapture.
    maximum_q = sum(case.q_lens)
    for stream in streams:
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            out, lse = call(out=None, return_lse=True, stream=stream, max_seqlen_q=maximum_q)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(3):
                    call(out=out, lse=lse, stream=stream, max_seqlen_q=maximum_q)
        main.wait_stream(stream)
        assert_close(case, backend, out, lse, causal)
        graphs.append(graph)
        outputs.append(out)
        lses.append(lse)
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert lses[0].data_ptr() != lses[1].data_ptr()
    compiled = frozenset(kernel._compiled)
    original_lengths = case.q_lens
    original_scales = case.qs.clone()
    prior = None
    # 2 streams x 3 captured calls x 256 replays x 4 epochs = 6,144 launches.
    # The all-empty epoch keeps storage/host maxima unchanged, exercising the
    # device-side scheduler rather than the host's empty-tensor fast path.
    for epoch, lengths in enumerate(((33, 65), (0, 98), (0, 0), (49, 49))):
        for stream in streams:
            main.wait_stream(stream)
        case.q_lens = lengths
        prefix = torch.tensor(list(itertools.accumulate(lengths, initial=0)), device=case.q.device, dtype=torch.int32)
        case.cq.copy_(prefix)
        if epoch == 1:
            case.page_order[:] = reversed(case.page_order)
            case.indices.copy_(torch.tensor(case.page_order, device=case.q.device, dtype=torch.int32))
            # Keep BF16/gfx942's zero-padded-tail contract after remapping.
            position = 0
            for length in case.kv_lens:
                count = (length + case.page - 1) // case.page
                if length % case.page:
                    physical = case.page_order[position + count - 1]
                    case.k_pages[physical, length % case.page:].zero_()
                    case.v_pages[physical, length % case.page:].zero_()
                position += count
            case.pack(copy=True)
            case.qs.copy_(original_scales * 0.75)
        elif epoch == 3:
            # ROCm PyTorch has no FP8 mul kernel; mutate via FP32 then cast
            # back without reallocating the tensors captured by the graph.
            case.v_pages.copy_((case.v_pages.float() * 0.5).to(case.v_pages.dtype))
            case.pack(copy=True)
            case.qs.copy_(original_scales * 1.25)
            if case.sinks is not None:
                case.sinks.fill_(2.0)
        for tensor in (*outputs, *lses):
            tensor.fill_(-123)
        for stream in streams:
            stream.wait_stream(main)
        for _ in range(256):
            for stream, graph in zip(streams, graphs):
                with torch.cuda.stream(stream):
                    graph.replay()
        for stream in streams:
            main.wait_stream(stream)
        for out, lse in zip(outputs, lses):
            assert_close(case, backend, out, lse, causal)
            end = sum(lengths)
            assert (out[end:] == -123).all() and (lse[end:] == -123).all()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(lses[0], lses[1], rtol=0, atol=0)
        if sum(lengths):
            if prior is not None:
                assert not torch.equal(outputs[0], prior), "graph reused stale metadata/cache content"
            prior = outputs[0].clone()
        assert frozenset(kernel._compiled) == compiled
    case.q_lens = original_lengths


@pytest.mark.parametrize("page", (32, 64, 128))
@pytest.mark.parametrize("dq", (128, 192))
def test_bf16_counter_matches_native_compute_units(monkeypatch, page, dq):
    if not BF16_942.available:
        pytest.skip("requires native gfx942; counter sizing is checked on the actual device")
    case = make_case((17, 259), (65, 321), dtype=torch.bfloat16, dq=dq, page=page,
                     heads=4, kv_heads=2, poison_tail=False)
    call, _, _ = make_call(case, BF16_942, False)
    counter_sizes = []
    original_zeros = torch.zeros

    def zeros(*args, **kwargs):
        tensor = original_zeros(*args, **kwargs)
        if tensor.is_cuda and tensor.dtype == torch.int32 and tensor.ndim == 1:
            counter_sizes.append(tensor.numel())
        return tensor

    with monkeypatch.context() as patch:
        patch.setattr(torch, "zeros", zeros)
        first = call().clone()
        for _ in range(4):
            torch.testing.assert_close(call(), first, rtol=0, atol=0)
    expected = torch.cuda.get_device_properties(case.q.device).multi_processor_count + 1
    assert counter_sizes == [expected] * 5
    assert_close(case, BF16_942, first, None, False)