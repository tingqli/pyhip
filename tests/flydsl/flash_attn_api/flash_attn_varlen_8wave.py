"""BF16 varlen attention with one paged-V conversion and one MHA launch."""

import functools
import importlib
import math
import weakref

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl._mlir.ir import IntegerType
from flydsl.expr import arith, range_constexpr


def _view_as_torch_tensor(pointer, shape):
    order = tuple(range(len(shape) - 1, -1, -1))
    return fx.make_view(pointer, fx.make_ordered_layout(shape, order))


def _s_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    fx.rocdl.s_waitcnt(
        vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)
    )


def _permute_bf16_pairs(lhs, rhs, selector):
    return llvm.inline_asm(
        IntegerType.get_signless(32),
        [arith.unwrap(lhs), arith.unwrap(rhs), arith.unwrap(selector)],
        "v_perm_b32 $0, $1, $2, $3",
        "=v,v,v,s",
        has_side_effects=True,
    )


@functools.cache
def _compile_paged_v_converter(
    num_heads,
    num_pages,
    total_tokens,
    head_dim=128,
    page_size=32,
):
    """Convert linear V to ``[P,H,page_size/vector_size,D,vector_size]``."""
    assert num_heads > 0
    assert num_pages > 0
    assert total_tokens > 0
    assert page_size == 32
    vector_size = 128 // fx.BFloat16.width
    assert vector_size == 8
    assert head_dim > 0 and head_dim % vector_size == 0

    num_threads = 256
    token_groups = page_size // vector_size
    dim_groups = head_dim // vector_size
    num_tiles = num_pages * num_heads * token_groups * dim_groups
    num_blocks = (num_tiles + num_threads - 1) // num_threads

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def convert_kernel(
        source_: fx.Tensor,
        page_token_starts: fx.Tensor,
        page_valid_lens: fx.Tensor,
        destination_: fx.Tensor,
    ):
        tile_index = fx.Int32(fx.block_idx.x) * num_threads + fx.thread_idx.x
        if tile_index < num_tiles:
            dim_group = tile_index % dim_groups
            page_head_group = tile_index // dim_groups
            token_group = page_head_group % token_groups
            page_and_head = page_head_group // token_groups
            head = page_and_head % num_heads
            page = page_and_head // num_heads
            dim_base = dim_group * vector_size
            token_in_page = token_group * vector_size

            source = fx.rocdl.make_buffer_tensor(
                _view_as_torch_tensor(
                    fx.get_iter(source_),
                    (total_tokens, num_heads, head_dim),
                ),
                max_size=False,
            )
            destination = fx.rocdl.make_buffer_tensor(
                _view_as_torch_tensor(
                    fx.get_iter(destination_),
                    (
                        num_pages,
                        num_heads,
                        token_groups,
                        head_dim,
                        vector_size,
                    ),
                ),
                max_size=False,
            )
            load_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), fx.BFloat16
            )
            store_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), fx.BFloat16
            )
            input_rows = fx.make_rmem_tensor(
                fx.make_layout(vector_size * vector_size, 1), fx.BFloat16
            )
            output_column = fx.make_rmem_tensor(
                fx.make_layout(vector_size, 1), fx.BFloat16
            )
            input_rows.fill(0)

            source_token_base = page_token_starts[page] + token_in_page
            valid_tokens = page_valid_lens[page]
            for row in range_constexpr(vector_size):
                if token_in_page + row < valid_tokens:
                    source_offset = source.layout(
                        source_token_base + row, head, dim_base
                    )
                    source_row = fx.make_view(
                        fx.get_iter(source) + source_offset,
                        fx.make_layout(vector_size, 1),
                    )
                    input_row = fx.make_view(
                        fx.get_iter(input_rows) + row * vector_size,
                        fx.make_layout(vector_size, 1),
                    )
                    fx.copy(load_atom, source_row, input_row)

            _s_waitcnt(vmcnt=0)
            input_words = input_rows.load().bitcast(fx.Uint32)
            transpose_low = fx.Uint32(0x01000504)
            transpose_high = fx.Uint32(0x03020706)
            for column in range_constexpr(vector_size):
                source_word = column // 2
                selector = transpose_high if column % 2 else transpose_low
                output_words = []
                for row_pair in range_constexpr(vector_size // 2):
                    first_row = row_pair * 2
                    output_words.append(
                        _permute_bf16_pairs(
                            input_words[first_row * 4 + source_word],
                            input_words[(first_row + 1) * 4 + source_word],
                            selector,
                        )
                    )
                output_column.store(
                    fx.Vector.from_elements(
                        output_words, dtype=fx.Uint32
                    ).bitcast(fx.BFloat16)
                )
                destination_offset = destination.layout(
                    page, head, token_group, dim_base + column, 0
                )
                destination_column = fx.make_view(
                    fx.get_iter(destination) + destination_offset,
                    fx.make_layout(vector_size, 1),
                )
                fx.copy(store_atom, output_column, destination_column)

    @flyc.jit
    def launch(
        source: fx.Tensor,
        page_token_starts: fx.Tensor,
        page_valid_lens: fx.Tensor,
        destination: fx.Tensor,
        stream: fx.Stream,
    ):
        convert_kernel(
            source, page_token_starts, page_valid_lens, destination
        ).launch(
            grid=(num_blocks, 1, 1),
            block=(num_threads, 1, 1),
            stream=stream,
        )

    def callable(
        source,
        page_token_starts,
        page_valid_lens,
        destination,
        stream=None,
    ):
        stream = torch.cuda.current_stream() if stream is None else stream
        tensors = (
            source,
            page_token_starts,
            page_valid_lens,
            destination,
        )
        assert all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
        assert source.dtype == destination.dtype == torch.bfloat16
        assert page_token_starts.dtype == page_valid_lens.dtype == torch.int32
        assert source.shape == (total_tokens, num_heads, head_dim)
        assert page_token_starts.shape == page_valid_lens.shape == (num_pages,)
        assert destination.shape == (
            num_pages,
            num_heads,
            token_groups,
            head_dim,
            vector_size,
        )

        cache_key = (
            torch.cuda.current_device(),
            torch.cuda.get_device_properties().gcnArchName,
            *(_tensor_signature(tensor) for tensor in tensors),
        )
        compiled_cache = getattr(launch, "_compiled", {})
        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            compiled = flyc.compile(
                launch,
                source,
                page_token_starts,
                page_valid_lens,
                destination,
                stream,
            )
            compiled_cache[cache_key] = compiled
            launch._compiled = compiled_cache
        else:
            compiled(
                source,
                page_token_starts,
                page_valid_lens,
                destination,
                stream,
            )
        return destination

    return callable


@functools.cache
def _compile_paged_k_converter(
    num_heads,
    num_pages,
    total_tokens,
    head_dim=128,
    page_size=32,
):
    """Convert linear K to legacy ``[P,H,D/vector,page,vector]`` layout."""
    assert num_heads > 0
    assert num_pages > 0
    assert total_tokens > 0
    assert page_size == 32
    vector_size = 128 // fx.BFloat16.width
    assert vector_size == 8
    assert head_dim > 0 and head_dim % vector_size == 0

    num_threads = 256
    dim_groups = head_dim // vector_size
    num_tiles = num_pages * num_heads * dim_groups * page_size
    num_blocks = (num_tiles + num_threads - 1) // num_threads

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def convert_kernel(
        source_: fx.Tensor,
        page_token_starts: fx.Tensor,
        page_valid_lens: fx.Tensor,
        destination_: fx.Tensor,
    ):
        tile_index = fx.Int32(fx.block_idx.x) * num_threads + fx.thread_idx.x
        if tile_index < num_tiles:
            token_in_page = tile_index % page_size
            page_head_dim = tile_index // page_size
            dim_group = page_head_dim % dim_groups
            page_and_head = page_head_dim // dim_groups
            head = page_and_head % num_heads
            page = page_and_head // num_heads

            source = fx.rocdl.make_buffer_tensor(
                _view_as_torch_tensor(
                    fx.get_iter(source_),
                    (total_tokens, num_heads, head_dim),
                ),
                max_size=False,
            )
            destination = fx.rocdl.make_buffer_tensor(
                _view_as_torch_tensor(
                    fx.get_iter(destination_),
                    (
                        num_pages,
                        num_heads,
                        dim_groups,
                        page_size,
                        vector_size,
                    ),
                ),
                max_size=False,
            )
            load_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), fx.BFloat16
            )
            store_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), fx.BFloat16
            )
            fragment = fx.make_rmem_tensor(
                fx.make_layout(vector_size, 1), fx.BFloat16
            )
            fragment.fill(0)
            if token_in_page < page_valid_lens[page]:
                source_offset = source.layout(
                    page_token_starts[page] + token_in_page,
                    head,
                    dim_group * vector_size,
                )
                source_vector = fx.make_view(
                    fx.get_iter(source) + source_offset,
                    fx.make_layout(vector_size, 1),
                )
                fx.copy(load_atom, source_vector, fragment)
                _s_waitcnt(vmcnt=0)

            destination_offset = destination.layout(
                page, head, dim_group, token_in_page, 0
            )
            destination_vector = fx.make_view(
                fx.get_iter(destination) + destination_offset,
                fx.make_layout(vector_size, 1),
            )
            fx.copy(store_atom, fragment, destination_vector)

    @flyc.jit
    def launch(
        source: fx.Tensor,
        page_token_starts: fx.Tensor,
        page_valid_lens: fx.Tensor,
        destination: fx.Tensor,
        stream: fx.Stream,
    ):
        convert_kernel(
            source, page_token_starts, page_valid_lens, destination
        ).launch(
            grid=(num_blocks, 1, 1),
            block=(num_threads, 1, 1),
            stream=stream,
        )

    def callable(
        source,
        page_token_starts,
        page_valid_lens,
        destination,
        stream=None,
    ):
        stream = torch.cuda.current_stream() if stream is None else stream
        tensors = (
            source,
            page_token_starts,
            page_valid_lens,
            destination,
        )
        assert all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
        assert source.dtype == destination.dtype == torch.bfloat16
        assert page_token_starts.dtype == page_valid_lens.dtype == torch.int32
        assert source.shape == (total_tokens, num_heads, head_dim)
        assert page_token_starts.shape == page_valid_lens.shape == (num_pages,)
        assert destination.shape == (
            num_pages,
            num_heads,
            dim_groups,
            page_size,
            vector_size,
        )

        cache_key = (
            torch.cuda.current_device(),
            torch.cuda.get_device_properties().gcnArchName,
            *(_tensor_signature(tensor) for tensor in tensors),
        )
        compiled_cache = getattr(launch, "_compiled", {})
        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            compiled = flyc.compile(
                launch,
                source,
                page_token_starts,
                page_valid_lens,
                destination,
                stream,
            )
            compiled_cache[cache_key] = compiled
            launch._compiled = compiled_cache
        else:
            compiled(
                source,
                page_token_starts,
                page_valid_lens,
                destination,
                stream,
            )
        return destination

    return callable


def _tensor_signature(tensor):
    return (
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _validated_varlen_bounds(
    cu_seqlens_q,
    cu_seqlens_k,
    total_q,
    total_k,
):
    key = (id(cu_seqlens_q), id(cu_seqlens_k))
    version = (
        cu_seqlens_q._version,
        cu_seqlens_k._version,
        total_q,
        total_k,
    )
    validated = getattr(
        _validated_varlen_bounds, "_validated", {}
    )
    cached = validated.get(key)
    if cached is not None and all(
        reference() is tensor
        for reference, tensor in zip(
            cached[:2], (cu_seqlens_q, cu_seqlens_k)
        )
    ) and cached[2] == version:
        return cached[3], cached[4]

    q_bounds = cu_seqlens_q.cpu()
    k_bounds = cu_seqlens_k.cpu()
    assert q_bounds[0].item() == k_bounds[0].item() == 0
    assert q_bounds[-1].item() == total_q
    assert k_bounds[-1].item() == total_k
    assert bool(((q_bounds[1:] - q_bounds[:-1]) > 0).all())
    assert bool(((k_bounds[1:] - k_bounds[:-1]) > 0).all())
    q_bounds_tuple = tuple(q_bounds.tolist())
    k_bounds_tuple = tuple(k_bounds.tolist())
    validated[key] = (
        weakref.ref(cu_seqlens_q),
        weakref.ref(cu_seqlens_k),
        version,
        q_bounds_tuple,
        k_bounds_tuple,
    )
    _validated_varlen_bounds._validated = validated
    return q_bounds_tuple, k_bounds_tuple


def _paged_metadata(cu_seqlens_k, page_size=32):
    key = (
        id(cu_seqlens_k),
        cu_seqlens_k._version,
        cu_seqlens_k.device.index,
        page_size,
    )
    cache = getattr(_paged_metadata, "_cache", {})
    cached = cache.get(key)
    if cached is not None and cached[0]() is cu_seqlens_k:
        return cached[1]

    bounds = cu_seqlens_k.cpu().tolist()
    lengths = [stop - start for start, stop in zip(bounds, bounds[1:])]
    page_counts = [(length + page_size - 1) // page_size for length in lengths]
    kv_indptr_host = [0]
    for page_count in page_counts:
        kv_indptr_host.append(kv_indptr_host[-1] + page_count)
    num_pages = kv_indptr_host[-1]
    page_token_starts = []
    page_valid_lens = []
    for sequence_start, length, page_count in zip(
        bounds, lengths, page_counts
    ):
        for page_index in range(page_count):
            token_offset = page_index * page_size
            page_token_starts.append(sequence_start + token_offset)
            page_valid_lens.append(min(page_size, length - token_offset))

    device = cu_seqlens_k.device
    metadata = (
        torch.tensor(kv_indptr_host, device=device, dtype=torch.int32),
        torch.arange(num_pages, device=device, dtype=torch.int32),
        torch.tensor(
            [page_valid_lens[end - 1] for end in kv_indptr_host[1:]],
            device=device,
            dtype=torch.int32,
        ),
        torch.tensor(page_token_starts, device=device, dtype=torch.int32),
        torch.tensor(page_valid_lens, device=device, dtype=torch.int32),
    )
    cache[key] = (weakref.ref(cu_seqlens_k), metadata)
    _paged_metadata._cache = cache
    return metadata


def _paged_value_workspace(
    value, num_pages, num_heads, head_dim, stream, page_size=32
):
    key = (
        value.device.index,
        stream.cuda_stream,
        num_pages,
        num_heads,
        head_dim,
        value.dtype,
    )
    cache = getattr(_paged_value_workspace, "_cache", {})
    workspace = cache.get(key)
    vector_size = 16 // value.element_size()
    expected_shape = (
        num_pages,
        num_heads,
        page_size // vector_size,
        head_dim,
        vector_size,
    )
    if workspace is None or workspace.shape != expected_shape:
        workspace = torch.empty(
            expected_shape, device=value.device, dtype=value.dtype
        )
        cache[key] = workspace
        _paged_value_workspace._cache = cache
    return workspace


def _paged_key_workspace(
    key, num_pages, num_heads, head_dim, stream, page_size=32
):
    vector_size = 16 // key.element_size()
    expected_shape = (
        num_pages,
        num_heads,
        head_dim // vector_size,
        page_size,
        vector_size,
    )
    cache_key = (
        key.device.index,
        stream.cuda_stream,
        *expected_shape,
        key.dtype,
    )
    cache = getattr(_paged_key_workspace, "_cache", {})
    workspace = cache.get(cache_key)
    if workspace is None or workspace.shape != expected_shape:
        workspace = torch.empty(
            expected_shape, device=key.device, dtype=key.dtype
        )
        cache[cache_key] = workspace
        _paged_key_workspace._cache = cache
    return workspace


def _descale_workspace(q, softmax_scale, stream):
    scale = (
        1.0
        if softmax_scale is None
        else float(softmax_scale) * math.sqrt(q.shape[-1])
    )
    key = (
        q.device.index,
        stream.cuda_stream,
        q.shape[0],
        q.shape[1],
        scale,
    )
    cache = getattr(_descale_workspace, "_cache", {})
    cached = cache.get(key)
    if cached is None:
        q_descale = torch.full(
            (q.shape[0], q.shape[1], 1),
            scale,
            device=q.device,
            dtype=torch.float32,
        )
        unit_descale = torch.ones(
            1, device=q.device, dtype=torch.float32
        )
        cached = (q_descale, unit_descale)
        cache[key] = cached
        _descale_workspace._cache = cache
    return cached


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    min_seqlen_q=0,
    dropout_p=0.0,
    softmax_scale=None,
    logits_soft_cap=0.0,
    causal=False,
    window_size=(-1, -1, 0),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    how_v3_bf16_cvt=1,
    block_table=None,
    out=None,
    cu_seqlens_q_padded=None,
    cu_seqlens_k_padded=None,
    sink_ptr=None,
    layout="linear",
    key_layout=None,
    num_waves=8,
):
    """Run varlen attention with one paged-V conversion and one MHA launch.

    Q, K, V, and O use linear ``[T,H,D]``. ``key_layout="vectorized"`` selects
    an internal one-launch K conversion to ``[page,H,D/8,32,8]`` before MHA;
    that conversion is part of this call and therefore part of event timing.
    In vectorized mode, ``cu_seqlens_k=None`` reuses ``cu_seqlens_q`` and
    therefore requires self-attention token boundaries.
    ``layout`` is retained as an alias for ``key_layout``.
    """
    layout = layout.lower()
    if key_layout is None:
        key_layout = layout
    else:
        key_layout = key_layout.lower()
        if layout != "linear" and layout != key_layout:
            raise ValueError("layout and key_layout disagree")
    assert key_layout in ("linear", "vectorized")
    assert num_waves in (4, 8)
    if cu_seqlens_k is None:
        if key_layout != "vectorized":
            raise ValueError(
                "cu_seqlens_k=None requires key_layout='vectorized'"
            )
        if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
            raise ValueError(
                "cu_seqlens_k=None requires self-attention token boundaries"
            )
        cu_seqlens_k = cu_seqlens_q
    tensors = (q, k, v, cu_seqlens_q, cu_seqlens_k)
    assert all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
    assert all(tensor.device == q.device for tensor in tensors)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert q.ndim == v.ndim == 3
    assert q.shape[2] == v.shape[2] == 128
    total_q, num_heads, head_dim = q.shape
    total_k = v.shape[0]
    assert v.shape[1:] == (num_heads, head_dim)
    assert k.shape == (total_k, num_heads, head_dim)
    assert max_seqlen_q > 0 and max_seqlen_k > 0
    assert 0 <= min_seqlen_q <= max_seqlen_q
    assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_q.ndim == cu_seqlens_k.ndim == 1
    assert cu_seqlens_q.shape == cu_seqlens_k.shape
    q_bounds, k_bounds = _validated_varlen_bounds(
        cu_seqlens_q,
        cu_seqlens_k,
        total_q,
        total_k,
    )
    q_lengths = tuple(
        stop - start for start, stop in zip(q_bounds, q_bounds[1:])
    )
    k_lengths = tuple(
        stop - start for start, stop in zip(k_bounds, k_bounds[1:])
    )
    assert max(q_lengths) <= max_seqlen_q
    assert max(k_lengths) <= max_seqlen_k

    unsupported = []
    if min_seqlen_q != 0:
        unsupported.append("min_seqlen_q")
    if dropout_p != 0.0:
        unsupported.append("dropout_p")
    if logits_soft_cap != 0.0:
        unsupported.append("logits_soft_cap")
    if causal:
        unsupported.append("causal")
    if tuple(window_size[:2]) != (-1, -1):
        unsupported.append("window_size")
    if len(window_size) > 2 and window_size[2] != 0:
        unsupported.append("sink_size")
    if bias is not None:
        unsupported.append("bias")
    if alibi_slopes is not None:
        unsupported.append("alibi_slopes")
    if deterministic:
        unsupported.append("deterministic")
    if return_lse:
        unsupported.append("return_lse")
    if return_attn_probs:
        unsupported.append("return_attn_probs")
    if how_v3_bf16_cvt != 1:
        unsupported.append("how_v3_bf16_cvt")
    if block_table is not None:
        unsupported.append("block_table")
    if cu_seqlens_q_padded is not None:
        unsupported.append("cu_seqlens_q_padded")
    if cu_seqlens_k_padded is not None:
        unsupported.append("cu_seqlens_k_padded")
    if sink_ptr is not None:
        unsupported.append("sink_ptr")
    if any(tensor.requires_grad for tensor in (q, k, v)):
        unsupported.append("backward")
    if unsupported:
        raise NotImplementedError(
            "8-wave varlen fast path does not support: "
            + ", ".join(unsupported)
        )

    if out is None:
        out = torch.empty_like(q)
    assert out.is_cuda and out.is_contiguous()
    assert out.device == q.device
    assert out.dtype == torch.bfloat16 and out.shape == q.shape

    stream = torch.cuda.current_stream(q.device)
    (
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        page_token_starts,
        page_valid_lens,
    ) = _paged_metadata(cu_seqlens_k)
    num_pages = page_token_starts.numel()
    mha_key = k
    if key_layout == "vectorized":
        mha_key = _paged_key_workspace(
            k, num_pages, num_heads, head_dim, stream
        )
        _compile_paged_k_converter(
            num_heads, num_pages, total_k, head_dim
        )(
            k,
            page_token_starts,
            page_valid_lens,
            mha_key,
            stream=stream,
        )
    vectorized_value = _paged_value_workspace(
        v, num_pages, num_heads, head_dim, stream
    )
    _compile_paged_v_converter(
        num_heads, num_pages, total_k, head_dim
    )(
        v,
        page_token_starts,
        page_valid_lens,
        vectorized_value,
        stream=stream,
    )
    q_descale, unit_descale = _descale_workspace(
        q, softmax_scale, stream
    )

    if num_waves == 4:
        import sys
        from pathlib import Path

        pa4_dir = Path(__file__).resolve().parents[1] / "pa_4wave"
        if str(pa4_dir) not in sys.path:
            sys.path.insert(0, str(pa4_dir))
        MHA = importlib.import_module("pa_prefill_4wave").MHA

        mha = MHA(
            num_heads,
            num_heads,
            head_dim,
            head_dim,
            32,
            causal,
            key_layout=key_layout,
        )
    else:
        import sys
        from pathlib import Path

        pa8_dir = Path(__file__).resolve().parents[1] / "pa_8wave"
        if str(pa8_dir) not in sys.path:
            sys.path.insert(0, str(pa8_dir))
        module = importlib.import_module("pa_prefill_8w32x32")
        PagedAttention = module.PagedAttention

        mha = PagedAttention(
            num_heads,
            num_heads,
            head_dim,
            head_dim,
            32,
            causal,
            key_layout=key_layout,
        )
    mha(
        q,
        mha_key,
        vectorized_value,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        q_descale,
        unit_descale,
        unit_descale,
        kv_last_page_lens,
        out,
        stream=stream,
    )
    return out


__all__ = [
    "flash_attn_varlen_func",
]