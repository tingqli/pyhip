import pytest
import torch

from flash_attn_varlen_8wave import flash_attn_varlen_func


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="requires gfx942",
)


def _reference(q, k, v, segments, scale=None):
    output = torch.empty_like(q)
    start = 0
    for length in segments:
        stop = start + length
        output[start:stop] = torch.nn.functional.scaled_dot_product_attention(
            q[start:stop].transpose(0, 1).unsqueeze(0),
            k[start:stop].transpose(0, 1).unsqueeze(0),
            v[start:stop].transpose(0, 1).unsqueeze(0),
            scale=scale,
        ).squeeze(0).transpose(0, 1)
        start = stop
    return output


@pytest.mark.parametrize("num_waves", [4, 8])
def test_linear_varlen_tail(num_waves):
    segments = (249, 7)
    total, heads, dim = sum(segments), 2, 128
    generator = torch.Generator(device="cuda").manual_seed(1101)
    q, k, v = (
        torch.randn(
            total,
            heads,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(segments).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max(segments),
        max(segments),
        key_layout="linear",
        num_waves=num_waves,
    )
    torch.cuda.synchronize()
    reference = _reference(q, k, v, segments)
    relative_l2 = (
        torch.linalg.vector_norm(output.float() - reference.float())
        / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    ).item()

    assert relative_l2 < 0.005
    assert torch.allclose(
        output[-segments[-1] :],
        reference[-segments[-1] :],
        rtol=0.005,
        atol=0.005,
    )


@pytest.mark.parametrize("num_waves", [4, 8])
def test_linear_and_vectorized_key_layouts_match(num_waves):
    segments = (64, 7)
    tokens, heads, dim = sum(segments), 2, 128
    generator = torch.Generator(device="cuda").manual_seed(7)
    q, k, v = (
        torch.randn(
            tokens,
            heads,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(segments).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    linear_output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max(segments),
        max(segments),
        key_layout="linear",
        num_waves=num_waves,
    )
    vectorized_output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max(segments),
        max(segments),
        key_layout="vectorized",
        num_waves=num_waves,
    )
    torch.cuda.synchronize()

    assert torch.equal(linear_output, vectorized_output)


@pytest.mark.parametrize("num_waves", [4, 8])
def test_vectorized_key_accepts_missing_k_boundaries(num_waves):
    segments = (64, 7)
    tokens, heads, dim = sum(segments), 2, 128
    generator = torch.Generator(device="cuda").manual_seed(41)
    q, k, v = (
        torch.randn(
            tokens,
            heads,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(segments).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    explicit = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max(segments),
        max(segments),
        key_layout="vectorized",
        num_waves=num_waves,
    )
    inferred = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        None,
        max(segments),
        max(segments),
        key_layout="vectorized",
        num_waves=num_waves,
    )
    torch.cuda.synchronize()

    assert torch.equal(explicit, inferred)


def test_linear_key_rejects_missing_k_boundaries():
    tokens, heads, dim = 32, 1, 128
    q = torch.zeros(
        tokens, heads, dim, device="cuda", dtype=torch.bfloat16
    )
    cu_seqlens = torch.tensor(
        [0, tokens], device="cuda", dtype=torch.int32
    )

    with pytest.raises(
        ValueError,
        match="cu_seqlens_k=None requires key_layout='vectorized'",
    ):
        flash_attn_varlen_func(
            q,
            q,
            q,
            cu_seqlens,
            None,
            tokens,
            tokens,
            key_layout="linear",
        )


@pytest.mark.parametrize("num_waves", [4, 8])
@pytest.mark.parametrize(
    ("segments", "heads"),
    [
        ((1024, 513, 33), 4),
        ((4097, 1025, 129, 7), 1),
    ],
)
def test_large_varlen_shape_combinations(num_waves, segments, heads):
    tokens, dim = sum(segments), 128
    generator = torch.Generator(device="cuda").manual_seed(
        1000 + tokens + heads
    )
    q, k, v = (
        torch.randn(
            tokens,
            heads,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(segments).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    linear_output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max(segments),
        max(segments),
        key_layout="linear",
        num_waves=num_waves,
    )
    vectorized_output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        None,
        max(segments),
        max(segments),
        key_layout="vectorized",
        num_waves=num_waves,
    )
    reference = _reference(q, k, v, segments)
    torch.cuda.synchronize()

    for output in (linear_output, vectorized_output):
        relative_l2 = (
            torch.linalg.vector_norm(output.float() - reference.float())
            / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
        ).item()
        assert relative_l2 < 0.005
    assert torch.equal(linear_output, vectorized_output)
