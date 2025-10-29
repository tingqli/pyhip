// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once
#include "streamk_gemm_tile_partitioner.hpp"
namespace ck_tile {

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::StreamKTilePartitionerBase(
    index_t m, index_t n, index_t k, index_t grid)
    : grid_{grid}, n_{n}
{
    iters_per_tile_ = integer_divide_ceil(k, KPerBlock);
    num_tiles_      = integer_divide_ceil(m, MPerBlock) * integer_divide_ceil(n_, NPerBlock);

    bool big_enough         = num_tiles_ > grid_;
    index_t remainder_tiles = num_tiles_ % grid_;

    if(remainder_tiles)
    {
        sk_tiles_       = big_enough ? full_tiles_ * grid_ + (num_tiles_ % grid_) : num_tiles_;
        sk_tiles_       = min(num_tiles_, sk_tiles_);
        sk_ctas_        = grid_;
        total_sk_iters_ = sk_tiles_ * iters_per_tile_;

        // If there still isn't enough work to saturate all CUs, then just revert to DP only.
        if(total_sk_iters_ < grid_)
        {
            sk_tiles_       = 0;
            sk_ctas_        = 0;
            total_sk_iters_ = 0;
        }
    }
    else // Full DP (i.e., no Stream-K)
    {
        sk_tiles_       = 0;
        sk_ctas_        = 0;
        total_sk_iters_ = 0;
    }

    iters_per_sk_cta_ = sk_ctas_ ? total_sk_iters_ / sk_ctas_ : 0;
    extra_iters_      = sk_ctas_ ? total_sk_iters_ % sk_ctas_ : 0;

    dp_tiles_       = num_tiles_ - sk_tiles_;
    total_dp_iters_ = dp_tiles_ * iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_partials_buffer_size(
    index_t acc_element_bytes) const noexcept
{
    return MPerBlock * NPerBlock * acc_element_bytes * sk_ctas_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_flags_buffer_size()
    const noexcept
{
    return sizeof(index_t) * sk_ctas_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE void
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_iter_boundaries(
    index_t& iter, index_t& iter_end, index_t cta_idx) const noexcept
{
    index_t extra_iters_before_me = ck_tile::min(cta_idx, extra_iters_);
    iter     = total_dp_iters_ + cta_idx * iters_per_sk_cta_ + extra_iters_before_me;
    iter_end = iter + iters_per_sk_cta_ + (cta_idx < extra_iters_);
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_tile_index(
    index_t iter) const noexcept
{
    return iter / iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE void
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_tile_boundaries(
    index_t& tile_iter, index_t& tile_iter_end, index_t tile_idx) const noexcept
{
    tile_iter     = tile_idx * iters_per_tile_;
    tile_iter_end = tile_iter + iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE /* static */ index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_local_iter(
    index_t iter, index_t tile_iter) noexcept
{
    return iter - tile_iter;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE /* static */ index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_local_iter_end(
    index_t tile_iter, index_t iter_end, index_t tile_iter_end) noexcept
{
    return ck_tile::min(iter_end, tile_iter_end) - tile_iter;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE auto
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_output_tile_index(
    index_t tile_idx) const noexcept -> tuple<index_t, index_t>
{
    const index_t n_macro_tiles = integer_divide_ceil(n_, NPerBlock);

    const index_t im = amd_wave_read_first_lane(tile_idx / n_macro_tiles);
    const index_t in = amd_wave_read_first_lane(tile_idx - im * n_macro_tiles);
    return make_tuple(im, in);
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_workspace_size(
    index_t acc_element_bytes) const noexcept
{
    if constexpr(ReductionStrategy == StreamKReductionStrategy::Reduction)
    {

        return get_partials_buffer_size(acc_element_bytes) + get_flags_buffer_size();
    }
    else // ReductionStrategy is Atomics
    {
        return 0;
    }
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_num_tiles()
    const noexcept
{
    return num_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_grid() const noexcept
{
    return grid_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_dp_tiles() const noexcept
{
    return dp_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_sk_tiles() const noexcept
{
    return sk_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_sk_ctas() const noexcept
{
    return sk_ctas_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_total_sk_iters()
    const noexcept
{
    return total_sk_iters_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_iters_per_tile()
    const noexcept
{
    return iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_iters_per_sk_cta()
    const noexcept
{
    return iters_per_sk_cta_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_extra_iters()
    const noexcept
{
    return extra_iters_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_total_dp_iters()
    const noexcept
{
    return total_dp_iters_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_n() const noexcept
{
    return n_;
}

template <typename BlockGemmShapeType,
          StreamKReductionStrategy ReductionStrategyType,
          bool Persistent>
struct StreamKTilePartitioner_v2;

// child class for Persistent Tile Partitioner
template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
StreamKTilePartitioner_v2<BlockGemmShapeType, ReductionStrategyType, true>::
    StreamKTilePartitioner_v2(ck_tile::index_t m,
                              ck_tile::index_t n,
                              ck_tile::index_t k,
                              ck_tile::index_t grid)
    : StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>(m, n, k, grid)
{ // inherit from base constructor
    dp_tiles_per_cta_ = this->dp_tiles_ / this->grid_;
    extra_dp_tiles_   = this->dp_tiles_ % this->grid_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST auto
StreamKTilePartitioner_v2<BlockGemmShapeType, ReductionStrategyType, true>::grid_size()
    const noexcept -> dim3
{
    if(extra_dp_tiles_ == 0)
    {
        return dim3(this->grid_, 1, 1);
    }
    else
    {
        return dim3(this->num_tiles_, 1, 1);
    }
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitioner_v2<BlockGemmShapeType, ReductionStrategyType, true>::get_dp_tiles_per_cta()
    const noexcept
{
    return dp_tiles_per_cta_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitioner_v2<BlockGemmShapeType, ReductionStrategyType, true>::get_extra_dp_tiles()
    const noexcept
{
    return extra_dp_tiles_;
}

// child class for Non-Persistent Tile Partitioner
template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
StreamKTilePartitioner_v2<BlockGemmShapeType, ReductionStrategyType, false>::
    StreamKTilePartitioner_v2(ck_tile::index_t m,
                              ck_tile::index_t n,
                              ck_tile::index_t k,
                              ck_tile::index_t grid)
    : StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>(m, n, k, grid)
{ // inherit from base constructor
    dp_ctas_            = this->dp_tiles_;
    dp_start_block_idx_ = 0;
    sk_start_block_idx_ = this->dp_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST auto
StreamKTilePartitioner_v2<BlockGemmShapeType, ReductionStrategyType, false>::grid_size()
    const noexcept -> dim3
{
    return dim3(dp_ctas_ + this->get_sk_ctas(), 1, 1);
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitioner_v2<BlockGemmShapeType, ReductionStrategyType, false>::get_dp_ctas()
    const noexcept
{
    return dp_ctas_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitioner_v2<BlockGemmShapeType, ReductionStrategyType, false>::
    get_dp_start_block_idx() const noexcept
{
    return dp_start_block_idx_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitioner_v2<BlockGemmShapeType, ReductionStrategyType, false>::
    get_sk_start_block_idx() const noexcept
{
    return sk_start_block_idx_;
}

} // namespace ck_tile
