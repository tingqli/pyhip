// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"

namespace ck_tile {

// TODO: currently only support 16 bit input, which means only support tr16_b128; will use ADataType
// to determine the layout in the future
template <typename Impl>
struct AWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<Impl::kRepeat>,
        tuple<sequence<Impl::kAMLane>,
              sequence<Impl::kABK0PerLane, Impl::kABKLane, Impl::kABK1PerLane>>,
        tuple<typename Impl::kABPs2RHssMajor>,
        tuple<typename Impl::kABPs2RHssMinor>,
        typename Impl::kABYs2RHsMajor,
        typename Impl::kABYs2RHsMinor>;
};

template <typename Impl>
struct BWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<Impl::kRepeat>,
        tuple<sequence<Impl::kBNLane>,
              sequence<Impl::kABK0PerLane, Impl::kABKLane, Impl::kABK1PerLane>>,
        tuple<typename Impl::kABPs2RHssMajor>,
        tuple<typename Impl::kABPs2RHssMinor>,
        typename Impl::kABYs2RHsMajor,
        typename Impl::kABYs2RHsMinor>;
};

template <typename Impl>
struct CWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<Impl::kCNLane>>,
        tuple<typename Impl::kCPs2RHssMajor>,
        tuple<typename Impl::kCPs2RHssMinor>,
        typename Impl::kCYs2RHsMajor,
        typename Impl::kCYs2RHsMinor>;
};

template <typename WarpGemmAttributeWmmaImpl_, bool kTransC = false>
struct WarpGemmAttributeWmma
{
    using Impl = remove_cvref_t<WarpGemmAttributeWmmaImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kM;
    static constexpr index_t kN          = Impl::kN;
    static constexpr index_t kK          = Impl::kK;
    static constexpr index_t kKPerThread = Impl::kABK0PerLane * Impl::kABK1PerLane;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    // 16 bit input, kAMLane = 16, kABK0PerLane = 4, kABKLane = 2, kABK1PerLane = 2
    // 8  bit input, kAMLane = 16, kABK0PerLane = 2, kABKLane = 2, kABK1PerLane = 4
    using AWarpDstrEncoding = typename AWarpDstrEncodingTrait<Impl>::type;
    using BWarpDstrEncoding = typename BWarpDstrEncodingTrait<Impl>::type;

    // kCM0PerLane = 4, kCMLane = 2, kCM1PerLane = 2, kCNLane = 16 for 16 bit input
    // kCM0PerLane = 2, kCMLane = 2, kCM1PerLane = 4, kCNLane = 16 for 8 bit input
    using CWarpDstrEncoding = typename CWarpDstrEncodingTrait<Impl>::type;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        if constexpr(kTransC)
        {
            Impl{}(c_vec, b_vec, a_vec, bool_constant<post_nop_>{});
        }
        else
        {
            Impl{}(c_vec, a_vec, b_vec, bool_constant<post_nop_>{});
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(kTransC)
        {
            return Impl{}(b_vec, a_vec);
        }
        else
        {
            return Impl{}(a_vec, b_vec);
        }
    }
};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          index_t M_Warp_Tile,
          index_t N_Warp_Tile,
          index_t K_Warp_Tile>
CK_TILE_HOST bool check_wmma_supported()
{
    if(is_gfx12_supported())
    {
        return has_wmma_traits_v<gfx12_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else if(is_gfx11_supported())
    {
        return has_wmma_traits_v<gfx11_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else
    {
        return false;
    }
}

} // namespace ck_tile
