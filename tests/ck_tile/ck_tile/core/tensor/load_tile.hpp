// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/tensor/tile_window_linear.hpp"
#include "ck_tile/core/tensor/null_tile_window.hpp"
#include "ck_tile/core/tensor/null_tensor.hpp"

namespace ck_tile {

template <typename TileWindow_, index_t i_access = -1, bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile(const TileWindow_& tile_window,
                              number<i_access>                     = {},
                              bool_constant<oob_conditional_check> = {})
{
    return tile_window.load(number<i_access>{}, bool_constant<oob_conditional_check>{});
}

/**
 * @brief Load tile with elementwise function
 *
 * @note This function is a modification of the existing load function.
 *       It has been extended with two additional parameters: it takes a tuple as input
 *       and an elementwise function. For each A = A0, A1… AN, the elementwise function
 *       is additionally applied during a single read.
 */
template <typename TileWindow_,
          typename ElementWise_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile_with_elementwise(const TileWindow_& tile_window,
                                               ElementWise_ elementwise,
                                               number<i_access>                     = {},
                                               bool_constant<oob_conditional_check> = {})
{
    // TODO: Tile windows should works with unknow number of params
    // Load element_wise API works only when the input typle is a tuple-tyupe
    return tile_window[number<0>{}].load(
        tile_window, elementwise, number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename DistributedTensor_,
          typename TileWindow_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile(DistributedTensor_& dst_tile,
                              const TileWindow_& tile_window,
                              number<i_access>                     = {},
                              bool_constant<oob_conditional_check> = {})
{
    return tile_window.load(dst_tile, number<i_access>{}, bool_constant<oob_conditional_check>{});
}

/**
 * @brief Loads a tile of data using inline assembly.
 *
 * @note Bare in mind that loading data this way, you have to manually initialize your
 *       thread buffer and synchronize load afterwards in order to make sure it's done before
 *       using loaded data from registers
 *       @see `tile_window_with_static_distribution::init_raw()` and `buffer_view.hpp`
 *       @see  `buffer_load_fence()`
 */
template <typename T,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto load_tile_raw(T& tile,
                                  const tile_window_with_static_distribution<BottomTensorView_,
                                                                             WindowLengths_,
                                                                             TileDistribution_,
                                                                             NumCoord>& tile_window,
                                  number<i_access>                     = {},
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {})
{
    tile_window.load_raw(
        tile, number<i_access>{}, bool_constant<oob_conditional_check>{}, bool_constant<pre_nop>{});
}

template <typename T,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto load_tile_raw(T& tile,
                                  const tile_window_linear<BottomTensorView_,
                                                           WindowLengths_,
                                                           TileDistribution_,
                                                           LinearBottomDims_>& tile_window,
                                  number<i_access>                     = {},
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {})
{
    tile_window.load_raw(
        tile, number<i_access>{}, bool_constant<oob_conditional_check>{}, bool_constant<pre_nop>{});
}

template <typename LdsTileWindow_,
          typename TileWindow_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto async_load_tile(LdsTileWindow_&& lds_tile,
                                    const TileWindow_& tile_window,
                                    number<i_access>                     = {},
                                    bool_constant<oob_conditional_check> = {})
{
    return tile_window.async_load(
        lds_tile, number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename LdsTileWindow_,
          typename TileWindow_,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto async_load_tile_raw(LdsTileWindow_&& lds_tile,
                                        const TileWindow_& tile_window,
                                        number<i_access>                     = {},
                                        bool_constant<oob_conditional_check> = {},
                                        bool_constant<pre_nop>               = {})
{
    return tile_window.async_load_raw(lds_tile,
                                      number<i_access>{},
                                      bool_constant<oob_conditional_check>{},
                                      bool_constant<pre_nop>{});
}

CK_TILE_DEVICE auto async_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

template <typename WindowLengths>
CK_TILE_DEVICE auto load_tile(const null_tile_window<WindowLengths>&)
{
    return null_tensor{};
}

template <typename T, typename WindowLengths>
CK_TILE_DEVICE auto load_tile_raw(T& /*null_tile*/, const null_tile_window<WindowLengths>&)
{
}

} // namespace ck_tile
