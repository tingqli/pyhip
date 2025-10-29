// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

#include <optional>

namespace ck_tile {

template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename DsLayout_,
          typename ELayout_,
          typename CDElementwise_,
          index_t kM_,
          index_t kN_,
          index_t MWave_,
          index_t NWave_,
          index_t MPerXdl_,
          index_t NPerXdl_,
          index_t KPerXdl_,
          bool isCTransposed_,
          memory_operation_enum MemoryOperation_,
          index_t kNumWaveGroups_      = 1,
          bool FixedVectorSize_        = false,
          index_t VectorSizeC_         = 1,
          bool TiledMMAPermuteN_       = false,
          index_t BlockedXDLN_PerWarp_ = 1> // The number of continuous xdl_output per warp
struct CShuffleEpilogueProblem
{
    using AsDataType                                       = remove_cvref_t<AsDataType_>;
    using BsDataType                                       = remove_cvref_t<BsDataType_>;
    using AccDataType                                      = remove_cvref_t<AccDataType_>;
    using ODataType                                        = remove_cvref_t<ODataType_>;
    using DsDataType                                       = remove_cvref_t<DsDataType_>;
    using DsLayout                                         = remove_cvref_t<DsLayout_>;
    using ELayout                                          = remove_cvref_t<ELayout_>;
    using CDElementwise                                    = remove_cvref_t<CDElementwise_>;
    static constexpr index_t kBlockSize                    = MWave_ * NWave_ * get_warp_size();
    static constexpr index_t kMPerBlock                    = kM_;
    static constexpr index_t kNPerBlock                    = kN_;
    static constexpr index_t MWave                         = MWave_;
    static constexpr index_t NWave                         = NWave_;
    static constexpr index_t MPerXdl                       = MPerXdl_;
    static constexpr index_t NPerXdl                       = NPerXdl_;
    static constexpr index_t KPerXdl                       = KPerXdl_;
    static constexpr index_t isCTransposed                 = isCTransposed_;
    static constexpr memory_operation_enum MemoryOperation = MemoryOperation_;
    static constexpr bool FixedVectorSize                  = FixedVectorSize_;
    static constexpr index_t VectorSizeC                   = VectorSizeC_;
    static constexpr index_t BlockedXDLN_PerWarp           = BlockedXDLN_PerWarp_;
    static constexpr bool TiledMMAPermuteN                 = TiledMMAPermuteN_;
    static constexpr index_t kNumWaveGroups                = kNumWaveGroups_;
    static constexpr index_t NumDTensor                    = DsDataType::size();

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");
};

template <typename Problem_, typename Policy_ = void>
struct CShuffleEpilogue
{
    using Problem     = remove_cvref_t<Problem_>;
    using AsDataType  = remove_cvref_t<typename Problem::AsDataType>;
    using BsDataType  = remove_cvref_t<typename Problem::BsDataType>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;
    using DsDataType  = remove_cvref_t<typename Problem::DsDataType>;
    using DsLayout    = remove_cvref_t<typename Problem::DsLayout>;

    static constexpr bool ADataTypeIsTuple = is_detected<is_tuple, AsDataType>::value;
    static constexpr bool BDataTypeIsTuple = is_detected<is_tuple, BsDataType>::value;

    using AsDataTypeTuple = std::conditional_t<ADataTypeIsTuple,
                                               remove_cvref_t<AsDataType>,
                                               remove_cvref_t<tuple<AsDataType>>>;

    using BsDataTypeTuple = std::conditional_t<BDataTypeIsTuple,
                                               remove_cvref_t<BsDataType>,
                                               remove_cvref_t<tuple<BsDataType>>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataTypeTuple>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataTypeTuple>>;

    using ATypeToUse =
        std::conditional_t<std::is_same_v<ADataType, pk_int4_t>, BDataType, ADataType>;
    // Used for weight-only quantization kernel, B would be dequantized to the same data type as A
    using BTypeToUse =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
    using ELayout       = remove_cvref_t<typename Problem::ELayout>;
    using CDElementwise = remove_cvref_t<typename Problem::CDElementwise>;
    static constexpr memory_operation_enum MemoryOperation = Problem::MemoryOperation;
    static constexpr index_t kBlockSize                    = Problem::kBlockSize;
    static constexpr index_t kMPerBlock                    = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock                    = Problem::kNPerBlock;
    static constexpr index_t MWave                         = Problem::MWave;
    static constexpr index_t NWave                         = Problem::NWave;
    static constexpr index_t MPerXdl                       = Problem::MPerXdl;
    static constexpr index_t NPerXdl                       = Problem::NPerXdl;
    static constexpr index_t KPerXdl                       = Problem::KPerXdl;
    static constexpr index_t isCTransposed                 = Problem::isCTransposed;
    static constexpr bool FixedVectorSize                  = Problem::FixedVectorSize;
    static constexpr bool TiledMMAPermuteN                 = Problem::TiledMMAPermuteN;
    static constexpr index_t BlockedXDLN_PerWarp           = Problem::BlockedXDLN_PerWarp;
    static constexpr index_t VectorSizeC                   = Problem::VectorSizeC;
    static constexpr index_t MPerIteration                 = MPerXdl * MWave;
    static constexpr index_t NPerIteration                 = NPerXdl * NWave;
    static constexpr index_t NumDTensor                    = Problem::NumDTensor;
    static constexpr index_t MRepeat                       = kMPerBlock / (MPerXdl * MWave);
    static constexpr index_t NRepeat                       = kNPerBlock / (NPerXdl * NWave);

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");
    /**
     * @brief Get the vector store size for C tensor.
     *
     * @note The vector store size for output C tensor would depend on multiple factors
     *       like its data layout and warp gemm C transposition. In general it would
     *       be the number of consecutive elements in contiguous C dimension hold by
     *       single thread.
     *
     * @return The vector store size for C tensor.
     */
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC()
    {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeC;
        }
        constexpr index_t max_vector_size = 16;
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    /**
     * @brief Get the vector store size for Di tensor.
     *
     * @return The vector store size for Di tensor.
     */
    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I> index)
    {
        constexpr index_t max_vector_size = 16;
        using DiDataType = remove_cvref_t<std::tuple_element_t<index.value, DsDataType>>;
        using DiLayout   = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
        if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else
        {
            static_assert(false, "Unsupported DLayout!");
        }
        return max_vector_size / sizeof(DiDataType);
    }
    /**
     * @brief Shuffle tile configuration parameters
     *
     * @details These parameters control the number of XDL tiles processed per wave in each shuffle
     * iteration:
     * - NumMXdlPerWavePerShuffle: Number of XDL tiles in M dimension processed per wave
     * - NumNXdlPerWavePerShuffle: Number of XDL tiles in N dimension processed per wave
     */
    static constexpr auto shuffle_tile_tuple = [] {
        constexpr index_t elem_per_thread = MPerXdl * NPerXdl / get_warp_size();
        if constexpr(elem_per_thread >= GetVectorSizeC())
        {
            return std::make_tuple(1, 1);
        }
        else
        {
            constexpr index_t num_xdl_shuffles = GetVectorSizeC() / elem_per_thread;
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                static_assert((kMPerBlock % (MPerXdl * MWave) == 0) &&
                                  (kMPerBlock % num_xdl_shuffles == 0),
                              "kMPerBlock must be divisible by MPerXdl*MWave and "
                              "num_xdl_shuffles for CShuffleEpilogue");
                return std::make_tuple(min(num_xdl_shuffles, kMPerBlock / (MPerXdl * MWave)), 1);
            }
            else
            {
                static_assert((kNPerBlock % (NPerXdl * NWave) == 0) &&
                                  (kNPerBlock % num_xdl_shuffles == 0),
                              "kNPerBlock must be divisible by NPerXdl*NWave and "
                              "num_xdl_shuffles for CShuffleEpilogue");
                return std::make_tuple(1, min(num_xdl_shuffles, kNPerBlock / (NPerXdl * NWave)));
            }
        }
    }();
    static constexpr index_t NumMXdlPerWavePerShuffle = std::get<0>(shuffle_tile_tuple);
    static constexpr index_t NumNXdlPerWavePerShuffle =
        max(BlockedXDLN_PerWarp, std::get<1>(shuffle_tile_tuple));

    static constexpr auto MNPerIterationShuffle = [] {
        constexpr index_t m_val = MPerXdl * MWave * NumMXdlPerWavePerShuffle;
        constexpr index_t n_val = NPerXdl * NWave * NumNXdlPerWavePerShuffle;
        if constexpr(kMPerBlock % m_val != 0 || kNPerBlock % n_val != 0)
            return std::make_tuple(MPerXdl * MWave, NPerXdl * NWave);
        else
            return std::make_tuple(m_val, n_val);
    }();
    static constexpr index_t MPerIterationShuffle = std::get<0>(MNPerIterationShuffle);
    static constexpr index_t NPerIterationShuffle = std::get<1>(MNPerIterationShuffle);

    using WG = WarpGemmDispatcher<ATypeToUse,
                                  BTypeToUse,
                                  AccDataType,
                                  MPerXdl,
                                  NPerXdl,
                                  KPerXdl,
                                  isCTransposed>;

    using CWarpDstr         = typename WG::CWarpDstr;
    using CWarpTensor       = typename WG::CWarpTensor;
    using CWarpDstrEncoding = typename WG::CWarpDstrEncoding;
    using SFC               = space_filling_curve<sequence<kMPerBlock, kNPerBlock>,
                                                  sequence<0, 1>,
                                                  sequence<MPerIterationShuffle, NPerIterationShuffle>>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<NPerIterationShuffle>{}, number<1>{}));
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<1>{}, number<MPerIterationShuffle>{}));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    CK_TILE_DEVICE static constexpr auto MakeLdsDistributionEncode()
    {
        constexpr auto block_outer_dstr_encoding = [] {
            if constexpr(BlockedXDLN_PerWarp == 1)
            {
                return tile_distribution_encoding<sequence<>,
                                                  tuple<sequence<NumMXdlPerWavePerShuffle, MWave>,
                                                        sequence<NumNXdlPerWavePerShuffle, NWave>>,
                                                  tuple<sequence<1, 2>>,
                                                  tuple<sequence<1, 1>>,
                                                  sequence<1, 2>,
                                                  sequence<0, 0>>{};
            }
            else
            {
                constexpr int RakedXDLN_PerWarp = NumNXdlPerWavePerShuffle / BlockedXDLN_PerWarp;
                // BlockedLayout
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<NumMXdlPerWavePerShuffle, MWave>,
                          sequence<RakedXDLN_PerWarp, NWave, BlockedXDLN_PerWarp>>,
                    tuple<sequence<1, 2>>,
                    tuple<sequence<1, 1>>,
                    sequence<1, 2, 2>,
                    sequence<0, 0, 2>>{};
            }
        }();
        constexpr auto block_dstr_encoding = detail::make_embed_tile_distribution_encoding(
            block_outer_dstr_encoding, typename CWarpDstr::DstrEncode{});

        return block_dstr_encoding;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType);
    }

    template <index_t iAccess, typename LdsTile, typename ScaleM, typename ScaleN>
    CK_TILE_DEVICE void
    scale_tile(LdsTile& lds_tile, ScaleM& scale_m_window, ScaleN& scale_n_window)
    {
        // Check if scales are EmptyScale first (no scaling needed)
        if constexpr(std::is_same_v<ScaleM, EmptyScale> && std::is_same_v<ScaleN, EmptyScale>)
        {
            // No scaling needed - this is a no-op
        }
        // Check if scales are scalar AccDataType
        else if constexpr(std::is_same_v<ScaleM, AccDataType> &&
                          std::is_same_v<ScaleN, AccDataType>)
        {
            // Handle scalar scales
            const AccDataType scale_m = scale_m_window;
            const AccDataType scale_n = scale_n_window;
            tile_elementwise_inout([&](auto& element) { element = element * scale_m * scale_n; },
                                   lds_tile);
        }
        // Otherwise, assume they are tile windows that can be loaded
        else
        {
            // Load tiles
            const auto scale_m_tile = load_tile(scale_m_window);
            const auto scale_n_tile = load_tile(scale_n_window);

            // Compute element-wise product in-place i.e. lds_tile = lds_tile * scale_m * scale_n
            tile_elementwise_inout(
                element_wise::MultiDMultiply{}, lds_tile, lds_tile, scale_m_tile, scale_n_tile);

            // Move scale windows
            constexpr index_t num_access = SFC::get_num_of_access();
            if constexpr(iAccess != num_access - 1)
            {
                constexpr auto step = SFC::get_forward_step(number<iAccess>{});

                move_tile_window(scale_m_window, {step.at(number<0>{}), step.at(number<1>{})});
                move_tile_window(scale_n_window, {step.at(number<0>{}), step.at(number<1>{})});
            }
        }
    }

    template <index_t iAccess, typename OAccTile, typename LdsTile>
    CK_TILE_DEVICE void slice_acc_tile(const OAccTile& o_acc_tile, LdsTile& lds_tile)
    {
        constexpr auto idx_y_start = SFC::get_index(number<iAccess>{});

        constexpr auto mIter = number<idx_y_start.at(number<0>{}) / (MPerIterationShuffle)>{};
        constexpr auto nIter = number<idx_y_start.at(number<1>{}) / (NPerIterationShuffle)>{};
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        lds_tile.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
            merge_sequences(
                sequence<mIter * NumMXdlPerWavePerShuffle, nIter * NumNXdlPerWavePerShuffle>{},
                c_warp_y_index_zeros),
            merge_sequences(sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                            c_warp_y_lengths));
    }

    template <typename LdsTile, typename InLdsWindow>
    CK_TILE_DEVICE void cast_lds_tile(LdsTile& lds_tile, InLdsWindow& in_lds_window)
    {
        const auto c_warptile_in_tensor_casted = cast_tile<ODataType>(lds_tile);

        store_tile(in_lds_window, c_warptile_in_tensor_casted);
    }

    template <typename DramWindows, typename COutTensor>
    CK_TILE_DEVICE void apply_d_tensors(DramWindows& d_dram_windows, COutTensor& c_out_tensor)
    {
        const auto ds_tensor = generate_tuple(
            [&](auto idx) { return load_tile(d_dram_windows[idx]); }, number<NumDTensor>{});

        const auto c_ds_tiles = concat_tuple_of_reference(
            tie(c_out_tensor, c_out_tensor),
            generate_tie([&](auto idx) -> const auto& { return ds_tensor[idx]; },
                         number<NumDTensor>{}));

        tile_elementwise_inout_unpack(typename Problem::CDElementwise{}, c_ds_tiles);
    }

    template <typename OutDramWindow, typename COutTensor>
    CK_TILE_DEVICE void store_to_dram(OutDramWindow& out_dram_window,
                                      const COutTensor& c_out_tensor)
    {
        if constexpr(MemoryOperation == memory_operation_enum::set)
        {
            store_tile(out_dram_window, c_out_tensor);
        }
        else
        {
            update_tile(out_dram_window, c_out_tensor);
        }
    }

    /**
     * @brief Move both the output and D tensors windows for the next access.
     */
    template <index_t iAccess, typename OutDramWindow, typename DDramWindows>
    CK_TILE_DEVICE void move_windows(OutDramWindow& out_dram_window, DDramWindows& d_dram_windows)
    {
        constexpr index_t num_access = SFC::get_num_of_access();
        if constexpr(iAccess != num_access - 1)
        {
            constexpr auto step = SFC::get_forward_step(number<iAccess>{});

            // move the output dram window
            move_tile_window(out_dram_window, {step.at(number<0>{}), step.at(number<1>{})});

            // move windows for each of the D matrices (inputs for element-wise)
            static_for<0, NumDTensor, 1>{}([&](auto idx) {
                move_tile_window(d_dram_windows[idx], {step.at(number<0>{}), step.at(number<1>{})});
            });
        }
    }

    // TODO: Check if there would be nicer ways to overload rather than with EmptyScale or nullptr_t
    struct EmptyScale
    {
    };

    template <typename, typename = void>
    struct ScaleDataType
    {
        using DataType = float;
    };

    template <typename T>
    struct ScaleDataType<T, std::void_t<typename T::DataType>>
    {
        using DataType = typename T::DataType;
    };

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM                         = EmptyScale,
              typename ScaleN                         = EmptyScale,
              int EnablePermuateN_                    = TiledMMAPermuteN,
              std::enable_if_t<EnablePermuateN_, int> = 0>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* /*p_smem*/,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        static constexpr int RowsPerLane = CWarpTensor::get_thread_buffer_size();

        static_assert(MPerXdl % RowsPerLane == 0,
                      "CShuffle (permuteN): MPerXdl must be divisible by per-lane row count.");
        constexpr int kM0 = MWave;
        constexpr int kM2 = RowsPerLane;
        constexpr int kM1 = MPerXdl / kM2;

        constexpr int kN0 = NWave;
        constexpr int kN1 = NPerXdl;
        constexpr int kN2 = NRepeat;

        using IntrThreadShuffleEncode =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<kM0, kM1, kM2>, sequence<kN0, kN1, kN2>>,
                                       tuple<sequence<1, 2>, sequence<1, 2>>,
                                       tuple<sequence<0, 0>, sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<2, 2>>;
        constexpr auto dram_tile_distribution =
            make_static_tile_distribution(IntrThreadShuffleEncode{});

        auto d_dram_windows = generate_tuple(
            [&](auto idx) {
                return make_tile_window(ds_dram_windows[idx], dram_tile_distribution);
            },
            number<NumDTensor>{});

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        auto shuffle_acc  = make_static_distributed_tensor<AccDataType>(dram_tile_distribution);
        auto c_out_tensor = make_static_distributed_tensor<ODataType>(dram_tile_distribution);

        // Optional scales (must share the same distribution to match per-thread indexing)
        constexpr bool has_scales =
            !std::is_same<ScaleM, EmptyScale>::value && !std::is_same<ScaleN, EmptyScale>::value;
        constexpr bool has_scalar_scales =
            std::is_same_v<ScaleM, AccDataType> && std::is_same_v<ScaleN, AccDataType>;

        // Tiles to hold row/col scales when present
        using SMType = typename ScaleDataType<ScaleM>::DataType;
        using SNType = typename ScaleDataType<ScaleN>::DataType;

        auto sm_tile = make_static_distributed_tensor<SMType>(dram_tile_distribution);
        auto sn_tile = make_static_distributed_tensor<SNType>(dram_tile_distribution);

        // Build windows only if non-scalar scales are provided
        auto scale_m_window = [&]() {
            if constexpr(has_scales && !has_scalar_scales)
            {
                return make_tile_window(scale_m, dram_tile_distribution);
            }
            else
            {
                return EmptyScale{};
            }
        }();
        auto scale_n_window = [&]() {
            if constexpr(has_scales && !has_scalar_scales)
            {
                return make_tile_window(scale_n, dram_tile_distribution);
            }
            else
            {
                return EmptyScale{};
            }
        }();

        static_for<0, MRepeat, 1>{}([&](auto mIter) {
            // Slice accumulators for this M repeat into the permuted layout
            shuffle_acc.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
                merge_sequences(sequence<mIter, 0>{}, c_warp_y_index_zeros),
                merge_sequences(sequence<1, NRepeat>{}, c_warp_y_lengths));

            // If non-scalar scales provided, load them with identical distribution
            if constexpr(has_scales && !has_scalar_scales)
            {
                sm_tile = load_tile(scale_m_window); // row scales in permuted layout
                sn_tile = load_tile(scale_n_window); // col scales in permuted layout
            }

            // Pack 4 “rows per lane” as you already do
            static_for<0, NRepeat, 1>{}([&](auto n_idx) {
                // source indices in shuffle_acc: (n_idx * product(Y) + row)
                const index_t plane = c_warp_y_lengths.product();

                // local lambda to fuse scale (if present) and convert
                static_for<0, kM2, 1>{}([&](auto m_lane) {
                    const int src = n_idx * plane + m_lane;   // source row in this N-plane
                    const int dst = n_idx + m_lane * NRepeat; // permuted N layout in output
                    AccDataType v = shuffle_acc.get_thread_buffer()[src];

                    if constexpr(has_scalar_scales)
                    {
                        v = static_cast<AccDataType>(v * scale_m * scale_n);
                    }
                    else if constexpr(has_scales && !has_scalar_scales)
                    {
                        const auto sm = static_cast<float>(sm_tile.get_thread_buffer()[dst]);
                        const auto sn = static_cast<float>(sn_tile.get_thread_buffer()[dst]);
                        v             = static_cast<AccDataType>(v * sm * sn);
                    }

                    c_out_tensor.get_thread_buffer()[dst] = type_convert<ODataType>(v);
                });
            });

            // store/update
            if constexpr(MemoryOperation == memory_operation_enum::set)
            {
                store_tile(out_dram_window, c_out_tensor);
            }
            else
            {
                update_tile(out_dram_window, c_out_tensor);
            }

            // advance output (and any D-tensors) by one MPerXdl*MWave chunk
            move_tile_window(out_dram_window, {number<MPerXdl * MWave>{}, number<0>{}});
            static_for<0, NumDTensor, 1>{}([&](auto idx) {
                move_tile_window(d_dram_windows[idx], {number<MPerXdl * MWave>{}, number<0>{}});
            });
        });
    }

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM                          = EmptyScale,
              typename ScaleN                          = EmptyScale,
              int EnablePermuateN_                     = TiledMMAPermuteN,
              std::enable_if_t<!EnablePermuateN_, int> = 0>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        constexpr auto LdsTileDistr = make_static_tile_distribution(MakeLdsDistributionEncode());

        auto lds_tile = make_static_distributed_tensor<AccDataType>(LdsTileDistr);

        constexpr auto lds_block_desc = MakeLdsBlockDescriptor<Problem>();
        auto o_lds_block              = make_tensor_view<address_space_enum::lds>(
            static_cast<ODataType*>(p_smem), lds_block_desc);

        auto in_lds_window = make_tile_window(
            o_lds_block,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0},
            LdsTileDistr);

        auto out_lds_window = make_tile_window(
            o_lds_block,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0});

        constexpr index_t num_access = SFC::get_num_of_access();

        static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                      "Currently, the CShuffle Epilogue only supports the Row Major Output layout");

        using TileEncodingPattern =
            tile_distribution_encoding_pattern_2d<kBlockSize,
                                                  MPerIterationShuffle,
                                                  NPerIterationShuffle,
                                                  GetVectorSizeC(),
                                                  tile_distribution_pattern::thread_raked,
                                                  Problem::kNumWaveGroups>;
        constexpr auto dram_tile_distribution =
            TileEncodingPattern::make_2d_static_tile_distribution();

        auto d_dram_windows = generate_tuple(
            [&](auto idx) {
                return make_tile_window(ds_dram_windows[idx], dram_tile_distribution);
            },
            number<NumDTensor>{});

        constexpr bool has_scales =
            !std::is_same_v<ScaleM, EmptyScale> && !std::is_same_v<ScaleN, EmptyScale>;
        constexpr bool has_scalar_scales =
            std::is_same_v<ScaleM, AccDataType> && std::is_same_v<ScaleN, AccDataType>;
        auto scale_m_window = [&]() {
            if constexpr(has_scalar_scales)
            {
                return scale_m;
            }
            else if constexpr(has_scales)
            {
                return make_tile_window(scale_m, lds_tile.get_tile_distribution());
            }
            else
            {
                return EmptyScale{};
            }
        }();
        auto scale_n_window = [&]() {
            if constexpr(has_scalar_scales)
            {
                return scale_n;
            }
            else if constexpr(has_scales)
            {
                return make_tile_window(scale_n, lds_tile.get_tile_distribution());
            }
            else
            {
                return EmptyScale{};
            }
        }();

        static_for<0, num_access, 1>{}([&](auto iAccess) {
            block_sync_lds();
            slice_acc_tile<iAccess>(o_acc_tile, lds_tile);

            if constexpr(has_scales)
            {
                scale_tile<iAccess>(lds_tile, scale_m_window, scale_n_window);
            }

            cast_lds_tile(lds_tile, in_lds_window);
            block_sync_lds();

            auto c_out_tensor = load_tile(make_tile_window(out_lds_window, dram_tile_distribution));

            apply_d_tensors(d_dram_windows, c_out_tensor);
            store_to_dram(out_dram_window, c_out_tensor);
            move_windows<iAccess>(out_dram_window, d_dram_windows);
        });
    }
};
} // namespace ck_tile
