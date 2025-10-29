// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/ops/common/load_interleaved_pk_type.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/elementwise.hpp"

namespace ck_tile {

template <typename Problem>
struct BlockGemmBQuantBase
{
    using BQDataType      = remove_cvref_t<typename Problem::BQDataType>;
    using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;

    template <typename T>
    CK_TILE_DEVICE static float cvt_scale_to_fp32(T scale)
    {
        float scale_reg_f = 0.f;
        if constexpr(std::is_same_v<BQDataType, ck_tile::fp8_t>)
        {
            scale_reg_f =
                ck_tile::element_wise::amd_assembly_fp8_to_fp32(static_cast<uint32_t>(scale));
        }
        else if constexpr(std::is_same_v<BQDataType, ck_tile::bf8_t>)
        {
            scale_reg_f =
                ck_tile::element_wise::amd_assembly_bf8_to_fp32(static_cast<uint32_t>(scale));
        }
        else if constexpr(std::is_same_v<BQDataType, float>)
        {
            scale_reg_f = ck_tile::bit_cast<float>(scale);
        }
        else
        {
            static_assert(false, "BQDataType must be float, fp8_t or bf8_t.");
        }
        return scale_reg_f;
    }
};

// A is block window on shared memory
// BQ (scale tensor) is block distributed tensor.
// Consecutive kQuantGroupSize elements of B are quantized with a separate scale.
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_,
          typename Policy_     = BlockGemmASmemBSmemCRegV1DefaultPolicy,
          index_t UnaryOpSize_ = 8>
struct BQuantBlockUniversalGemmAsBsCr : public BlockGemmBQuantBase<Problem_>
{
    private:
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem         = remove_cvref_t<PipelineProblem_>;
        using Policy          = remove_cvref_t<GemmPolicy_>;
        using ADataType       = remove_cvref_t<typename Problem::ADataType>;
        using BDataType       = remove_cvref_t<typename Problem::BDataType>;
        using BQDataType      = remove_cvref_t<typename Problem::BQDataType>;
        using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
        using CDataType       = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;

        static constexpr index_t kQuantGroupSize = Problem::kQuantGroupSize;
        static constexpr index_t kBlockSize      = Problem::kBlockSize;
        static constexpr auto Scheduler          = Problem::Scheduler;

        // Threadblock GEMM tile size
        static constexpr index_t MPerBlock  = BlockGemmShape::kM;
        static constexpr index_t NPerBlock  = BlockGemmShape::kN;
        static constexpr index_t KPerBlock  = BlockGemmShape::kK;
        static constexpr index_t BQPerBlock = KPerBlock / kQuantGroupSize;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm               = remove_cvref_t<decltype(config.template at<0>())>;

        // number of warps along M and N for threadblock's GEMM problem size
        static constexpr index_t MWarp = config.template at<1>();
        static constexpr index_t NWarp = config.template at<2>();

        using I0 = number<0>;
        using I1 = number<1>;

        static_assert(MWarp == BlockGemmShape::BlockWarps::at(I0{}),
                      "Error! WarpGemm's MWarp is not consistent with BlockGemmShape!");
        static_assert(NWarp == BlockGemmShape::BlockWarps::at(I1{}),
                      "Error! WarpGemm's NWarp is not consistent with BlockGemmShape!");
        static_assert(WarpGemm::kM == BlockGemmShape::WarpTile::at(I0{}),
                      "Error! WarpGemm's M is not consistent with BlockGemmShape!");
        static_assert(WarpGemm::kN == BlockGemmShape::WarpTile::at(I1{}),
                      "Error! WarpGemm's N is not consistent with BlockGemmShape!");

        static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

        static constexpr index_t QScalesPerBlockRow =
            (KPerBlock + kQuantGroupSize - 1) / kQuantGroupSize;
        static constexpr index_t QScalesPerWarpGemmRow =
            (WarpGemm::kK + kQuantGroupSize - 1) / kQuantGroupSize;

        static constexpr index_t KIterPerQScale = KIterPerWarp / QScalesPerBlockRow;

        static_assert(kQuantGroupSize % WarpGemm::kK == 0,
                      "Error! WarpGemm::kK should be a multiple of kQuantGroupSize");
        static_assert(QScalesPerWarpGemmRow == 1,
                      "Error! kQuantGroupSize shouldn't be smaller than WarpGemm::kK");
        static_assert(KIterPerWarp % QScalesPerBlockRow == 0,
                      "Error! KItersPerWarp should be a multiple of QscalesPerBlockRow");

        static_assert(KPerBlock / kQuantGroupSize > 0,
                      "Error! Each row of blockgemm should have a separate scale");

        static_assert(MIterPerWarp * MWarp * WarpGemm::kM == MPerBlock,
                      "Error! Warps should cover all Block tile!");
        static_assert(NIterPerWarp * NWarp * WarpGemm::kN == NPerBlock,
                      "Error! Warps should cover all Block tile!");

        // Currently tested combinations (A, B, BQ)
        // 1. fp8, fp8, fp32 -> f32
        // 2. bf8, bf8, fp32 -> f32
        // 3. i4,  fp8, (fp8/fp32) -> f32
        // 4. i4,  bf8, (fp8/fp32) -> f32
        static_assert((std::is_same_v<ADataType, fp8_t> || std::is_same_v<ADataType, bf8_t>) &&
                      (std::is_same_v<BDataType, fp8_t> || std::is_same_v<BDataType, bf8_t> ||
                       std::is_same_v<BDataType, ck_tile::pk_int4_t>) &&
                      (std::is_same_v<BQDataType, float> ||
                       std::is_same_v<BQDataType, ck_tile::fp8_t> ||
                       std::is_same_v<BQDataType, ck_tile::bf8_t>) &&
                      (std::is_same_v<ComputeDataType, fp8_t> ||
                       std::is_same_v<ComputeDataType, bf8_t>) &&
                      std::is_same_v<CDataType, fp32_t>);

        static constexpr index_t InterWaveSchedulingMacClusters = 1;

        static constexpr index_t KPack      = WarpGemm::kKPerThread;
        static constexpr index_t KPerThread = KIterPerWarp * WarpGemm::kKPerThread;
    };

    public:
    using Traits = GemmTraits_<Problem_, Policy_>;

    using ADataType       = remove_cvref_t<typename Traits::ADataType>;
    using BDataType       = remove_cvref_t<typename Traits::BDataType>;
    using BQDataType      = remove_cvref_t<typename Traits::BQDataType>;
    using ComputeDataType = remove_cvref_t<typename Traits::ComputeDataType>;
    using CDataType       = remove_cvref_t<typename Traits::CDataType>;

    using Base = BlockGemmBQuantBase<Problem_>;

    using Loader   = remove_cvref_t<InterleavedPKTypeLoader<ComputeDataType, UnaryOpSize_>>;
    using WarpGemm = remove_cvref_t<typename Traits::WarpGemm>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;

    static constexpr index_t MWarp = Traits::MWarp;
    static constexpr index_t NWarp = Traits::NWarp;

    static constexpr auto Scheduler = Traits::Scheduler;

    using AWarpDstr = typename WarpGemm::AWarpDstr;
    using BWarpDstr = typename WarpGemm::BWarpDstr;
    using CWarpDstr = typename WarpGemm::CWarpDstr;

    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;

    static_assert(std::is_same_v<typename WarpGemm::CDataType, float>);

    static constexpr auto a_warp_y_lengths =
        to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto b_warp_y_lengths =
        to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

    static constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    static constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
    static constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    using I0 = number<0>;
    using I1 = number<1>;

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        constexpr index_t KPerThread     = Traits::KPerThread;
        constexpr index_t NumMacClusters = Traits::InterWaveSchedulingMacClusters;

        constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);

        constexpr index_t KIterInterwave = KPerInnerLoop / WarpGemm::kKPerThread;

        using KIterSeq = std::conditional_t<Scheduler == GemmPipelineScheduler::Interwave,
                                            sequence<KIterInterwave>,
                                            sequence<KIterPerWarp>>;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, KIterSeq>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        return a_block_dstr_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        constexpr index_t KPerThread     = Traits::KPerThread;
        constexpr index_t NumMacClusters = Traits::InterWaveSchedulingMacClusters;
        constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        constexpr index_t KIterInterwave = KPerInnerLoop / WarpGemm::kKPerThread;

        using KIterSeq = std::conditional_t<Scheduler == GemmPipelineScheduler::Interwave,
                                            sequence<KIterInterwave>,
                                            sequence<KIterPerWarp>>;

        constexpr auto b_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, KIterSeq>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        return b_block_dstr_encode;
    }

    private:
    template <GemmPipelineScheduler Scheduler, typename GemmTraits>
    struct BlockGemmImpl
    {
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Intrawave, GemmTraits>
    {
        static constexpr auto ALdsTileDistr =
            decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
        static constexpr auto BLdsTileDistr =
            decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};

        using ALdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(BLdsTileDistr));

        ALdsTile a_warp_tile_;
        BLdsTile b_warp_tile_;

        template <typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                          const BSmemBlockWindow& b_block_window)
        {
            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                static_assert(std::is_same_v<ComputeDataType, fp8_t> ||
                              std::is_same_v<ComputeDataType, bf8_t>);
                Loader::load_interleaved_pk_type(a_warp_tile_, a_block_window);
            }
            else
            {
                load_tile(a_warp_tile_, a_block_window);
            }
            if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            {
                static_assert(std::is_same_v<ComputeDataType, fp8_t> ||
                              std::is_same_v<ComputeDataType, bf8_t>);
                Loader::load_interleaved_pk_type(b_warp_tile_, b_block_window);
            }
            else
            {
                load_tile(b_warp_tile_, b_block_window);
            }
        }

        // C += A * B
        template <typename CBlockTensor,
                  typename BQBlockTensor,
                  typename ASmemBlockWindow,
                  typename BSmemBlockWindow>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       BQBlockTensor& bq_block_tensor,
                                       [[maybe_unused]] ASmemBlockWindow& a_block_window,
                                       [[maybe_unused]] BSmemBlockWindow& b_block_window)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as corresponding "
                          "C block tensor data type!");

            // hot loop:
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    CWarpTensor c_warp_tensor;

                    static_for<0, Traits::QScalesPerBlockRow, 1>{}([&](auto kQScale) {
                        static_for<0, Traits::KIterPerQScale, 1>{}([&](auto kIterInQScale) {
                            constexpr auto kIter = kQScale * Traits::KIterPerQScale + kIterInQScale;

                            AWarpTensor a_warp_tensor;
                            a_warp_tensor.get_thread_buffer() =
                                a_warp_tile_.get_y_sliced_thread_data(
                                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                            BWarpTensor b_warp_tensor;
                            b_warp_tensor.get_thread_buffer() =
                                b_warp_tile_.get_y_sliced_thread_data(
                                    merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                            if constexpr(kIterInQScale == 0)
                            {
                                c_warp_tensor = WarpGemm{}(a_warp_tensor, b_warp_tensor);
                            }
                            else
                            {
                                WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                            }
                        });

                        // Need to multiply bquant with accumulated C
                        //
                        // The accumulated C tile has the standard distribution. For example
                        // lane 0 holds elements [0,0], [1,0], [2,0], [3,0], [8,0], [9,0],
                        // [10,0], [11,0], [16,0], [17,0], [18,0], [19,0], [24,0], [25,0],
                        // [26,0], [27,0].
                        //
                        // These elements are in different rows, need to get the scale value
                        // for the corresponding row.
                        // Based on bquant's tile distribution, it can be inferred which
                        // lane holds the relevant scale. For example, the scales corresponding
                        // to the 16 elements held by lane 0 are held by lanes 0, 1, 2, 3, 8, 9,
                        // 10, 11, 16, 17, 18, 19, 24, 25, 26, 27 respectively.
                        //
                        // These scales can be obtained using __builtin_amdgcn_ds_bpermute.

                        constexpr index_t reg_offset = nIter * Traits::BQPerBlock + kQScale;

                        constexpr auto tbuf_offset =
                            number<typename CBlockTensor::ThreadTensorDesc{}.calculate_offset(
                                       merge_sequences(sequence<mIter, nIter>{},
                                                       c_warp_y_index_zeros)) /
                                   CBlockTensor::PackedSize>{};

                        auto& scale_reg   = bq_block_tensor.get_thread_buffer()[reg_offset];
                        float scale_reg_f = Base::cvt_scale_to_fp32(scale_reg);
                        static_for<0, WarpGemm::kM / 2, 1>{}([&](auto c_row) {
                            c_block_tensor.get_thread_buffer()[tbuf_offset + c_row] +=
                                (c_warp_tensor.get_thread_buffer()[c_row] * scale_reg_f);
                        });
                    });
                });
            });
        }
    };

    public:
    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);

        return c_block_tensor;
    }

    template <typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                      const BSmemBlockWindow& b_block_window)
    {
        block_gemm_impl_.LocalPrefetch(a_block_window, b_block_window);
    }

    // C += A * B
    template <typename CBlockTensor,
              typename BQBlockTensor,
              typename ASmemBlockWindow,
              typename BSmemBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   BQBlockTensor& bq_block_tensor,
                                   const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window)
    {
        block_gemm_impl_(c_block_tensor, bq_block_tensor, a_block_window, b_block_window);
    }

    private:
    BlockGemmImpl<Scheduler, Traits> block_gemm_impl_{};
};

} // namespace ck_tile
