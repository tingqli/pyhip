#include "ck_tile/core/config.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include <ck_tile/core/numeric/integral_constant.hpp>
#include <ck_tile/core/arch/arch.hpp>
#include <ck_tile/core/container/tuple.hpp>
#include <ck_tile/core/tensor/tensor_view.hpp>

namespace ck_tile {
#define TILE_M 256
#define TILE_N 256
#define TILE_K 32
#define WAVE_M 2   // 2 waves in M dimension, for gemm loop
#define WAVE_N 2   // 2 waves in N dimension

#define TILE_M_WAVE (TILE_M / WAVE_M)
#define TILE_N_WAVE (TILE_N / WAVE_N)

__device__ void print_t(auto&& tensor, int len=16)
{
    if (get_thread_id() == 0)
    {
        auto& tb = tensor.get_thread_buffer();
        index_t a_sz = tensor.get_thread_buffer_size();
        printf("thread buffer size=%d\n", int(a_sz));
        for(int j = 0; j < (a_sz < len ? a_sz : len); ++j)
        {
            float v = type_convert<float>(tb.get(j));
            printf("  [%d]=%.2f\n", j, v);
        }
    }
}

__device__ void gemm_tile_256x256x32(const fp16_t* A, const fp16_t* B, float* C, index_t M, index_t N, index_t K) {
    auto a_view = make_naive_tensor_view<address_space_enum::global>(A, make_tuple(M, K), make_tuple(K, 1), number<8>(), number<1>());
    auto b_view = make_naive_tensor_view<address_space_enum::global>(B, make_tuple(N, K), make_tuple(K, 1), number<8>(), number<1>());
    auto c_view = make_naive_tensor_view<address_space_enum::global>(C, make_tuple(M, N), make_tuple(N, 1), number<8>(), number<1>());
    __shared__ fp16_t share_buf[TILE_M * TILE_K + TILE_N * TILE_K];
    // auto a_ds_view = make_naive_tensor_view<address_space_enum::lds>(share_buf, make_tuple(TILE_M, TILE_K), make_tuple(TILE_K, 1));
    constexpr auto rows_per_banks = 32 * 4 / (TILE_K * sizeof(fp16_t)) >= 1 ? 32 * 4 / (TILE_K * sizeof(fp16_t)) : 1;
    auto a_ds_desc0 = make_naive_tensor_descriptor(make_tuple(TILE_M / rows_per_banks, rows_per_banks * TILE_K / 8, 8), make_tuple(TILE_K * rows_per_banks, 8, 1));
    auto a_ds_desc1 = transform_tensor_descriptor(a_ds_desc0, 
        // k = k^(m % K)
        make_tuple(make_xor_transform(make_tuple(TILE_M / rows_per_banks, rows_per_banks * TILE_K / 8)), make_pass_through_transform(number<8>())),
        make_tuple(sequence<0, 1>(), sequence<2>()),
        make_tuple(sequence<0, 1>(), sequence<2>()));
    auto a_ds_desc2 = transform_tensor_descriptor(a_ds_desc1,
        make_tuple(make_pass_through_transform(number<TILE_M / rows_per_banks>()), make_unmerge_transform(make_tuple(rows_per_banks, TILE_K / 8)), make_pass_through_transform(number<8>())),
        make_tuple(sequence<0>(), sequence<1>(), sequence<2>()),
        make_tuple(sequence<0>(), sequence<1, 2>(), sequence<3>()));
    auto a_ds_desc = transform_tensor_descriptor(a_ds_desc2, 
        make_tuple(make_merge_transform(make_tuple(TILE_M / rows_per_banks, rows_per_banks)), make_merge_transform(make_tuple(TILE_K / 8, 8))),
        make_tuple(sequence<0, 1>(), sequence<2, 3>()),
        make_tuple(sequence<0>(), sequence<1>()));
    auto a_ds_view = make_tensor_view<address_space_enum::lds>(share_buf, a_ds_desc);
    // auto b_ds_view = make_naive_tensor_view<address_space_enum::lds>(share_buf + TILE_M * TILE_K, make_tuple(TILE_N, TILE_K), make_tuple(TILE_K, 1));
    auto b_ds_desc0 = make_naive_tensor_descriptor(make_tuple(TILE_N / rows_per_banks, rows_per_banks * TILE_K / 8, 8), make_tuple(TILE_K * rows_per_banks, 8, 1));
    auto b_ds_desc1 = transform_tensor_descriptor(b_ds_desc0, 
        make_tuple(make_xor_transform(make_tuple(TILE_N / rows_per_banks, rows_per_banks * TILE_K / 8)), make_pass_through_transform(number<8>())),
        make_tuple(sequence<0, 1>(), sequence<2>()),
        make_tuple(sequence<0, 1>(), sequence<2>()));
    auto b_ds_desc2 = transform_tensor_descriptor(b_ds_desc1,
        make_tuple(make_pass_through_transform(number<TILE_N / rows_per_banks>()), make_unmerge_transform(make_tuple(rows_per_banks, TILE_K / 8)), make_pass_through_transform(number<8>())),
        make_tuple(sequence<0>(), sequence<1>(), sequence<2>()),
        make_tuple(sequence<0>(), sequence<1, 2>(), sequence<3>()));
    auto b_ds_desc = transform_tensor_descriptor(b_ds_desc2, 
        make_tuple(make_merge_transform(make_tuple(TILE_M / rows_per_banks, rows_per_banks)), make_merge_transform(make_tuple(TILE_K / 8, 8))),
        make_tuple(sequence<0, 1>(), sequence<2, 3>()),
        make_tuple(sequence<0>(), sequence<1>()));
    auto b_ds_view = make_tensor_view<address_space_enum::lds>(share_buf + TILE_M * TILE_K, b_ds_desc);

    auto n_blocks = N / TILE_N;
    auto k_blocks = K / TILE_K;
    auto block_idx = get_block_id();
    auto m_block_idx = block_idx / n_blocks;
    auto n_block_idx = block_idx % n_blocks;
    auto wave_idx = get_warp_id();
    auto m_wave_idx = wave_idx / WAVE_N;
    auto n_wave_idx = wave_idx % WAVE_N;

    auto a_dram_win = make_tile_window(a_view, make_tuple(number<TILE_M>(), number<TILE_K>()), make_multi_index(m_block_idx * TILE_M, 0),
        make_static_tile_distribution(tile_distribution_encoding<sequence<>,
            tuple<sequence<4, 4, 16>, sequence<4, 8>>,
            // wave distriutes in the outer most in M dimension, the distribution is [16, 4(x8)] for each wave
            tuple<sequence<1>, sequence<1, 2>>,
            tuple<sequence<0>, sequence<2, 0>>,
            // repeation is second 4 in M dimension
            sequence<1, 2>,
            sequence<1, 1>>()));
    auto b_dram_win = make_tile_window(b_view, make_tuple(number<TILE_N>(), number<TILE_K>()), make_multi_index(n_block_idx * TILE_N, 0),
        make_static_tile_distribution(tile_distribution_encoding<sequence<>,
            tuple<sequence<4, 4, 16>, sequence<4, 8>>,
            tuple<sequence<1>, sequence<1, 2>>,
            tuple<sequence<0>, sequence<2, 0>>,
            sequence<1, 2>,
            sequence<1, 1>>()));
    auto a_ds_write_win = make_tile_window(a_ds_view, make_tuple(number<TILE_M>(), number<TILE_K>()), make_multi_index(0, 0),
        a_dram_win.get_tile_distribution());
    auto b_ds_write_win = make_tile_window(b_ds_view, make_tuple(number<TILE_N>(), number<TILE_K>()), make_multi_index(0, 0),
        b_dram_win.get_tile_distribution());

    using Gemm = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution<>;
    auto a_dist = detail::make_embed_tile_distribution_encoding(
        tile_distribution_encoding<sequence<WAVE_N>, 
            // M=2x4xM of Gemm N=2xK of Gemm
            tuple<sequence<TILE_M / TILE_M_WAVE, TILE_M_WAVE / Gemm::kM>, sequence<TILE_K / Gemm::kK>>,
            // wave parallel(M, N)
            tuple<sequence<1, 0>>,
            tuple<sequence<0, 0>>,
            // repeatition
            sequence<1, 2>,
            sequence<1, 0>>(),
        Gemm::AWarpDstrEncoding{});
    auto b_dist = detail::make_embed_tile_distribution_encoding(
        tile_distribution_encoding<sequence<WAVE_M>, 
            // M=2x4xM of Gemm N=2xK of Gemm
            tuple<sequence<TILE_N / TILE_N_WAVE, TILE_N_WAVE / Gemm::kN>, sequence<TILE_K / Gemm::kK>>,
            // wave parallel(M, N)?
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 0>>,
            // repeatition
            sequence<1, 2>,
            sequence<1, 0>>(),
        Gemm::BWarpDstrEncoding{});
    auto c_dist = detail::make_embed_tile_distribution_encoding(
        tile_distribution_encoding<sequence<>, 
            tuple<sequence<2, 4>, sequence<2, 4>>, 
            // wave parallel: N first, then M(from right to left)
            tuple<sequence<1, 2>>,
            tuple<sequence<0, 0>>, 
            // repeatition: N first, then M
            sequence<1, 2>, 
            sequence<1, 1>>(),
        Gemm::CWarpDstrEncoding{});
    auto a_ds_read_win = make_tile_window(a_ds_view, make_tuple(number<TILE_M>(), number<TILE_K>()), make_multi_index(0, 0),
        make_static_tile_distribution(a_dist));
    auto b_ds_read_win = make_tile_window(b_ds_view, make_tuple(number<TILE_N>(), number<TILE_K>()), make_multi_index(0, 0),
        make_static_tile_distribution(b_dist));

    auto c_tensor = make_static_distributed_tensor<float>(make_static_tile_distribution(c_dist));
    decltype(a_ds_read_win.load()) a_tensor0, a_tensor1;
    decltype(b_ds_read_win.load()) b_tensor0, b_tensor1;
    clear_tile(c_tensor);
    constexpr auto a_warp_y_lengths =
        to_sequence(Gemm::AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto b_warp_y_lengths =
        to_sequence(Gemm::BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_lengths =
        to_sequence(Gemm::CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<Gemm::AWarpDstr::NDimY, 0>{};
    constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<Gemm::BWarpDstr::NDimY, 0>{};
    constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<Gemm::CWarpDstr::NDimY, 0>{};

    auto gemm = [&] (auto&& a_tensor, auto&& b_tensor) {
        static_for<0, TILE_K / Gemm::kK, 1>{}([&](auto k) {
        static_for<0, TILE_M_WAVE / Gemm::kM, 1>{} ([&] (auto m) {
            Gemm::AWarpTensor a;
            a.get_thread_buffer() = a_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<m, k>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
            static_for<0, TILE_N_WAVE / Gemm::kN, 1>{} ([&] (auto n) {
                Gemm::BWarpTensor b;
                b.get_thread_buffer() = b_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<n, k>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
                Gemm::CWarpTensor c;
                c.get_thread_buffer() = c_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<m, n>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                Gemm{}(c, a, b);
                c_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<m, n>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c.get_thread_buffer());
            });
        });
        });
    };

    // dram(block0)->reg
    auto a_dram_t = a_dram_win.load();
    auto b_dram_t = b_dram_win.load();
    move_tile_window(a_dram_win, {0, TILE_K});
    move_tile_window(b_dram_win, {0, TILE_K});
    // reg(block0)->ds
    a_ds_write_win.store(a_dram_t);
    b_ds_write_win.store(b_dram_t);

    // dram(block1)->reg
    a_dram_t = a_dram_win.load();
    b_dram_t = b_dram_win.load();
    move_tile_window(a_dram_win, {0, TILE_K});
    move_tile_window(b_dram_win, {0, TILE_K});

    block_sync_lds();
    // ds->reg0(block0)
    a_tensor0 = a_ds_read_win.load();
    b_tensor0 = b_ds_read_win.load();
    block_sync_lds(); //?
    // reg(block2)->ds
    a_ds_write_win.store(a_dram_t);
    b_ds_write_win.store(b_dram_t);

    // dram(block2)->reg
    a_dram_t = a_dram_win.load();
    b_dram_t = b_dram_win.load();
    move_tile_window(a_dram_win, {0, TILE_K});
    move_tile_window(b_dram_win, {0, TILE_K});

    while (k_blocks > 3) {
        block_sync_lds();
        // ds->reg1(block1)
        a_tensor1 = a_ds_read_win.load();
        b_tensor1 = b_ds_read_win.load();
        // matmul(block0)
        gemm(a_tensor0, b_tensor0);
        block_sync_lds();

        // reg(block2)->ds
        a_ds_write_win.store(a_dram_t);
        b_ds_write_win.store(b_dram_t);

        // dram(block3)->reg
        a_dram_t = a_dram_win.load();
        b_dram_t = b_dram_win.load();
        move_tile_window(a_dram_win, {0, TILE_K});
        move_tile_window(b_dram_win, {0, TILE_K});

        block_sync_lds();
        // ds->reg0(block2)
        a_tensor0 = a_ds_read_win.load();
        b_tensor0 = b_ds_read_win.load();
        // matmul(block1)
        gemm(a_tensor1, b_tensor1);
        block_sync_lds();

        // reg(block3)->ds
        a_ds_write_win.store(a_dram_t);
        b_ds_write_win.store(b_dram_t);

        // dram(block4)->reg
        a_dram_t = a_dram_win.load();
        b_dram_t = b_dram_win.load();
        move_tile_window(a_dram_win, {0, TILE_K});
        move_tile_window(b_dram_win, {0, TILE_K});

        k_blocks -= 2;
    }
    // tail
    {
        // tail - 3
        block_sync_lds();
        // ds->reg1(block3)
        a_tensor1 = a_ds_read_win.load();
        b_tensor1 = b_ds_read_win.load();
        // matmul(block2)
        gemm(a_tensor0, b_tensor0);
        block_sync_lds();

        // reg(block4)->ds
        a_ds_write_win.store(a_dram_t);
        b_ds_write_win.store(b_dram_t);

        // tail - 2
        block_sync_lds();
        // ds->reg0(block4)
        a_tensor0 = a_ds_read_win.load();
        b_tensor0 = b_ds_read_win.load();
        // matmul(block3)
        gemm(a_tensor1, b_tensor1);

        // tail - 1
        if (k_blocks > 2)
            gemm(a_tensor0, b_tensor0);
    }
    auto c_dram_win = make_tile_window(c_view, make_tuple(number<TILE_M>(), number<TILE_N>()), make_multi_index(m_block_idx * TILE_M, n_block_idx * TILE_N),
        make_static_tile_distribution(c_dist));
    
    store_tile(c_dram_win, c_tensor);
}

}

__global__ void __launch_bounds__(256, 1) gemm_tile_256x256x32(const ck_tile::fp16_t* A, const ck_tile::fp16_t* B, float* C, int M, int N, int K) {
    ck_tile::gemm_tile_256x256x32(A, B, C, M, N, K);
}
