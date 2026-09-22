import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_tb_swizzle():
    @pyhip.jit()
    def tb_swizzle(J, wg_M, wg_N, N, M01, GroupNum, block_1d_id:"int", M:"int", output:"int*"):
        M0 = J.gpr(J.div_up(M, wg_M))
        N0 = J.div_up(N, wg_N)
        group_size    = J.div_up(M0 * N0, GroupNum)
        big_group_num = J.gpr(GroupNum - (group_size * GroupNum - M0 * N0))
        group_id_y    = J.gpr(block_1d_id // GroupNum)
        group_id_x    = J.gpr(block_1d_id - group_id_y * GroupNum) 

        remap_block_1d_id = J.gpr(group_id_x * group_size + group_id_y)

        with J.If(group_id_x > big_group_num):
            remap_block_1d_id[0] = remap_block_1d_id[0] + (big_group_num - group_id_x)

        idx_M0 = J.gpr(remap_block_1d_id // N0)
        idx_N0 = J.gpr(remap_block_1d_id - idx_M0 * N0)

        M0_tmp     = J.gpr(M0 // M01)
        M0_mod_M01 = J.gpr(M0 - M0_tmp * M01)

        M01_adapt = J.gpr("su32")
        J.SetMask("scc", idx_M0 < M0 - M0_mod_M01)
        J.s_cselect_b32(M01_adapt, M01, M0_mod_M01)
        
        # M01_adapt = (idx_M0 < M0 - M0_mod_M01) ? M01 : M0_mod_M01;

        idx_M00          = J.gpr(idx_M0 // M01)
        idx_M01          = J.gpr(idx_M0 - idx_M00 * M01)
        idx_N0_M01_local = J.gpr(idx_N0 + idx_M01 * N0)

        N_out           = J.gpr(idx_N0_M01_local // M01_adapt)
        idx_loc_mod_M01 = J.gpr(idx_N0_M01_local - N_out * M01_adapt)

        M_out = J.gpr(idx_loc_mod_M01 + idx_M00 * M01)
        J.s_store_dword(M_out, output, 0, mod="glc")
        J.s_store_dword(N_out, output, 4, mod="glc")

    wg_M = 128
    wg_N = 128
    M = 5*wg_M
    N = 4*wg_N
    M01 = 2
    GroupNum = 8

    output = torch.zeros(2, dtype=torch.int32)
    expect = [
        [0,0],[1,1],[0,3],[3,0],
        [2,2],[2,3],[4,0],[4,2],
        [1,0],[0,2],[1,3],[2,1],
        [3,2],[3,3],[4,1],[4,3],
        [0,1],[1,2],[2,0],[3,1]
    ]
    for block_1d_id in range(5*4):
        tb_swizzle([1],[64],wg_M, wg_N, N, M01, GroupNum, block_1d_id, M, output.data_ptr())
        assert output.tolist() == expect[block_1d_id], f"{output.tolist()} != {expect[block_1d_id]}"

if __name__ == "__main__":
    test_tb_swizzle()