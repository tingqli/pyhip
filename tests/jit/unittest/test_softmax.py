import pyhip
import torch

torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device("cuda")
torch.manual_seed(0)


class Buffer:
    def __init__(self, J):
        self.J = J
        self.desc = J.new_gpr("s", 4, align=4)
        self.base = self.desc[0:1]
        self.range = self.desc[2]
        self.config = self.desc[3]
        J.s_mov_b32(self.config, 0x00020000)

    def setup(self, base, range):
        self.base[0] = base[0]
        self.base[1] = base[1]
        self.range[0] = range[0]

    def load_dword(self, vdst, voffset, soffset, offset12=0):
        # vdst,     vaddr,           srsrc, soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12, int)  # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset12:{offset12}"
        self.J.buffer_load_dword(vdst, voffset, self.desc, soffset, mod=mod)

    def load_dwordx4(self, vdst, voffset, soffset, offset12=0):
        # vdst,     vaddr,           srsrc, soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12, int)  # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset12:{offset12}"
        self.J.buffer_load_dwordx4(vdst, voffset, self.desc, soffset, mod=mod)

    def store_dwordx4(self, vdata, voffset, soffset, offset12=0):
        # vdata,    vaddr,        srsrc,  soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12, int)  # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset12:{offset12}"
        self.J.buffer_store_dwordx4(vdata, voffset, self.desc, soffset, mod=mod)

    def store_dword(self, vdata, voffset, soffset, offset12=0):
        # vdata,    vaddr,        srsrc,  soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12, int)  # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset12:{offset12}"
        self.J.buffer_store_dword(vdata, voffset, self.desc, soffset, mod=mod)


import math


# def test_softmax(kv_len):
def test_softmax():

    Q = 16
    K = 64
    # how many elements(keys) in one lane.
    LANE_SZ = 4
    ROW_LANES = Q
    COL_LANES = 64 // ROW_LANES
    # how many key elements in one tile. one tile means one lane only needs to handle 'LANE_SZ' data.
    KEYS_PER_TILE = COL_LANES * LANE_SZ
    COL_REPEAT = K // KEYS_PER_TILE
    assert math.log2(
        COL_LANES
    ).is_integer(), f"{COL_LANES} must by be integer power of two"
    SWAPS_PER_ROW = int(math.log2(COL_LANES))

    def toshift(a):
        assert math.log2(a).is_integer(), f"{a} must by be integer power of two"
        return int(math.log2(a))

    @pyhip.jit()
    def kernel(J, pIn: "float*", pOut: "float*", kv_len: "uint"):
        lane_id = J.gpr(J.threadIdx.x[0] & 63)
        row_id = J.gpr(lane_id & (ROW_LANES - 1))
        col_id = J.gpr(lane_id >> toshift(ROW_LANES))
        buff_in = Buffer(J)
        buff_out = Buffer(J)
        size = J.new_gpr("s", 1, dtype="u32", align=1)
        size[0] = Q * K * 4
        kv_stop = J.new_gpr("v", 1, dtype="u32", align=1)
        kv_stop[0] = kv_len[0]
        buff_in.setup(pIn, size)
        buff_out.setup(pOut, size)
        voffset = J.new_gpr("v", 1, dtype="u32")
        vdata = J.gpr(COL_REPEAT*LANE_SZ, "vf32")
        voffset[0] = (row_id[0] << (toshift(K * 4))) + (
            (col_id[0]) << (toshift(4 * LANE_SZ)))
        # voffset[0] = (((J.threadIdx.x[0]&(ROW_LANES-1)) << (toshift(COL_LANES*COL_REPEAT))) + (J.threadIdx.x[0]>>toshift(ROW_LANES)))<< (toshift(4*LANE_SZ))
        voffset_1 = J.new_gpr("v", 1, dtype="u32")
        voffset_1[0] = voffset[0]
        for i in range(0, COL_REPEAT):
            buff_in.load_dwordx4(vdata[i * 4 : i * 4 + 3], (voffset_1[0]), 0)
            voffset_1[0] = voffset_1[0] + (COL_LANES * LANE_SZ * 4)
        J.s_waitcnt(mod=f"vmcnt(0)")
        temp = J.new_gpr("v", 1)

        vmax = J.new_gpr("v", 1)
        # k_end = J.new_gpr("v", 1, dtype="u32")
        k_pos = J.new_gpr("v", 1, dtype="u32")

        temp0 = J.gpr("vf32")
        temp1 = J.new_gpr("v", 1, dtype="u32")
        float_min = J.gpr("vf32")
        # float_min = -FLOAT_MAX
        J.v_mov_b32_e32(float_min[0], 0xFF7FFFFF)
        k_pos[0] = col_id[0] * LANE_SZ
        vmax[0] = float_min[0]
        # k_end[0] = KV_LEN
        for i in range(0, COL_REPEAT):
            temp1[0] = k_pos[0]
            for j in range(0, LANE_SZ):
                J.v_cmp_lt_u32_e32("vcc", temp1[0], kv_stop[0])
                J.v_cndmask_b32_e32(temp0, float_min, vdata[i * LANE_SZ + j], "vcc")
                J.v_max_f32_e32(vmax, temp0, vmax)
                temp1[0] = temp1[0] + 1
            k_pos[0] = k_pos[0] + KEYS_PER_TILE

        vlaneid = J.new_gpr("v", 1)
        vlane_id = J.threadIdx.x[0] & 63
        vlaneid_xor = J.new_gpr("v", 1)
        # find row_max across lane
        # for mask in range(16,33,16):
        for i in range(0, SWAPS_PER_ROW):
            mask = (i + 1) * ROW_LANES
            vlaneid_xor = (vlane_id ^ mask) << 2
            J.ds_bpermute_b32(temp, vlaneid_xor, vmax)
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.v_max_f32_e32(vmax, temp, vmax)
        # exponent(x-row_max)
        for i in range(0, COL_REPEAT * LANE_SZ):
            J.v_sub_f32_e32(vdata[i], vdata[i], vmax)
            # packe 2 multiply? only multiply can pack for f32
            J.v_mul_f32_e32(vdata[i], 0x3FB8AA3B, vdata[i])
            J.v_exp_f32_e32(vdata[i], vdata[i])

        # sum(exponent(x-row_max))
        vsum = J.gpr("vf32")
        vsum[0] = 0
        k_pos[0] = col_id[0] * LANE_SZ
        for i in range(0, COL_REPEAT):
            temp1[0] = k_pos[0]
            for j in range(0, LANE_SZ):
                J.v_cmp_lt_u32_e32("vcc", temp1[0], kv_stop[0])
                J.v_cndmask_b32_e32(temp0, 0, vdata[i * LANE_SZ + j], "vcc")
                J.v_add_f32_e32(vsum, temp0, vsum)
                temp1[0] = temp1[0] + 1
            k_pos[0] = k_pos[0] + KEYS_PER_TILE

        for i in range(0, SWAPS_PER_ROW):
            mask = (i + 1) * ROW_LANES
            vlaneid_xor = (vlane_id ^ mask) << 2
            J.ds_bpermute_b32(temp, vlaneid_xor, vsum)
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.v_add_f32_e32(vsum, temp, vsum)
        inv_sum_scale = J.gpr(1, "vf32")
        J.v_rcp_f32(inv_sum_scale[0], vsum[0] + 1e-6)
        for i in range(0, COL_REPEAT * LANE_SZ):
            J.v_mul_f32(vdata[i], vdata[i], inv_sum_scale)

        # vdata[0] = vmax[0]
        # vdata[1] = vmax[0]
        # vdata[2] = vmax[0]
        # vdata[3] = vmax[0]

        # vdata[0] = vsum[0]
        # vdata[1] = vsum[0]
        # vdata[2] = vsum[0]
        # vdata[3] = vsum[0]
        voffset_1[0] = voffset[0]
        for i in range(0, COL_REPEAT):
            buff_out.store_dwordx4(vdata[i * 4 : i * 4 + 3], (voffset_1[0]), 0)
            voffset_1[0] = voffset_1[0] + (COL_LANES * LANE_SZ * 4)
        J.s_waitcnt(mod=f"vmcnt(0)")

    softmax = torch.nn.Softmax(dim=1)
    input_buf = torch.randint(-4, 5, (Q, K)).to(dtype=torch.float32) / 5.0
    output_buf = torch.zeros(Q, K, dtype=torch.float32)

    for KV_LEN in range(1, K + 1):
        print(f"------------------------------------------------")
        # for KV_LEN in range(1, K+1):
        input_ref = input_buf[:, 0:KV_LEN]
        kernel([1], [64], input_buf.data_ptr(), output_buf.data_ptr(), int(KV_LEN))
        output_ref = softmax(input_ref)
        output = output_buf[:, 0:KV_LEN]
        # print(output-output_ref)
        assert torch.allclose(output_ref, output, atol=0.001, rtol=0.005)
        print(f"KV_LEN={KV_LEN} PASSED")
    # max=input_1.amax(dim=1,keepdim=True)
    # exponent=torch.exp(input_1-max)
    # sum=torch.sum(exponent,dim=1,keepdim=True)
    # for r in range(0, 16):
    #     for c in range(0, KV_LEN):
    #         assert abs((sum[r,0]-output1[r,c])/output1[r,c]) <0.01, f'sum[{r}][0]={sum[r,0]}, output1[{r},{c}]={output1[r,c]}'
    # assert max[r, 0] == output1[r,c], f'max[{r}][0]={max[r, 0]}, output1[{r},{c}]={output1[r,c]}'
    # assert abs((exponent[r,c]-output1[r,c])/output1[r,c]) <0.001, f'suexponentm[{r}][c]={exponent[r,c]}, output1[{r},{c}]={output1[r,c]}'


if __name__ == "__main__":
    test_softmax()
