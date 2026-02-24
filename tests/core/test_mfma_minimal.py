import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.set_default_device('cuda')
torch.manual_seed(0)

from functools import cache

"""
本例为最简单的 单个mfma
图可以参考 https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
"""
def test_mfma_16x16x16_bf16():
    """
    test C= A @ B^T
    hip version be like
    __global__ void mfma_fp32_16x16x16_fp16_A_BT(const fp16_t* A, const fp16_t* B, float* C) {
        fp16x4_t a_reg;
        fp16x4_t b_reg;
        fp32x4_t c_reg {};

        a_reg = *reinterpret_cast<const fp16x4_t*>(A + 4 * (threadIdx.x / 16) + 16 * (threadIdx.x % 16));
        b_reg = *reinterpret_cast<const fp16x4_t*>(B + 4 * (threadIdx.x / 16) + 16 * (threadIdx.x % 16));

        c_reg = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_reg, 0, 0, 0);

        for (int i = 0; i < 4; i++) {
            *(C + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64) = c_reg[i];
        }
   }

    """
    @pyhip.jit()
    def mfma_16x16x16_bf16(J: pyhip.JIT,
           pA: "void*",
           pB: "void*",
           pC: "void*"):

        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)

        lane_id = J.gpr(J.threadIdx.x[0] % 64)

        r = J.gpr(lane_id &15) #lane_id %16
        cblock = J.gpr(lane_id >> 4) #lane_id /16

   
        A_buf = J.Buffer(pA, 16 * 16 * 2)
        B_buf = J.Buffer(pB, 16 * 16 * 2)
        C_buf = J.Buffer(pC, 16 * 16 * 4)

        a_frag = J.gpr(2, "bf16x2")   # 4 bf16 total (dwordx2)
        b_frag = J.gpr(2, "bf16x2")

        voffset = J.gpr(r * (16 ) + cblock * (4 ))
        voffset = voffset << 1
        A_buf.load_dwordx2(a_frag, voffset, 0)       
        B_buf.load_dwordx2(b_frag, voffset, 0)
        J.s_waitcnt(mod="vmcnt(0)")

        c_frag = J.gpr(4, "f32")

        J.v_mfma_f32_16x16x16_bf16(
            c_frag,
            a_frag,
            b_frag,
            0
        )

        base_offset = J.gpr(r * 4 + cblock * 64*4)

        C_buf.store_dword(c_frag[0], base_offset, 0, offset12=0)
        C_buf.store_dword(c_frag[1], base_offset, 0, offset12=64)
        C_buf.store_dword(c_frag[2], base_offset, 0, offset12=128)
        C_buf.store_dword(c_frag[3], base_offset, 0, offset12=192)

        return


    A = torch.randn(16, 16, dtype=torch.bfloat16, device = "cuda")
    B = torch.randn(16, 16, dtype=torch.bfloat16, device = "cuda")
    C = torch.randn(16, 16, dtype=torch.float, device = "cuda")
    mfma_16x16x16_bf16([1],[64], A.data_ptr(), B.data_ptr(), C.data_ptr())
    ref = torch.mm(A.to(torch.float32), B.to(torch.float32).T)
    torch.cuda.synchronize()
    if not torch.allclose(ref, C, atol=0.01, rtol=0.01):
        print(ref)
        print("===================")
        print(C)
    else:
        print("PASS: test_mfma_16x16x16_bf16")


def test_mfma_32x32x8_bf16():
    """
    test C= A @ B^T
    hip version be like:
    __global__ void mfma_fp32_32x32x8_fp16_A_BT(const fp16_t* A, const fp16_t* B, float* C) {
        fp16x4_t a_reg;
        fp16x4_t b_reg;
        fp32x16_t c_reg {};

        a_reg = *reinterpret_cast<const fp16x4_t*>(A + 4 * (threadIdx.x / 32) + 8 * (threadIdx.x % 32));
        b_reg = *reinterpret_cast<const fp16x4_t*>(B + 4 * (threadIdx.x / 32) + 8 * (threadIdx.x % 32));
    

        c_reg = __builtin_amdgcn_mfma_f32_32x32x8f16(a_reg, b_reg, c_reg, 0, 0, 0);

        for (int i = 0; i < 4; i++) {
            C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + i * 32 * 8]          = c_reg[i * 4];
            C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 1 + i * 32 * 8] = c_reg[i * 4 + 1];
            C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 2 + i * 32 * 8] = c_reg[i * 4 + 2];
            C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 3 + i * 32 * 8] = c_reg[i * 4 + 3];
        }
    }
    """
    @pyhip.jit()
    def mfma_32x32x8_bf16(J: pyhip.JIT,
           pA: "void*",
           pB: "void*",
           pC: "void*"):

        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)

        lane_id = J.gpr(J.threadIdx.x[0] % 64)

        r = J.gpr(lane_id &31) #lane_id %32
        cblock = J.gpr(lane_id >> 5) #lane_id /32

   
        A_buf = J.Buffer(pA, 32 * 8 * 2)
        B_buf = J.Buffer(pB, 32 * 8 * 2)
        C_buf = J.Buffer(pC, 32 * 32 * 4)

        a_frag = J.gpr(2, "bf16x2")   # 4 bf16 total (dwordx2)
        b_frag = J.gpr(2, "bf16x2")

        voffset = J.gpr(cblock * 4 + r * 8)
        voffset = voffset << 1
        A_buf.load_dwordx2(a_frag, voffset, 0)       
        B_buf.load_dwordx2(b_frag, voffset, 0)
        J.s_waitcnt(mod="vmcnt(0)")

        c_frag = J.gpr(16, "f32")

        J.v_mfma_f32_32x32x8_bf16(
            c_frag,
            a_frag,
            b_frag,
            0
        )
        base_offset = J.gpr(r + cblock * 4 * 32)
        base_offset = base_offset << 2
        for i in range(4):
            # 这个地方 跟hip版本对应的，因为是float 所以指针的offset都多乘了4
            # base_offset 对应 threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 
            # 256*4 对应 i * 32 * 8 
            # mfma的C为32X32 的pattern基本都差不多 同一个thread 会在32X32的矩阵 中间隔地出现四次
            #  每次都间隔 32X32/4 = 256 个元素
            # 0 128 256 384 对应 0，32 32*2 32*3 这个index 表示C上隔了一行
            C_buf.store_dword(c_frag[i*4+0], base_offset, 0, offset12=i*256*4 + 0)
            C_buf.store_dword(c_frag[i*4+1], base_offset, 0, offset12=i*256*4 + 128)
            C_buf.store_dword(c_frag[i*4+2], base_offset, 0, offset12=i*256*4 + 256)
            C_buf.store_dword(c_frag[i*4+3], base_offset, 0, offset12=i*256*4 + 384)

        return


    A = torch.randn(32, 8, dtype=torch.bfloat16, device = "cuda")
    B = torch.randn(32, 8, dtype=torch.bfloat16, device = "cuda")
    C = torch.randn(32, 32, dtype=torch.float, device = "cuda")
    mfma_32x32x8_bf16([1],[64], A.data_ptr(), B.data_ptr(), C.data_ptr())
    ref = torch.mm(A.to(torch.float32), B.to(torch.float32).T)
    torch.cuda.synchronize()
    if not torch.allclose(ref, C, atol=0.01, rtol=0.01):
        print(ref)
        print("===================")
        print(C)
    else:
        print("PASS: test_mfma_32x32x8_bf16")


def test_mfma_32x32x16_fp8():
    """
    C = A @ B^T，A 32x16 fp8，B 32x16 fp8，C 32x32 f32。
    hip version be like:
    __global__ void mfma_fp32_32x32x16_fp8_fp8(const fp8_t* A, const fp8_t* B, float* C) {
        fp8x8_t a_reg;
        fp8x8_t b_reg;
        fp32x16_t_fp8 c_reg {};

        a_reg = *reinterpret_cast<const fp8x8_t*>(A + (threadIdx.x / 32) * 8 + (threadIdx.x % 32) * 16);
        b_reg = *reinterpret_cast<const fp8x8_t*>(B + (threadIdx.x / 32) * 8 + (threadIdx.x % 32) * 16);
      
        c_reg = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8((long)a_reg, (long)b_reg, c_reg, 0, 0, 0);

        for (int i = 0; i < 4; i++) {
            C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + i * 32 * 8]          = c_reg[i * 4];
            C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 1 + i * 32 * 8] = c_reg[i * 4 + 1];
            C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 2 + i * 32 * 8] = c_reg[i * 4 + 2];
            C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 3 + i * 32 * 8] = c_reg[i * 4 + 3];
        }
    }
    
    """
    arch = torch.cuda.get_device_properties().gcnArchName
    if "gfx942" in arch:
        A = torch.randn(32, 16, device="cuda").to(torch.float8_e4m3fnuz)
        B = torch.randn(32, 16, device="cuda").to(torch.float8_e4m3fnuz)
    elif "gfx950" in arch:
        A = torch.randn(32, 16, device="cuda").to(torch.float8_e4m3fn)
        B = torch.randn(32, 16, device="cuda").to(torch.float8_e4m3fn)
    else:
        print("SKIP: test_mfma_32x32x16_fp8 (no torch.float8_e4m3fnuz or torch.float8_e4m3fn)")
        return

    @pyhip.jit()
    def mfma_32x32x16_fp8(J: pyhip.JIT,
           pA: "void*",
           pB: "void*",
           pC: "void*"):

        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)

        lane_id = J.gpr(J.threadIdx.x[0] % 64)

        r = J.gpr(lane_id &31) #lane_id %32
        cblock = J.gpr(lane_id >> 5) #lane_id /32

   
        A_buf = J.Buffer(pA, 32 * 16 * 1)
        B_buf = J.Buffer(pB, 32 * 16 * 1)
        C_buf = J.Buffer(pC, 32 * 32 * 4)

        a_frag = J.gpr(2, "vu32")   
        b_frag = J.gpr(2, "vu32")

        voffset = J.gpr(cblock * 8 + r * 16)
        # voffset = voffset << 1
        A_buf.load_dwordx2(a_frag, voffset, 0)       
        B_buf.load_dwordx2(b_frag, voffset, 0)
        J.s_waitcnt(mod="vmcnt(0)")

        c_frag = J.gpr(16, "f32")

        J.v_mfma_f32_32x32x16_fp8_fp8(
            c_frag,
            a_frag,
            b_frag,
            0
        )
        base_offset = J.gpr(r + cblock * 4 * 32)
        base_offset = base_offset << 2
        for i in range(4):
            C_buf.store_dword(c_frag[i*4+0], base_offset, 0, offset12=i*256*4 + 0)
            C_buf.store_dword(c_frag[i*4+1], base_offset, 0, offset12=i*256*4 + 128)
            C_buf.store_dword(c_frag[i*4+2], base_offset, 0, offset12=i*256*4 + 256)
            C_buf.store_dword(c_frag[i*4+3], base_offset, 0, offset12=i*256*4 + 384)
        return

    C = torch.zeros(32, 32, dtype=torch.float32, device="cuda")
    mfma_32x32x16_fp8([1], [64], A.data_ptr(), B.data_ptr(), C.data_ptr())
    torch.cuda.synchronize()
    ref = torch.mm(A.to(torch.float32), B.to(torch.float32).T)
    if not torch.allclose(ref, C, atol=0.1, rtol=0.1):
        print("ref (sample):", ref[:4, :4])
        print("C (sample):", C[:4, :4])
    else:
        print("PASS: test_mfma_32x32x16_fp8")


def test_mfma_16x16x32_fp8():
    """
    C = A @ B^T，A 32x16 fp8，B 32x16 fp8，C 32x32 f32。
    hip version be like:
    __global__ void mfma_fp32_16x16x32_fp8_fp8_A_BT(const fp8_t* A, const fp8_t* B, float* C) {
        fp8x8_t a_reg;
        fp8x8_t b_reg;
        fp32x4_t c_reg {};

        a_reg = *reinterpret_cast<const fp8x8_t*>(A + (threadIdx.x / 16) * 8 + (threadIdx.x % 16) * 32);
        b_reg = *reinterpret_cast<const fp8x8_t*>(B + (threadIdx.x / 16) * 8 + (threadIdx.x % 16) * 32);


        c_reg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8((long)a_reg, (long)b_reg, c_reg, 0, 0, 0);

        for (int i = 0; i < 4; i++) {
            *(C + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64) = c_reg[i];
        }
    }

    A (B^T) is logically 32 x 16 

    0   :  0 0 0 0 0 0 0 0 | 16 16 16 16 16 16 16 16 | 32 32 32 32 32 32 32 32 | 48 48 48 48 48 48 48 48
    1   :  1 1 1 1 1 1 1 1 | 17 17 17 17 17 17 17 17 | 33 33 33 33 33 33 33 33 | 49 49 49 49 49 49 49 49
    2   :  2 2 2 2 2 2 2 2 | 18 18 18 18 18 18 18 18 | 34 34 34 34 34 34 34 34 | 50 50 50 50 50 50 50 50
    3   :  3 3 3 3 3 3 3 3 | 19 19 19 19 19 19 19 19 | 35 35 35 35 35 35 35 35 | 51 51 51 51 51 51 51 51
    4   :  4 4 4 4 4 4 4 4 | 20 20 20 20 20 20 20 20 | 36 36 36 36 36 36 36 36 | 52 52 52 52 52 52 52 52
    5   :  5 5 5 5 5 5 5 5 | 21 21 21 21 21 21 21 21 | 37 37 37 37 37 37 37 37 | 53 53 53 53 53 53 53 53
    6   :  6 6 6 6 6 6 6 6 | 22 22 22 22 22 22 22 22 | 38 38 38 38 38 38 38 38 | 54 54 54 54 54 54 54 54
    7   :  7 7 7 7 7 7 7 7 | 23 23 23 23 23 23 23 23 | 39 39 39 39 39 39 39 39 | 55 55 55 55 55 55 55 55
    8   :  8 8 8 8 8 8 8 8 | 24 24 24 24 24 24 24 24 | 40 40 40 40 40 40 40 40 | 56 56 56 56 56 56 56 56
    9   :  9 9 9 9 9 9 9 9 | 25 25 25 25 25 25 25 25 | 41 41 41 41 41 41 41 41 | 57 57 57 57 57 57 57 57
    10  : 10 10 10 10 10 10 10 10 | 26 26 26 26 26 26 26 26 | 42 42 42 42 42 42 42 42 | 58 58 58 58 58 58 58 58
    11  : 11 11 11 11 11 11 11 11 | 27 27 27 27 27 27 27 27 | 43 43 43 43 43 43 43 43 | 59 59 59 59 59 59 59 59
    12  : 12 12 12 12 12 12 12 12 | 28 28 28 28 28 28 28 28 | 44 44 44 44 44 44 44 44 | 60 60 60 60 60 60 60 60
    13  : 13 13 13 13 13 13 13 13 | 29 29 29 29 29 29 29 29 | 45 45 45 45 45 45 45 45 | 61 61 61 61 61 61 61 61
    14  : 14 14 14 14 14 14 14 14 | 30 30 30 30 30 30 30 30 | 46 46 46 46 46 46 46 46 | 62 62 62 62 62 62 62 62
    15  : 15 15 15 15 15 15 15 15 | 31 31 31 31 31 31 31 31 | 47 47 47 47 47 47 47 47 | 63 63 63 63 63 63 63 63
    """
    arch = torch.cuda.get_device_properties().gcnArchName
    if "gfx942" in arch:
        A = torch.randn(16, 32, device="cuda").to(torch.float8_e4m3fnuz)
        B = torch.randn(16, 32, device="cuda").to(torch.float8_e4m3fnuz)
    elif "gfx950" in arch:
        A = torch.randn(16, 32, device="cuda").to(torch.float8_e4m3fn)
        B = torch.randn(16, 32, device="cuda").to(torch.float8_e4m3fn)
    else:
        print("SKIP: test_mfma_32x32x16_fp8 (no torch.float8_e4m3fnuz or torch.float8_e4m3fn)")
        return

    @pyhip.jit()
    def mfma_32x32x16_fp8(J: pyhip.JIT,
           pA: "void*",
           pB: "void*",
           pC: "void*"):

        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)

        lane_id = J.gpr(J.threadIdx.x[0] % 64)

        r = J.gpr(lane_id &15) #lane_id %16
        cblock = J.gpr(lane_id >> 4) #lane_id /16

   
        A_buf = J.Buffer(pA, 32 * 16 * 1)
        B_buf = J.Buffer(pB, 32 * 16 * 1)
        C_buf = J.Buffer(pC, 16 * 16 * 4)

        a_frag = J.gpr(2, "vu32")   
        b_frag = J.gpr(2, "vu32")

        voffset = J.gpr(cblock * 8 + r * 32)
 
        A_buf.load_dwordx2(a_frag, voffset, 0)       
        B_buf.load_dwordx2(b_frag, voffset, 0)
        J.s_waitcnt(mod="vmcnt(0)")

        c_frag = J.gpr(4, "f32")

        J.v_mfma_f32_16x16x32_fp8_fp8(
            c_frag,
            a_frag,
            b_frag,
            0
        )

        base_offset = J.gpr(r * 4 + cblock * 64*4)

        C_buf.store_dword(c_frag[0], base_offset, 0, offset12=0)
        C_buf.store_dword(c_frag[1], base_offset, 0, offset12=64)
        C_buf.store_dword(c_frag[2], base_offset, 0, offset12=128)
        C_buf.store_dword(c_frag[3], base_offset, 0, offset12=192)

        return

    C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    mfma_32x32x16_fp8([1], [64], A.data_ptr(), B.data_ptr(), C.data_ptr())
    torch.cuda.synchronize()
    ref = torch.mm(A.to(torch.float32), B.to(torch.float32).T)
    if not torch.allclose(ref, C, atol=0.1, rtol=0.1):
        print("ref (sample):", ref[:4, :4])
        print("C (sample):", C[:4, :4])
    else:
        print("PASS: test_mfma_16x16x32_fp8")

if __name__ == "__main__":
    test_mfma_16x16x16_bf16()
    test_mfma_32x32x8_bf16()
    test_mfma_32x32x16_fp8()
    test_mfma_16x16x32_fp8()
