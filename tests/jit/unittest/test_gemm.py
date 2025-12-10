import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

def ptr_add(J, ptr, offset32):
    J.s_add_u32(ptr[0], ptr[0], offset32)
    J.s_addc_u32(ptr[1], ptr[1], 0)

'''
tile级别的封装会希望直接连续使用buffer load指令把整个LDSBuff对应的内容预取到寄存器
但是这些指令最终排布时可能更适合跟计算指令交织起来避免MFMA空闲，
另外预取的目的地是寄存器而不是LDS，预取一个2d-tile到寄存器，需要描述寄存器tile的layout
如果缩小一点，64个dwordx4，按照mem-coalescing方式排列可以代表最小单位是面积为
1024-fp8/512-fp16/256-fp32 的任何尺寸的2D-tile，宽度最小为16xfp8/8xfp16/4xfp32

这样的2D加载需要计算per-lane的偏移地址，这个偏移地址应该计算一次就能代表tile的view
也就是layout相关的部分，后面在整个外存tile中移动这个2d-tile只需要使用soffset/offset12
就可以高效完成了(修改voffset的话也可以，但会占用VALU)

另外这样的2D加载如果同一个WG内的全部warp都参与执行的话，可以使用更复杂的分布表达这个block级别的
寄存器变量，根据warp的分布（比如4个warp可能是1x4,4x1,2x2）每个warp需要负责加载的数据偏移也会不同
但是warp之间因为指令流是完全独立执行的，并没有warp内部线程lock-step的限制，因此从灵活性角度
我们可以仅仅封装wave级别的Tile即可


这个DW4Tile也应该能表达LDS中的数据，此时lane跟row/col的映射方式 跟 外存读入时就不同了，需要考虑swizzle
以及MFMA的layout，也即是这个数据结构的本质是想要表达 DWORDx4 寄存器对应到 VMEM/DS指令时的映射关系，也就是layout。

逻辑layout上，DW4Tile永远只是一个2D tensor（例如MFMA是32x8/16x16），并且我们总是假定cols方向连续(column-major)

物理layout上，DW4Tile把逻辑layout上的元素映射到64-lane的寄存器上，方式就很多种，可以是符合 mem-coalescing 
的水平密集摆放，也可以是符合MFMA input/output矩阵摆放要求的先4列，再32行，再2列等。也可以是在上面两种layout
基础上再叠加swizzle的,也就是根据逻辑rows/cols进行swizzle之后适合存入LDS的那种。

不论如何，其内部都是一个per-lane的偏移地址，逻辑上都是一个2D-tensor，可以在另一个更大的2D-tensor中偏移
因为不论怎么排布，一个64-lane的vgpr偏移都能完美表达了

排布的编码其实可以简单理解为对2d tensor本身进行重排，例如可以假设初始shape就是(32,8), 经过一些加工变为MFMA要求的样子

tile0 = Tile(shape=[32,8], dtype=torch.float16)
tile0 = tile0.reshape([32,2,4])
tile0 = tile0.transpose(0,1)  #[2,32,4]

这样就代表了MFMA的32x8的layout了

本质上layout是使用类似reshape/permute组合表达的一种对数据排布的变换
对于寄存器layout而言，简化为根据lane_id推导行列坐标的过程，这个过程决定了
一个2D tile加载到寄存器中之后是什么样子。这个过程可以归结为一个voffset偏移量

但是swizzle之后的坐标可能一个voffset不足以表达，而是组成LDSBuff的每个DW4Tile都要
构造一个voffset,又或者可以根据swizzle构造一组voffset, 例如指定
'''

class DW4Tile:
    def __init__(self, J, rows:int, cols:int, ele_size:int, nstride_bytes):
        assert isinstance(rows, int)
        assert isinstance(cols, int)
        assert isinstance(ele_size, int)
        assert (cols*ele_size) % 16 == 0, "row size must be multiple of dwordx4"
        assert rows*cols*ele_size == 64*16, "total size must equal to 64-lane x dwordx4"
        lanes_per_row = (cols * ele_size)//16
        assert lanes_per_row > 0 and (lanes_per_row & (lanes_per_row-1)) == 0, f"{lanes_per_row=} is not power of 2"
        shift_bits = lanes_per_row.bit_length() - 1

        assert ele_size == 1 or ele_size == 2 or ele_size == 4
        ele_size_shift_bits = ele_size.bit_length() - 1

        self.J = J

        # self.voffset 只在初始化时计算一次，后面移动时不再重复计算
        self.voffset = J.gpr(f"vu32")
        lane_id = J.gpr(J.threadIdx.x[0] & 63)
        if lanes_per_row < 64:
            self.voffset[0] = ((lane_id & (lanes_per_row - 1)) << 4) + (lane_id >> shift_bits) * nstride_bytes
        else:
            self.voffset[0] = (lane_id << 4)

    def load(self, buff, soff):
        buff


class LDSBuff:
    def __init__(self, J, base, rows, cols):
        self.J = J
        self.base = base
        self.rows = rows
        self.cols = cols


    def prefetch_dwordx4(buff, row, soffset, nstride):
        pass


def txest_gemm():
    INST_M = 32
    INST_N = 32
    INST_K = 8
    WARP_M = 4
    WARP_N = 4
    WARP_K = 4
    BLK_M = INST_M * WARP_M * 2
    BLK_N = INST_N * WARP_N * 2
    BLK_K = INST_K * WARP_K
    @pyhip.jit()
    def gemm_kernel(J, 
               pA:"__fp16*", pB:"__fp16*", nstrideAB:"int",
               pC:"float*", nstrideC:"int",
               K:"int"):
        sizeof__fp16 = 2
        sizeof__float = 4
        blkX = J.blockIdx.x
        blkY = J.blockIdx.y
        offset = J.gpr(blkY * nstrideAB * BLK_M * sizeof__fp16)
        ptr_add(J, pA, offset)
        ptr_add(J, pB, offset)
        offset = J.gpr((blkX * BLK_N + blkY * BLK_M * nstrideC) * sizeof__float)
        ptr_add(J, pC, offset)

        buff1 = J.Buffer(pA, nstrideAB * (BLK_M*sizeof__fp16))
        buff2 = J.Buffer(pB, nstrideAB * (BLK_N*sizeof__fp16))

        Abuff = J.alloc_lds(BLK_M*BLK_K*sizeof__fp16)
        Bbuff = J.alloc_lds(BLK_N*BLK_K*sizeof__fp16)
        lds_nstride = BLK_K

        '''
        A: 256x32*2 = 16KB = 64 regs = 16(4 per SIMD) buffer_load_dwordx4 = 16(8 per SIMD) ds_read_b128
        B: 256x32*2 = 16KB = 64 regs = 16(4 per SIMD) buffer_load_dwordx4 = 16(8 per SIMD) ds_read_b128
        MFMA_32x32x8: 16*4 = 64 

        因为2x2的SIMD协作模式，分摊到每个SIMD上的外存数据加载指令比较少，远远少于MFMA计算指令
        因此 LDS ping-pong 的话是一种极大的浪费

        load_to_Regs => lds_write => sync => lds_read => MFMA

        MFMA计算分为3部分 C1/C2/C3

        先把可复用的简单工具做好

        在这种加载数据到LDS，LDS读出给MFMA用的模式中，可复用的就是数据加载/swizzle/读出
        可以认为MFMA_LDS_Buff就是可复用的封装。它具有编译期已知的layout和swizzle策略，
        以dwordx4/b128为单位的两种方式的加载和写入
        或者从另一个角度，按照dwordx4为基本单位swizzle是LDS缓存的属性，任何该LDS的vgpr读写
        其per-lane的行列转换为LDS 偏移之前都要经过swizzle

        # row/col是per-lane的，offset是per-lane的LDS偏移，可以直接给ds-read/ds-write指令
        lds.offset(row, col)

        协助手工排布指令，在维护vmcnt/lkgmcnt, 在关键点记录 vmcnt/lgkmcnt，需要访问这些关键点
        数据的时候可以很方便的插入s_waitcnt。

        交织指令比较

        '''

        return
        idx0 = J.gpr(J.blockIdx.x[0] * block_count)
        idx1 = J.gpr(idx0[0] + block_count)
        J.s_min_u32(idx1, idx1, count)
        J.s_sub_u32(block_count[0], idx1[0], idx0[0])

        J.s_add_u32(pA[0], pA[0], idx0 << 2)
        J.s_addc_u32(pA[1], pA[1], 0)

        buff = J.Buffer(pA, block_count[0] << 2)

        voffset = J.gpr(J.threadIdx.x[0] << 4) # 16bytes per-lane
        
        vtemp = J.gpr(f"vf32x4")
        vtemp[0] = 0
        vtemp[1] = 0
        vtemp[2] = 0
        vtemp[3] = 0

        vsum = J.gpr(f"vf32x4")
        vsum[0] = 0
        vsum[1] = 0
        vsum[2] = 0
        vsum[3] = 0
        
        FIFO_CNT = 2
        dwordx4_per_step = 8
        offset12 = 0

        fifo = []
        for f in range(FIFO_CNT):
            fifo.append({
                "vgprs": J.gpr(f'vf32x{4*dwordx4_per_step}'),
                # how vmem load instruction is on the fly
                # when this vgprs are being loaded
                "vmcnt": 0,
            })

        vmcnt = 0
        # python closure 闭包 (类似C++ lambda) 表达pipeline的步骤
        def load_to_reg(step):
            nonlocal offset12, vmcnt
            fitem = fifo[step%FIFO_CNT]
            for i in range(dwordx4_per_step):
                buff.load_dwordx4(fitem["vgprs"][i*4+0:i*4+3], voffset, 0, offset12=offset12)
                vmcnt += 1
                offset12 += 64*16
                if offset12 >= (1<<12):
                    voffset[0] = voffset[0] + offset12
                    offset12 = 0
            fitem["vmcnt"] = vmcnt

        def compute_on_reg(step):
            fitem = fifo[step%FIFO_CNT]
            # 当前指令位置跟我们所依赖的fifo项之间又插入了多少条新的vmcnt，就是我们需要等待的vm数目
            J.s_waitcnt(mod=f"vmcnt({vmcnt - fitem['vmcnt']})")
            
            if 0:
                nop_count = 80//4*dwordx4_per_step
                while nop_count >= 0:
                    vtemp[nop_count%4] = vtemp[nop_count%4] + 1
                    nop_count -= 1
            
            for i in range(dwordx4_per_step):
                for n in range(4):
                    vsum[n] = vsum[n] + fitem["vgprs"][i*4 + n]

        load_to_reg(0)

        idx0[0] = 0
        with J.While(idx0[0] < block_count[0]):
            load_to_reg(1)
            compute_on_reg(0)

            load_to_reg(2)
            compute_on_reg(1)
            idx0[0] = idx0[0] + 64*4*dwordx4_per_step*2

        vsum[0] = vsum[0] + vsum[1]
        vsum[2] = vsum[2] + vsum[3]
        vsum[0] = vsum[0] + vsum[2]
        vsum = J.reduce("v_add_f32", vsum[0])
        #  sdata:dst,       sbase,    soffset        offset20u glc
        with J.ExecMask(J.threadIdx.x[0] == 0):
            J.global_atomic_add_f32(J.threadIdx.x[0] << 2, vsum[0], pDst)
            #J.s_store_dword(count, pA, 0, mod="glc")
        J.s_waitcnt(mod=f"vmcnt(0) lgkmcnt(0)")
        return

    CU_rows = 8 #best_rows
    CU_cols = 10 #num_CU//CU_rows
    M = BLK_M * CU_rows*4
    N = BLK_N * CU_cols*4
    K = 8192
    print(f" {M}x{N}x{K}  CUs: {CU_rows} x {CU_cols} = {CU_rows*CU_cols} ")
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(N, K, dtype=torch.float16)
    C = torch.randn(M, N, dtype=torch.float)
    A_bytes = A.element_size() * A.nelement()
    B_bytes = B.element_size() * B.nelement()
    C_bytes = C.element_size() * C.nelement()
    buffers = []
    nbytes = 0
    while nbytes < 20e9:
        buffers.append([A.clone(), B.clone(), C.clone()])
        nbytes += A_bytes + B_bytes + C_bytes
    print(f"total buffer size:  {(A_bytes+B_bytes+C_bytes)*1e-6:.1f}MB x {len(buffers)} = {nbytes*1e-9:.3f} GB")

    di = 0
    for i in range(4):
        with pyhip.cudaPerf(M*N*K*2, name=f"torch-linear-{di}"):
            ref = torch.nn.functional.linear(buffers[di][0], buffers[di][1])
        di = (di + 1) % len(buffers)
    ref = ref.to(dtype=torch.float)

    for i in range(4):
        A,B,C = buffers[di]
        with pyhip.cudaPerf(M*N*K*2, (M*K*2+K*N*2), name=f"gemm_kernel-{di}"):
            gemm_kernel([(N//BLK_N), (M//BLK_M)],[256],
                A.data_ptr(), B.data_ptr(), K, C.data_ptr(), N, K)
        di = (di + 1) % len(buffers)

    pass_flag = torch.allclose(ref, C, atol=0.1, rtol=0.1)
    if not pass_flag:
        print(ref[:,0])
        print(C[:,0])
        assert 0

if __name__ == "__main__":
    txest_gemm()

