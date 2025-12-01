import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

'''
计算耗时比数据搬运耗时低的时候，mem-bound比较明显

计算耗时稍微高于搬运耗时的时候，哪怕是有间隔的发起外存搬运请求，
仍然有一定概率引起issue-stall，从而无法使用计算完美遮盖内存搬运 ？？？

除非计算耗时远超搬运耗时（例如是搬运耗时的2倍或者以上），搬运耗时引起的
issue-stall基本消失，遮盖也接近完全
---------------------------------------------------------------------------------
如果4个SIMD都密集发起load_dwordx4指令的话，可以达到4TB/s带宽
按照主频1.4GHz来算，每个SIMD的每条 load_dwordx4 需要115个cycle完成
但是如果碰巧某个 load_dwordx4 指令发生了一点波动，超时了，由此引发后继计算和交织其中的下一条load指令也晚发起了
---------------------------------------------------------------------------------
GROUP_SIZE 似乎帮助很大，相邻4/8/16个外存加载指令作为一组连续发起:
 - 利用offset12降低地址计算指令个数
 - 大幅减少s_waitcnt指令密度
 - 密集发起似乎比间隔发起更不容易受到波动影响
因此过于细粒度的同步反而不太好
---------------------------------------------------------------------------------
使用可移动步骤描述计算流程：load_HBM_to_Reg => Use_Reg_to_compute
这样的步骤进行pipeline摆放就可以达到并行计算和搬运的目的
如下所示：

load0

load1
wait0-compute0

load0
wait1-compute1

更复杂的pipeline可能包含跟多步骤： load_HBM_to_Reg => Store_Reg_to_LDS => sync => load_LDS_to_Reg => MFMA

自动插入s_waitcnt很难识别到这种group级别的等待，从而可能插入太多的等待指令，浪费了一些issue资源
'''

def test_sum():
    @pyhip.jit()
    def kernel(J, pA:"int*", count:"int", wave_count:"int", pDst:"int*"):
        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)
        lane_id = J.gpr(J.threadIdx.x[0] % 64)

        idx0 = J.gpr((J.blockIdx.x[0] * 4 + warp_id) * wave_count)
        idx1 = J.gpr(idx0[0] + wave_count)
        J.s_min_u32(idx1, idx1, count)
        J.s_sub_u32(wave_count[0], idx1[0], idx0[0])

        J.s_add_u32(pA[0], pA[0], idx0 << 2)
        J.s_addc_u32(pA[1], pA[1], 0)

        buff = J.Buffer(pA, wave_count[0] << 2)

        voffset = J.gpr(lane_id << 4) # 16bytes per-lane
        
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
        with J.While(idx0[0] < wave_count[0]):
            load_to_reg(1)
            compute_on_reg(0)

            load_to_reg(2)
            compute_on_reg(1)
            J.s_barrier()
            idx0[0] = idx0[0] + 64*4*dwordx4_per_step*2

        vsum[0] = vsum[0] + vsum[1]
        vsum[2] = vsum[2] + vsum[3]
        vsum[0] = vsum[0] + vsum[2]
        vsum = J.reduce("v_add_f32", vsum[0])
        #  sdata:dst,       sbase,    soffset        offset20u glc
        with J.ExecMask(lane_id[0] == 0):
            J.global_atomic_add_f32(lane_id << 2, vsum[0], pDst)
            #J.s_store_dword(count, pA, 0, mod="glc")
        J.s_waitcnt(mod=f"vmcnt(0) lgkmcnt(0)")
        return

    num_blocks = 80
    waves_per_block = 4
    wave_count = 1024*256
    count = num_blocks * wave_count * waves_per_block
    A = torch.randn(count, dtype=torch.float)

    A_bytes =  A.element_size() * A.nelement()
    buffers = []
    nbytes = 0
    while nbytes < 20e9:
        buffers.append(A.clone())
        nbytes += A_bytes
    print(f"total buffer size:  {A_bytes*1e-6:.1f}MB x {len(buffers)} = {nbytes*1e-9:.3f} GB")

    for i in range(10):
        B = torch.zeros(64, dtype=torch.float)
        with pyhip.cudaPerf(rw_bytes=A_bytes, name="kernel"):
            kernel([num_blocks],[64*waves_per_block], buffers[i%len(buffers)].data_ptr(), count, wave_count, B.data_ptr())
        torch.cuda.synchronize()
    if not torch.allclose(B[0], torch.sum(A)):
        print(B[:8])
        print(torch.sum(A).item())
        assert 0

if __name__ == "__main__":
    test_sum()

