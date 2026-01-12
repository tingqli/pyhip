import pyhip
import pytest
import functools
import torch

"""
work-group协作，每次预取 wg_M * row_bytes 大小的内容到预取寄存器，写入LDS
"""
class MFMA_DW4Loader:
    def __init__(self, J:pyhip.JIT, ptr, buff_size,
                 wg_M:int, row_bytes:int, stride_bytes:int,
                 wave_cnt:int, swizzle_row_div:int,
                 skip_load:bool = False):
        self.buff = J.Buffer(ptr, buff_size) # wg_M*stride_bytes)
        self.wave_cnt = wave_cnt
        self.wg_M = wg_M
        self.J = J
        sizeof_DWORDX4 = 16
        assert (row_bytes) % sizeof_DWORDX4 == 0 # each lane prefetch DWORDx4 which is 8xhalf
        num_lanes_per_row = row_bytes // sizeof_DWORDX4
        assert 64 % num_lanes_per_row == 0
        dw4_prefetch_MN = (self.wave_cnt*64//num_lanes_per_row)
        assert dw4_prefetch_MN >= 1
        assert self.wg_M % dw4_prefetch_MN == 0
        # print(f"{self.wg_M=} {num_lanes_per_row=} {dw4_prefetch_MN=}")
        num_prefetch_M = self.wg_M // dw4_prefetch_MN
        self.prefetch_reg = J.gpr(num_prefetch_M, 4, "vu32")
        self.num_lanes_per_row = num_lanes_per_row
        self.prefetch_voffset = J.gpr((J.threadIdx.x % num_lanes_per_row) * sizeof_DWORDX4 + (J.threadIdx.x //num_lanes_per_row) * stride_bytes)
        self.prefetch_soffset = J.gpr("su32")
        self.prefetch_step_size = (dw4_prefetch_MN)*stride_bytes
        self.num_prefetch_M = num_prefetch_M
        self.stride_bytes = stride_bytes
        self.swizzle_row_div = swizzle_row_div
        if skip_load:
            self.prefetch_step_size = 0
        
        # precompute ds_write vaddr
        self.ds_write_b128_vaddr = [] 
        for index in range(num_prefetch_M):
            vaddr = J.gpr("vu32")
            col = J.threadIdx.x % num_lanes_per_row
            row = (J.threadIdx.x // num_lanes_per_row) + index*dw4_prefetch_MN
            swizzle_col = ((row//swizzle_row_div) ^ col) % (num_lanes_per_row)
            vaddr[0] = J.gpr((row * row_bytes) + swizzle_col*(sizeof_DWORDX4))
            self.ds_write_b128_vaddr.append(vaddr) 

    def __len__(self):
        return self.num_prefetch_M

    def reset_offset(self, koff):
        self.prefetch_soffset[0] = koff[0]

    def prefetch(self, index):
        assert index < self.num_prefetch_M
        self.buff.load_dwordx4(self.prefetch_reg[index], self.prefetch_voffset, self.prefetch_soffset)
        self.prefetch_soffset[0] = self.prefetch_soffset[0] + self.prefetch_step_size

    def ds_write(self, index, lds_base):
        assert index < self.num_prefetch_M
        self.J.ds_write_b128(self.ds_write_b128_vaddr[index], self.prefetch_reg[index], mod=f"offset:{lds_base}") #  vaddr, vdata offset gds

"""
外存数据经过按照 mfma_MN x  mfma_K 尺寸 preshuffle，
work-group协作，每次预取 wg_M * row_bytes 大小的内容到预取寄存器
"""
class MFMA_DW4Loader_preshuffled:
    def __init__(self, J, ptr, buff_size, mfma_MN, 
                 wg_M:int, row_bytes:int, stride_bytes:int,
                 wave_cnt:int, swizzle_row_div:int,
                 skip_load:bool = False):
        self.buff = J.Buffer(ptr, buff_size)
        sizeof_DWORDX4 = 16
        assert row_bytes % sizeof_DWORDX4 == 0
        row_lanes = row_bytes // sizeof_DWORDX4
        mfma_K_lanes = 64 // mfma_MN
        assert row_lanes % mfma_K_lanes == 0, f"{row_lanes=} {mfma_K_lanes=}"
        assert wg_M % mfma_MN == 0
        num_prefetch_m = wg_M // mfma_MN
        num_prefetch_n = row_lanes // mfma_K_lanes
        assert (num_prefetch_m * num_prefetch_n) % wave_cnt == 0
        wave_prefetches = num_prefetch_m * num_prefetch_n // wave_cnt
        # big_waves = wave_cnt - (wave_prefetches * wave_cnt - num_prefetch_m * num_prefetch_n)
        num_lanes_per_row = row_bytes // sizeof_DWORDX4

        warp_id = J.warp_id
        lane_id = J.lane_id
        # at compile time, precompute offsets for waves and generate conditional assign
        self.ds_write_b128_vaddr = [J.gpr("vu32") for _ in range(wave_prefetches)]
        self.prefetch_reg = J.gpr(wave_prefetches, 4, "vu32")
        self.prefetch_sbase = J.gpr("su32")
        self.prefetch_offsets = J.gpr(wave_prefetches, "su32")
        self.prefetch_voffset = lane_id * sizeof_DWORDX4
        prefetch_id = 0
        for wave in range(wave_cnt):
            with J.If(warp_id[0] == wave):
                # pre-shuffled data unit size: [mfma_MN, mfma_K_lanes * sizeof_DWORDX4]
                for i in range(wave_prefetches):
                    prefetch_n = prefetch_id % num_prefetch_n
                    prefetch_m = prefetch_id // num_prefetch_n
                    assert prefetch_m < num_prefetch_m
                    prefetch_id += 1
                    offset = prefetch_m * (mfma_MN * stride_bytes) + prefetch_n * (mfma_MN * mfma_K_lanes * sizeof_DWORDX4)
                    self.prefetch_offsets[i] = offset

                    #ds_offset = prefetch_m * (mfma_MN * row_bytes) + prefetch_n * (mfma_MN * mfma_K_lanes * sizeof_DWORDX4)
                    col = (lane_id // mfma_MN) + prefetch_n * mfma_K_lanes
                    row = (lane_id % mfma_MN) + prefetch_m * mfma_MN
                    swizzle_col = ((row//swizzle_row_div) ^ col) % (num_lanes_per_row)
                    self.ds_write_b128_vaddr[i][0] = (row * row_bytes) + swizzle_col*(sizeof_DWORDX4)

        self.wave_prefetches = wave_prefetches
        self.mfma_MN = mfma_MN
        self.J = J

    def __len__(self):
        return self.wave_prefetches

    def reset_offset(self, koff_bytes):
        self.prefetch_sbase[0] = koff_bytes[0] * self.mfma_MN

    def prefetch(self, index):
        assert index < self.wave_prefetches
        self.buff.load_dwordx4(self.prefetch_reg[index], self.prefetch_voffset, self.prefetch_offsets[index] + self.prefetch_sbase[0])
        #self.prefetch_sbase[0] = self.prefetch_sbase[0] + self.prefetch_step_size

    def ds_write(self, index, lds_base):
        assert index < self.wave_prefetches
        self.J.ds_write_b128(self.ds_write_b128_vaddr[index], self.prefetch_reg[index], mod=f"offset:{lds_base}") #  vaddr, vdata offset gds


class UGEMM:
    def __init__(self, J, 
                 mfma_MN:int,
                 wave_size:list,
                 wave_cnt:list,
                 K, N):
        self.K = K
        self.N = N
        
        wave_size_M, wave_size_N = wave_size
        wave_cnt_M, wave_cnt_N = wave_cnt
        assert mfma_MN in [16, 32]
        assert wave_size_M % mfma_MN == 0
        assert wave_size_N % mfma_MN == 0
        # *2 for 8-bf16/fp16 so DWORDx4 lane-size can be used
        self.mfma_K = (2*8 if mfma_MN == 32 else 2*16)

        # number of C/D regs per wave
        wave_nCM = wave_size_M // mfma_MN
        wave_nCN = wave_size_N // mfma_MN
        wave_nCK = 2

        # wnM,wnN: 2x2, 1x4, 4x1, 1x1, 1x2, 2x1, ....
        self.J = J
        self.mfma_MN = mfma_MN
        self.wave_size_M = wave_size_M
        self.wave_size_N = wave_size_N
        self.wave_cnt_M = wave_cnt_M
        self.wave_cnt_N = wave_cnt_N
        self.wave_cnt = wave_cnt_M * wave_cnt_N

        self.wg_M = self.wave_size_M * self.wave_cnt_M
        self.wg_N = self.wave_size_N * self.wave_cnt_N
        self.wg_K = self.mfma_K * wave_nCK          # 2*(2*8) or 2*(2*16)

        assert self.K % self.wg_K == 0, f"{self.K=} {self.wg_K=}"
        # prefetch evenly distrubuted on all waves
        assert self.wg_M % self.wave_cnt == 0
        assert self.wg_N % self.wave_cnt == 0
        self.wave_nCM = wave_nCM
        self.wave_nCN = wave_nCN
        self.wave_nCK = wave_nCK

    def run(self, loaderA, loaderB, buff_c, M, debug_warp, skip_load):
        J = self.J

        sizeof_bf16 = 2
        sizeof_fp32 = 4
        sizeof_DWORDX4 = 16        

        LDSA_size = self.wg_M * self.wg_K * sizeof_bf16
        LDSB_size = self.wg_N * self.wg_K * sizeof_bf16
        ldsA = J.alloc_lds(LDSA_size)
        ldsB = J.alloc_lds(LDSB_size)

        # prefetch in memory-coalescing way
        assert (sizeof_bf16*self.wg_K) % sizeof_DWORDX4 == 0 # each lane prefetch DWORDx4 which is 8xhalf
        num_lanes_per_row = sizeof_bf16 * self.wg_K // sizeof_DWORDX4
        assert 64 % num_lanes_per_row == 0
        dw4_prefetch_MN = (self.wave_cnt*64//num_lanes_per_row)
        assert dw4_prefetch_MN >= 1
        print(f"{self.wg_M=} {num_lanes_per_row=} {dw4_prefetch_MN=}")

        # 4-wave within a WG coorperatively loads VRAM data in DWORDx4
        def swizzle(row, col):
            swizzle_row_div = 1 if self.mfma_MN == 16 else 2
            return (row//swizzle_row_div) ^ col

        # each swizzle generates a new vaddr pattern, precompute all of them
        # each wave reads its own part
        ds_readA_vaddr = J.gpr(self.wave_nCM, self.wave_nCK, "vu32")
        ds_readB_vaddr = J.gpr(self.wave_nCN, self.wave_nCK, "vu32")
        # wave location
        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)
        lane_id = J.gpr(J.threadIdx.x % 64)
        warp_id_m = warp_id // self.wave_cnt_N
        warp_id_n = warp_id % self.wave_cnt_N
        warp_offset_m = warp_id_m * self.wave_size_M
        warp_offset_n = warp_id_n * self.wave_size_N
        for m in range(self.wave_nCM):
            for k in range(self.wave_nCK):
                row = lane_id % self.mfma_MN + warp_offset_m + m*self.mfma_MN
                col = lane_id // self.mfma_MN + (k * self.mfma_K * sizeof_bf16) // sizeof_DWORDX4
                swizzle_col = swizzle(row, col) % (num_lanes_per_row)
                ds_readA_vaddr[m, k] = J.gpr((row * (self.wg_K * sizeof_bf16)) + swizzle_col*(sizeof_DWORDX4))

        for n in range(self.wave_nCN):
            for k in range(self.wave_nCK):
                row = lane_id % self.mfma_MN + warp_offset_n + n*self.mfma_MN
                col = lane_id // self.mfma_MN + (k * self.mfma_K * sizeof_bf16) // sizeof_DWORDX4
                swizzle_col = swizzle(row, col) % (num_lanes_per_row)
                ds_readB_vaddr[n, k] = J.gpr((row * (self.wg_K * sizeof_bf16)) + swizzle_col*(sizeof_DWORDX4))

        Creg_size = (self.mfma_MN * self.mfma_MN)//64
        mfma_C = self.J.gpr(self.wave_nCM, self.wave_nCN, Creg_size, f"af32")
        ABReg_size = (self.mfma_MN * self.mfma_K * 2//4)//64
        mfma_A = J.gpr(self.wave_nCM, self.wave_nCK, ABReg_size, "vbf16x2")
        mfma_B = J.gpr(self.wave_nCN, self.wave_nCK, ABReg_size, "vbf16x2")

        def ds_readA(m, k):
            J.ds_read_b128(mfma_A[m,k], ds_readA_vaddr[m, k], mod=f"offset:{ldsA}") #  vaddr, vdata offset gds

        def ds_readB(n, k):
            J.ds_read_b128(mfma_B[n,k], ds_readB_vaddr[n,k], mod=f"offset:{ldsB}") #  vaddr, vdata offset gds

        J.debug_setup((J.blockIdx.x[0] == 0) & (J.blockIdx.y[0] == 0) & (warp_id[0] == debug_warp))

        #======================== prelog 0 ===========================
        #                     prefetch [0]
        #======================== prelog 1 ===========================
        #               | ds_write[0]  +  prefetch [1]    |
        #    ....       |                                 | read0[0]
        #    ....       |  ... ....                       | ......  
        #======================== loop-body ==========================
        #               | ds_write[i+1] + prefetch [i+2]  |
        # |  read1 [i]  |                                 | read0[i+1]
        # |  mfma0 [i]  |  mfma0[i] + mfma1[i]            | mfma1[i]
        #
        #       | means s_waitcnt(lgkmcnt) (+s_barrier maybe)
        #==============================================================

        k_offset = J.gpr("su32")        
        
        num_prefetch_M = len(loaderA)
        num_prefetch_N = len(loaderB)

        k_offset[0] = 0
        loaderA.reset_offset(k_offset)
        loaderB.reset_offset(k_offset)
        # prelog 0
        for r in range(num_prefetch_M): loaderA.prefetch(r)
        for r in range(num_prefetch_N): loaderB.prefetch(r)
        mfma_C[:] = 0 

        # prelog 1: ds_write + prefetch
        k_offset[0] = 0 if skip_load else (k_offset[0] + self.wg_K * sizeof_bf16)
        loaderA.reset_offset(k_offset)
        loaderB.reset_offset(k_offset)

        for r in range(num_prefetch_M):
            J.s_waitcnt(mod=f"vmcnt({num_prefetch_M + num_prefetch_N - 1})")
            loaderA.ds_write(r, ldsA)
            loaderA.prefetch(r)
        for r in range(num_prefetch_N):
            J.s_waitcnt(mod=f"vmcnt({num_prefetch_M + num_prefetch_N - 1})")
            loaderB.ds_write(r, ldsB)
            loaderB.prefetch(r)

        # prelog 1: wait ds_write
        J.s_waitcnt(mod="lgkmcnt(0)")
        J.s_barrier()

        # prelog 1: ds_read0
        for k in range(0,self.wave_nCK//2):
            for m in range(self.wave_nCM):
                ds_readA(m, k)
            for n in range(self.wave_nCN):
                ds_readB(n, k)
        J.s_waitcnt(mod=f"lgkmcnt(0)")

        if 0:
            J.debug_log(mfma_A[0], torch.bfloat16, "2h.4h.16v.8h")
            J.debug_log(mfma_A[1], torch.bfloat16, "2h.4h.16v.8h")
            J.debug_log(mfma_A[2], torch.bfloat16, "2h.4h.16v.8h")
            J.debug_log(mfma_A[3], torch.bfloat16, "2h.4h.16v.8h")

            J.debug_log(mfma_B[0], torch.bfloat16, "2h.4h.16v.8h")
            J.debug_log(mfma_B[1], torch.bfloat16, "2h.4h.16v.8h")
            J.debug_log(mfma_B[2], torch.bfloat16, "2h.4h.16v.8h")
            J.debug_log(mfma_B[3], torch.bfloat16, "2h.4h.16v.8h")

        mfma_info={
            16:("v_mfma_f32_16x16x16_bf16",16),
            32:("v_mfma_f32_32x32x8_bf16",32)
        }
        mfma_name = mfma_info[self.mfma_MN][0]
        mfma_cycles = mfma_info[self.mfma_MN][1]

        # mfma reg stage1 | ds_read/hidden
        # ---------------------------------
        #  wait LDS buffer to be finished
        #
        #   J.s_waitcnt ds_read finished
        #   J.s_barrier() 
        #
        def mfma_generator0():
            for k in range(0,self.wave_nCK//2):
                for m in range(self.wave_nCM):
                    for n in range(self.wave_nCN):
                        yield getattr(J, mfma_name)(mfma_C[m,n],
                                            mfma_B[n, k, 0:1],
                                            mfma_A[m, k, 0:1],
                                            mfma_C[m,n])
                for m in range(self.wave_nCM):
                    for n in range(self.wave_nCN):
                        yield getattr(J, mfma_name)(mfma_C[m,n],
                                            mfma_B[n, k, 2:3],
                                            mfma_A[m, k, 2:3],
                                            mfma_C[m,n])
        def mfma_generator1():
            for k in range(self.wave_nCK//2, self.wave_nCK):
                for m in range(self.wave_nCM):
                    for n in range(self.wave_nCN):
                        yield getattr(J, mfma_name)(mfma_C[m,n],
                                            mfma_B[n, k, 0:1],
                                            mfma_A[m, k, 0:1],
                                            mfma_C[m,n])
                for m in range(self.wave_nCM):
                    for n in range(self.wave_nCN):
                        yield getattr(J, mfma_name)(mfma_C[m,n],
                                            mfma_B[n, k, 2:3],
                                            mfma_A[m, k, 2:3],
                                            mfma_C[m,n])

        def emit_mfma(generators:list, cycles:int):
            while cycles > 0:
                found = False
                for g in generators:
                    if next(g, None) is not None:
                        found = True
                        break
                if not found:
                    break
                cycles -= mfma_cycles

        cur_k = J.gpr("su32")
        cur_k[0] = 0
        k_loop_cnt = self.K//self.wg_K

        with J.While(cur_k[0] < k_loop_cnt):
            #for unroll in range(k_loop_cnt):

            k_offset[0] = 0 if skip_load else (k_offset[0] + self.wg_K * sizeof_bf16)
            loaderA.reset_offset(k_offset)
            loaderB.reset_offset(k_offset)

            mfma0 = mfma_generator0()
            mfma1 = mfma_generator1()

            # ds_read from last iteration is most-likely finished already
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            # ================= ds_read1(for current iteration) & mfma0 ==============
            for k in range(self.wave_nCK//2, self.wave_nCK):
                for m in range(self.wave_nCM):
                    ds_readA(m, k)
                    emit_mfma([mfma0], 16*2)
                for n in range(self.wave_nCN):
                    ds_readB(n, k)
                    emit_mfma([mfma0], 16*2)

            # ensure all waves has been finished reading LDS, so ds_write can overwrite it
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()

            # ================= ds_write + buffer_load ==============
            for r in range(num_prefetch_M):
                emit_mfma([mfma0, mfma1], 16)
                J.s_waitcnt(mod=f"vmcnt({num_prefetch_N + num_prefetch_M - 1})")
                loaderA.ds_write(r, ldsA)
                emit_mfma([mfma0, mfma1], 16*2)
                loaderA.prefetch(r)
                emit_mfma([mfma0, mfma1], 16*8)

            for r in range(num_prefetch_N):
                emit_mfma([mfma0, mfma1], 16)
                J.s_waitcnt(mod=f"vmcnt({num_prefetch_N + num_prefetch_M - 1})")
                loaderB.ds_write(r, ldsB)
                emit_mfma([mfma0, mfma1], 16*2)
                loaderB.prefetch(r)
                emit_mfma([mfma0, mfma1], 16*8)

            # enure mfma0 finished using part0 of mfma_A/mfma_B, before ds_read0 overwrites them
            # (most likely already empty and some part of mfma1 has been consumed)
            emit_mfma([mfma0], 999999999)
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()
            # ================= ds_read0(for next iteration) & mfma1 ==============
            for k in range(0,self.wave_nCK//2):
                for m in range(self.wave_nCM):
                    ds_readA(m, k)
                    emit_mfma([mfma1], 16*2)
                for n in range(self.wave_nCN):
                    ds_readB(n, k)
                    emit_mfma([mfma1], 16*2)
            emit_mfma([mfma1], 999999999)
            cur_k[0] = cur_k[0] + 1

        # write out
        J.s_waitcnt(mod=f"vmcnt(0)")
        if 0:
            J.debug_log(mfma_C[0], torch.float, "4h.4h.16v.4h")
            J.debug_log(mfma_C[1], torch.float, "4h.4h.16v.4h")
            J.debug_log(mfma_C[2], torch.float, "4h.4h.16v.4h")
            J.debug_log(mfma_C[3], torch.float, "4h.4h.16v.4h")

        # store to LDS for mem coealecing 
        J.free_lds(ldsA)
        J.free_lds(ldsB)

        if buff_c is None:
            # skip store c
            return mfma_C
        if 1:
            # swizzle-LDS to form better memory-coelascing VMEM
            # each warp has its own [mfma_M x self.wave_size_N] output buffer
            lds_out = J.alloc_lds(4 * self.mfma_MN * self.wave_size_N * sizeof_fp32)
            lds_warp_offset = warp_id * (self.mfma_MN * self.wave_size_N * sizeof_fp32)
            num_lanes_per_row = self.wave_size_N * sizeof_fp32 // sizeof_DWORDX4
            assert 64 % num_lanes_per_row == 0
            rows_per_read = 64//num_lanes_per_row
            assert rows_per_read > 0
            assert self.mfma_MN % rows_per_read == 0            

            vdata = J.gpr(4, "vu32")

            for m in range(self.wave_nCM):
                for n in range(self.wave_nCN):
                    row = lane_id % self.mfma_MN
                    if self.mfma_MN == 16:
                        col = lane_id // self.mfma_MN + n * (self.mfma_MN * sizeof_fp32 // sizeof_DWORDX4)
                        swizzle_col = swizzle(row, col) % (num_lanes_per_row)
                        vaddr_w = J.gpr((row) * (self.wave_size_N * sizeof_fp32) + \
                                        lds_warp_offset + \
                                        (swizzle_col) * sizeof_DWORDX4)
                        J.ds_write_b128(vaddr_w, mfma_C[m,n], mod=f"offset:{lds_out}")
                    else:
                        for ni in range(4):
                            noff = ni*4
                            col = lane_id // self.mfma_MN + (n*self.mfma_MN + ni*8) * sizeof_fp32 // sizeof_DWORDX4
                            swizzle_col = swizzle(row, col) % (num_lanes_per_row)
                            vaddr_w = J.gpr((row) * (self.wave_size_N * sizeof_fp32) + \
                                            lds_warp_offset + \
                                            (swizzle_col) * sizeof_DWORDX4)
                            J.ds_write_b128(vaddr_w, mfma_C[m,n,noff:noff+3],
                                            mod=f"offset:{lds_out}")

                voffset = J.gpr((lane_id % num_lanes_per_row) * sizeof_DWORDX4 + \
                                (lane_id // num_lanes_per_row + m*self.mfma_MN + warp_offset_m) * (self.N*sizeof_fp32) + \
                                (warp_offset_n*sizeof_fp32))

                for r in range(0, self.mfma_MN, rows_per_read):
                    row = lane_id // num_lanes_per_row + r
                    col = lane_id % num_lanes_per_row
                    swizzle_col = swizzle(row, col) % num_lanes_per_row
                    vaddr_r = J.gpr((swizzle_col) * sizeof_DWORDX4 + \
                                    (row) * (self.wave_size_N * sizeof_fp32) + \
                                    lds_warp_offset)
                    J.ds_read_b128(vdata, vaddr_r, mod=f"offset:{lds_out}")
                    J.s_waitcnt(mod=f"lgkmcnt(0)")
                    buff_c.store_dwordx4(vdata, voffset, 0)
                    voffset[0] = voffset[0] + rows_per_read * (self.N*sizeof_fp32)
        else:
            # store_dwordx4(self, vdata, voffset, soffset, offset12=0):
            for m in range(self.wave_nCM):
                for n in range(self.wave_nCN):
                    row = lane_id % self.mfma_MN + warp_offset_m + m*self.mfma_MN
                    col = lane_id // self.mfma_MN + n * (self.mfma_MN * sizeof_fp32 // sizeof_DWORDX4)
                    voffset = J.gpr(row * (N*sizeof_fp32) + warp_offset_n*sizeof_fp32 + col*sizeof_DWORDX4)
                    if self.mfma_MN == 16:
                        buff_c.store_dwordx4(mfma_C[m,n], voffset, 0)
                    elif self.mfma_MN == 32:
                        for ni in range(4):
                            noff = ni*4
                            buff_c.store_dwordx4(mfma_C[m,n,noff:noff+3], voffset, 0, offset12=ni*8*sizeof_fp32)
                    else:
                        assert 0


def tb_swizzle(J, block_1d_id:"sgpr", M:"sgpr", wg_M:int, wg_N:int, N:int, M01:int, GroupNum:int):
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

    # M01_adapt = (idx_M0 < M0 - M0_mod_M01) ? M01 : M0_mod_M01;
    M01_adapt = J.gpr("su32")
    J.SetMask("scc", idx_M0 < M0 - M0_mod_M01)
    J.s_cselect_b32(M01_adapt, M01, M0_mod_M01)

    idx_M00          = J.gpr(idx_M0 // M01)
    idx_M01          = J.gpr(idx_M0 - idx_M00 * M01)
    idx_N0_M01_local = J.gpr(idx_N0 + idx_M01 * N0)

    N_out           = J.gpr(idx_N0_M01_local // M01_adapt)
    idx_loc_mod_M01 = J.gpr(idx_N0_M01_local - N_out * M01_adapt)

    M_out = J.gpr(idx_loc_mod_M01 + idx_M00 * M01)
    return M_out, N_out

@pyhip.jit(with_debug_log=False, force_recompile=True)
def gemm_kernel(J, K, N, M01, GroupNum,
                mfma_MN, wave_size, wave_cnt, A_preshuffled, B_preshuffled,
                debug_warp, skip_load,
                pA:"void*", pB:"void*", pC:"float*", M:"int"):

    block_1d_id = J.blockIdx.x

    gemm = UGEMM(J, mfma_MN, wave_size, wave_cnt, K, N)
    if GroupNum <= 1 and M01 <= 1:
        N0 = J.div_up(N, gemm.wg_N)
        blk_m = J.gpr(block_1d_id // N0)
        blk_n = J.gpr(block_1d_id - blk_m*N0)
    else:
        blk_m, blk_n = tb_swizzle(J, block_1d_id, M, gemm.wg_M, gemm.wg_N, N, M01, GroupNum)

    sizeof_bf16 = 2
    sizeof_fp32 = 4
    if not skip_load:
        pA[:] = pA[:] + blk_m * (gemm.wg_M * K * sizeof_bf16)
        pB[:] = pB[:] + blk_n * (gemm.wg_N * K * sizeof_bf16)
        pC[:] = pC[:] + (blk_m * (gemm.wg_M * N * sizeof_fp32) + blk_n * (gemm.wg_N * sizeof_fp32))

    min_M1 = J.gpr("su32")
    J.s_min_u32(min_M1, blk_m * gemm.wg_M + gemm.wg_M, M)
    actual_wg_M = J.gpr(min_M1 - blk_m * gemm.wg_M)


    stride_bytes = K * sizeof_bf16
    total_wave_cnt = wave_cnt[0] * wave_cnt[1]
    swizzle_row_div = 1 if mfma_MN == 16 else 2
    if not A_preshuffled:
        loaderA = MFMA_DW4Loader(J, pA, actual_wg_M * K * sizeof_bf16,
                                gemm.wg_M, sizeof_bf16*gemm.wg_K, stride_bytes,
                                total_wave_cnt, swizzle_row_div, skip_load)
    else:
        loaderA = MFMA_DW4Loader_preshuffled(J, pA, actual_wg_M * K * sizeof_bf16, mfma_MN,
                                gemm.wg_M, sizeof_bf16*gemm.wg_K, stride_bytes,
                                total_wave_cnt, swizzle_row_div, skip_load)
    if not B_preshuffled:
        loaderB = MFMA_DW4Loader(J, pB, gemm.wg_N * K * sizeof_bf16,
                                gemm.wg_N, sizeof_bf16*gemm.wg_K, stride_bytes,
                                total_wave_cnt, swizzle_row_div, skip_load)
    else:
        loaderB = MFMA_DW4Loader_preshuffled(J, pB, actual_wg_M * K * sizeof_bf16, mfma_MN,
                                gemm.wg_N, sizeof_bf16*gemm.wg_K, stride_bytes,
                                total_wave_cnt, swizzle_row_div, skip_load)
    
    buff_c = J.Buffer(pC, actual_wg_M * N * sizeof_fp32)

    gemm.run(loaderA, loaderB, buff_c, M, debug_warp, skip_load)
    return


def pre_shuffle(x, mfma_MN):
    M, K = x.shape
    K_bytes = K * x.itemsize
    sizeof_DWORDX4 = 16
    mfma_K_lanes = 64 // mfma_MN
    mfma_K_L = sizeof_DWORDX4//x.itemsize
    mfma_K = mfma_K_lanes * mfma_K_L 

    assert M % mfma_MN == 0
    mfma_K_bytes = mfma_K_lanes * sizeof_DWORDX4
    assert K_bytes % mfma_K_bytes == 0

    x = x.reshape(M//mfma_MN, mfma_MN, K//mfma_K, mfma_K_lanes, mfma_K_L)
    x = x.permute(0,2,3,1,4)
    return x.contiguous()


@pytest.mark.parametrize("mfma_MN", [32,16])
@pytest.mark.parametrize("wave_size", [[128,128],[64,64],[32,32],[64,32]])
@pytest.mark.parametrize("wave_cnt", [[2,2],[1,2],[2,1],[1,1]])
def test_gemm(mfma_MN, wave_size, wave_cnt, A_preshuffled = False, B_preshuffled = False):
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    # following cases consumes too much VGPRs for prefetch_Areg/prefetch_Breg
    # due to lack of cooperative waves
    if mfma_MN == 16 and wave_size == [128,128] and wave_cnt in [[1,1],[1,2],[2,1]]:
        pytest.skip(f"Skipping combination: {mfma_MN}, {wave_size}, {wave_cnt}")

    perf_ratio = 0
    assert mfma_MN in [16, 32]
    assert len(wave_cnt) == 2
    assert len(wave_size) == 2
    # only test performance when setting is optimal
    if mfma_MN in [32, 16] and wave_size == [128, 128] and wave_cnt == [2, 2]:
        perf_ratio = 0.95
    
    props = torch.cuda.get_device_properties()
    num_CUs = props.multi_processor_count

    if num_CUs % 16 == 0:
        CU_rows = num_CUs//16
        CU_cols = 16
    elif num_CUs % 8 == 0:
        CU_rows = 8
        CU_cols = num_CUs//CU_rows
    else:
        assert 0, f"{num_CUs=}"

    debug_warp = 0
    skip_load = 0
    M01 = 8
    GroupNum = 8

    #mfma_MN = 16
    #wave_size = [128, 128]
    #wave_cnt = [2,2]
    wg_size = [wave_size[i]*wave_cnt[i] for i in range(2)]

    M,N,K = wg_size[0]*CU_rows*2,wg_size[1]*CU_cols*2,8192
    # M -= 14

    blk_cnt = (N//wg_size[1]) * ((M + wg_size[0] - 1)//wg_size[0])
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)
    out = torch.randn(M, N, dtype=torch.float)

    A1 = A
    B1 = B
    if A_preshuffled:
        A1 = pre_shuffle(A1, mfma_MN)
    if B_preshuffled:
        B1 = pre_shuffle(B1, mfma_MN)
        
    gemm_kernel([blk_cnt],[wave_cnt[0] * wave_cnt[1] * 64],
                K, N, M01, GroupNum,
                mfma_MN, wave_size, wave_cnt, A_preshuffled, B_preshuffled,
                debug_warp, skip_load,
                A1.data_ptr(), B1.data_ptr(), out.data_ptr(), M)
    ref_out = A @ B.t()

    debug_warp_m = debug_warp//2
    debug_warp_n = debug_warp%2
    m0 = debug_warp_m*64 
    n0 = debug_warp_n*64 
    #print(B[m0:m0+64, :64])
    #print(ref_out[m0:m0+64, n0:n0+64])

    cur_out = out.to(torch.bfloat16)
    if not torch.allclose(ref_out, cur_out, rtol=0.02, atol=0.02):
        print("====================ref_out")
        print(ref_out)
        print("====================cur_out")
        print(cur_out)
        idx = torch.where(torch.abs(ref_out - cur_out) > 1.5)
        if len(idx[0]):
            print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
        #assert 0
        acc_flag = False
    else:
        acc_flag = True

    DATA_CLONES = 40
    As = [torch.clone(A1) for _ in range(DATA_CLONES)]
    Bs = [torch.clone(B1) for _ in range(DATA_CLONES)]
    Cs = [torch.clone(out) for _ in range(DATA_CLONES)]
    di = 0

    tflops_res = []
    for i in range(10):
        di = (di + 1)%DATA_CLONES
        #di = 0
        with pyhip.cudaPerf(M*N*K*2, (M*K*2+K*N*2), name=f"gemm_{di}") as p0:
            gemm_kernel([blk_cnt],[256],
                        K, N, M01, GroupNum,
                        mfma_MN, wave_size, wave_cnt, A_preshuffled, B_preshuffled,
                        debug_warp, skip_load,
                        As[di].data_ptr(), Bs[di].data_ptr(), Cs[di].data_ptr(), M)
        tflops_res.append(p0.tflops())

    As = [torch.clone(A) for _ in range(DATA_CLONES)]
    Bs = [torch.clone(B) for _ in range(DATA_CLONES)]
    Cs = [torch.clone(out) for _ in range(DATA_CLONES)]
    tflops_ref = []
    for i in range(10):
        di = (di + 1)%DATA_CLONES
        with pyhip.cudaPerf(M*N*K*2, name=f"torch-linear-{di}") as p0:
            ref = torch.nn.functional.linear(As[di], Bs[di])
        tflops_ref.append(p0.tflops())

    torch.cuda.synchronize()
    if acc_flag:
        acc_info = "acc passed"
    else:
        color_id = 1
        color0 = f"\033[0;{30+(color_id % 8)}m"
        color1 = f"\033[0m"
        acc_info = color0 + "acc failed" + color1

    print(f"{M=} {N=} {K=} {mfma_MN=} {wave_size=} {wave_cnt=} {wg_size=} {acc_info}")
    avg_tflops_ref = sum(tflops_ref)/len(tflops_ref)
    avg_tflops_res = sum(tflops_res)/len(tflops_res)
    ratio = avg_tflops_res/avg_tflops_ref
    print(f"TFLOPS: {avg_tflops_res:.2f}/{avg_tflops_ref:.2f} ~ {ratio:.2f}")
    assert acc_flag is True
    assert ratio > perf_ratio

if __name__ == "__main__":
    #test_gemm(16, [128, 128], [1, 1])
    #assert 0
    #test_gemm(32, [128, 128], [2, 2], A_preshuffled = False, B_preshuffled = False) 
    test_gemm(16, [64, 64], [2, 2], A_preshuffled = False, B_preshuffled = True)
    #test_gemm(16, [128, 128], [2, 2], A_preshuffled = True, B_preshuffled = True)
    #test_gemm(32, [128, 128], [2, 2], A_preshuffled = True, B_preshuffled = True)
    assert 0
    test_gemm(32, [64, 128], [2, 2])
    test_gemm(32, [128, 64], [2, 2])
    test_gemm(32, [64, 64], [2, 2])
    test_gemm(32, [64, 64], [1, 2])
    test_gemm(32, [64, 64], [2, 1])
    test_gemm(32, [64, 64], [1, 1])
    test_gemm(32, [32, 32], [2, 2]) 

