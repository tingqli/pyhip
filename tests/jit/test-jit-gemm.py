import pyhip

'''
jit 程序也能实现面向对象吗？
 C++代码包含更多语法可以产生复杂的运行期和编译期行为
 jit程序宿主对象（此处是python对象）也可以模仿这种行为
 并且高性能C++模板对象更多依赖编译期行为，jit也应如此

 传统意义的对象包含了相关功能的运行时资源（数据成员：寄存器或者LDS资源）和相关方法
 jit对象也是如此，只是其生命周期维护涉及到寄存器资源管理，其方法调用都会直接emit代码完成相关功能
 一些比较有用的功能都需要临时寄存器，这个方式要想成功必须解决临时寄存器分配释放问题

 一种方式是，按照目前的方法，分配的寄存器都是从不释放，不设置上限，绝对不存在冲突，但是也大量浪费，
 可以理解为类似虚拟寄存器，需要一个后期寄存器重分配过程把他们最有效的映射为物理寄存器。

 首先需要一个按照BB(基本块 BasicBlock)方式组织的代码执行次序关系图结构，然后按照这个结构遍历BB
 保证所有BB都能遍历到。并且一定是按照执行次序，但是多孩子分支BB（比如if/else导致的）不能确定谁会先被执行
 二者也是不确定次序的，或者从寄存器生命周期维护角度出发，认为二者是同时的。分支BB还可能进一步分支，最终
 一定会重新聚合。经过这样的分析，给每个BB都赋值一个时间戳，无法判断先后的就赋予相同时间戳。
 根据时间戳遍历，找到每个虚拟寄存器第一次出现的地方和最后出现的地方，就是它的生命周期，然后根据生命周期
 遍历摆放方式。

'''
class Buffer:
    def __init__(self, J):
        self.J = J
        self.desc = J.new_gpr('s', 4, align=4)
        self.base = self.desc[0:1]
        self.range = self.desc[2]
        self.config = self.desc[3]
        J.s_mov_b32(self.config, 0x00020000)

    def setup(self, sgpr_args_base, offset_base:int, offset_count:int, sizeof_ele:int):
        J = self.J
        J.s_load_dwordx2(self.base, sgpr_args_base, offset_base)
        J.s_load_dword(self.range, sgpr_args_base, offset_count)
        J.s_waitcnt(mod=f"lgkmcnt({0})")
        if sizeof_ele != 1:
            J.s_mulk_i32(self.range, sizeof_ele)

    def load_dwordx4(self, vdst, voffset, soffset, offset12=0):
        # vdst,     vaddr,           srsrc, soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset12:{offset12}"
        self.J.buffer_load_dwordx4(vdst, voffset, self.desc, soffset, mod=mod)

    def store_dwordx4(self, vdata, voffset, soffset, offset12=0):
        # vdata,    vaddr,        srsrc,  soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset12:{offset12}"
        self.J.buffer_store_dwordx4(vdata, voffset, self.desc, soffset, mod=mod)

'''
jit 程序也能表达 Tile 吗？
'''

def gemm_kernel(J):
    p_kargs = J.new_gpr('s',[0,1])
    threadId_X = J.new_gpr('v',[0,0])
    vtemp =J.new_gpr('v',4) 
    temp = J.new_gpr('s',2)
    K = J.new_gpr('s',1)

    #J.s_load_dwordx2(temp, p_kargs, 0)
    #J.s_load_dword(K, p_kargs, 8)
    #J.s_waitcnt(mod=f"lgkmcnt({0})")
    #J.s_mul_i32(temp[0], K, 4)

    buff_a = Buffer(J)
    buff_a.setup(p_kargs, 0, 8, sizeof_ele=4)

    v4 = J.new_gpr('v', 4, align=2)
    
    J.v_lshl_add_u32(vtemp[0], threadId_X, 2, 0)
    buff_a.load_dwordx4(v4, vtemp[0], 0)
    J.s_waitcnt(mod=f"vmcnt(0)")

    for i in range(4):
        J.v_add_u32(v4[i], 102, v4[i])

    buff_a.store_dwordx4(v4, vtemp[0], 0)
    J.s_waitcnt(mod=f"vmcnt(0)")
    return
    acc = J.new_gpr("a", 4)
    for i in range(4):
        J.v_accvgpr_write_b32(acc[i], 0)

    
    J.s_waitcnt(mod=f"lgkmcnt({0})")
    #T.v_mov_b32(v[2], s[2])
    # v_mov_b32(v3, s3)
    #with J.BB():
    #    J.v_lshl_add_u32(v[2], v[0], 2, v[2])
    s_idx = J.new_gpr('s',1)
    s_temp = J.new_gpr('s',2)

    J.s_mov_b32(s_idx, 0)

    J.Label("bb0")

    #J.s_lshl_b32(s_temp[1],1, s_idx)

    J.s_lshl_b32(s_temp[0],s_idx,2)
    #s_temp[0] = s_idx[0] << 2

    J.s_add_u32(s_temp[0], pA[0], s_temp[0])
    J.s_addc_u32(s_temp[1], pA[1], 0)

    J.s_store_dword(K, s_temp, 0, mod="glc")

    J.s_addk_i32(s_idx, 1)
    J.s_cmp_lt_i32(s_idx, 32)
    J.s_cbranch_scc0(mod="bb1%=")
    J.s_branch(mod="bb0%=")

    J.Label("bb1")

    # flat_store_dword(v[2:3], v0)

jit = pyhip.JIT()
gemm_kernel(jit)
kernel = jit.build("test(int* p, int K)")

import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

A = torch.ones(64, dtype=torch.int)
print(A)
kernel([1],[64], A.data_ptr(), 64)
torch.cuda.synchronize()
print(A)