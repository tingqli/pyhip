
# Dump all IRs

Passing `-mllvm -print-after-all` to clang (or just `-print-after-all` to llc), all IRs after every pass will be dumpped to console during compilation.

# LDS异步写入指令编译产生多余vmcnt的问题


```c++
// hipcc -x hip --offload-device-only --offload-arch=gfx942 -std=c++20 -I. -O2 -S -o p2.s ./p.cpp

__device__ int devfunc(int32x4_t rsrc, int * ptr0, int * ptr1) {
    auto ret0 = ptr1[threadIdx.x];
    auto ret1 = ptr1[threadIdx.x + 64*1];
    auto ret2 = ptr1[threadIdx.x + 64*2];
    auto ret3 = ptr1[threadIdx.x + 64*3];
    raw_buffer_load_lds(rsrc, (as3_uint32_ptr )ptr0, 4, threadIdx.x, 0,64*sizeof(int)*0,0);
    raw_buffer_load_lds(rsrc, (as3_uint32_ptr )ptr0, 4, threadIdx.x, 0,64*sizeof(int)*1,0);
    raw_buffer_load_lds(rsrc, (as3_uint32_ptr )ptr0, 4, threadIdx.x, 0,64*sizeof(int)*2,0);
    raw_buffer_load_lds(rsrc, (as3_uint32_ptr )ptr0, 4, threadIdx.x, 0,64*sizeof(int)*3,0);
    //__builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, (as3_uint32_ptr)dst0, 4, threadIdx.x, 0,0,0);
    return ret0 + ret1 + ret2 + ret3;
}
__global__ void test(int * src, int * dst, int offset) {
    __shared__ int lds[16*1024];
    BufferResource buff_src(src, 16*1024);
    int sum = 0;
    for(int i = 0 ; i < 1; i++) {
        sum += devfunc(buff_src.descriptor, lds + i*64*4, lds + offset + i*64*4);
    }
    dst[threadIdx.x] = sum;
}
/*
产生如下汇编，看起来，LDS的读写入口是有序的，因此不需要加任何等待，
先发起的ds_read不会受到后发起的 buffer_load的影响返回被修改过的结果
*/
	ds_read2st64_b32 v[2:3], v4 offset1:1
	ds_read2st64_b32 v[4:5], v4 offset0:2 offset1:3
	buffer_load_dword v0, s[0:3], 0 offen lds
	buffer_load_dword v0, s[0:3], 0 offen offset:256 lds
	buffer_load_dword v0, s[0:3], 0 offen offset:512 lds
	buffer_load_dword v0, s[0:3], 0 offen offset:768 lds
	s_waitcnt lgkmcnt(1)
	v_add_u32_e32 v0, v2, v3
	s_waitcnt lgkmcnt(0)
	v_add3_u32 v0, v0, v4, v5
	global_store_dword v1, v0, s[6:7]

/*
修改次序，先写入LDS,再从LDS读取数据:

__device__ int devfunc(int32x4_t rsrc, int * ptr0, int * ptr1) {
    raw_buffer_load_lds(rsrc, (as3_uint32_ptr )ptr0, 4, threadIdx.x, 0,64*sizeof(int)*0,0);
    raw_buffer_load_lds(rsrc, (as3_uint32_ptr )ptr0, 4, threadIdx.x, 0,64*sizeof(int)*1,0);
    raw_buffer_load_lds(rsrc, (as3_uint32_ptr )ptr0, 4, threadIdx.x, 0,64*sizeof(int)*2,0);
    raw_buffer_load_lds(rsrc, (as3_uint32_ptr )ptr0, 4, threadIdx.x, 0,64*sizeof(int)*3,0);
    auto ret0 = ptr1[threadIdx.x];
    auto ret1 = ptr1[threadIdx.x + 64*1];
    auto ret2 = ptr1[threadIdx.x + 64*2];
    auto ret3 = ptr1[threadIdx.x + 64*3];
    //__builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, (as3_uint32_ptr)dst0, 4, threadIdx.x, 0,0,0);
    return ret0 + ret1 + ret2 + ret3;
}

因为无法确定写入和读出是否针对相同地址或者重叠内存区域
所以必须加入 s_waitcnt vmcnt(0) 保证异步写入LDS已经完成，再发起 LDS读取
*/
	buffer_load_dword v0, s[0:3], 0 offen lds
	buffer_load_dword v0, s[0:3], 0 offen offset:256 lds
	buffer_load_dword v0, s[0:3], 0 offen offset:512 lds
	buffer_load_dword v0, s[0:3], 0 offen offset:768 lds
	v_lshlrev_b32_e32 v4, 2, v0
	v_lshl_add_u32 v2, s8, 2, v4
	s_waitcnt vmcnt(0)
	ds_read2st64_b32 v[0:1], v2 offset1:1
	ds_read2st64_b32 v[2:3], v2 offset0:2 offset1:3
	s_waitcnt lgkmcnt(1)
	v_add_u32_e32 v0, v0, v1
	s_waitcnt lgkmcnt(0)
	v_add3_u32 v0, v0, v2, v3
	global_store_dword v4, v0, s[6:7]


/*
仅仅把 ptr1 加上 restrict 修饰 (ptr0的 restrict 修饰加不加不影响结果汇编)
__device__ int devfunc(int32x4_t rsrc, int * ptr0, int * __restrict__ ptr1) {
产生的汇编开始先从内存写入LDS，然后再从LDS读出，因为 restrict 语义已经保证 二者不会
访问相同地址的内容，因此先发起写入，可以利用 ds_read 更好的并行。

但是这里的 s_waitcnt vmcnt(0) 感觉是多余的。因为 restrict 语义已经保证了，从LDS读取无需等待
这涉及到使用该异步LDS读入进行高性能计算的核心逻辑，因此必须找到优化掉这个多余 s_waitcnt vmcnt(0) 的方法。

*/

	buffer_load_dword v0, s[0:3], 0 offen lds
	buffer_load_dword v0, s[0:3], 0 offen offset:256 lds
	buffer_load_dword v0, s[0:3], 0 offen offset:512 lds
	buffer_load_dword v0, s[0:3], 0 offen offset:768 lds
	v_lshlrev_b32_e32 v4, 2, v0
	v_lshl_add_u32 v2, s8, 2, v4
	s_waitcnt vmcnt(0)
	ds_read2st64_b32 v[0:1], v2 offset1:1
	ds_read2st64_b32 v[2:3], v2 offset0:2 offset1:3
	s_waitcnt lgkmcnt(1)
	v_add_u32_e32 v0, v0, v1
	s_waitcnt lgkmcnt(0)
	v_add3_u32 v0, v0, v2, v3
	global_store_dword v4, v0, s[6:7]


/*
该错误似乎在clang-22中被修复了， s_waitcnt vmcnt(0) 只有不存在 restrict 修饰符的情况下才会被
插入。 
*/

	buffer_load_dword v1, s[0:3], 0 offen lds
	buffer_load_dword v1, s[0:3], 0 offen offset:256 lds
	buffer_load_dword v1, s[0:3], 0 offen offset:512 lds
	buffer_load_dword v1, s[0:3], 0 offen offset:768 lds
	v_lshlrev_b32_e32 v1, 2, v1
	v_lshl_add_u32 v1, s8, 2, v1
	ds_read2st64_b32 v[2:3], v1 offset1:1
	ds_read2st64_b32 v[4:5], v1 offset0:2 offset1:3
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(1)
	v_add_u32_e32 v1, v2, v3
	s_waitcnt lgkmcnt(0)
	v_add3_u32 v1, v1, v4, v5
	global_store_dword v0, v1, s[6:7]

```


# 如何优化多余的vmcnt等待


llvm的执行选项可以通过clang传入:`-mllvm -print-after-all`，这会打印全部的IR，即使发布版也有，非常利于调试。


```python
# 在最终的log里面搜索第一次出现 isel （instruction selection）关键字可以发现，llvm IR在restrict存在和不存在就存在区别：没有 restrict 时，按照原始代码次序，先触发 LDS DMA，再从LDS加载数据到寄存器，因为二者可能引用重叠内存地址, 并且两个指令都没有 alias 相关的属性
'''
  %12 = getelementptr inbounds i32, ptr addrspace(3) %9, i32 %11
  tail call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> noundef %8, ptr addrspace(3) noundef @_ZZ4testPiS_iE3lds, i32 noundef 4, i32 noundef %11, i32 noundef 0, i32 noundef 0, i32 noundef 0) #3
  %13 = load i32, ptr addrspace(3) %12, align 4, !tbaa !6
'''

# 带有 restrict 时，先从LDS读取再触发LDS DMA加载数据，因为二者数据无关，并且先出现的指令 带有 alias.scope, 后出现的带有 noalias, 这一点跟 lds-dma-waits.ll 中的测试用例不同，但是我们通过构造类似 lds-dma-waits.ll 中的用例，就是声明多个 __shared__ 数组的方式，发现虽然isel之前的llvm IR中没有alias信息，但是在(amdgpu-lower-module-lds)步骤之后却出现了正确的 alias.scope 和 noalias 属性，每个lds数组被生成了一个独立的 alias.scope, 以供后继分析使用

'''
  %21 = getelementptr inbounds i32, ptr addrspace(3) %18, i32 %20
  %22 = load i32, ptr addrspace(3) %21, align 4, !tbaa !8, !alias.scope !12
  tail call void @llvm.experimental.noalias.scope.decl(metadata !12)
  tail call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> noundef %17, ptr addrspace(3) noundef @llvm.amdgcn.kernel._Z4testPiS_i.lds, i32 noundef 4, i32 noundef %20, i32 noundef 0, i32 noundef 0, i32 noundef 0) #7, !noalias !12
'''

  # 但是这个次序在 Machine Instruction Scheduler (machine-scheduler) 之后反转，又变回了先触发LDS DMA，再读LDS,这也没问题，越早触发越早完成，反正restrict语义保证了二者引用的内存地址不重叠，谁先谁后无所谓, 但是这里出现一点问题就是按理说应该先出现的指令先声明alias.scope，但是此处指令调度之后，后出现的指令才有scope!!!

'''
  320B	  BUFFER_LOAD_DWORD_LDS_OFFEN %14:vgpr_32, %12:sgpr_128, 0, 0, 0, 0, implicit $exec, implicit $m0 :: (dereferenceable load (s32) from `ptr addrspace(1) poison`, align 1, !noalias !12, addrspace 1), (dereferenceable store (s32) into @llvm.amdgcn.kernel._Z4testPiS_i.lds, align 1, !noalias !12, addrspace 3)
  328B	  %16:vgpr_32 = V_LSHLREV_B32_e32 2, %14:vgpr_32, implicit $exec
  336B	  %17:vgpr_32 = V_LSHL_ADD_U32_e64 %4:sreg_32_xm0_xexec, 2, %16:vgpr_32, implicit $exec
  344B	  %19:vgpr_32 = DS_READ_B32_gfx9 %17:vgpr_32, 0, 0, implicit $exec :: (load (s32) from %ir.21, !tbaa !8, !alias.scope !12, addrspace 3)
'''
  # 对比第一次出现 S_WAITCNT 3952 （也就是s_waitcnt vmcnt(0)的编码）的pass，可知是`SIInsertWaitcnts.cpp`生成的wait:
'''
  BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr1, killed $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec, implicit $m0 :: (dereferenceable load (s32) from `ptr addrspace(1) poison`, align 1, !noalias !12, addrspace 1), (dereferenceable store (s32) into @llvm.amdgcn.kernel._Z4testPiS_i.lds, align 1, !noalias !12, addrspace 3)
  renamable $vgpr1 = V_LSHLREV_B32_e32 2, killed $vgpr1, implicit $exec
  renamable $vgpr1 = V_LSHL_ADD_U32_e64 killed $sgpr8, 2, killed $vgpr1, implicit $exec
  S_WAITCNT 3952
  renamable $vgpr1 = DS_READ_B32_gfx9 killed renamable $vgpr1, 0, 0, implicit $exec :: (load (s32) from %ir.21, !tbaa !8, !alias.scope !12, addrspace 3)
'''
#  这一步中，根据 ds_read 中的 !alias.scope !12 以及 BUFFER_LOAD_DWORD_LDS_OFFEN 中的 !noalias !12 应该可以肯定二者无关。
#  从而不必插入s_waitcnt vmcnt(0)进行等待。

```

通过跟踪插入waitcnt的代码在`restrict`修饰符存在和不存在两种情况下的不同执行路径发现，在restrict存在时，`ds_read`指令会跳过一些逻辑不会产生vmcnt0的wait，而关键代码是下面的commit添加的：

```txt

021def6c2278fd932d18b4d891c2e75c1d8e6f1d

[AMDGPU] Use alias info to relax waitcounts for LDS DMA (#74537)

LDA DMA loads increase VMCNT and a load from the LDS stored must wait on
this counter to only read memory after it is written. Wait count
insertion pass does not track memory dependencies, it tracks register
dependencies. To model the LDS dependency a pseudo register is used in
the scoreboard, acting like if LDS DMA writes it and LDS load reads it.

This patch adds 8 more pseudo registers to use for independent LDS
locations if we can prove they are disjoint using alias analysis.

Fixes: SWDEV-433427

这里提到了一些关键点和scoreboard概念：
 - 修改前只有track register dependencies, 例如load指令读入数据到vgpr寄存器，后面又对该寄存器进行ALU计算，此时该vgpr在被ALU指令使用之前应该插入s_waitcnt指令保证数据就位，但是应该wait多少个cnt，则取决指令调度/排布阶段，在使用该寄存器和读取该寄存器之间插入了多少条其他vm指令。这些信息就是编译器内部维护的所谓scoreboard完成的。
 - 修改增加了memory dependencies的tracking逻辑，因为buffer_load_to_lds指令写入的是LDS内存而非寄存器，后继读取LDS内存到寄存器的指令潜在会需要等待buffer_load_to_lds的目的LDS内存区域ready，这个等待也是通过插入合适的wait vmcnt指令完成，但是跟寄存器不同，此时写入的是LDS，而且地址还可能未知（alias）

但是该commit出问题的clang中也包含，因此不是该commit解决的问题
对比了无问题的最新代码和有问题的分支发现下面的commit才是fix，
该commit基本上就是说，只需要ds_read中存在 alias.scope 修饰符
哪怕在`LDSDMAStores`中找不到匹配项，仍然不插入vmcnt(0).
这里提到一个 `inter-thread dependences` 有点无法理解。

e75f586b813a081cffcafb8b5d34b5547e52e548

[AMDGPU] Relax lds dma waitcnt with no aliasing pair (#131842)

If we cannot find any lds DMA instruction that is aliased by some load
from lds, we will still insert vmcnt(0). This is overly cautious since
handling inter-thread dependences is normally managed by the memory
model instead of the waitcnt pass, so this change updates the behavior
to be more inline with how other types of memory events are handled.

这个commit也修改了下面的测试用例，从测试用例可以更清楚的了解该修改的主要逻辑
应该可以用debug模式跟踪测试用例的编译器的表现来更深入的理解这些改动。

llvm\test\CodeGen\AMDGPU\lds-dma-waits.ll (e75f586)


; There are 8 pseudo registers defined to track LDS DMA dependencies.
; When exhausted we default to vmcnt(0). <============== removed

; GCN-LABEL: {{^}}buffer_load_lds_dword_10_arrays:
; GCN-COUNT-10: buffer_load_dword
; GCN: s_waitcnt vmcnt(8)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(7)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(6)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(5)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(4)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(3)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(2)
; GCN-NOT: s_waitcnt vmcnt
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(0)      <============== removed
; GCN: ds_read_b32
define amdgpu_kernel void @buffer_load_lds_dword_10_arrays(<4 x i32> %rsrc, i32 %i1, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, i32 %i8, i32 %i9, ptr addrspace(1) %out) {
main_body:
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.0, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.1, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.2, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.3, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.4, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.5, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.6, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.7, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.8, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.9, i32 4, i32 0, i32 0, i32 0, i32 0)
  %gep.0 = getelementptr float, ptr addrspace(3) @lds.0, i32 %i1
  %gep.1 = getelementptr float, ptr addrspace(3) @lds.1, i32 %i2
  %gep.2 = getelementptr float, ptr addrspace(3) @lds.2, i32 %i2
  %gep.3 = getelementptr float, ptr addrspace(3) @lds.3, i32 %i2
  %gep.4 = getelementptr float, ptr addrspace(3) @lds.4, i32 %i2
  %gep.5 = getelementptr float, ptr addrspace(3) @lds.5, i32 %i2
  %gep.6 = getelementptr float, ptr addrspace(3) @lds.6, i32 %i2
  %gep.7 = getelementptr float, ptr addrspace(3) @lds.7, i32 %i2
  %gep.8 = getelementptr float, ptr addrspace(3) @lds.8, i32 %i2
  %gep.9 = getelementptr float, ptr addrspace(3) @lds.9, i32 %i2
  %val.0 = load float, ptr addrspace(3) %gep.0, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.1 = load float, ptr addrspace(3) %gep.1, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.2 = load float, ptr addrspace(3) %gep.2, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.3 = load float, ptr addrspace(3) %gep.3, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.4 = load float, ptr addrspace(3) %gep.4, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.5 = load float, ptr addrspace(3) %gep.5, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.6 = load float, ptr addrspace(3) %gep.6, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.7 = load float, ptr addrspace(3) %gep.7, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.8 = load float, ptr addrspace(3) %gep.8, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.9 = load float, ptr addrspace(3) %gep.9, align 4
  %out.gep.1 = getelementptr float, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr float, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr float, ptr addrspace(1) %out, i32 3
  %out.gep.4 = getelementptr float, ptr addrspace(1) %out, i32 4
  %out.gep.5 = getelementptr float, ptr addrspace(1) %out, i32 5
  %out.gep.6 = getelementptr float, ptr addrspace(1) %out, i32 6
  %out.gep.7 = getelementptr float, ptr addrspace(1) %out, i32 7
  %out.gep.8 = getelementptr float, ptr addrspace(1) %out, i32 8
  %out.gep.9 = getelementptr float, ptr addrspace(1) %out, i32 9
  store float %val.0, ptr addrspace(1) %out
  store float %val.1, ptr addrspace(1) %out.gep.1
  store float %val.2, ptr addrspace(1) %out.gep.2
  store float %val.3, ptr addrspace(1) %out.gep.3
  store float %val.4, ptr addrspace(1) %out.gep.4
  store float %val.5, ptr addrspace(1) %out.gep.5
  store float %val.6, ptr addrspace(1) %out.gep.6
  store float %val.7, ptr addrspace(1) %out.gep.7
  store float %val.8, ptr addrspace(1) %out.gep.8
  store float %val.9, ptr addrspace(1) %out.gep.9
  ret void
}


下面是新引入的测试用例，LDS DMA写入的区域(scope)和load的区域完全无关, generateWaitcntInstBefore因此就会比对之前的检查

define amdgpu_kernel void @global_load_lds_no_alias_ds_read(ptr addrspace(1) nocapture %gptr, i32 %i1, i32 %i2, ptr addrspace(1) %out) {
; GFX9-LABEL: global_load_lds_no_alias_ds_read:
; GFX9: global_load_dword
; GFX9: global_load_dword
; GFX9: s_waitcnt vmcnt(1)
; GFX9-NOT: s_waitcnt vmcnt(0)
; GFX9: ds_read_b32
; GFX9: s_waitcnt vmcnt(0)
; GFX9: ds_read_b32
; GFX9: s_endpgm
body:
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) @lds.0, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) @lds.1, i32 4, i32 4, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 3953)
  %gep.0 = getelementptr float, ptr addrspace(3) @lds.2, i32 %i1
  %val.0 = load float, ptr addrspace(3) %gep.0, align 4
  call void @llvm.amdgcn.s.waitcnt(i32 3952)
  %gep.1 = getelementptr float, ptr addrspace(3) @lds.3, i32 %i2
  %val.1 = load float, ptr addrspace(3) %gep.1, align 4
  %tmp = insertelement <2 x float> poison, float %val.0, i32 0
  %res = insertelement <2 x float> %tmp, float %val.1, i32 1
  store <2 x float> %res, ptr addrspace(1) %out
  ret void
}
```


```c++

# *** IR Dump After SI insert wait instructions (si-insert-waitcnts) ***:
# Machine code for function _Z4testPiS_i: NoPHIs, TracksLiveness, NoVRegs, TiedOpsRewritten, 

BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec, implicit $m0 :: (dereferenceable load (s32) from `ptr addrspace(1) poison`, align 1, !noalias !13, addrspace 1), (dereferenceable store (s32) into @llvm.amdgcn.kernel._Z4testPiS_i.lds, align 1, !noalias !13, addrspace 3)

renamable $vgpr2_vgpr3 = DS_READ2ST64_B32_gfx9 renamable $vgpr1, 0, 1, 0, implicit $exec :: (load (s32) from %ir.20, !tbaa !9, !alias.scope !13, addrspace 3), (load (s32) from %ir.22, !tbaa !9, !alias.scope !13, addrspace 3)


```


WaitcntBrackets
现在我们对问题的大致情况有所了解，但是还是缺乏一个全局概念，理解函数`SIInsertWaitcnts::run`的行为就是最后一步。一个比读代码更为非常有效的，直接获得函数行为的感性认识的方式，就是观察代码的行为，而作者非常清楚这一点，下面的参数可以产生我们感兴趣的pass的行为debug-log `-debug-only=si-insert-waitcnts -print-after-all`, 这个log还需要配合一个非常简单的测试用例，我们直接从作者构造的用例`lds-dma-waits.ll`中摘取一个函数即可。

跟llvm IR有异曲同工之妙，log也是高度抽象的，简洁的表达，因此需要观察产生log的地方的代码才能理解其含义，我们发现打印的是核心数据结构`WaitcntBrackets`的内容，以及发生重大修改，也就是插入新s_waitcnt指令的地方。

每遇到一条会引起 vmcnt/lgkmcnt/... 等计数器增加的指令类型时，编译器就会模拟硬件执行期行为，对`WaitcntBrackets`中的计数器进行增加，同时记录是哪些目标寄存器（对加载到LDS的指令，引入了8个伪寄存器）会被修改，例如`s[8:15]=S_LOAD_DWORDX8`就会引起`LGKM_CNT`增加1并由编译器记录`s[8:15]`会被更新。后面遇到需要访问`s[8:15]`寄存器的指令时，就知道需要插入合适的s_waitcnt了。

每个寄存器维护了一个score，代表的是其是被第几个issue（且未完成）的指令所写入的，每次插入s_waitcnt指令要求count降低时，低score的寄存器（也就是较早发起读入的寄存器）就优先被完成。

s0 = s_load_dword : lgkm_cnt(1) 0:s0
s1 = s_load_dword : lgkm_cnt(2) 0:s0 1:s1
s2 = s_load_dword : lgkm_cnt(3) 0:s0 1:s1 2:s2

s_waitcnt lgkm_cnt(1)  等待lgkm_cnt小于等于1，因此score小于等于1的寄存器就完成了
                    lgkm_cnt <=1 1:s2  


Summary:
Instead of storing the "score" (last time point) of the various relevant
events, only store whether an event is pending or not.

This is sufficient, because whenever only one event of a count type is
pending, its last time point is naturally the upper bound of all time
points of this count type, and when multiple event types are pending,
the count type has gone out of order and an s_waitcnt to 0 is required
to clear any pending event type (and will then clear all pending event
types for that count type).


阅读代码的方法，收集简单事实，理解变量含义：
 - 每种counter有自己的LB/UB: getScoreUB(T)/setScoreUB(T, CurrScore);
 - 每个寄存器，每种counter有自己的score: getRegScore(RegNo, T)/setScoreByInterval(Interval, T, CurrScore);
 - counter的UB在 updateEventWaitcntAfter 函数中，根据指令类型增长1，这就是在追踪硬件counter对相应指令issue阶段加1的行为，并且搜索代码发现UB永远不减少，所以**UB的含义就是该类型指令的总的issue的数量**。
 - counter的LB在 generateWaitcnt/applyWaitcnt 函数中，根据生成的s_waitcnt的类型和等待数量Count, 修改为`(UB-Count)`, 根据`s_waitcnt type(Count)`指令的含义，**LB可以理解为已经完成的此类型的指令的数量**。
 - 从注释中得知，WaitcntBrackets中的bracket指的就是`[LB, UB]`之间的范围

 - `[LB,UB]`以及寄存器Score的最主要作用就是**判断何时需要插入s_waitcnt，插入多少**，为此：
  - updateByEvent()中：
   1. (对应当前指令类型的)UB自加1 (UB永不减少)
   2. 检查指令目标寄存器(Inst.defs())，利用UB决定目标寄存器的score, 一旦score生成就不再修改，因此**寄存器的Score也就是代表了，会修改其值的指令的issue的序号或者时间点**
   3. 检查指令的内存操作数(Inst.memoperands())，如果指令可能会写入LDS--不论是从寄存器还是直接从外存DMA搬运，则根据alias的scope分配一个伪目标寄存器表达LDS目标区域并设置其score，并且记录alias信息到LDSDMAStores的一个slot中，另外不论alias信息是否存在，都用FIRST_LDS_VGPR记录最新的写入LDS的时间点---也即是score.

  - generateWaitcntInstBefore()/determineWait()中：
   1. 指令需要读取的src寄存器：检查其score是否在`(LB,UB]`之间，也就是会被已issue但是尚未完成的指令所修改，是的话，就预备插入s_waitcnt，而等待的cnt是 `UB-Score`，也就是保证score对应的时间点的指令执行完成。
   2. 指令的内存操作数(Inst.memoperands())：如果是有可能从LDS读取--也就是内存操作数目的AddrSpace属于LDS或者FLAT(编译期无法确定地址空间)，则检查其alias.scope，如果跟LDSDMAStores中某个LDS区域重叠，就使用该区域的score确定需要插入等待的cnt，如果没有重叠就不插入任何cnt，如果没有alias.scope则根据FIRST_LDS_VGPR进行最保守的等待。
    
 - WaitcntBrackets::print()中打印出来的寄存器的score其实是相对时间`RegScore - (LB + 1)`，小于等于LB的不会打印，因此看到的debug log中的值都是`[0,UB-LB)`范围内的，代表目前尚未完成的指令中issue的相对序号。





















感叹：复杂事务结构化，简单化，模块化，代码行为可视化
理解基本概念才是最核心的，因为相同的基本概念会以不同的形式实现出来。例如基本的`BasicBlock`概念在llvm就实现了多次，例如CodeGen中的`MachineBasicBlock`:
 - `Predecessor`: 任何会跳转到该BB的BB
 - `Sucessor`：任何会从该BB跳转到的BB

llvm/include/llvm/Support/GenericLoopInfo.h：

高级语言使用`for/while/do-while`构造的结构化的loop或者嵌套loop，在汇编级别有着特定的结构：由一个入口组成，这个入口负责初始化循环变量，因为最内层循环体仍然可能包含`if/continue/break`，因此通常的最内层循环体可能是多个BB组成，除了循环体最末尾的BB,组成循环体的其他的BB，也可能跳转回入口形成循环。



llvm/include/llvm/PassSupport.h

编译pass需要以一种有机方式组织，例如passX依赖passA, 而passY也依赖passA, 而passZ依赖passX和passY, 则最终passA先运行，然后是X,Y,最后是Z，也就是pass之间的依赖链跟函数调用依赖链不同，passA不会运行多次。所有的pass也不是无关的或者依赖人为决定运行次序的。当passX运行时(`runOnMachineFunction`)，passA的运行保证已经结束，其分析结果数据可以使用`getAnalysis`获取。


MachineLoopInfo *MLI
MachinePostDominatorTree *PDT
AliasAnalysis *AA

SIInsertWaitcnts::run

因为需要逐条处理指令，这些pass都是高度状态化的，很多成员变量用以维护大量生命周期不明的信息。而且很多if判断用来应对各种可能出现的情况，不相关的逻辑会交织在一起。因此通过逐行阅读代码理解这些逻辑是不现实的，而应该针对一个测试用例使用debug手段跟踪命中的逻辑，理解这些命中的逻辑。



