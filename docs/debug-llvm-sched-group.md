# Dump all IRs

Passing `-mllvm -print-after-all` to clang (or just `-print-after-all` to llc), all IRs after every pass will be dumpped to console during compilation.

reproduce 

# Basics on Instruction Scheduler

 - Data_hazards: https://en.wikipedia.org/wiki/Data_dependency#Data_hazards
    - read after write (RAW), a true dependency
    - write after read (WAR), an anti-dependency
    - write after write (WAW), an output dependency
    - read after read (RAR), a false dependency
 - compile-time instruction scheduling for in-order core: https://myhsu.xyz/llvm-machine-scheduler/


# Relevant Commits

Search `llvm.amdgcn.sched.group.barrier` `amdgcn_sched_group_barrier` in llvm-project, found commit:

 - [[AMDGPU] Add amdgcn_sched_group_barrier builtin](https://github.com/llvm/llvm-project/commit/f5b21680d1221d7acaa1b174d0b86fa907c71eb8)
    - Major changes : llvm/lib/Target/AMDGPU/AMDGPUIGroupLP.cpp
    - test case: llvm/test/CodeGen/AMDGPU/llvm.amdgcn.sched.group.barrier.ll
 - [[MLIR][ROCDL] Added SchedGroupBarrier and IglpOpt ops](https://github.com/llvm/llvm-project/pull/112237)
    - MLIR based AI compiler can call rocdl dialect for this function: https://github.com/triton-lang/triton/blob/main/test/TritonGPU/amd/amd-schedule-hint.mlir

# __builtin_amdgcn_sched_group_barrier

To reach maximum performance with limited resources, often all VGPRs/AccGPRs are being exclusively allocted to a single warp for an SIMD, in this case, thread-level-parallelsim doesn't work, and we have to fall-back to instruction-level-parallelsim([ILP](https://en.wikipedia.org/wiki/Instruction-level_parallelism)).

GPU HW has limited power for ILP comparing to CPU(x86):
 - Out-of-order execution
 - Superscalar

```llvm
; /opt/rocm/llvm/bin/llc -march=amdgcn  -mcpu=gfx942 < sched-group.ll

define amdgpu_kernel void @test_sched_group_barrier_simple_pipeline(<8 x i32> addrspace(1)* noalias %in, <8 x i32> addrspace(1)* noalias %out) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #2
  %gep1 = getelementptr <8 x i32>, <8 x i32> addrspace(1)* %in, i32 %tid
  %load = load <8 x i32>, <8 x i32> addrspace(1)* %gep1
  %mul = mul <8 x i32> %load, %load
  %gep2 = getelementptr <8 x i32>, <8 x i32> addrspace(1)* %out, i32 %tid
  store <8 x i32> %mul, <8 x i32> addrspace(1)* %gep2

  call void @llvm.amdgcn.sched.group.barrier(i32 32, i32 1, i32 0) ; VMEM read
  call void @llvm.amdgcn.sched.group.barrier(i32 2,  i32 4, i32 0)  ; VALU
  call void @llvm.amdgcn.sched.group.barrier(i32 64, i32 1, i32 0) ; VMEM write

  call void @llvm.amdgcn.sched.group.barrier(i32 32, i32 1, i32 0) ; VMEM read
  call void @llvm.amdgcn.sched.group.barrier(i32 2,  i32 4, i32 0)  ; VALU
  call void @llvm.amdgcn.sched.group.barrier(i32 64, i32 1, i32 0) ; VMEM write
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #2
declare void @llvm.amdgcn.sched.group.barrier(i32, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone speculatable }
```

check LLVM behaviour:

```bash
/opt/rocm/llvm/bin/llc -march=amdgcn  -mcpu=gfx942 -print-after-all < sched.ll > sched.s 2> sched.ir
```

check `sched.ir`, we found that after `machine-scheduler` pass, the instructions are scheduled according to `llvm.amdgcn.sched.group.barrier`.

```s
# *** IR Dump After Machine Instruction Scheduler (machine-scheduler) ***:
# Machine code for function test_sched_group_barrier_simple_pipeline: NoPHIs, TracksLiveness, TiedOpsRewritten
Function Live Ins: $vgpr0 in %0, $sgpr4_sgpr5 in %3

0B	bb.0 (%ir-block.0):
	  liveins: $vgpr0, $sgpr4_sgpr5
16B	  %3:sgpr_64(p4) = COPY $sgpr4_sgpr5
32B	  %0:vgpr_32(s32) = COPY $vgpr0
48B	  early-clobber %46:sgpr_128 = S_LOAD_DWORDX4_IMM_ec %3:sgpr_64(p4), 36, 0 :: (dereferenceable invariant load (s128), align 4, addrspace 4)
64B	  %11:vgpr_32 = V_AND_B32_e32 1023, %0:vgpr_32(s32), implicit $exec
80B	  %13:vgpr_32 = nuw nsw V_LSHLREV_B32_e32 5, %11:vgpr_32, implicit $exec
112B	  %19:vreg_128_align2 = GLOBAL_LOAD_DWORDX4_SADDR %46.sub0_sub1:sgpr_128, %13:vgpr_32, 0, 0, implicit $exec :: (load (s128) from %ir.gep1, align 32, addrspace 1)
128B	  undef %44.sub3:vreg_128_align2 = V_MUL_LO_U32_e64 %19.sub3:vreg_128_align2, %19.sub3:vreg_128_align2, implicit $exec
144B	  %44.sub2:vreg_128_align2 = V_MUL_LO_U32_e64 %19.sub2:vreg_128_align2, %19.sub2:vreg_128_align2, implicit $exec
160B	  %44.sub1:vreg_128_align2 = V_MUL_LO_U32_e64 %19.sub1:vreg_128_align2, %19.sub1:vreg_128_align2, implicit $exec
176B	  %44.sub0:vreg_128_align2 = V_MUL_LO_U32_e64 %19.sub0:vreg_128_align2, %19.sub0:vreg_128_align2, implicit $exec
400B	  GLOBAL_STORE_DWORDX4_SADDR %13:vgpr_32, %44:vreg_128_align2, %46.sub2_sub3:sgpr_128, 0, 0, implicit $exec :: (store (s128) into %ir.gep2, align 32, addrspace 1)
404B	  %14:vreg_128_align2 = GLOBAL_LOAD_DWORDX4_SADDR %46.sub0_sub1:sgpr_128, %13:vgpr_32, 16, 0, implicit $exec :: (load (s128) from %ir.gep1 + 16, addrspace 1)
408B	  undef %45.sub3:vreg_128_align2 = V_MUL_LO_U32_e64 %14.sub3:vreg_128_align2, %14.sub3:vreg_128_align2, implicit $exec
416B	  %45.sub2:vreg_128_align2 = V_MUL_LO_U32_e64 %14.sub2:vreg_128_align2, %14.sub2:vreg_128_align2, implicit $exec
424B	  %45.sub1:vreg_128_align2 = V_MUL_LO_U32_e64 %14.sub1:vreg_128_align2, %14.sub1:vreg_128_align2, implicit $exec
432B	  %45.sub0:vreg_128_align2 = V_MUL_LO_U32_e64 %14.sub0:vreg_128_align2, %14.sub0:vreg_128_align2, implicit $exec
440B	  GLOBAL_STORE_DWORDX4_SADDR %13:vgpr_32, %45:vreg_128_align2, %46.sub2_sub3:sgpr_128, 16, 0, implicit $exec :: (store (s128) into %ir.gep2 + 16, addrspace 1)
448B	  SCHED_GROUP_BARRIER 32, 1, 0
456B	  SCHED_GROUP_BARRIER 2, 4, 0
464B	  SCHED_GROUP_BARRIER 64, 1, 0
472B	  SCHED_GROUP_BARRIER 32, 1, 0
480B	  SCHED_GROUP_BARRIER 2, 4, 0
496B	  SCHED_GROUP_BARRIER 64, 1, 0
512B	  S_ENDPGM 0

```

Check behaviour of passes`machine-scheduler`&`igrouplp`:

```bash
/home/tingqli/repo/llvm-project/build/bin/llc -mtriple=amdgcn  -mcpu=gfx942 -debug-only=machine-scheduler,igrouplp < sched.ll 2> sched.debug.txt
```

we can see extra dependencies(edges) between `ScheduleDAG`'s nodes when `llvm.amdgcn.sched.group.barrier` was used. these dependencies were marked as `Order/Artificial`. and they influences scheduler's behaviour later.


`ScheduleDAGMILive::schedule` calls：
  - `postProcessDAG();`  calls 
    - `IGroupLPDAGMutation::apply` scan region(BB w/o call) for sched intrinsics:
      - SCHED_BARRIER:  `ScheduleDAGInstrs::addEdge` 从region(BB)中查找指定类型的指令，该指令到SCHED_BARRIER或者SCHED_BARRIER到该指令的额外的`dependency edge`，来防止后继schedule algo改变此类型指令相对SCHED_BARRIER的未知关系
      - SCHED_GROUP_BARRIER: 因为 `sched.group`仅仅提供了`(指令类型,个数,次序)`信息，因此首先要在当前region(不包含call的Basic Block)中找到所有可能匹配到sched.group的指令，然后使用`PipelineSolver::solve`算法完成匹配，然后根据匹配引入新的`dependency edge`，这个过程容易出现误匹配，但是因为新引入`dependency edge`时会检查已有的依赖，如果出现环，违背了已有依赖就放弃添加，所以不会对正确性产生不良影响。比较常见的VALU指令在匹配阶段很容易出现错误匹配导致失效。


