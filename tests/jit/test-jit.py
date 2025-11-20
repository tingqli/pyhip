from typing import List, Optional, Set

# https://llvm.org/docs/AMDGPUInstructionSyntax.html#amdgpu-syn-instructions
class Instruction:
    def __init__(self, parent_bb:'BasicBlock', opcode):
        self.opcode = opcode
        self.parent_bb = parent_bb

    def __call__(self, *operands, mod:str=""):
        self.operands = operands
        self.mod = mod
        for i, op in enumerate(operands):
            assert isinstance(op, GPROp) or isinstance(op, int) or isinstance(op, GPRs), f"arg {i} type is {type(op)}"
        self.parent_bb.add_instruction(self)

    def __repr__(self):
        return f"{self.opcode} {','.join([repr(op) for op in self.operands])} {self.mod}"

class GPROp:
    def __init__(self, gpr, key):
        self.gpr = gpr
        self.key = key

    def __repr__(self):
        base_id = self.gpr.start_id
        if isinstance(self.key, slice):
            s = self.key
            assert s.step == 1
            return f"{self.gpr.rtype}[{base_id + s.start}:{base_id + s.stop}]"  # s[0:1] v[16:20]
        elif isinstance(self.key, int):
            return f"{self.gpr.rtype}{base_id + self.key}"  # v9, a9, s9
        else:
            assert 0

# a continous region of GPRs allocated
# to reference GPR in it, we need getitem:
#   v[0:1]
#   v[2]
# we cann't use v2 as pure-asm does
# GPRs should be allocated by JIT
class GPRs:
    def __init__(self, rtype, start_id, count):
        self.rtype = rtype
        self.start_id = start_id
        self.count = count

    def __getitem__(self, key):
        return GPROp(self, key)

    def __repr__(self):
        if self.count == 1:
            return f"{self.rtype}{self.start_id}"
        else:
            return f"{self.rtype}[{self.start_id}:{self.start_id + self.count - 1}]"

# every basic block has a Label in asm
class BasicBlock:
    def __init__(self, jit, label_name):
        self.jit = jit
        self.label_name = label_name
        self.instructions: List[Instruction] = []
        self.predecessors: List['BasicBlock'] = []
        self.successors: List['BasicBlock'] = []

    def add_instruction(self, instr: Instruction):
        self.instructions.append(instr)

    def add_successor(self, successor: 'BasicBlock'):
        if successor not in self.successors:
            self.successors.append(successor)
        if self not in successor.predecessors:
            successor.predecessors.append(self)

    def __repr__(self):
        # https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html#Special-format-strings
        asm = f"{self.label_name}%=:\n"
        for inst in self.instructions:
            asm += f"\t{inst}\n"
        return asm

# JIT emits instructions into BBs
class JIT:
    def __init__(self):
        self.current_bb = BasicBlock(self, "main")
        self.blocks = []
        # increased freely
        self.free_gpr_id = {'s':0, 'v':0, 'a': 0}

    def __getattr__(self, instruction):
        return Instruction(self.current_bb, instruction)

    def _finish_bb(self, bb, next_bb_name=""):
        assert bb is self.current_bb
        self.blocks.append(bb)
        next_bb = BasicBlock(self, next_bb_name)
        bb.add_successor(next_bb)
        self.current_bb = next_bb

    def Label(self, name=""):
        # creat new basic block, archieve current bb
        self._finish_bb(self.current_bb, name)
        return self.current_bb

    def generate(self):
        self._finish_bb(self.current_bb)
        asm=""
        for bb in self.blocks:
            asm += repr(bb)
        return asm

    def new_gpr(self, reg_type, count_range):
        assert reg_type == 's' or reg_type == 'v' or reg_type == 'a'
        if isinstance(count_range, int):
            # allocate do not reuse, just increase the index
            count = count_range
            start_id = self.free_gpr_id[reg_type]
            self.free_gpr_id[reg_type] += count
            return GPRs(reg_type, start_id, count)
        elif isinstance(count_range, tuple) or isinstance(count_range, list):
            assert len(count_range) == 2
            first_id, last_id = count_range
            assert self.free_gpr_id[reg_type] <= first_id, "specified reg has been allocated"
            assert self.free_gpr_id[reg_type] <= last_id, "specified reg has been allocated"
            self.free_gpr_id[reg_type] = last_id + 1
            return GPRs(reg_type, first_id, last_id - first_id + 1)
        else:
            assert 0

def kernel(J):
    p_kargs = J.new_gpr('s',[0,1])
    pA = J.new_gpr('s',2)
    K = J.new_gpr('s',1)

    J.s_load_dwordx2(pA, p_kargs, 0)

    acc = J.new_gpr("a", 4)
    for i in range(4):
        J.v_accvgpr_write_b32(acc[i], 0)

    J.s_load_dword(K, p_kargs, 8)
    J.s_waitcnt(mod=f"lgkmcnt({0})")
    #T.v_mov_b32(v[2], s[2])
    # v_mov_b32(v3, s3)
    #with J.BB():
    #    J.v_lshl_add_u32(v[2], v[0], 2, v[2])
    s_idx = J.new_gpr('s',1)
    s_temp = J.new_gpr('s',2)

    J.s_mov_b32(s_idx, 0)

    J.Label("bb0")

    J.s_lshl_b32(s_temp[0],s_idx,2)
    J.s_add_u32(s_temp[0], pA[0], s_temp[0])
    J.s_addc_u32(s_temp[1], pA[1], 0)

    J.s_store_dword(K, s_temp, 0, mod="glc")

    J.s_addk_i32(s_idx, 1)
    J.s_cmp_lt_i32(s_idx, 32)
    J.s_cbranch_scc0(mod="bb1%=")
    J.s_branch(mod="bb0%=")

    J.Label("bb1")

    # flat_store_dword(v[2:3], v0)

jit = JIT()
kernel(jit)
asm = jit.generate()
print(asm)

import asm_build

header='''
;kernel test
;arg int* A
;arg int K
'''

code = header + asm
print(code)
kernel = asm_build.build(code)

import torch

A = torch.ones(64, dtype=torch.int)
print(A)
kernel([1],[64], A.data_ptr(), 64)
torch.cuda.synchronize()
print(A)