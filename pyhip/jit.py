from typing import List, Optional, Set, Union
from .hiptools import module

# https://llvm.org/docs/AMDGPUInstructionSyntax.html#amdgpu-syn-instructions
class Instruction:
    def __init__(self, parent_bb:'BasicBlock', opcode):
        self.opcode = opcode
        self.parent_bb = parent_bb
        assert not opcode.startswith("s_call")
        self.is_branch = self.opcode.startswith("s_branch")
        self.is_cbranch = self.opcode.startswith("s_cbranch_")
        self.debug_info = ""

    def __call__(self, *operands, mod:str=""):
        self.operands = operands
        self.mod = mod
        for i, op in enumerate(operands):
            assert isinstance(op, GPRExpr) or isinstance(op, int) or isinstance(op, GPRs), f"arg {i} type is {type(op)}"
        self.parent_bb.add_instruction(self)

    def __repr__(self):
        return f"{self.opcode} {','.join([repr(op) for op in self.operands])} {self.mod} ; {self.debug_info}"

class GPRExpr:
    def __init__(self, op:str, src0=None, src1=None, src2=None):
        self.op=op
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.depth = 1
        if isinstance(src0, GPRExpr):
            self.depth = max(self.depth, src0.depth + 1)
        if isinstance(src1, GPRExpr):
            self.depth = max(self.depth, src1.depth + 1)
        if isinstance(src2, GPRExpr):
            self.depth = max(self.depth, src2.depth + 1)

    def match(self, other: 'GPRExpr'):
        if self.op == "pattern":
            # pattern node handles the rest
            return self.src0.match(other)
        if self.op != other.op:
            return False
        if (self.src0 is not other.src0) and (not self.src0.match(other.src0)):
            return False
        if (self.src1 is not other.src1) and (not self.src1.match(other.src1)):
            return False
        return True

    def __add__(self, other):
        return GPRExpr("+", self, other)
    def __sub__(self, other):
        return GPRExpr("-", self, other)
    def __mul__(self, other):
        return GPRExpr("*", self, other)
    def __lshift__(self, other):
        return GPRExpr("<<", self, other)
    def __rshift__(self, other):
        return GPRExpr(">>", self, other)

    def __repr__(self):
        if self.op == "getitem":
            base_id = self.src0.start_id
            first = self.src1
            last = self.src2
            if last > first:
                return f"{self.src0.rtype}[{base_id + first}:{base_id + last}]"  # s[0:1] v[16:20]
            else:
                return f"{self.src0.rtype}{base_id + first}"  # v9, a9, s9
        elif (self.src1 is None) and (self.src2 is None):
            return f"{self.op} {self.src0}"
        elif (self.src2 is None):
            return f"{self.src0} {self.op} {self.src1}"
        else:
            return f"{self.op}({self.src0},{self.src1},{self.src2})"

class GPRItemPat:
    def __init__(self, rtype:str, count:int = 1):
        self.rtype = rtype
        self.count = count

    def match(self, other):
        if self.rtype == "i":
            return isinstance(other, int)
        self.other = other
        if not isinstance(other, GPRExpr): return False
        if other.op != "getitem": return False
        if self.rtype != other.src0.rtype: return False
        other_count = other.src2 - other.src1 + 1
        return self.count == other_count

# a continous region of GPRs allocated
# to reference GPR in it, we need getitem:
#   v[0:1]
#   v[2]
# we cann't use v2 as pure-asm does
# GPRs should be allocated by JIT
class GPRs:
    patterns = None
    def __init__(self, jit, rtype, start_id, count, align=0):
        self.jit = jit
        self.rtype = rtype
        self.start_id = start_id
        self.count = count
        self.align = align

    def __getitem__(self, key):
        if isinstance(key, slice):
            s = key
            assert s.step == 1 or s.step is None, f"slice {s} with step not eq 1"
            return GPRExpr("getitem", self, s.start, s.stop)
        elif isinstance(key, int):
            idx = key
            return GPRExpr("getitem", self, idx, idx)
        else:
            assert 0, f"unsupported key {key}"

    def __setitem__(self, key, value:GPRExpr):
        dst = self[key]
        i0 = GPRExpr("pattern", GPRItemPat("i", 1))
        s1 = GPRExpr("pattern", GPRItemPat("s", 1))
        s2 = GPRExpr("pattern", GPRItemPat("s", 1))
        ds1 = GPRExpr("pattern", GPRItemPat("s", 2))
        ds2 = GPRExpr("pattern", GPRItemPat("s", 2))

        if (s1 << s2).match(value):
            inst = Instruction(self.jit.current_bb, "s_lshl_b32")
            inst(dst, s1.src0.other, s2.src0.other)
        elif (s1 << i0).match(value):
            inst = Instruction(self.jit.current_bb, "s_lshl_b32")
            inst(dst, s1.src0.other, s2.src0.other)
        elif (s1 + s2).match(value):
            inst = Instruction(self.jit.current_bb, "s_add_u32")
            inst(dst, s1.src0.other, s2.src0.other)
        else:
            assert 0

    def __repr__(self):
        if self.count == 1:
            return f"{self.rtype}{self.start_id}"
        else:
            return f"{self.rtype}[{self.start_id}:{self.start_id + self.count - 1}]"

# every basic block has a Label in asm
class BasicBlock:
    index = 0
    def __init__(self, jit, label_name):
        self.bb_index = BasicBlock.index
        BasicBlock.index += 1
        self.jit = jit
        self.label_name = label_name
        if label_name == "":
            self.label_name = f"_bb_no_name_{self.bb_index}"
        
        jit.label2bb[self.label_name] = self
        self.instructions: List[Instruction] = []
        self.predecessors: List['BasicBlock'] = []
        self.successors: List['BasicBlock'] = []
        self.unsolved_successores : List[str] = []

    # all bb has been defined, solve label target bb
    def solve_succesors(self):
        for label in self.unsolved_successores:
            successor = self.jit.label2bb[label]
            self.add_successor(successor)

    def add_instruction(self, instr: Instruction):
        self.instructions.append(instr)
        # branch or cbranch can tell successors
        if instr.is_branch:
            # one possible successor
            self.jit._finish_bb(self, "")
            target_lable_str = instr.mod
            self.add_successor(target_lable_str)
        elif instr.is_cbranch:
            # two possible successor
            self.jit._finish_bb(self, "")
            target_lable_str = instr.mod
            self.add_successor(target_lable_str)
            self.add_successor(self.jit.current_bb)
    
    def add_successor(self, successor: Union['BasicBlock', str]):
        if isinstance(successor, BasicBlock):
            if successor not in self.successors:
                self.successors.append(successor)
            if self not in successor.predecessors:
                successor.predecessors.append(self)
        else:
            assert isinstance(successor, str)
            if successor in self.jit.label2bb:
                successor = self.jit.label2bb[successor]
                self.add_successor(successor)
            else:
                # unknown BB in the future
                self.unsolved_successores.append(successor)

    def __repr__(self):
        # https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html#Special-format-strings
        asm = f"{self.label_name}:\n"
        asm += f";BB#{self.bb_index}\n"
        asm += f";predecessors:[{','.join([p.label_name for p in self.predecessors])}]\n"
        asm += f";successors:[{','.join([p.label_name for p in self.successors])}]\n"
        for inst in self.instructions:
            asm += f"\t{inst}\n"
        return asm

# JIT emits instructions into BBs
class JIT:
    def __init__(self):
        self.blocks = []
        # increased freely
        self.free_gpr_id = {'s':0, 'v':0, 'a': 0}
        self.label2bb = {}
        self.current_bb = BasicBlock(self, "main")
        self.relocatable_gprs = []
        self.fixed_gprs = []

    def __getattr__(self, instruction):
        return Instruction(self.current_bb, instruction)

    def _finish_bb(self, bb, next_bb_name=""):
        assert bb is self.current_bb
        self.blocks.append(bb)
        next_bb = BasicBlock(self, next_bb_name)
        self.current_bb = next_bb

    def Label(self, name=""):
        # creat new basic block, archieve current bb
        old_bb = self.current_bb
        self._finish_bb(self.current_bb, name)
        old_bb.add_successor(self.current_bb)
        return self.current_bb

    def _align_up(self, a, align):
        return ((a + align - 1)//align) * align

    def new_gpr(self, reg_type, count_range, align=1):
        assert reg_type == 's' or reg_type == 'v' or reg_type == 'a'
        if isinstance(count_range, int):
            # allocate do not reuse, just increase the index
            count = count_range
            start_id = self._align_up(self.free_gpr_id[reg_type], align)
            self.free_gpr_id[reg_type] = start_id + count
            gprs = GPRs(self, reg_type, start_id, count, align=align)
            self.relocatable_gprs.append(gprs)
            return gprs
        elif isinstance(count_range, tuple) or isinstance(count_range, list):
            assert len(count_range) == 2
            assert align == 1
            first_id, last_id = count_range
            assert self.free_gpr_id[reg_type] <= first_id, "specified reg has been allocated"
            assert self.free_gpr_id[reg_type] <= last_id, "specified reg has been allocated"
            self.free_gpr_id[reg_type] = last_id + 1
            gprs = GPRs(self, reg_type, first_id, last_id - first_id + 1, align=align)
            self.fixed_gprs.append(gprs)
            return gprs
        else:
            assert 0

    # this step is not mandatory，but it allows more registers to use
    # linear-scan:
    #    try to shrink register usage on `relocatable_gprs`
    #    do not touch `fixed_gprs`
    def register_allocation_linear_scan(self):
        # note: BB in self.blocks has been ordered, assign each instruction an serial
        serial_index = 0
        for bb in self.blocks:
            for inst in bb.instructions:
                inst.sid = serial_index
                serial_index += 1

        # find live-intervals of gprs in `relocatable_gprs`
        live_intervals = {}
        bb_access_gprs = {}
        for bb in self.blocks:
            bb_access_gprs[bb] = []
            for inst in bb.instructions:
                # inst.sid
                for op in inst.operands:
                    if isinstance(op, GPRExpr):
                        gprs = op.src0
                    elif isinstance(op, GPRs):
                        gprs = op
                    else:
                        continue
                    if gprs in self.fixed_gprs:
                        continue
                    assert gprs in self.relocatable_gprs
                    if gprs not in live_intervals:
                        live_intervals[gprs] = [inst.sid, inst.sid]
                    live_intervals[gprs][0] = min(live_intervals[gprs][0], inst.sid)
                    live_intervals[gprs][1] = max(live_intervals[gprs][1], inst.sid)
                    bb_access_gprs[bb].append(gprs)

        # extend live-interval of gprs within loop
        # if jump to bb with higher sid, then we don't need to handle since interval logic works fine
        # based on sid, but if we found a jump back(successor with smaller sid), we need to handle
        # it, and it only affect register which's life ends within current bb
        # so in bb where register ends, we check if such loop-back exists, if so extend it to the jump-back
        # instruction.
        for bb in self.blocks:
            if len(bb.instructions) == 0: continue
            sid0 = bb.instructions[0].sid
            for parent_bb in bb.predecessors:
                if parent_bb.instructions[0].sid > sid0:
                    # backward jump detected, enlarge all gpr's interval
                    jump_back_sid = parent_bb.instructions[-1].sid
                    for gprs in live_intervals:
                        first_sid,last_sid = live_intervals[gprs]
                        if first_sid < sid0 and last_sid >= sid0 and last_sid < jump_back_sid:
                            live_intervals[gprs][1] = jump_back_sid

        # add debug-info to inst
        for bb in self.blocks:
            for inst in bb.instructions:
                debug_info = f" #{inst.sid}  regs:"
                for gprs in live_intervals:
                    first_sid, last_sid = live_intervals[gprs]
                    if inst.sid >= first_sid and inst.sid <= last_sid:
                        debug_info += f"{gprs},"
                inst.debug_info = debug_info

        self.asm_debug_info += ";============ register live interval ==============\n"
        for gprs in live_intervals:
            first_sid, last_sid = live_intervals[gprs]
            self.asm_debug_info += f";{repr(gprs):10s}  {first_sid} ~ {last_sid}\n"

        # re-assign each gprs's start_id (linear_scan)
        # sorted_live_interval = [(live_intervals[gprs][0], live_intervals[gprs][1], gprs) for gprs in live_intervals].sort()
        event_list = []
        for gprs in live_intervals:
            first_sid, last_sid = live_intervals[gprs]
            event_list.append((first_sid, 0, repr(gprs), gprs)) # 0 means first-use
            event_list.append((last_sid, 1, repr(gprs), gprs)) # 1 means last-use

        # linear_scan all events according to time
        reg_resource = {"s":[0 for _ in range(103)],
                        "v": [0 for _ in range(256)],
                        "a": [0 for _ in range(256)]}
        # reserve fixed gprs
        for gprs in self.fixed_gprs:
            i0 = gprs.start_id
            i1 = gprs.start_id + gprs.count
            reg_resource[gprs.rtype][i0:i1] = [1]*gprs.count
        
        '''
        self.rtype = rtype
        self.start_id = start_id
        self.count = count
        self.align = align
        '''
        from collections import OrderedDict
        # there may be multiple first/last use events at same sid
        # we group them and then do allocation before free, because
        #   - alloc happens before instruction
        #   - free happends after instruction
        event_groups = OrderedDict()
        for sid, event, _, gprs in sorted(event_list):
            if sid not in event_groups:
                event_groups[sid] = []
            event_groups[sid].append([event, gprs])

        def alloc_gpr(gprs):
            rtype = gprs.rtype
            count = gprs.count
            align = gprs.align
            slots = reg_resource[rtype]
            for i in range(0, len(slots), align):
                if all(s == 0 for s in slots[i:(i+count)]):
                    gprs.start_id = i
                    slots[i:(i+count)] = [1]*count # mark as used
                    return
            assert 0, f"cannot allocate {gprs}, not enough resources"

        def free_gpr(gprs):
            rtype = gprs.rtype
            count = gprs.count
            slots = reg_resource[rtype]
            i = gprs.start_id
            slots[i:(i+count)] = [0]*count

        self.asm_debug_info += ";============ register allocation ==============\n"
        for sid, events in event_groups.items():
            # at same first allocation 
            for ev,gprs in events:
                if ev == 0: # first use
                    alloc_gpr(gprs)
                    self.asm_debug_info += f";alloc {gprs} at #{sid}\n"
            # now free
            for ev,gprs in events:
                if ev == 1: # last use
                    free_gpr(gprs)
                    self.asm_debug_info += f";free {gprs} at #{sid}\n"

        # (following code-gen phase will use the new start_id automatically)

        return

    def build(self, signature):
        self.asm_debug_info = ""
        self._finish_bb(self.current_bb)

        # remove unreachable bb
        bb_to_remove = []
        for bb in self.blocks[1:]:
            if len(bb.predecessors) == 0:
                print(f"remove unreachable code block {bb.label_name}")
                bb_to_remove.append(bb)
        for bb in bb_to_remove:
            self.blocks.remove(bb)

        # resolve successors from label
        for bb in self.blocks:
            bb.solve_succesors()

        self.register_allocation_linear_scan()

        # generate asm: basic blocks are in natural order
        asm=""
        for bb in self.blocks:
            asm += repr(bb)
        asm += self.asm_debug_info

        used_gprs = []
        for a in asm.splitlines():
            if a[0] == ";": continue
            for op in a.replace(","," ").split()[1:]:
                if (op.startswith("s") or op.startswith("v") or op.startswith("a")) and (op[1].isdigit()):
                    gpr_type = op[0]
                    str_gpr = f"{gpr_type}{int(op[1:])}"
                    if str_gpr not in used_gprs: used_gprs.append(str_gpr)
                elif op.startswith("s[") or op.startswith("v[") or op.startswith("a["):
                    gpr_type = op[0]
                    idx0, idx1 = op[2:-1].split(":")
                    idx0 = int(idx0)
                    idx1 = int(idx1)
                    for idx in range(idx0, idx1 + 1):
                        str_gpr = f"{gpr_type}{idx}"
                        if str_gpr not in used_gprs: used_gprs.append(str_gpr)
        '''
        str_arg_c_del = ",".join([f"{a[0]} {a[1]}" for a in args])
        signature = f"{kernel_name}({str_arg_c_del})"
        '''
        inline_asm = "\n".join([ f'"{line}\\n"' if len(line) else "" for line in asm.splitlines()])
        kernel_name = signature.split("(")[0]
        str_used_gprs = ",".join([f'\"{s}\"' for s in used_gprs])
        print(f" kernel: {signature}  used_gprs={used_gprs}")

        hip_src =r'''
#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
__global__ void __launch_bounds__(256, 1) ''' + signature +  r''' {
    //A[threadIdx.x] = K;
    asm volatile("\n"
''' + "\n" + inline_asm + \
r'''
                ::
                :"memory",''' + str_used_gprs + r''');
}
        '''
        import os
        cpp_src_fpath = os.path.join(os.getcwd(),f'.jit-gen-{kernel_name}.cpp')
        with open(cpp_src_fpath, 'w', encoding='utf-8') as f:
            f.write(hip_src)
        hip = module(cpp_src_fpath)
        return getattr(hip, kernel_name)
