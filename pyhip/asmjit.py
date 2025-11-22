from typing import List, Optional, Set, Union
from .hiptools import module
import inspect
import os

# https://llvm.org/docs/AMDGPUInstructionSyntax.html#amdgpu-syn-instructions
class Instruction:
    def __init__(self, parent_bb:'BasicBlock', opcode):
        self.opcode = opcode
        self.parent_bb = parent_bb
        assert not opcode.startswith("s_call")
        self.is_branch = self.opcode.startswith("s_branch")
        self.is_cbranch = self.opcode.startswith("s_cbranch_")
        self.debug_info = ""
        self.sid = 0

    def __call__(self, *operands, mod:str="", insert_bb_pos = None):
        self.operands = operands
        self.mod = mod
        for i, op in enumerate(operands):
            assert isinstance(op, GPRExpr) or \
                   isinstance(op, int) or isinstance(op, float) or \
                   op=="scc" or op == "vcc" or op == "off" or \
                   (isinstance(op, str) and op.startswith("0x")) or \
                   isinstance(op, GPRs), \
            f"arg {i} type is {type(op)}"
        self.parent_bb.add_instruction(self, insert_bb_pos)

    def op_repr(self, op):
        if isinstance(op, str):
            return op
        return repr(op)

    def __repr__(self):
        return f"{self.opcode} {','.join([self.op_repr(op) for op in self.operands])} {self.mod} ; {self.debug_info}"

class GPRExpr:
    def __init__(self, op:str, src0=None, src1=None, src2=None):
        self.op=op
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2

    def __add__(self, other):
        return GPRExpr("+", self, other)
    def __radd__(self, other):
        return GPRExpr("+", other, self)
    def __sub__(self, other):
        return GPRExpr("-", self, other)
    def __rsub__(self, other):
        return GPRExpr("-", other, self)
    def __mul__(self, other):
        return GPRExpr("*", self, other)
    def __rmul__(self, other):
        return GPRExpr("*", self, other)
    def __lshift__(self, other):
        return GPRExpr("<<", self, other)
    def __rlshift__(self, other):
        return GPRExpr("<<", other, self)
    def __rshift__(self, other):
        return GPRExpr(">>", self, other)
    def __rrshift__(self, other):
        return GPRExpr(">>", other, self)
    def __and__(self, other):
        return GPRExpr("&", self, other)
    def __rand__(self, other):
        return GPRExpr("&", self, other)
    def __or__(self, other):
        return GPRExpr("|", self, other)
    def __ror__(self, other):
        return GPRExpr("|", self, other)
    def __xor__(self, other):
        return GPRExpr("^", self, other)
    def __rxor__(self, other):
        return GPRExpr("^", self, other)
    def __eq__(self, other):
        return GPRExpr("eq", self, other)
    def __ne__(self, other):
        return GPRExpr("ne", self, other)
    def __lt__(self, other):
        return GPRExpr("lt", self, other)
    def __gt__(self, other):
        return GPRExpr("gt", self, other)
    def __le__(self, other):
        return GPRExpr("le", self, other)
    def __ge__(self, other):
        return GPRExpr("ge", self, other)

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

# a continous region of GPRs allocated
# to reference GPR in it, we need getitem:
#   v[0:1]
#   v[2]
# we cann't use v2 as pure-asm does
# GPRs should be allocated by JIT
class GPRs:
    patterns = None
    def __init__(self, jit, rtype, start_id, count, dtype, align=0, name=""):
        self.jit = jit
        self.rtype = rtype
        self.start_id = start_id
        self.count = count
        self.align = align
        self.dtype = dtype
        self.name = name # for debugging purpose only

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

    def __setitem__(self, key, value:Union[GPRExpr,int,float]):
        dst = self[key]
        inst = Instruction(self.jit.current_bb, "expression_place_holder")
        inst(dst, value)
        # expression_place_holder will be compiled later when all program is ready
        # all expressions within same BB can be processed together

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

    def add_instruction(self, instr: Instruction, insert_bb_pos=None):
        if insert_bb_pos is None:
            self.instructions.append(instr)
        else:
            self.instructions.insert(insert_bb_pos, instr)
            assert not (instr.is_branch or instr.is_cbranch), "do not insert branch please!"

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
        asm = f"{self.label_name}:\t"
        asm += f" ;BB#{self.bb_index}"
        asm += f" predecessors:[{','.join([p.label_name for p in self.predecessors])}]"
        asm += f" successors:[{','.join([p.label_name for p in self.successors])}]\n"
        for inst in self.instructions:
            asm += f"\t{inst}\n"
        return asm

import struct

def float_to_ieee754_bits_little(f):
    packed = struct.pack('<f', f)    # 小端序
    bits = struct.unpack('<I', packed)[0]  # 解包为无符号整数
    return bits

# JIT emits instructions into BBs
all_kernel_hip_src_names = {}
class JIT:
    def __init__(self):
        self.blocks = []
        # increased freely
        self.free_gpr_id = {'s':0, 'v':0, 'a': 0}
        self.label2bb = {}
        self.current_bb = BasicBlock(self, "_jit_main")
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

    '''
    # scalar jump (wave level, no divergent)
    def Jump(self, label:str, cond:GPRExpr = None):
        if cond is None:
            self.s_branch(mod=label)
        else:
    '''

    def _align_up(self, a, align):
        return ((a + align - 1)//align) * align

    def new_gpr(self, reg_type, count_range, dtype="", align=1, name=""):
        assert reg_type == 's' or reg_type == 'v' or reg_type == 'a'
        if isinstance(count_range, int):
            # allocate do not reuse, just increase the index
            count = count_range
            start_id = self._align_up(self.free_gpr_id[reg_type], align)
            self.free_gpr_id[reg_type] = start_id + count
            gprs = GPRs(self, reg_type, start_id, count, dtype=dtype, align=align, name=name)
            self.relocatable_gprs.append(gprs)
            return gprs
        elif isinstance(count_range, tuple) or isinstance(count_range, list):
            assert len(count_range) == 2
            assert align == 1
            first_id, last_id = count_range
            assert self.free_gpr_id[reg_type] <= first_id, "specified reg has been allocated"
            assert self.free_gpr_id[reg_type] <= last_id, "specified reg has been allocated"
            self.free_gpr_id[reg_type] = last_id + 1
            gprs = GPRs(self, reg_type, first_id, last_id - first_id + 1, dtype=dtype, align=align, name=name)
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
                assert len(parent_bb.instructions) > 0, f"{parent_bb.label_name} is empty"
                if parent_bb.instructions[0].sid > sid0:
                    # backward jump detected, enlarge all gpr's interval
                    jump_back_sid = parent_bb.instructions[-1].sid
                    for gprs in live_intervals:
                        first_sid,last_sid = live_intervals[gprs]
                        if first_sid < sid0 and last_sid >= sid0 and last_sid < jump_back_sid:
                            live_intervals[gprs][1] = jump_back_sid

        # add debug-info to inst
        if 0:
            for bb in self.blocks:
                for inst in bb.instructions:
                    debug_info = f" #{inst.sid}  regs:"
                    for gprs in live_intervals:
                        first_sid, last_sid = live_intervals[gprs]
                        if inst.sid >= first_sid and inst.sid <= last_sid:
                            debug_info += f"'{gprs.name}'{gprs} "
                    inst.debug_info = debug_info

        self.asm_debug_info += ";============ register live interval ==============\n"
        for gprs in live_intervals:
            first_sid, last_sid = live_intervals[gprs]
            self.asm_debug_info += f";'{gprs.name}'{repr(gprs):10s}  {first_sid} ~ {last_sid}\n"

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
            assert 0, f"cannot allocate '{gprs.name}'{gprs}, not enough resources"

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
                    self.asm_debug_info += f";alloc '{gprs.name}'{gprs} at #{sid}\n"
            # now free
            for ev,gprs in events:
                if ev == 1: # last use
                    free_gpr(gprs)
                    self.asm_debug_info += f";free '{gprs.name}'{gprs} at #{sid}\n"

        # (following code-gen phase will use the new start_id automatically)
        # add debug info to asm instruction
        for bb in self.blocks:
            for inst in bb.instructions:
                debug_info = f"#{inst.sid} "
                for op in inst.operands:
                    if isinstance(op, GPRExpr):
                        op = op.src0
                    if isinstance(op, GPRs):
                        debug_info += f"{op.name},"
                    else:
                        debug_info += f"{op},"
                inst.debug_info = debug_info
        return

    '''
    with the help of temp vars, complex expression can be recursively generated & appended into bb.
    '''
    def recursive_expr_gen(self, bb, dst_expr:GPRExpr,  expr:Union['GPRExpr',int,float]):
        assert dst_expr.op == "getitem"
        dst_gprs = dst_expr.src0
        dst_idx0 = dst_expr.src1
        dst_idx1 = dst_expr.src2
        assert dst_idx1 == dst_idx0
        rtype = dst_gprs.rtype
        dtype = dst_gprs.dtype        
        if isinstance(expr, float):
            expr = float_to_ieee754_bits_little(expr)
        if isinstance(expr, int):
            new_inst = Instruction(bb, f"{rtype}_mov_b32")
            new_inst(dst_expr, expr)
            return
        assert isinstance(expr, GPRExpr)
        if expr.op == "getitem":
            # assign
            src_cnt = expr.src2 - expr.src1 + 1
            assert src_cnt == 1
            new_inst = Instruction(bb, f"{rtype}_mov_b32")
            new_inst(dst_expr, expr)
            return

        if isinstance(expr.src0, GPRExpr):
            if expr.src0.op == "getitem":
                # "getitem" expr can be used as operand directly
                src0_operand = expr.src0
            else:
                src0_gprs = self.new_gpr(rtype, 1, dtype=dtype, align=1, name="")
                src0_operand = GPRExpr("getitem", src0_gprs, 0, 0)
                self.recursive_expr_gen(bb, src0_operand, expr.src0)
        else:
            src0_operand = expr.src0 # int,float,...
            if isinstance(src0_operand, float): # convert float-const into int?
                src0_operand = float_to_ieee754_bits_little(src0_operand)

        if isinstance(expr.src1, GPRExpr):
            if expr.src1.op == "getitem":
                # "getitem" expr can be used as operand directly
                src1_operand = expr.src1
            else:
                src1_gprs = self.new_gpr(rtype, 1, dtype=dtype, align=1, name="")
                src1_operand = GPRExpr("getitem", src1_gprs, 0, 0)
                self.recursive_expr_gen(bb, src1_operand, expr.src1)
        else:
            src1_operand = expr.src1 # int,float,...
            if isinstance(src1_operand, float): # convert float-const into int?
                src1_operand = float_to_ieee754_bits_little(src1_operand)

        def vgpr_add_sub_ui32(op):
            nonlocal src0_operand, src1_operand
            if isinstance(src0_operand, GPRExpr) and isinstance(src1_operand, GPRExpr):
                return Instruction(bb, f"v_add_u32_e32" if op == '+' else f"v_sub_u32_e32")
            if isinstance(src0_operand, GPRExpr):
                if op == '-':
                    src1_operand = hex(-src1_operand & 0xffffffff)
                # v_add_u32_e32's src0 can be const
                src1_operand, src0_operand = src0_operand, src1_operand
                return Instruction(bb, f"v_add_u32_e32") # vgpr + (+/- const)
            if isinstance(src1_operand, GPRExpr):
                if op == '-':
                    return Instruction(bb, f"v_sub_u32_e32") # const - vgpr
                return Instruction(bb, f"v_add_u32_e32") # const + vgpr
            assert 0

        # now src0_operand & src1_operand are generated, we can generate our result
        if expr.op == "+":
            assert dtype != ""
            if rtype == "s":
                new_inst = Instruction(bb, f"{rtype}_add_{dtype}")
            elif rtype == "v" and dtype in ["u32", "i32"]:
                new_inst = vgpr_add_sub_ui32(expr.op)
            else:
                assert 0
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op == "-":
            assert dtype != ""
            if rtype == "s":
                new_inst = Instruction(bb, f"{rtype}_sub_{dtype}")
            elif rtype == "v" and dtype in ["u32", "i32"]:
                new_inst = vgpr_add_sub_ui32(expr.op)
            else:
                assert 0
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op == "*":
            assert dtype != ""
            if rtype == "s":
                if dtype == "u32":
                    print("s_mul_u32 not exist, using s_mul_i32 instead")
                    new_inst = Instruction(bb, f"s_mul_i32")
                else:
                    new_inst = Instruction(bb, f"s_mul_{dtype}")
            elif rtype == "v":
                if dtype in ["u32", "u16"]:
                    new_inst = Instruction(bb, f"v_mul_lo_{dtype}")
                elif dtype in ["f32", "f16"]:
                    new_inst = Instruction(bb, f"v_mul_{dtype}")
                elif dtype in ["i32"]:
                    new_inst = Instruction(bb, f"v_mul_i32_i24")
                else:
                    assert 0, f"unsupported v_mul dtype: {dtype}"
            else:
                assert 0, f"unsupported v_mul rtype: {rtype}"
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op == "<<":
            if rtype == "s":
                new_inst = Instruction(bb, f"s_lshl_b32")
                new_inst(dst_expr, src0_operand, src1_operand)
            elif rtype == "v":
                new_inst = Instruction(bb, f"v_lshlrev_b32")
                new_inst(dst_expr, src1_operand, src0_operand)
            else:
                assert 0, f"unsupported v_mul rtype: {rtype}"
        elif expr.op == ">>":
            if rtype == "s":
                if dtype == "u32":
                    new_inst = Instruction(bb, f"s_lshr_b32")
                elif dtype == "i32":
                    new_inst = Instruction(bb, f"s_ashr_i32")
                else:
                    assert 0, f"unsupported sgpr shift right dtype {dtype}"
                new_inst(dst_expr, src0_operand, src1_operand)
            elif rtype == "v":
                if dtype == "u32":
                    new_inst = Instruction(bb, f"v_lshrrev_b32")
                elif dtype == "i32":
                    new_inst = Instruction(bb, f"v_ashrrev_i32")
                else:
                    assert 0, f"unsupported vgpr shift right dtype {dtype}"
                new_inst(dst_expr, src1_operand, src0_operand)
            else:
                assert 0, f"unsupported rtype: {rtype}"
        elif expr.op == "&":
            new_inst = Instruction(bb, f"{rtype}_and_b32")
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op == "|":
            new_inst = Instruction(bb, f"{rtype}_or_b32")
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op == "^":
            new_inst = Instruction(bb, f"{rtype}_xor_b32")
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op in ["eq","ne","lt","gt","le","ge"]:
            if rtype == "s":
                op = expr.op
                if op == "ne" : op = "lg"
                cmp = Instruction(bb, f"s_cmp_{op}_{dtype}")
                mov = Instruction(bb, f"s_mov_b32")
                cmp(src0_operand, src1_operand)
                mov(dst_expr, "scc")
            elif rtype == "v":
                op = expr.op
                # if op == "ne" : op = "lg"
                # only src0 can be const
                if not isinstance(src1_operand, GPRExpr):
                    src1_operand, src0_operand = src0_operand, src1_operand
                    cmp_op_reverse = {"gt":"le", "lt":"ge", "le":"gt", "ge":"lt", "eq":"eq", "ne":"ne"}
                    op = cmp_op_reverse[op]
                cmp = Instruction(bb, f"v_cmp_{op}_{dtype}_e32")
                mov = Instruction(bb, f"v_cndmask_b32_e64")
                cmp("vcc", src0_operand, src1_operand)
                mov(dst_expr, 0, 1, "vcc")
            else:
                assert 0, f"unsupported rtype: {rtype}"
        else:
            assert 0, f"unsupported expression {expr}"


    def compile_bb_expr(self, bb, expr_insts):
        for inst in expr_insts:
            pos = bb.instructions.index(inst)
            bb.instructions.remove(inst)
            
            # insert new instructions at pos
            assert inst.opcode == "expression_place_holder"
            dst_expr = inst.operands[0]
            src_expr = inst.operands[1]

            tempbb = BasicBlock(self, "")
            self.recursive_expr_gen(tempbb, dst_expr, src_expr)
            for i, newinst in enumerate(tempbb.instructions):
                bb.instructions.insert(pos + i, newinst)

    def compile_expressions(self):
        # first version, only simple binary expressions
        for bb in self.blocks:
            expr_insts = []
            for inst in bb.instructions:
                if inst.opcode == "expression_place_holder":
                    expr_insts.append(inst)
            self.compile_bb_expr(bb, expr_insts)

    def build(self, kernel_name, signature, extra_compiler_options, temp_filename):
        self.asm_debug_info = ""
        self._finish_bb(self.current_bb)

        # remove unreachable bb
        bb_to_remove = []
        for bb in self.blocks[1:]:
            if len(bb.predecessors) == 0:
                print(f"remove unreachable code block {bb.label_name}")
                bb_to_remove.append(bb)
        for empty_bb in bb_to_remove:
            for bb in empty_bb.predecessors:
                bb.successors.remove(empty_bb)
            for bb in empty_bb.successors:
                bb.predecessors.remove(empty_bb)
            self.blocks.remove(empty_bb)    

        # resolve successors from label
        for bb in self.blocks:
            bb.solve_succesors()

        self.compile_expressions()
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
        str_used_gprs = ""
        if len(used_gprs):
            str_used_gprs = "," + ",".join([f'\"{s}\"' for s in used_gprs])
        print(f" kernel: {kernel_name}{signature}  used_gprs={used_gprs}")

        hip_src =r'''
#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
__global__ void __launch_bounds__(256, 1) ''' + kernel_name + signature +  r''' {
    //A[threadIdx.x] = K;
    asm volatile("\n"
''' + "\n" + inline_asm + \
r'''
                ::
                :"memory"''' + str_used_gprs + r''');
}
        '''
        import os
        cpp_src_fpath = os.path.join(os.getcwd(),f'{temp_filename}.cpp')
        with open(cpp_src_fpath, 'w', encoding='utf-8') as f:
            f.write(hip_src)
        hip = module(cpp_src_fpath, extra_compiler_options)
        return getattr(hip, kernel_name)

'''
@jit(signature="(type arg, ...)")
def kernel(J):
    ...
'''
class jit:
    def __init__(self, signature, extra_compiler_options = ""):
        self.signature = signature
        self.extra_compiler_options = extra_compiler_options

    def __call__(self, func):
        assert callable(func)
        # here we need to avoid generating same temp HIP source name for different kernel with same name.
        # otherwise it will try to generate same CO binary and causes conflict when being loaded & used
        file_info = inspect.getfile(func)
        line_no = inspect.getsourcelines(func)[1]
        filename = os.path.basename(file_info)
        J = JIT()
        func(J)
        func_name = func.__name__
        return J.build(func_name, self.signature, self.extra_compiler_options, temp_filename = f"jit-{filename}-{line_no}-{func_name}")

