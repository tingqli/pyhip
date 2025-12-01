from typing import List, Optional, Set, Union
from .hiptools import module
import inspect
import os
from contextlib import contextmanager

# https://llvm.org/docs/AMDGPUInstructionSyntax.html#amdgpu-syn-instructions
class Instruction:
    def __init__(self, parent_bb:'BasicBlock', opcode, loc=""):
        self.opcode = opcode
        self.parent_bb = parent_bb
        assert not opcode.startswith("s_call")
        self.is_branch = self.opcode.startswith("s_branch")
        self.is_cbranch = self.opcode.startswith("s_cbranch_")
        self.debug_info = ""
        self.sid = 0
        self.loc = loc

    def __call__(self, *operands, mod:str="", insert_bb_pos = None):
        self.operands = list(operands)
        self.mod = mod
        jit = self.parent_bb.jit
        for i, op in enumerate(operands):
            # generate expression
            if isinstance(op, GPRExpr) and op.op != "getitem":
                rtype = op.find_rtype()
                dtype = op.find_dtype()
                dst_gprs = jit.new_gpr(rtype, 1, dtype=dtype, align=1, name="")
                dst_expr = GPRExpr("getitem", dst_gprs, 0, 0)
                jit.recursive_expr_gen(self.parent_bb, dst_expr, op, loc=inspect.currentframe().f_back.f_lineno)
                self.operands[i] = dst_expr

            assert isinstance(op, GPRExpr) or \
                   isinstance(op, int) or isinstance(op, float) or \
                   (isinstance(op, str) and (op in ["scc", "vcc", "exec", "off", "execz", "m0"])) or \
                   (isinstance(op, str) and op.startswith("0x")) or \
                   isinstance(op, GPRs), \
            f"arg {i} : {type(op)} {op}"
        self.parent_bb.add_instruction(self, insert_bb_pos)

    def isVALU(self):
        return self.opcode.startswith("v_")

    def isTransOp(self):
        return self.opcode[:6] in ["v_exp_","v_log_","v_rcp_","v_rsq_","v_sqrt","v_sin_","v_cos_"]

    def op_repr(self, op):
        if isinstance(op, str):
            return op
        if isinstance(op, int):
            return hex(op)
        return repr(op)

    def __repr__(self):
        return f"{self.opcode} {','.join([self.op_repr(op) for op in self.operands])} {self.mod} ; {self.debug_info}"

class GPRExpr:
    def __init__(self, op:str, src0=None, src1=None, src2=None):
        self.op=op
        # some instructions only support const on operand src0
        if op in ["+", "*", "&", "|", "^"] and (isinstance(src1, int) or isinstance(src1, float)):
            src0, src1 = src1, src0
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
    def __floordiv__(self, other):
        assert isinstance(other, int)
        if other <= 0 or (other & (other - 1)) != 0:
            assert "the divisor is not power of 2"
        shift_right_bits = other.bit_length() - 1
        return GPRExpr(">>", self, shift_right_bits)
    def __mod__(self, other):
        assert isinstance(other, int)
        if other <= 0 or (other & (other - 1)) != 0:
            assert "the divisor is not power of 2"
        return GPRExpr("&", self, (other-1))
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

    def overlap(self, other: Union['GPRExpr','GPRs']):
        assert self.op == "getitem"
        if isinstance(other, GPRs):
            other = other[0:other.count-1]
        assert other.op == "getitem"
        gprs0 = self.src0
        gprs1 = other.src0
        if gprs0.rtype != gprs1.rtype:
            return False
        first0 = gprs0.start_id + self.src1
        last0 = gprs0.start_id + self.src2
        
        first1 = gprs1.start_id + other.src1
        last1 = gprs1.start_id + other.src2

        if first1 > last0 or first0 > last1:
            return False
        return True

    # given: a = gprs[4:7]
    #   a[0] -> gprs[4]
    #   a[0:1] -> gprs[4:5]
    def __getitem__(self, key):
        assert self.op == "getitem", f"{self}"
        gprs = self.src0
        base_offset = self.src1
        if isinstance(key, slice):
            s = key
            assert s.step == 1 or s.step is None, f"slice {s} with step not eq 1"
            return GPRExpr("getitem", gprs, base_offset + s.start, base_offset + s.stop)
        elif isinstance(key, int):
            idx = key
            return GPRExpr("getitem", gprs, base_offset + idx, base_offset + idx)
        else:
            assert 0, f"unsupported key {key}"

    # given: a = gprs[4:7]
    #    a[0:1] = ...  is equivalent to gprs[4:5]=...
    def __setitem__(self, key, value:Union['GPRExpr',int,float]):
        assert self.op == "getitem"
        gprs = self.src0
        dst = self[key]
        #inst = Instruction(gprs.jit.current_bb, "expression_place_holder")
        #inst(dst, value)
        gprs.jit.recursive_expr_gen(gprs.jit.current_bb, dst, value, loc=inspect.currentframe().f_back.f_lineno)

    def find_dtype(self):
        if self.op == "getitem":
            return self.src0.dtype
        if isinstance(self.src0, GPRExpr):
            return self.src0.find_dtype()
        if isinstance(self.src1, GPRExpr):
            return self.src1.find_dtype()
        assert 0

    def find_rtype(self):
        if self.op == "getitem":
            return self.src0.rtype
        rtype0 = None
        rtype1 = None
        if isinstance(self.src0, GPRExpr):
            rtype0 = self.src0.find_rtype()
            if rtype0 == "v": return rtype0
        if isinstance(self.src1, GPRExpr):
            rtype1 = self.src1.find_rtype()
            if rtype1 == "v": return rtype1
        assert rtype0 != "a"
        assert rtype1 != "a"
        if rtype0 == "s": return rtype0
        if rtype1 == "s": return rtype1
        assert 0

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

    def overlap(self, other: Union['GPRExpr','GPRs']):
        return self[0:self.count-1].overlap(other)

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
        #inst = Instruction(self.jit.current_bb, "expression_place_holder")
        #inst(dst, value)
        self.jit.recursive_expr_gen(self.jit.current_bb, dst, value, loc=inspect.currentframe().f_back.f_lineno)
        # expression_place_holder will be compiled later when all program is ready
        # all expressions within same BB can be processed together

    '''
    all magic methods GPRExpr supports can also be supported
    GPRs will convert self into a GPRExpr before compose expression
    '''
    def to_expr(self):
        if self.count == 1:
            return self[0]
        else:
            return self[0:(self.count - 1)]

    def __add__(self, other):
        return GPRExpr("+", self.to_expr(), other)
    def __radd__(self, other):
        return GPRExpr("+", other, self.to_expr())
    def __sub__(self, other):
        return GPRExpr("-", self.to_expr(), other)
    def __rsub__(self, other):
        return GPRExpr("-", other, self.to_expr())
    def __mul__(self, other):
        return GPRExpr("*", self.to_expr(), other)
    def __rmul__(self, other):
        return GPRExpr("*", self.to_expr(), other)
    def __lshift__(self, other):
        return GPRExpr("<<", self.to_expr(), other)
    def __rlshift__(self, other):
        return GPRExpr("<<", other, self.to_expr())
    def __rshift__(self, other):
        return GPRExpr(">>", self.to_expr(), other)
    def __rrshift__(self, other):
        return GPRExpr(">>", other, self.to_expr())
    def __and__(self, other):
        return GPRExpr("&", self.to_expr(), other)
    def __rand__(self, other):
        return GPRExpr("&", self.to_expr(), other)
    def __or__(self, other):
        return GPRExpr("|", self.to_expr(), other)
    def __ror__(self, other):
        return GPRExpr("|", self.to_expr(), other)
    def __xor__(self, other):
        return GPRExpr("^", self.to_expr(), other)
    def __rxor__(self, other):
        return GPRExpr("^", self.to_expr(), other)
    # overloading __eq__ methods also effects compile-time behaviour
    # for example we cannot correctly using list's index or `in` methods
    #def __eq__(self, other):
    #    return GPRExpr("eq", self.to_expr(), other)
    #def __ne__(self, other):
    #    return GPRExpr("ne", self.to_expr(), other)
    def __lt__(self, other):
        return GPRExpr("lt", self.to_expr(), other)
    def __gt__(self, other):
        return GPRExpr("gt", self.to_expr(), other)
    def __le__(self, other):
        return GPRExpr("le", self.to_expr(), other)
    def __ge__(self, other):
        return GPRExpr("ge", self.to_expr(), other)

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
            self.jit._finish_bb(self)
            target_lable_str = instr.mod
            self.add_successor(target_lable_str)
        elif instr.is_cbranch:
            # two possible successor
            self.jit._finish_bb(self)
            self.jit.current_bb = BasicBlock(self.jit, "")
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

class Buffer:
    def __init__(self, J):
        self.J = J
        self.desc = J.new_gpr('s', 4, align=4)
        self.base = self.desc[0:1]
        self.range = self.desc[2]
        self.config = self.desc[3]
        J.s_mov_b32(self.config, 0x00020000)

    def setup(self, base, size):
        self.base[0] = base[0]
        self.base[1] = base[1]
        self.range[0] = size # size可以是GPRExpr

    def load_dwordx4(self, vdst, voffset, soffset, offset12=0):
        # vdst,     vaddr,           srsrc, soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        self.J.buffer_load_dwordx4(vdst, voffset, self.desc, soffset, mod=mod)

    def store_dwordx4(self, vdata, voffset, soffset, offset12=0):
        # vdata,    vaddr,        srsrc,  soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        self.J.buffer_store_dwordx4(vdata, voffset, self.desc, soffset, mod=mod)

    def load_dword_lds(self, voffset, soffset, offset12=0):
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        mod += " lds"
        self.J.buffer_load_dword(voffset, self.desc, soffset, mod=mod)

    def store_dword(self, vdata, voffset, soffset, offset12=0):
        # vdata,    vaddr,        srsrc,  soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        self.J.buffer_store_dword(vdata, voffset, self.desc, soffset, mod=mod)

# JIT emits instructions into BBs
all_kernel_hip_src_names = {}
class JIT:
    def __init__(self):
        self.blocks = []
        # increased freely
        self.free_gpr_id = {'s':0, 'v':0, 'a': 0}
        self.free_lds_offset = 0
        self.label2bb = {}
        self.current_bb = BasicBlock(self, "_jit_main")
        self.relocatable_gprs = []
        self.fixed_gprs = []


    def __getattr__(self, instruction):
        if self.current_bb is None:
            self.current_bb = BasicBlock(self, label_name="")
        return Instruction(self.current_bb, instruction, loc=inspect.currentframe().f_back.f_lineno)

    def _finish_bb(self, bb):
        if bb is None:
            self.current_bb = None
            return
        assert bb is self.current_bb
        self.blocks.append(bb)
        self.current_bb = None

    def Label(self, name=""):
        # creat new basic block, archieve current bb
        old_bb = self.current_bb
        if old_bb is None:
            old_bb = self.blocks[-1]
        self._finish_bb(self.current_bb)
        self.current_bb = BasicBlock(self, name)
        old_bb.add_successor(self.current_bb)
        return self.current_bb

    def log(self, *args, **kwargs):
        PYHIP_JIT_LOG = int(os.getenv("PYHIP_JIT_LOG", "1"))
        if PYHIP_JIT_LOG:
            color_id = 3
            color0 = f"\033[0;{30+(color_id % 8)}m"
            color1 = f"\033[0m"
            print(color0, f"[{PYHIP_JIT_LOG=}] ", *args, color1, **kwargs)

    def Buffer(self, sgpr_base, sgpr_size):
        buff = Buffer(self)
        buff.setup(sgpr_base, sgpr_size)
        return buff

    '''
    # scalar jump (wave level, no divergent)
    '''
    def Jump(self, label:str, cond:GPRExpr = None, reverse = False):
        if cond is None:
            self.s_branch(mod=label)
        else:
            # generate expression into scc
            dtype = cond.find_dtype()
            dst_gprs = self.new_gpr("s", 1, dtype=dtype, align=1, name="")
            dst_expr = GPRExpr("getitem", dst_gprs, 0, 0)
            self.recursive_expr_gen(self.current_bb, dst_expr, cond, loc=inspect.currentframe().f_back.f_lineno)
            # optimize 
            last_inst = self.current_bb.instructions[-1]
            if last_inst.opcode == "s_mov_b32" and last_inst.operands[0] == dst_expr and last_inst.operands[1] == "scc":
                self.log("s_mov_b32 scc optimized")
                self.current_bb.instructions.pop()
            else:
                # we need to mov dst_expr into scc, s_or can update scc:
                # D0.u32 = (S0.u32 | S1.u32);
                # SCC = D0.u32 != 0U
                self.s_or_b32(dst_expr, dst_expr, 0)
            if reverse:
                self.s_cbranch_scc0(mod=label)
            else:
                self.s_cbranch_scc1(mod=label)

    @contextmanager
    def While(self, cond:GPRExpr = None):
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back.f_back
        lineno = caller_frame.f_lineno
        label_begin = f"_while_begin_{lineno}"
        label_end = f"_while_end_{lineno}"
        self.Label(label_begin)
        if cond is not None:
            self.Jump(label_end, cond, reverse=True)
        try:
            # following dict is for loop body code to continue or break
            yield {"begin":label_begin, "end":label_end}
        finally:
            self.Jump(label_begin)
            self.Label(label_end)

    '''
    Use this to set exec-mask to handle the tail/vari-SIMD-length problem
    '''
    @contextmanager
    def ExecMask(self, cond:GPRExpr = None):
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back.f_back
        lineno = caller_frame.f_lineno
        label_begin = f"_execmask_begin_{lineno}"
        label_end = f"_execmask_end_{lineno}"
        
        dtype = cond.find_dtype()
        dst_gprs = self.new_gpr("v", 1, dtype=dtype, align=1)
        dst_expr = GPRExpr("getitem", dst_gprs, 0, 0)
        self.recursive_expr_gen(self.current_bb, dst_expr, cond, loc=inspect.currentframe().f_back.f_lineno)
        last_inst = self.current_bb.instructions[-1]
        if last_inst.opcode == "v_cndmask_b32_e64" and \
            last_inst.operands[0] == dst_expr and \
            last_inst.operands[1] == 0 and \
            last_inst.operands[2] == 1 and \
            last_inst.operands[3] == "vcc" :
            self.log("v_cndmask_b32_e64 dst,0,1,vcc is optimized")
            self.current_bb.instructions.pop()
        else:
            self.v_cmp_ne_u32_e64("vcc", 0, dst_gprs)
        exec_backup = self.new_gpr("s", 2, dtype="i32", align=2)
        self.s_and_saveexec_b64(exec_backup, "vcc") # scc = (exec!=0)
        self.s_cbranch_execz(mod=label_end) # early skip
        try:
            yield # the body of computation with ExecMask
        finally:
            self.Label(label_end)
            self.s_mov_b64("exec", exec_backup)
            # if we want to do something when execz happens, we can use scc0

    '''
    SIMT while: just a prototype with bugs, do not use
    '''
    @contextmanager
    def SIMTWhile(self, cond:GPRExpr = None):
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back.f_back
        lineno = caller_frame.f_lineno
        label_begin = f"_simtwhile_begin_{lineno}"
        label_end = f"_simtwhile_end_{lineno}"
        exec_backup = self.new_gpr("s", 2, dtype="i32", align=2)
        # exec_stopped = self.new_gpr("s", 2, dtype="i32", align=2)

        # generate expression into vcc
        dtype = cond.find_dtype()
        def generate_not_cond_in_vcc():
            dst_gprs = self.new_gpr("v", 1, dtype=dtype, align=1)
            dst_expr = GPRExpr("getitem", dst_gprs, 0, 0)
            self.recursive_expr_gen(self.current_bb, dst_expr, cond)
            last_inst = self.current_bb.instructions[-1]
            if last_inst.opcode == "v_cndmask_b32_e64" and \
                last_inst.operands[0] == dst_expr and \
                last_inst.operands[1] == 0 and \
                last_inst.operands[1] == 1 and \
                last_inst.operands[1] == "vcc" :
                self.log("v_cndmask_b32_e64 dst,0,1,vcc is optimized")
                self.current_bb.instructions.pop()
                # above code generate condition to keep alive
                # we need vcc[lane]=1 when cond is false
                self.s_not_b64("vcc","vcc")
            else:
                self.v_cmp_eq_u32_e64("vcc", 0, dst_gprs)

        generate_not_cond_in_vcc()
        # backup exec, set exec-mask based on vcc
        self.s_and_saveexec_b64(exec_backup, "vcc")
        # jump to end if no lanes are alive
        self.s_cbranch_execz(mod=label_end)
        self.Label(label_begin)
        try:
            # following dict is for loop body code to continue or break
            yield {"begin":label_begin, "end":label_end}
        finally:
            generate_not_cond_in_vcc()
            self.s_andn2_b64("exec", "exec", "vcc")
            self.s_cbranch_execnz(mod=label_begin)
            self.Label(label_end)
            self.s_or_b64("exec", "exec", exec_backup)

    def _align_up(self, a, align):
        return ((a + align - 1)//align) * align

    def auto_gpr(self, expr:GPRExpr, name="", loc=""):
        if name == "":
            # try to reover python var's name from code_context
            stack = inspect.stack()
            caller_frame = stack[1]
            if caller_frame.code_context:
                src_line = caller_frame.code_context[0].strip()
                name = src_line.split("=")[0].strip()
        # derive dtype reg_type from expr & allocate
        rtype = expr.find_rtype()
        dtype = expr.find_dtype()
        gprs = self.new_gpr(rtype, 1, dtype=dtype, align=1, name=name)
        if loc == "":
            loc = inspect.currentframe().f_back.f_lineno
        self.recursive_expr_gen(self.current_bb, gprs[0], expr, loc=loc)
        return gprs


    '''
    si32[0:1]   reserve s[0:1] as two i32
    vi32[0]     reserve v[0] as i32
    af32[0:255] reserve a[0:255] as f32

    si32x2      alloc two scalar i32 
    su32x4      alloc four scalar i32
    vf32x4      alloc four vector f32
    '''
    def gpr(self, desc:Union[str,GPRExpr], align=0, name=""):
        if name == "":
            # try to reover python var's name from code_context
            stack = inspect.stack()
            caller_frame = stack[1]
            if caller_frame.code_context:
                src_line = caller_frame.code_context[0].strip()
                if "=" in src_line:
                    items = src_line.split("=")
                    if len(items) > 1:
                        if ".gpr(" in items[1]:
                            name = items[0].strip()

        if isinstance(desc, GPRExpr):
            return self.auto_gpr(desc, name=name, loc=inspect.currentframe().f_back.f_lineno)

        desc = desc.strip()
        rtype = desc[0]
        if "[" in desc[1:]:
            dtype, range = desc[1:].split("[")
            assert range[-1] == "]"
            range = range[:-1]
            if ":" in range:
                first, last = range.split(":")
                count_range = (int(first), int(last))
            else:
                count_range = (int(range),int(range))
        else:
            dtype_cnt = desc[1:].split("x")
            dtype = dtype_cnt[0]
            if len(dtype_cnt) == 2:
                count_range = int(dtype_cnt[1])
            else:
                count_range = 1
            if align == 0: # derive alignment automatically
                align = min(count_range, 4)//2*2
        align = max(align, 1)
        return self.new_gpr(rtype, count_range, dtype, align, name)


    def new_gpr(self, reg_type, count_range, dtype="", align=1, name=""):
        if name == "":
            # try to reover python var's name from code_context
            stack = inspect.stack()
            caller_frame = stack[1]
            if caller_frame.code_context:
                src_line = caller_frame.code_context[0].strip()
                name = src_line.split("=")[0].strip()

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
                if len(parent_bb.instructions) == 0: continue
                # assert len(parent_bb.instructions) > 0, f"{parent_bb.label_name} is empty"
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
            gprs.start_id = -1 # set gprs status to "not allocated"

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
            # if one instruction happens to be the first & last use-site of one reg
            # then, it uses the last one as src, and first one as dst, and they can be same
            # so we free befor alloc.
            # free
            unalloc_gprs = []
            for ev,gprs in events:
                if ev == 1: # last use(as src)
                    if gprs.start_id >= 0:
                        free_gpr(gprs)
                        self.asm_debug_info += f";free '{gprs.name}'{gprs} at #{sid}\n"
                    else:
                        unalloc_gprs.append(gprs)
            # allocation 
            for ev,gprs in events:
                if ev == 0: # first use(as dst)
                    alloc_gpr(gprs)
                    self.asm_debug_info += f";alloc '{gprs.name}'{gprs} at #{sid}\n"

            # in case some gpr's last & first sids are the same
            for gprs in unalloc_gprs:
                free_gpr(gprs)

        # (following code-gen phase will use the new start_id automatically)
        # add debug info to asm instruction
        for bb in self.blocks:
            for inst in bb.instructions:
                debug_info = f"#{inst.loc}   sid:{inst.sid} "
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
    def recursive_expr_gen(self, bb, dst_expr:GPRExpr,  expr:Union['GPRExpr',int,float], loc=""):
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
            new_inst = Instruction(bb, f"{rtype}_mov_b32", loc=loc)
            new_inst(dst_expr, expr)
            return
        assert isinstance(expr, GPRExpr)
        if expr.op == "getitem":
            # assign
            src_cnt = expr.src2 - expr.src1 + 1
            assert src_cnt == 1
            new_inst = Instruction(bb, f"{rtype}_mov_b32", loc=loc)
            new_inst(dst_expr, expr)
            return

        if dtype == "":
            if expr.op not in ["&","|","^"]:
                dtype = expr.find_dtype()
                self.log(f"infer dtype={dtype} by expr {expr}")
                assert dtype != ""

        if isinstance(expr.src0, GPRExpr):
            if expr.src0.op == "getitem":
                # "getitem" expr can be used as operand directly
                src0_operand = expr.src0
            else:
                src0_gprs = self.new_gpr(rtype, 1, dtype=dtype, align=1, name="")
                src0_operand = GPRExpr("getitem", src0_gprs, 0, 0)
                self.recursive_expr_gen(bb, src0_operand, expr.src0, loc=loc)
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
                self.recursive_expr_gen(bb, src1_operand, expr.src1, loc=loc)
        else:
            src1_operand = expr.src1 # int,float,...
            if isinstance(src1_operand, float): # convert float-const into int?
                src1_operand = float_to_ieee754_bits_little(src1_operand)

        def vgpr_add_sub_ui32(op):
            nonlocal src0_operand, src1_operand
            if isinstance(src0_operand, GPRExpr) and isinstance(src1_operand, GPRExpr):
                return Instruction(bb, f"v_add_u32_e32" if op == '+' else f"v_sub_u32_e32", loc=loc)
            if isinstance(src0_operand, GPRExpr):
                if op == '-':
                    src1_operand = hex(-src1_operand & 0xffffffff)
                # v_add_u32_e32's src0 can be const
                src1_operand, src0_operand = src0_operand, src1_operand
                return Instruction(bb, f"v_add_u32_e32", loc=loc) # vgpr + (+/- const)
            if isinstance(src1_operand, GPRExpr):
                if op == '-':
                    return Instruction(bb, f"v_sub_u32_e32") # const - vgpr
                return Instruction(bb, f"v_add_u32_e32", loc=loc) # const + vgpr
            assert 0

        # now src0_operand & src1_operand are generated, we can generate our result
        if expr.op == "+":
            if rtype == "s":
                new_inst = Instruction(bb, f"{rtype}_add_{dtype}", loc=loc)
            elif rtype == "v":
                if dtype in ["u32", "i32"]:
                    new_inst = vgpr_add_sub_ui32(expr.op)
                else:
                    new_inst = Instruction(bb, f"{rtype}_add_{dtype}", loc=loc)
            else:
                assert 0
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op == "-":
            if rtype == "s":
                new_inst = Instruction(bb, f"{rtype}_sub_{dtype}", loc=loc)
            elif rtype == "v":
                if dtype in ["u32", "i32"]:
                    new_inst = vgpr_add_sub_ui32(expr.op)
                else:
                    new_inst = Instruction(bb, f"{rtype}_sub_{dtype}", loc=loc)
            else:
                assert 0
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op == "*":
            if rtype == "s":
                if dtype == "u32":
                    self.log("s_mul_u32 not exist, using s_mul_i32 instead")
                    new_inst = Instruction(bb, f"s_mul_i32", loc=loc)
                else:
                    new_inst = Instruction(bb, f"s_mul_{dtype}", loc=loc)
            elif rtype == "v":
                if dtype in ["u32", "u16"]:
                    new_inst = Instruction(bb, f"v_mul_lo_{dtype}", loc=loc)
                elif dtype in ["f32", "f16"]:
                    new_inst = Instruction(bb, f"v_mul_{dtype}", loc=loc)
                elif dtype in ["i32"]:
                    new_inst = Instruction(bb, f"v_mul_i32_i24", loc=loc)
                else:
                    assert 0, f"unsupported v_mul dtype: {dtype}"
            else:
                assert 0, f"unsupported v_mul rtype: {rtype}"
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op == "<<":
            if rtype == "s":
                new_inst = Instruction(bb, f"s_lshl_b32", loc=loc)
                new_inst(dst_expr, src0_operand, src1_operand)
            elif rtype == "v":
                new_inst = Instruction(bb, f"v_lshlrev_b32", loc=loc)
                new_inst(dst_expr, src1_operand, src0_operand)
            else:
                assert 0, f"unsupported v_mul rtype: {rtype}"
        elif expr.op == ">>":
            if rtype == "s":
                if dtype == "u32":
                    new_inst = Instruction(bb, f"s_lshr_b32", loc=loc)
                elif dtype == "i32":
                    new_inst = Instruction(bb, f"s_ashr_i32", loc=loc)
                else:
                    assert 0, f"unsupported sgpr shift right dtype {dtype}"
                new_inst(dst_expr, src0_operand, src1_operand)
            elif rtype == "v":
                if dtype in ["u32","f32"]:
                    new_inst = Instruction(bb, f"v_lshrrev_b32", loc=loc)
                elif dtype == "i32":
                    new_inst = Instruction(bb, f"v_ashrrev_i32", loc=loc)
                else:
                    assert 0, f"unsupported vgpr shift right dtype {dtype}"
                new_inst(dst_expr, src1_operand, src0_operand)
            else:
                assert 0, f"unsupported rtype: {rtype}"
        elif expr.op == "&":
            new_inst = Instruction(bb, f"{rtype}_and_b32", loc=loc)
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op == "|":
            new_inst = Instruction(bb, f"{rtype}_or_b32", loc=loc)
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op == "^":
            new_inst = Instruction(bb, f"{rtype}_xor_b32", loc=loc)
            new_inst(dst_expr, src0_operand, src1_operand)
        elif expr.op in ["eq","ne","lt","gt","le","ge"]:
            if rtype == "s":
                op = expr.op
                if op == "ne" : op = "lg"
                cmp = Instruction(bb, f"s_cmp_{op}_{dtype}", loc=loc)
                mov = Instruction(bb, f"s_mov_b32", loc=loc)
                cmp(src0_operand, src1_operand)
                mov(dst_expr, "scc")
            elif rtype == "v":
                op = expr.op
                # if op == "ne" : op = "lg"
                # only src0 can be const
                if not isinstance(src1_operand, GPRExpr):
                    src1_operand, src0_operand = src0_operand, src1_operand
                    cmp_op_reverse = {"gt":"lt", "lt":"gt", "le":"ge", "ge":"le", "eq":"eq", "ne":"ne"}
                    op = cmp_op_reverse[op]
                cmp = Instruction(bb, f"v_cmp_{op}_{dtype}_e32", loc=loc)
                mov = Instruction(bb, f"v_cndmask_b32_e64", loc=loc)
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

    def insert_nop(self):
        for bb in self.blocks:
            i = 1
            while i < len(bb.instructions):
                prev = bb.instructions[i-1]
                cur = bb.instructions[i]
                loc = cur.loc
                n_nops = -1
                if cur.opcode == "v_readfirstlane_b32":
                    vsrc0 = cur.operands[1]
                    if prev.opcode.startswith("v_"):
                        vdst = prev.operands[0]
                        if vdst.overlap(vsrc0):
                            n_nops = 1
                            self.log(f"insert s_nop({n_nops}) at #{loc} : [VALU writes VGPRn,v_readlane vsrc0 reads VGPRn]")
                    
                if prev.isTransOp() and (not cur.isTransOp()) and (cur.isVALU()):
                    vdst = prev.operands[0]
                    for op in cur.operands:
                        if (isinstance(op, GPRExpr) or isinstance(op, GPRs)) and vdst.overlap(op):
                            n_nops = 1
                            self.log(f"insert s_nop({n_nops}) at #{loc} : [VALU Trans op, Non-trans VALU op consumes result of that op]")
                            break

                if n_nops >= 0:
                    inst = Instruction(bb, "s_nop")
                    inst(n_nops, insert_bb_pos = i)

                i += 1

    def run_threads(self, *thread_jits):
        # generate instructions to special BB
        oldbb = self.current_bb

        # collect all instructions from each thread
        class VThread:
            def __init__(self, bb:BasicBlock, name, signal_table):
                self.name = name
                self.bb = bb
                self.index = 0      # current instruction
                self.waitting = False
                self.finished = False
                self.next()  # initialize self.inst
                self.signal_table = signal_table

            def progress():
                return self.index * 4

            def next(self):
                if self.finished or self.index >= len(self.bb.instructions):
                    self.finished = True
                    return None

                while True:
                    self.inst = self.bb.instructions[self.index]
                    self.index += 1
                    if self.inst.opcode.startswith == "signal":
                        signal_name = self.inst.mod
                        if signal_name in self.signal_table:
                            assert self.signal_table[signal_name][0] == 0
                            self.signal_table[signal_name][0] = 1
                            wait_thr = self.signal_table[signal_name][1]
                            wait_thr.waitting = False
                        else:
                            self.signal_table[signal_name] = [0, None]
                        continue # signal thread keep fetching next instruction

                    if self.inst.opcode.startswith == "wait":
                        # go into wait status
                        signal_name = self.inst.mod
                        if signal_name in self.signal_table:
                            signaled, thr = self.signal_table[signal_name]
                            assert (thr is self) or (thr is None), "only 1 thread can wait on a signal only once"
                        else:
                            self.signal_table[signal_name] = [0, self] # so other thread can wake me up
                            signaled = 0
                        self.waitting = not signaled
                        if self.waitting:
                            return
                        else:
                            # has signaled,just fetch next instruction
                            continue
                    break

                self.is_mem_load = self.inst.opcode.startswith("global_load_") or self.inst.opcode.startswith("buffer_load_")
                self.is_mfma = self.inst.opcode.startswith("v_mfma_")
                self.is_dsrd = self.inst.opcode.startswith("ds_read")
                self.is_dswr = self.inst.opcode.startswith("ds_write")
                self.is_ds = self.is_dsrd or self.is_dswr
                return self.inst

        vthreads = []
        for thr in thread_jits:
            bb = BasicBlock(self, thr.__name__)
            self.current_bb = bb
            thr(self)
            vthreads.append(VThread(bb, thr.__name__))
        
        self.current_bb = oldbb
        # schedule instructions from bbs into self.current_bb
        ISSUE_INTERVAL = { "vmem": 113,"ds":32, "mfma": 16}
        last_global_load_cycle = -100000
        last_ds_cycle = -100000
        last_mfma_cycle = -100000
        clock_cycle = 0

        is_all_finished = False
        while not is_all_finished:
            # select instruction : also handle signal & wait pseudo instruction
            min_progress = 1e12
            selected_thr = None
            is_all_finished = True
            for thr in vthreads:
                if thr.finished: continue
                is_all_finished = False
                if thr.waitting: continue
                # check issue-interval
                if thr.is_mem_load and (clock_cycle - last_global_load_cycle) < ISSUE_INTERVAL["vmem"]: continue
                if thr.is_ds and (clock_cycle - last_ds_cycle) < ISSUE_INTERVAL["vmem"]: continue
                if thr.is_mfma and (clock_cycle - last_mfma_cycle) < ISSUE_INTERVAL["mfma"]: continue
                # select the candidate with minimal timeslice
                progress = thr.progress()
                if min_progress > progress:
                    min_progress = progress
                    selected_thr = thr

            if selected_thr is not None:
                self.current_bb.add_instruction(selected_thr.inst)
                selected_thr.next()
                if selected_thr.is_mem_load: last_global_load_cycle = clock_cycle
                if selected_thr.is_ds: last_ds_cycle = clock_cycle
                if selected_thr.is_mfma: last_mfma_cycle = clock_cycle
            else:
                pass # just wait for some threads ready

            # assume each issue took 4 cycles
            clock_cycle += 4

    def alloc_lds(self, num_bytes, align=4):
        # aways alloc, no free
        offset = ((self.free_lds_offset + align - 1)//align) * align
        self.free_lds_offset = offset + num_bytes
        return offset

    def build(self, kernel_name, signature, extra_compiler_options, temp_filename):
        self.asm_debug_info = ""
        self._finish_bb(self.current_bb)

        # remove unreachable bb
        bb_to_remove = []
        for bb in self.blocks[1:]:
            if len(bb.predecessors) == 0:
                self.log(f"remove unreachable code block {bb.label_name}")
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
        
        # remove empty bb
        '''
        asm=""
        for bb in self.blocks:
            asm += repr(bb)
        print(asm)
        bb_to_remove = []
        for bb in self.blocks[:-1]:
            if len(bb.instructions) == 0:
                assert len(bb.predecessors) == 1, f"{bb.label_name}"
                assert len(bb.successors) == 1
                bb.predecessors[0].successors.remove(bb)
                bb.successors[0].predecessors.remove(bb)
                bb.predecessors[0].add_successor(bb.successors[0])
                bb_to_remove.append(bb)
        for empty_bb in bb_to_remove:
            self.blocks.remove(empty_bb)   
        '''

        # use following way only
        # if we want to do multi-expression optimization (like common expr extraction...)
        # self.compile_expressions()
        self.insert_nop()
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
        self.log(f" kernel: {kernel_name}{signature}  used_gprs={used_gprs}")

        decl_lds = ""
        if self.free_lds_offset > 0:
            # (as3_uint32_ptr)(lds_buffer) is always 0
            # but this 2 lines of code has side effect: compiler may use s0 as `lds_buffer`` input 
            # which causes damage to kernal-arg pointer s[0:1], we can work-around it by putting
            # these 2 lines of codes at the end of the kernel's source code.
            decl_lds += f"    __shared__ uint lds_buffer[{self.free_lds_offset//4}];\n"
            decl_lds += f'    asm(" ; lds_buffer %0 "::"s"((as3_uint32_ptr)(lds_buffer)));'
            pass

        hip_src =r'''
#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;
__global__ void ''' + kernel_name + signature +  r''' {
    // force HIP to initialize sgpr2/sgpr3/sgpr4, TODO: add threadIdx.y/z
    asm volatile(" ; blockIdx.x %0, blockIdx.y %1, blockIdx.z %2"::"s"(blockIdx.x),"s"(blockIdx.y),"s"(blockIdx.z));
    asm volatile("\n"
''' + "\n" + inline_asm + \
r'''
                ::
                :"memory"''' + str_used_gprs + r''');
''' + decl_lds + r'''
}
        '''
        import os
        cpp_src_fpath = os.path.join(os.getcwd(),f'{temp_filename}.cpp')
        with open(cpp_src_fpath, 'w', encoding='utf-8') as f:
            f.write(hip_src)
        hip = module(cpp_src_fpath, extra_compiler_options)
        return getattr(hip, kernel_name)

    '''
    reduce("v_max_f32", vinput)
    reduce("v_add_f32", vinput)
    '''
    def reduce(self, vinst, vinput):
        self.s_nop(mod="2")
        v1 = self.new_gpr('v',1,dtype="i32", align=1)
        getattr(self, vinst)(v1, vinput, vinput, mod="row_shr:8 bound_ctrl:0")
        self.s_nop(mod="2")
        getattr(self, vinst)(v1, v1, v1, mod="row_shr:4 bound_ctrl:0")
        self.s_nop(mod="2")
        getattr(self, vinst)(v1, v1, v1, mod="row_shr:2 bound_ctrl:0")
        self.s_nop(mod="2")
        getattr(self, vinst)(v1, v1, v1, mod="row_shr:1 bound_ctrl:0")
        self.s_nop(mod="2")
        getattr(self, vinst)(v1, v1, v1, mod="row_bcast:15 bound_ctrl:0")
        self.s_nop(mod="2")
        getattr(self, vinst)(v1, v1, v1, mod="row_bcast:31 bound_ctrl:0")
        self.s_nop(mod="2")
        vaddr = self.new_gpr('v',1,dtype="i32", align=1)
        vaddr[0] = 63*4 # broadcast last lane to all lanes
        self.ds_bpermute_b32(v1, vaddr, v1) # vdst,  vaddr,    vdata   offset
        self.s_waitcnt(mod=f"lgkmcnt({0})")
        return v1

'''
@jit(signature="(type arg, ...)")
def kernel(J):
    ...
'''
gen_hip_file_unique_id = 0

class Idx3D:
    def __init__(self):
        pass

class jit:
    def __init__(self, extra_compiler_options = ""):
        self.extra_compiler_options = extra_compiler_options

    def __call__(self, func):
        assert callable(func)
        # here we need to avoid generating same temp HIP source name for different kernel with same name.
        # otherwise it will try to generate same CO binary and causes conflict when being loaded & used
        file_info = inspect.getfile(func)
        line_no = inspect.getsourcelines(func)[1]
        filename = os.path.basename(file_info)

        argspec = inspect.getfullargspec(func)
        argtypes = func.__annotations__

        J = JIT()
        # create special sgpr for args
        # and generate codes to load these args
        signatures = []
        sgpr_args = []
        J.kargs = J.new_gpr('s',[0,1],name="kargs")
        J.threadIdx = Idx3D()
        J.blockIdx = Idx3D()
        J.threadIdx.x = J.new_gpr('v',[0,0], dtype="u32", name="threadIdx.x")
        J.blockIdx.x = J.new_gpr('s',[2,2], dtype="u32", name="blockIdx.x")
        J.blockIdx.y = J.new_gpr('s',[3,3], dtype="u32", name="blockIdx.y")
        J.blockIdx.z = J.new_gpr('s',[4,4], dtype="u32", name="blockIdx.z")
        arg_offset = 0
        for arg_name in argspec.args[1:]:
            assert arg_name in argtypes
            atype = argtypes[arg_name].strip()
            assert isinstance(atype, str)
            signatures.append(f"{atype} {arg_name}")
            if atype.endswith("*"):
                arg_offset = ((arg_offset + 7) // 8) * 8
                sgpr = J.new_gpr('s',2, dtype="u32", align=2, name=arg_name)
                J.s_load_dwordx2(sgpr, J.kargs, arg_offset)
                sgpr_args.append(sgpr)
                arg_offset += 8
                continue
            if atype in ["int","uint","unsigned int"]:
                arg_offset = ((arg_offset + 3) // 4) * 4
                sgpr = J.new_gpr('s',1, dtype=f"{atype[0]}32", align=1, name=arg_name)
                J.s_load_dword(sgpr, J.kargs, arg_offset)
                sgpr_args.append(sgpr)
                arg_offset += 4
                continue

        if len(sgpr_args) > 0:
            J.s_waitcnt(mod=f"lgkmcnt(0)")

        # now generate your kernel code
        func(J, *sgpr_args)
        func_name = func.__name__
        global gen_hip_file_unique_id
        gen_hip_file_unique_id += 1
        return J.build(func_name, f"({','.join(signatures)})", self.extra_compiler_options,
                       temp_filename = f".jit-{gen_hip_file_unique_id}-{filename}-{line_no}-{func_name}")

