from functools import cache
from typing import List, Optional, Set, Union

import filelock
from .hiptools import module, amdgpu_arch
import inspect
import os
from contextlib import contextmanager
import math

from .mem_allocator import SimpleMemoryAllocator

import hashlib
import subprocess

import types

def get_caller_loc():
    frame = inspect.currentframe().f_back.f_back
    loc = ""
    while True:
        loc += f" {os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
        if frame.f_code.co_filename != __file__:
            break
        frame = frame.f_back
    return loc

def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        print(f'warning: get git revision error: {e}')
        return 'unknown'

@cache
def _utils_dst_operand_id(opcode, mod):
    if opcode.startswith("buffer_load_") and "lds" in mod:
        return -1
    if opcode.startswith("ds_write") or \
        opcode.startswith("global_store") or \
        opcode.startswith("global_load_lds_") or \
        opcode.startswith("flat_store_") or \
        opcode.startswith("scratch_load_lds_") or \
        opcode.startswith("buffer_store_"):
        return -1
    return 0

@cache
def _utils_is_trans_op(opcode):
    return opcode[:6] in ["v_exp_","v_log_","v_rcp_","v_rsq_","v_sqrt","v_sin_","v_cos_"]

@cache
def _utils_is_simple_operand0_store(opcode, mod):
    '''
      simple operand0_store :
        - operands[0] is the only state-change of this instruction
        - no other side-effect other than store to 1st op
    '''
    if opcode.startswith("s_load_"):
        return True
    elif opcode.startswith("s_scratch_load_"):
        return True
    elif opcode.startswith("s_buffer_load"):
        return True
    elif opcode.startswith("ds_read"):
        return True
    elif opcode.startswith("flat_load_"):
        return True
    elif opcode.startswith("global_load_") and (not opcode.startswith("global_load_lds_")):
        return True
    elif opcode.startswith("tbuffer_load_"):
        return True
    elif opcode.startswith("buffer_load_") and ("lds" not in mod):
        return True
    elif opcode.startswith("scratch_load_") and (not opcode.startswith("scratch_load_lds_")):
        return True
    elif opcode.startswith("ds_swizzle_"):
        return True
    elif opcode.startswith("ds_permute_") or opcode.startswith("ds_bpermute_"):
        return True
    elif opcode.startswith("ds_consume"):
        return True
    elif opcode.startswith("ds_") and ("_rtn_" in opcode):
        return True

    return False

@cache
def _utils_accept_accvgpr(opcode, mod):
    if opcode.startswith("v_accvgpr_"):
        return True
    if opcode.startswith("v_mfma_"):
        return True
    if opcode.startswith("v_smfmac_"):
        return True
    if opcode.startswith("flat_"):
        return True
    if opcode.startswith("ds_"):
        return True
    if opcode.startswith("buffer_"):
        return True
    if opcode.startswith("tbuffer_"):
        return True
    if opcode.startswith("global_"):
        return True
    if opcode.startswith("scratch_"):
        return True
    return False

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
        self.is_dead = False
        self.loc = loc

    def __call__(self, *operands, mod:str="", insert_bb_pos = None):
        self.operands = []
        self.mod = mod
        jit = self.parent_bb.jit
        for i, op in enumerate(operands):
            # generate expression
            if isinstance(op, GPRExpr) and op.op != "getitem":
                rtype = op.find_rtype()
                dtype = op.find_dtype()
                dst_gprs = jit.new_gpr(rtype, 1, dtype=dtype, align=1, name="idst")
                dst_expr = GPRExpr("getitem", dst_gprs, 0, 0)
                jit.recursive_expr_gen(self.parent_bb, dst_expr, op, loc=self.loc)
                self.operands.append(dst_expr)
                continue

            # canonicalize GPR operand into GPRExpr
            if isinstance(op, GPRs):
                op = op.to_expr()

            assert isinstance(op, GPRExpr) or \
                   isinstance(op, int) or isinstance(op, float) or \
                   (isinstance(op, str) and (op in ["scc", "vcc", "exec", "off", "execz", "m0"])) or \
                   (isinstance(op, str) and op.startswith("0x")), f"arg {i} : {type(op)} {op}"

            self.operands.append(op)

        self.parent_bb.add_instruction(self, insert_bb_pos)
        return self

    def isVALU(self):
        return self.opcode.startswith("v_")

    def dst_operand_id(self):
        return _utils_dst_operand_id(self.opcode, self.mod)

    def is_simple_operand0_store(self):
        return _utils_is_simple_operand0_store(self.opcode, self.mod)

    def isTransOp(self):
        return _utils_is_trans_op(self.opcode)

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

    def __len__(self):
        # number of registers
        if self.op == "getitem":
            return self.src2 - self.src1 + 1
        return None

    def __iter__(self):
        assert self.op == "getitem"
        gprs = self.src0
        for i in range(self.src1, self.src2 + 1):
            yield GPRExpr("getitem", gprs, i, i)

    def split_const_terms(self):
        const_terms = 0
        cur_expr = self
        while True:
            if cur_expr.op == "+":
                if isinstance(cur_expr.src0, int):
                    const_terms += self.src0
                    cur_expr = cur_expr.src1
                    continue
                if isinstance(cur_expr.src1, int):
                    const_terms += self.src1
                    cur_expr = cur_expr.src0
                    continue
            if cur_expr.op == "-" and isinstance(cur_expr.src1, int):
                const_terms -= self.src1
                cur_expr = cur_expr.src0
                continue
            break
        return const_terms, cur_expr

    def __add__(self, other):
        return GPRExpr("+", self, other)
    def __radd__(self, other):
        return GPRExpr("+", other, self)
    def __sub__(self, other):
        return GPRExpr("-", self, other)
    def __rsub__(self, other):
        return GPRExpr("-", other, self)
    def __mul__(self, other):
        if isinstance(other, int) and other >= 0:
            if (other & (other - 1)) == 0:
                shift_left_bits = other.bit_length() - 1
                return GPRExpr("<<", self, shift_left_bits)
        return GPRExpr("*", self, other)
    def __rmul__(self, other):
        if isinstance(other, int) and other >= 0:
            if (other & (other - 1)) == 0:
                shift_left_bits = other.bit_length() - 1
                return GPRExpr("<<", self, shift_left_bits)
        return GPRExpr("*", self, other)
    def __floordiv__(self, other):
        if isinstance(other, int):
            if other <= 0 or (other & (other - 1)) != 0:
                # the divisor is not power of 2
                return GPRExpr("floordiv", self, other)
            else:
                shift_right_bits = other.bit_length() - 1
                return GPRExpr(">>", self, shift_right_bits)
        else:
            return GPRExpr("floordiv", self, other)
    def __mod__(self, other):
        assert isinstance(other, int)
        if other <= 0 or (other & (other - 1)) != 0:
            assert 0, "the divisor is not power of 2"
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

    def overlap(self, other: Union['GPRExpr','GPRs'], fully_match = False):
        assert self.op == "getitem"
        if isinstance(other, GPRs):
            other = other[...]
        assert other.op == "getitem"
        gprs0 = self.src0
        gprs1 = other.src0
        if gprs0.rtype != gprs1.rtype:
            return False
        first0 = gprs0.start_id + self.src1
        last0 = gprs0.start_id + self.src2
        
        first1 = gprs1.start_id + other.src1
        last1 = gprs1.start_id + other.src2

        if fully_match:
            return first1 == first0 and last0 == last1
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
        if key is Ellipsis:
            # support a[...]
            return GPRExpr("getitem", gprs, self.src1, self.src2)        
        elif isinstance(key, slice):
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
        gprs.jit.recursive_expr_gen(gprs.jit.current_bb, dst, value, loc=get_caller_loc())

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

# a logical continous region of GPRs allocated
# may not physically continous
class GPRs:

    def __init__(self, jit, rtype, start_id, count, dtype, align=0, name="", is_fixed=False):
        self.jit = jit
        self.rtype = rtype
        self.start_id = start_id
        self.count = count
        self.align = align
        self.dtype = dtype
        self.name = name # for debugging purpose only
        self.is_fixed = is_fixed
        self.set_shape([count]) # default shape: 1D

    def set_shape(self, shape):
        self.shape = shape # multi-dimension shape
        self.strides = [1 for _ in shape]
        for i in range(len(shape)-2, -1, -1):
            self.strides[i] = self.strides[i+1] * shape[i+1]

    def __len__(self):
        return self.count

    def __iter__(self):
        for i in range(self.count):
            yield GPRExpr("getitem", self, i, i)

    def overlap(self, other: Union['GPRExpr','GPRs'], fully_match = False):
        return self[0:self.count-1].overlap(other, fully_match)

    def split_const_terms(self):
        return 0, GPRExpr("getitem", self, 0, self.count - 1)

    '''
    GPRs following AMDGPU's slicing rules instead of python:
         [first:last]  instead of  [start:stop)
    '''
    def __getitem__(self, key):
        if key is Ellipsis:
            # support a[...]
            return GPRExpr("getitem", self, 0, self.count - 1)
        # in multi-dimension shape case, key can be a tuple
        # only the last key in the tuple can be slice
        if not isinstance(key, tuple):
            key = (key,)

        base_idx = 0
        last_dim_id = len(key) - 1
        for i,k in enumerate(key[:-1]):
            assert isinstance(k, int)
            assert k < self.shape[i]
            base_idx += k * self.strides[i]

        last_k = key[-1]

        if isinstance(last_k, slice):
            s = last_k
            first, last = s.start,s.stop
            if first is None: first = 0
            if last is None: last = self.shape[last_dim_id] - 1
            assert s.step == 1 or s.step is None, f"slice {s} with step not eq 1"

            assert first >= 0 and first < self.shape[last_dim_id]
            assert last >= 0 and last < self.shape[last_dim_id]

            first = first * self.strides[last_dim_id]
            last = last * self.strides[last_dim_id] + self.strides[last_dim_id] - 1

            assert base_idx + first < self.count
            assert base_idx + last < self.count
            return GPRExpr("getitem", self, base_idx + first, base_idx + last)
        elif isinstance(last_k, int):
            # print(f"{last_k=} {last_dim_id=} {self.shape=} {self.strides=}")
            first = last_k * self.strides[last_dim_id]
            last = last_k * self.strides[last_dim_id] + self.strides[last_dim_id] - 1
            assert base_idx + first < self.count
            assert base_idx + last < self.count
            return GPRExpr("getitem", self, base_idx + first, base_idx + last)
        else:
            assert 0, f"unsupported key {last_k}"

    def __setitem__(self, key, value:Union[GPRExpr,int,float]):
        dst = self[key]
        #inst = Instruction(self.jit.current_bb, "expression_place_holder")
        #inst(dst, value)
        self.jit.recursive_expr_gen(self.jit.current_bb, dst, value, loc=get_caller_loc())
        # expression_place_holder will be compiled later when all program is ready
        # all expressions within same BB can be processed together

    '''
    all magic methods GPRExpr supports can also be supported
    GPRs will convert self into a GPRExpr before compose expression
    '''
    def to_expr(self):
        return self[...]

    def __add__(self, other):
        return GPRExpr("+", self.to_expr(), other)
    def __radd__(self, other):
        return GPRExpr("+", other, self.to_expr())
    def __sub__(self, other):
        return GPRExpr("-", self.to_expr(), other)
    def __rsub__(self, other):
        return GPRExpr("-", other, self.to_expr())
    def __mul__(self, other):
        return self.to_expr().__mul__(other)
    def __rmul__(self, other):
        return self.to_expr().__rmul__(other)
    def __floordiv__(self, other):
        return self.to_expr().__floordiv__(other)
    def __mod__(self, other):
        return self.to_expr().__mod__(other)
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

    def debug_str(self):
        asm = f"{self.label_name}:\t"
        asm += f" ;BB#{self.bb_index}"
        asm += f" predecessors:[{','.join([p.label_name for p in self.predecessors])}]"
        asm += f" successors:[{','.join([p.label_name for p in self.successors])}]\n"
        for i,inst in enumerate(self.instructions):
            asm += f"\t /*{i} #{inst.loc} */ {inst}\n"
        return asm

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
        self.desc = J.new_gpr('s', 4, dtype="u32", align=4)
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
            assert offset12.bit_length() <= 12
            mod += f" offset:{offset12}"
        if vdst is None:
            return self.J.buffer_load_dwordx4(voffset, self.desc, soffset, mod = mod + " lds")
        return self.J.buffer_load_dwordx4(vdst, voffset, self.desc, soffset, mod=mod)

    def load_dwordx2(self, vdst, voffset, soffset, offset12=0):
        # vdst,     vaddr,           srsrc, soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        return self.J.buffer_load_dwordx2(vdst, voffset, self.desc, soffset, mod=mod)

    def load_dword(self, vdst, voffset, soffset, offset12=0):
        # vdst,     vaddr,           srsrc, soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        if vdst is None:
            return self.J.buffer_load_dword(voffset, self.desc, soffset, mod = mod + " lds")
        return self.J.buffer_load_dword(vdst, voffset, self.desc, soffset, mod=mod)

    def store_dwordx4(self, vdata, voffset, soffset, offset12=0):
        # vdata,    vaddr,        srsrc,  soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        return self.J.buffer_store_dwordx4(vdata, voffset, self.desc, soffset, mod=mod)

    def store_dwordx2(self, vdata, voffset, soffset, offset12=0):
        # vdata,    vaddr,        srsrc,  soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        return self.J.buffer_store_dwordx2(vdata, voffset, self.desc, soffset, mod=mod)

    def load_dword_lds(self, voffset, soffset, offset12=0):
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        mod += " lds"
        return self.J.buffer_load_dword(voffset, self.desc, soffset, mod=mod)

    def store_dword(self, vdata, voffset, soffset, offset12=0):
        # vdata,    vaddr,        srsrc,  soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        return self.J.buffer_store_dword(vdata, voffset, self.desc, soffset, mod=mod)

class LDSTensor:
    def __init__(self, J, shape, dtype, lds_base=None):
        self.J = J
        self.shape = shape
        stride_bytes = [dtype.itemsize]
        for i,dim in enumerate(reversed(shape)):
            cur = dim * stride_bytes[-1]
            stride_bytes.append(cur)
        self.size_bytes = stride_bytes[-1]
        self.stride_bytes = list(reversed(stride_bytes))[1:]
        self.dtype = dtype
        if lds_base is None:
            self.lds_base = self.J.alloc_lds(self.size_bytes)
            self.own = True
        else:
            self.lds_base = lds_base
            self.own = False

    # Python do not guarantee `__del__` happens immediatly at `del obj`
    # it may delay which is not intended. call free instead
    """
    def __del__(self):
        if self.own:
            self.J.free_lds(self.lds_base)
            self.own = False
    """
    def free(self):
        if self.own:
            self.J.free_lds(self.lds_base)
            self.own = False

    def read(self, dtype:str, vdst, *coord_exprs, offset1=None):
        self._access("read", dtype, vdst, *coord_exprs, offset1=offset1)

    def write(self, dtype:str, vdst, *coord_exprs, offset1=None):
        self._access("write", dtype, vdst, *coord_exprs, offset1=offset1)

    # reuse the same share buffer
    def view_as(self, shape, dtype):
        t = LDSTensor(self.J, shape, dtype, self.lds_base)
        assert t.size_bytes <= self.size_bytes, f'new view size must be less than the original, new={t.size_bytes:,} old={self.size_bytes:,}'
        return t

    def _access(self, op_name, dtype:str, vdst, *coord_exprs, offset1=None):
        # extract const part
        const_terms = 0
        exprs = []
        for i, ep in enumerate(coord_exprs):
            if isinstance(ep, int):
                const = ep
                nc_ep = None
            else:
                # assert isinstance(ep,GPRExpr)
                const, nc_ep = ep.split_const_terms()
            const_terms += const*self.stride_bytes[i]
            exprs.append(nc_ep)

        # combine non-const part
        for i, ep in enumerate(exprs):
            if ep is None: continue
            if i == 0:
                cur_expr = ep * self.stride_bytes[i]
            else:
                cur_expr = cur_expr + ep * self.stride_bytes[i]

        # mapping const part to offset
        if offset1 is not None:
            assert self.lds_base == 0, f"offset1 form has limited range, needs {self.lds_base=} to be zero"
            if const_terms < 0:
                offset0 = 0
                cur_expr = cur_expr + const_terms
            else:
                item_size = self.stride_bytes[-1]
                if (const_terms % item_size) == 0 and \
                   (const_terms//item_size) + offset1 <= 0xFF:
                    offset0 = const_terms//item_size
                    offset1 = offset0 + offset1
                else:
                    offset0 = 0
                    cur_expr = cur_expr + const_terms
            assert offset0 <= 0xFF and offset0 >= 0
            assert offset1 <= 0xFF and offset1 >= 0
            mod = f"offset0:{offset0} offset1:{offset1}"
            tag2 = "2"
        else:
            if const_terms < 0:
                offset0 = 0
                cur_expr = cur_expr + const_terms
            else:
                if const_terms < 64*1024:
                    offset0 = const_terms
                else:
                    offset0 = 0
                    cur_expr = cur_expr + const_terms
            offset0 += self.lds_base
            assert offset0 >= 0 and offset0 < 160*1024
            mod = f"offset:{offset0}"
            tag2 = ""

        loc = get_caller_loc()
        dst_gprs = self.J.gpr("vu32")
        self.J.recursive_expr_gen(self.J.current_bb, dst_gprs[0], cur_expr, loc=loc)
        if op_name == "read":
            getattr(self.J, f"ds_{op_name}{tag2}_{dtype}")(vdst, dst_gprs[0], mod=mod)
        elif op_name == "write":
            getattr(self.J, f"ds_{op_name}{tag2}_{dtype}")(dst_gprs[0], vdst, mod=mod)
        else:
            assert 0, op_name

# JIT emits instructions into BBs
all_kernel_hip_src_names = {}
replace_index = 0
dump_serial_id = 0

PYHIP_CACHE_DIR = os.getenv("PYHIP_CACHE_DIR", os.path.expanduser("~/.pyhip"))
os.makedirs(PYHIP_CACHE_DIR, exist_ok=True)

PYHIP_DEBUG_LOG = os.getenv("PYHIP_DEBUG_LOG", "")
PYHIP_JIT_LOG = int(os.getenv("PYHIP_JIT_LOG", "1"))
PYHIP_DUMP_DIR = os.getenv("PYHIP_DUMP_DIR", "")
PYHIP_RECOMPILE = int(os.getenv("PYHIP_RECOMPILE", "0"))
PYHIP_NOPASS = os.getenv("PYHIP_NOPASS", "").split(":")

if len(PYHIP_DEBUG_LOG):
    PYHIP_RECOMPILE = 1
if len(PYHIP_DUMP_DIR):
    # remove temp-cache to force recompile once 
    PYHIP_RECOMPILE = 1
    os.makedirs(PYHIP_DUMP_DIR, exist_ok=True)

if PYHIP_RECOMPILE:
    os.system(f'rm -rf {PYHIP_CACHE_DIR}/*')

_arch_lds_size = {
    "gfx950": 160*1024
}
class JIT:
    arch = amdgpu_arch()
    assert arch.startswith("gfx")
    gfx = int(arch[3:])
    cdna = 4 if gfx >= 950 else 3
    warp_size = 64

    def __init__(self, kernel_tag = "", no_pass = None):
        self.no_pass = PYHIP_NOPASS if no_pass is None else no_pass
        self.blocks = []
        # increased freely
        self.free_gpr_id = {'s':0, 'v':0, 'a': 0}
        self.label2bb = {}
        self.current_bb = BasicBlock(self, "_jit_main")
        self.relocatable_gprs = []
        self.fixed_gprs = []
        self.debug_log_info = []
        self.mark_idx = 0
        self.debug_cond_sgpr = None
        self.debug_log_ptr = None
        self.lds_allocator = SimpleMemoryAllocator(_arch_lds_size.get(self.arch, 64*1024))
        self.special_vars = {}
        self.kernel_tag = kernel_tag
        # sizeof mnemonics
        self.sizeof_f32 = 4
        self.sizeof_f16 = 2
        self.sizeof_s32 = 4
        self.sizeof_u32 = 4
        self.sizeof_DW = 4
        self.sizeof_DW2 = 8
        self.sizeof_DW4 = 16
        self.sizeof_fp32 = 4
        self.sizeof_bf16 = 2
        self.sizeof_fp16 = 2
        self.sizeof_fp8 = 1
        self.sizeof_bf8 = 1
        self.sizeof_fp4x2 = 1
        # allows J.sizeof(dtype), dtype can be string or torch dtype
        self._sizeof = {
            "dw4" : 16,
            "dwx4" : 16,
            "DW4" : 16,
            "DWx4" : 16,
            "dwordx4":16,
            "DWORDx4":16,

            "f32" : 4,
            "fp32" : 4,
            "float" : 4,
            "int" : 4,
            "uint" : 4,
            "DWORD" : 4,
            "DW" : 4,
            "s32": 4,
            "u32": 4,

            "bf16" : 2,
            "bfloat16":2,
            "fp16":2,
            "float16":2,
            "half":2,
            "f16":2,
            "s16": 2,
            "u16": 2,

            "s8": 1,
            "u8": 1,

            # https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
            "fp8" : 1,
            "bf8" : 1,
            # CDNA4/gfx950 : FP8-OCP (Open Compute Project)
            # https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
            #   torch.float8_e4m3fn:1,
            #   torch.float8_e5m2:1,
            # CDNA3/gfx940 : FP8-FNUZ (Finite and NaN Only, no Inf support)
            #  there is a 2 scaling factor between :
            #    fp8/float8_e4m3fnuz & bf8/float8_e5m2fnuz
            # torch.float8_e4m3fnuz:1,
            # torch.float8_e5m2fnuz:1,
            "fp4x2" : 1
        }

    def sizeof(self, dtype, cnt=1):
        if str(dtype) in ["torch.float4_e2m1fn_x2", "fp4x2"]:
            assert cnt % 2 == 0
            return cnt//2
        if isinstance(dtype ,str):
            return self._sizeof[dtype] * cnt
        # assume it's torch dype
        return dtype.itemsize * cnt

    def __getattr__(self, instruction):
        if self.current_bb is None:
            self.current_bb = BasicBlock(self, label_name="")
        return Instruction(self.current_bb, instruction, loc=get_caller_loc())

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
        if PYHIP_JIT_LOG or len(PYHIP_DEBUG_LOG):
            color_id = 3
            color0 = f"\033[0;{30+(color_id % 8)}m"
            color1 = f"\033[0m"
            print(color0, f"[{PYHIP_JIT_LOG=}] ", *args, color1, **kwargs)

    def debug_print(self, *args, **kwargs):
        # caller's function name must be
        if len(PYHIP_DEBUG_LOG) == 0: return
        caller_name = inspect.currentframe().f_back.f_code.co_name
        for item in PYHIP_DEBUG_LOG.split(":"):
            if item == "":continue
            if item in caller_name or item == "*":
                color_id = 3
                color0 = f"\033[0;{30+(color_id % 8)}m"
                color1 = f"\033[0m"
                print(color0, f"[PYHIP_DEBUG_LOG: {caller_name}] ", *args, color1, **kwargs)
                return True
        return False

    def is_no_pass(self, *args, **kwargs):
        if len(self.no_pass) == 0: return
        caller_name = inspect.currentframe().f_back.f_code.co_name
        return caller_name in self.no_pass

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
            # generate expression into scc
        else:
            dtype = cond.find_dtype()
            dst_gprs = self.new_gpr("s", 1, dtype=dtype, align=1, name="Jump_cond")
            dst_expr = GPRExpr("getitem", dst_gprs, 0, 0)
            self.recursive_expr_gen(self.current_bb, dst_expr, cond, loc=get_caller_loc())
            # optimize 
            last_inst = self.current_bb.instructions[-1]
            if last_inst.opcode == "s_mov_b32" and \
               last_inst.operands[0] is dst_expr and \
               last_inst.operands[1] == "scc":
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
        label_begin = f"_while_begin_{lineno}_{self.mark_idx}"
        label_end = f"_while_end_{lineno}_{self.mark_idx}"
        self.mark_idx += 1
        self.Label(label_begin)
        if cond is not None:
            self.Jump(label_end, cond, reverse=True)
        try:
            # following dict is for loop body code to continue or break
            yield {"begin":label_begin, "end":label_end}
        finally:
            self.Jump(label_begin)
            self.Label(label_end)

    @contextmanager
    def If(self, cond:GPRExpr):
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back.f_back
        lineno = caller_frame.f_lineno
        label_end = f"_if_end_{lineno}_{self.mark_idx}"
        self.mark_idx += 1
        self.Jump(label_end, cond, reverse=True)
        try:
            yield None
        finally:
            self.Label(label_end)

    '''
    for setting VCC based on condition expression
    '''
    def SetMask(self, dst, cond:GPRExpr = None):
        dst_is_sgprx2 = isinstance(dst, GPRs) and dst.rtype == "s" and dst.count == 2
        dst_is_vcc = isinstance(dst, str) and dst == "vcc"
        dst_is_exec = isinstance(dst, str) and dst == "exec"
        dst_is_scc = isinstance(dst, str) and dst == "scc"
        #assert dst_is_vcc or dst_is_sgprx2
        rtype = "v"
        if dst_is_scc:
            rtype = "s"
        if isinstance(dst, GPRs):
            rtype = dst.rtype
        dtype = cond.find_dtype()
        dst_gprs = self.new_gpr(rtype, 1, dtype=dtype, align=1, name="SetMask_dst")
        dst_expr = GPRExpr("getitem", dst_gprs, 0, 0)
        self.recursive_expr_gen(self.current_bb, dst_expr, cond, loc=get_caller_loc())
        last_inst = self.current_bb.instructions[-1]
        if dst_is_exec:
            # use v_cmpx_
            assert last_inst.opcode == "v_cndmask_b32_e64" and \
                    last_inst.operands[0] is dst_expr and \
                    last_inst.operands[1] == 0 and \
                    last_inst.operands[2] == 1 and \
                    last_inst.operands[3] == "vcc"
            secondlast_inst = self.current_bb.instructions[-2]
            assert secondlast_inst.opcode.startswith("v_cmp_")
            secondlast_inst.opcode = secondlast_inst.opcode.replace("v_cmp_","v_cmpx_")
            self.log("v_cndmask_b32_e64 dst,0,1,vcc is optimized")
            self.log("v_cmp_ is replaced by v_cmpx_")
            self.current_bb.instructions.pop()
        elif dst_is_vcc and \
            last_inst.opcode == "v_cndmask_b32_e64" and \
            last_inst.operands[0] is dst_expr and \
            last_inst.operands[1] == 0 and \
            last_inst.operands[2] == 1 and \
            last_inst.operands[3] == "vcc" :
            self.log("v_cndmask_b32_e64 dst,0,1,vcc is optimized")
            self.current_bb.instructions.pop()
        elif dst_is_scc and \
            last_inst.opcode == "s_mov_b32" and \
            last_inst.operands[0] is dst_expr and \
            last_inst.operands[1] == "scc" :
            self.log("s_mov_b32 dst,scc is optimized")
            self.current_bb.instructions.pop()
        elif dst_is_scc:
            # generate scc
            self.s_cmp_lg_i32(0, dst_gprs)
        else:
            # generate mask into dst (vcc or sgprx2)
            self.v_cmp_ne_u32_e64(dst, 0, dst_gprs)
        return


    '''
    Use this to set exec-mask to handle the tail/vari-SIMD-length problem
    '''
    @contextmanager
    def ExecMask(self, cond:GPRExpr = None, early_skip = True):
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back.f_back
        lineno = caller_frame.f_lineno
        label_begin = f"_execmask_begin_{lineno}_{self.mark_idx}"
        label_end = f"_execmask_end_{lineno}_{self.mark_idx}"
        self.mark_idx += 1
        
        self.SetMask("vcc", cond)
        exec_backup = self.new_gpr("s", 2, dtype="i32", align=2, name="ExecMask_exec_backup")
        self.s_and_saveexec_b64(exec_backup, "vcc") # scc = (exec!=0)
        if early_skip: self.s_cbranch_execz(mod=label_end) # early skip
        try:
            yield # the body of computation with ExecMask
        finally:
            if early_skip: 
                self.Label(label_end)
            self.s_mov_b64("exec", exec_backup)
            # if we want to do something when execz happens, we can use scc0

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
            loc = get_caller_loc()
        self.recursive_expr_gen(self.current_bb, gprs[0], expr, loc=loc)
        return gprs

    '''
    a = J.gpr(4, 4, 4,"abf16x2"): alloc 4*4*4 AccVGPRs, each 32-bit gpr has bf16x2 type
    a[2,0,0]   : referencing single gpr
    a[2,0,0:3] : referencing gpr groups (continous, no more than instruction needed)
    a[2,0]     : same as above
    a[2]       : referencing

    a = J.gpr("vu32", 0)    : alloc 1 vgpr and set initial value to 0
    a = J.gpr(4, "af32", 0) : alloc 4 agpr and set initial value to 0
    a = J.gpr(4, "af32", 0, 3) : alloc 4 agpr and set initial value to 0,3,3,3
    '''
    def gpr(self, *desc, align=0, name=""):
        if name == "":
            # try to reover python var's name from code_context
            code_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context
            if code_context:
                src_line = code_context[0].strip()
                if "=" in src_line:
                    items = src_line.split("=")
                    if len(items) > 1:
                        if ".gpr(" in items[1]:
                            name = items[0].strip()

        if len(desc) == 1 and isinstance(desc[0], GPRExpr):
            return self.auto_gpr(desc[0], name=name, loc=get_caller_loc())

        # allocate GPRs

        num_gprs = 1
        shape = []
        i = 0
        for i in range(len(desc)):
            dim = desc[i]
            if not isinstance(dim, int):
                break
            shape.append(dim)
            num_gprs *= dim

        dtype = desc[i]
        assert isinstance(dtype, str), f"{desc=} {i=} {dtype=}"

        if i + 1 < len(desc):
            initial_values = desc[(i+1):]
        else:
            initial_values = None

        # allows empty shape info: `J.gpr("su32")`
        if len(shape) == 0:
            shape = [1]

        dtype = dtype.strip()
        if dtype[0] in ["v", "a", "s"]:
            rtype = dtype[0]
            dtype = dtype[1:]
        else:
            rtype = "v"

        if align == 0: # derive alignment automatically
            align = min(num_gprs, 4)//2*2
        align = max(align, 1)
        gprs = self.new_gpr(rtype, num_gprs, dtype, align, name)
        # set additional shape meta data for easier indexing & slicing
        gprs.set_shape(shape)

        if initial_values is not None:
            if isinstance(initial_values, list) or isinstance(initial_values, tuple):
                n_values = len(initial_values)
                flatten_gprs = gprs[...]
                for i in range(len(flatten_gprs)):
                    flatten_gprs[i] = initial_values[min(i, n_values-1)]
            else:
                gprs[...] = initial_values

        return gprs


    def new_gpr(self, reg_type, count_range, dtype="u32", align=1, name=""):
        if name == "":
            # try to reover python var's name from code_context
            code_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context
            if code_context:
                src_line = code_context[0].strip()
                name = src_line.split("=")[0].strip()

        dtype = dtype.replace("fp32","f32")
        assert dtype in ["u32","i32","f32","bf16x2","fp16x2","bf8x4","fp8x4"]
        assert reg_type == 's' or reg_type == 'v' or reg_type == 'a'
        if isinstance(count_range, int):
            # allocate do not reuse, just increase the index
            count = count_range
            start_id = self._align_up(self.free_gpr_id[reg_type], align)
            self.free_gpr_id[reg_type] = start_id + count
            gprs = GPRs(self, reg_type, start_id, count, dtype=dtype, align=align, name=name, is_fixed=False)
            self.relocatable_gprs.append(gprs)
            return gprs
        elif isinstance(count_range, tuple) or isinstance(count_range, list):
            assert len(count_range) == 2
            assert align == 1
            first_id, last_id = count_range
            assert self.free_gpr_id[reg_type] <= first_id, "specified reg has been allocated"
            assert self.free_gpr_id[reg_type] <= last_id, "specified reg has been allocated"
            self.free_gpr_id[reg_type] = last_id + 1
            gprs = GPRs(self, reg_type, first_id, last_id - first_id + 1, dtype=dtype, align=align, name=name, is_fixed=True)
            self.fixed_gprs.append(gprs)
            return gprs
        else:
            assert 0

    # Dead Store Elimination
    def pass_dse(self):
        if self.is_no_pass(): return
        # if a register is written multiple times w/o any read, all such writes can be removed
        # dead loads, if a load dst is used by no one, remove such loads
        gpr_stores = {}
        gpr_src = set()
        for bid, bb in enumerate(self.blocks):
            for iid, inst in enumerate(bb.instructions):
                possible_src_id0 = 0
                if inst.is_simple_operand0_store():
                    dst_gpr = inst.operands[0]
                    flatten_gpr = dst_gpr[...]
                    for i in range(len(flatten_gpr)):
                        key = repr(flatten_gpr[i])
                        if key not in gpr_stores:
                            gpr_stores[key] = []
                        gpr_stores[key].append((bid, iid))
                    possible_src_id0 = 1 # skip dst-gpr for src-gpr tests
                for op in inst.operands[possible_src_id0:]:
                    if not isinstance(op, GPRExpr):
                        continue
                    flatten_gpr = op[...]
                    for i in range(len(flatten_gpr)):
                        key = repr(flatten_gpr[i])
                        gpr_src.add(key)

        # keep store-only gprs
        for key in gpr_src:
            if key in gpr_stores:
                del gpr_stores[key]

        for key in gpr_stores:
            # just store, no use/read
            for bid, iid in gpr_stores[key]:
                inst = self.blocks[bid].instructions[iid]
                if inst.is_dead: continue
                # all component of dst must also be marked as dead-store
                dst_gpr = inst.operands[0]
                flatten_dst = dst_gpr[...]
                if all([repr(r) in gpr_stores for r in flatten_dst]):
                    inst.is_dead = True

        # remove dead-loads
        for bb in self.blocks:
            for i in range(len(bb.instructions)-1, -1, -1):
                if bb.instructions[i].is_dead:
                    bb.instructions.pop(i)

    def pass_dce(self):
        if self.is_no_pass(): return
        inst_list = []
        serial_index = 0
        for bb in self.blocks:
            for inst in bb.instructions:
                inst.sid = serial_index
                inst_list.append(inst)
                serial_index += 1

        # in unit of one gpr
        def is_normal_gpr(op):
            if not isinstance(op, GPRExpr):
                return False
            assert op.op == "getitem"
            gprs = op.src0
            assert isinstance(gprs, GPRs), f"{type(gprs)}"
            return not (gprs in self.fixed_gprs)

        live_intervals = {}
        for bid, bb in enumerate(self.blocks):
            for iid, inst in enumerate(bb.instructions):
                for op in inst.operands:
                    if is_normal_gpr(op):
                        flatten_gpr = op[...]
                        for i in range(len(flatten_gpr)):
                            key = repr(flatten_gpr[i])
                            if key not in live_intervals:
                                live_intervals[key] = [bid, iid]
                            live_intervals[key].append(inst.sid)

        # recursively remove all unused gpr & instrustions producing them
        while True:
            useless_gprs = {}
            for gpr_repr in live_intervals:
                ivs = live_intervals[gpr_repr]
                bid, iid  = ivs[:2]
                if len(set(ivs[2:])) == 1:
                    useless_gprs[gpr_repr] = ivs

            if len(useless_gprs) == 0: break
            self.debug_print("useless_gprs: ", useless_gprs)

            for gpr_repr in useless_gprs:
                del live_intervals[gpr_repr]

                ivs = useless_gprs[gpr_repr]
                if len(ivs) < 3:
                    continue # the instruction has been marked as dead
                bid, iid, sid  = useless_gprs[gpr_repr][:3]
                
                inst = self.blocks[bid].instructions[iid]
                if inst.is_dead:
                    continue
                assert sid == inst.sid
                dst_gpr = inst.operands[0]
                if all([repr(dst_gpr[i]) in useless_gprs for i in range(len(dst_gpr))]):
                    inst.is_dead = True
                    self.debug_print(gpr_repr, sid, "============", inst)
                    # other operand used by this inst also disappears
                    for op in inst.operands[1:]:
                        if is_normal_gpr(op):
                            flatten_gpr = op[...]
                            for i in range(len(flatten_gpr)):
                                key = repr(flatten_gpr[i])
                                if key in live_intervals:
                                    self.debug_print(key, live_intervals[key])
                                    id = live_intervals[key].index(sid, 2)
                                    del live_intervals[key][id]

        # delete dead instructions
        for bb in self.blocks:
            for i in range(len(bb.instructions)-1, -1, -1):
                if bb.instructions[i].is_dead:
                    bb.instructions.pop(i)

    def pass_break_down_gprs(self):
        if self.is_no_pass(): return
        # gprs are allocate in unit of GPRs, break GPRs into smaller pieces
        # can help to reduce register space fragmentation issue.
        # each instruction operand reference to gpr represents a 
        # requirement on physical adjacency of a subset of GPRs
        # and these requirement are overlapping, for example:
        #     v[0:3] requires v[0,1,2,3] to be physically adjacent to each other
        #      - v[0:1] requirements are weaker than above one.
        #      - v[2:5] further extends physical adjacency of 0:3 to 0:5
        #     so basically these requirements, if overlapping with each other
        #     they will form a bigger adjacency requirement. all such requirements
        #     are maintained in GPRs.
        # some referencing are not used to compose instructions and such reference
        # are just logical references and not a physical requirement, so we need 
        # go through all instructions and collect real adjacency requirements.
        # these adjacency requirements help to break GPRs into smaller pieces
        class GPRparts:
            def __init__(self, name):
                self.parts = []
                self.name = name
            def update(self, first, last, loc):
                merged_parts = []
                merged_locs = [loc]
                for i,p in enumerate(self.parts):
                    i0,i1,locs = p
                    n0,n1 = min(i0, first), max(i1, last)
                    full_range = n1 - n0 + 1
                    max_range = (i1-i0+1) + (last-first+1)
                    if full_range < max_range:
                        # overlapping
                        merged_parts.append(i)
                        merged_locs.extend(locs)
                        first, last = n0, n1 # new range
                # delete merged parts
                for i in range(len(self.parts)-1, -1, -1):
                    if i in merged_parts:
                        self.parts.pop(i)
                self.parts.append((first, last, merged_locs))

        gpr_parts = {}
        for bid, bb in enumerate(self.blocks):
            for iid, inst in enumerate(bb.instructions):
                for oid,op in enumerate(inst.operands):
                    if not isinstance(op, GPRExpr): continue
                    assert op.op == "getitem"
                    gprs = repr(op.src0)
                    first = op.src1
                    last = op.src2
                    if op.src0.count == 1 or op.src0.rtype == 's': continue
                    if gprs not in gpr_parts:
                        gpr_parts[gprs] = GPRparts(op.src0.name)
                    gpr_parts[gprs].update(first, last, (bid,iid,oid))

        for k,parts in gpr_parts.items():
            if len(parts.parts) == 1: continue
            #print("#### ", k)
            for first, last, locs in parts.parts:
                gprs = self.gpr(last-first+1, k[0]+"u32", name=f"{parts.name}[{first}:{last}]")
                #print(first, last, locs)
                for bid,iid,oid in locs:
                    inst = self.blocks[bid].instructions[iid]
                    #sin = repr(inst)
                    op = inst.operands[oid]
                    # some instructions may share same GPRExpr
                    if op.src0 is gprs: continue
                    #op0 = repr(op)
                    op.src1 = op.src1 - first
                    op.src2 = op.src2 - first
                    op.src0 = gprs
                    #op1 = repr(op)
                    #print(op0,"=====>",op1, "       ", sin, "====>", inst)

    # this step is not mandatory，but it allows more registers to use
    # linear-scan:
    #    try to shrink register usage on `relocatable_gprs`
    #    do not touch `fixed_gprs`
    def register_allocation_linear_scan(self):
        # note: BB in self.blocks has been ordered, assign each instruction an serial
        inst_dict = {}
        inst_list = []
        serial_index = 0
        for bb in self.blocks:
            for inst in bb.instructions:
                inst.sid = serial_index
                inst_dict[inst.sid] = inst
                inst_list.append(inst)
                serial_index += 1

        # find live-intervals of gprs in `relocatable_gprs`
        live_intervals = {}
        for bb in self.blocks:
            for inst in bb.instructions:
                # inst.sid
                for op in inst.operands:
                    if not isinstance(op, GPRExpr): continue
                    gprs = op.src0
                    if gprs.is_fixed: continue
                    if gprs not in live_intervals:
                        live_intervals[gprs] = [inst.sid, inst.sid]
                    live_intervals[gprs][0] = min(live_intervals[gprs][0], inst.sid)
                    live_intervals[gprs][1] = max(live_intervals[gprs][1], inst.sid)

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
        reg_resource = {"s": [0 for _ in range(103)],
                        "v": [0 for _ in range(256)],
                        "a": [0 for _ in range(256)]}
        gpr_used = {"s": [0 for _ in range(103)],
                    "v": [0 for _ in range(256)],
                    "a": [0 for _ in range(256)]}
        # reserve fixed gprs
        for gprs in self.fixed_gprs:
            i0 = gprs.start_id
            i1 = gprs.start_id + gprs.count
            reg_resource[gprs.rtype][i0:i1] = [1]*gprs.count
            gpr_used[gprs.rtype][i0:i1] = [1]*gprs.count

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

        alive_gprs = []
        def gpr_usage():
            ret = ""
            for rtype, slots in reg_resource.items():
                used = 0
                last_id = 0
                for i,s in enumerate(slots):
                    if s:
                        used += 1
                        last_id = i
                ret += f"{rtype}:{used}({last_id}) "
            return ret

        def alloc_gpr(gprs, sid):
            rtype = gprs.rtype
            count = gprs.count
            align = gprs.align
            slots = reg_resource[rtype]
            for i in range(0, len(slots), align):
                if all(s == 0 for s in slots[i:(i+count)]) and i+count <= len(slots):
                    gprs.start_id = i
                    gprs.sid = sid
                    slots[i:(i+count)] = [1]*count # mark as used
                    alive_gprs.append(gprs)
                    gpr_used[rtype][i:(i+count)] = [1]*count
                    return
            # summary for diagnose GPR overflow issue
            summary = gpr_usage() + "\n"
            for g in alive_gprs:
                summary += f"\t{str(g.count):5s} {g.rtype}GPRs  {repr(g):15s} {g.name:20s}  {inst_list[g.sid].loc}\n"
            assert 0, f"{inst_list[sid].loc} cannot allocate '{gprs.name}'  {rtype}GPRs x {count} {align=}, not enough resource:\n {summary}"

        def free_gpr(gprs):
            rtype = gprs.rtype
            count = gprs.count
            slots = reg_resource[rtype]
            i = gprs.start_id
            slots[i:(i+count)] = [0]*count
            alive_gprs.remove(gprs)

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
                        inst_dict[sid].loc += f" free '{gprs.name}'{gprs}"
                        #self.asm_debug_info += f";free '{gprs.name}'{gprs} at #{sid}\n"
                    else:
                        unalloc_gprs.append(gprs)
            # allocation 
            for ev,gprs in events:
                if ev == 0: # first use(as dst)
                    alloc_gpr(gprs, sid)
                    inst_dict[sid].loc += f" alloc '{gprs.name}'{gprs}    {gpr_usage()}"
                    #self.asm_debug_info += f";alloc '{gprs.name}'{gprs} at #{sid}      {gpr_usage()}\n"

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
                        debug_info += f"{op.src0.name},"
                    else:
                        debug_info += f"{op},"
                inst.debug_info = debug_info

        used_gprs = []
        for rtype in gpr_used:
            for i, cnt in enumerate(gpr_used[rtype]):
                if cnt > 0:
                    used_gprs.append(f"{rtype}{i}")
        return used_gprs

    def is_iconst(self, i):
        return isinstance(i, int) and (i >= -16) and (i <= 64)
    
    def special_expr_64bits(self, bb, dst_expr:GPRExpr,  expr:Union['GPRExpr'], loc=""):
        # we only support very limited 64bit expression which can be used for addressing
        # s_add_u32/s_addc_u32
        #   s[:] = q[:] + k[:]  k can be 32bit number
        assert isinstance(expr, GPRExpr)
        assert dst_expr.op == "getitem"
        dst_gprs = dst_expr.src0
        dst_idx0 = dst_expr.src1
        dst_idx1 = dst_expr.src2
        assert (dst_idx1 - dst_idx0) == 1
        if dst_gprs.rtype == "s":
            assert expr.find_dtype() == "u32"
            assert expr.op == "+"
            lhs = expr.src0
            rhs = expr.src1
            if isinstance(lhs, int) and isinstance(rhs, GPRExpr):
                assert rhs.op == "getitem"
                assert lhs.bit_length() <= 32
                rlhs_gprs = rhs.src0
                add_low = Instruction(bb, f"s_add_u32", loc=loc)
                add_low(dst_gprs[dst_idx0], lhs, rlhs_gprs[rhs.src1])
                add_high = Instruction(bb, f"s_addc_u32", loc=loc)
                add_high(dst_gprs[dst_idx1], 0, 0 if rhs.src2 == rhs.src1 else rlhs_gprs[rhs.src2])
                return
            assert lhs.op == "getitem"
            lhs_gprs = lhs.src0
            assert (lhs.src2 - lhs.src1) <= 1
            assert (lhs_gprs.rtype == "s")
            if rhs.op == "getitem":
                rlhs_gprs = rhs.src0
                assert (rhs.src2 - rhs.src1) <= 1
                add_low = Instruction(bb, f"s_add_u32", loc=loc)
                add_low(dst_gprs[dst_idx0], lhs_gprs[lhs.src1], rlhs_gprs[rhs.src1])
                add_high = Instruction(bb, f"s_addc_u32", loc=loc)
                add_high(dst_gprs[dst_idx1],
                         0 if lhs.src2 == lhs.src1 else lhs_gprs[lhs.src2],
                         0 if rhs.src2 == rhs.src1 else rlhs_gprs[rhs.src2])
                return
            else:
                assert isinstance(rhs, GPRExpr)
                src1_gprs = self.new_gpr("s", 1, dtype="u32", align=1, name="src1_gprs")
                src1_operand = GPRExpr("getitem", src1_gprs, 0, 0)
                self.recursive_expr_gen(bb, src1_operand, rhs, loc=loc)
                add_low = Instruction(bb, f"s_add_u32", loc=loc)
                add_low(dst_gprs[dst_idx0], lhs_gprs[lhs.src1], src1_operand)
                add_high = Instruction(bb, f"s_addc_u32", loc=loc)
                add_high(dst_gprs[dst_idx1],
                         0 if lhs.src2 == lhs.src1 else lhs_gprs[lhs.src2],
                         0)
        else:
            # v_lshl_add_u64
            #   a[:] = b[:] + c[:]
            #   a[:] = b[:] + c[:] << k  # k in [0,1,2,3,4]
            #   a[:] = b[:] + c[:] * k  # k in [1,2,4,8,16]
            assert dst_gprs.rtype == "v"
            assert expr.op == "+"
            lhs, rhs = expr.src0, expr.src1
            assert lhs.op == "getitem"
            lhs_gprs = lhs.src0
            assert lhs_gprs.rtype == "v"
            assert (lhs.src2 - lhs.src1) == 1
            if rhs.op == "getitem":
                rhs_gprs = rhs
                shift_left = 0
            elif rhs.op == "<<" and rhs.src0.op == "getitem":
                rhs_gprs = rhs.src0
                assert (rhs.src0.src2 - rhs.src0.src1) == 1
                assert rhs.src1 in [0,1,2,3,4]
                shift_left = rhs.src1
            elif rhs.op == "*" and rhs.src0.op == "getitem":
                rhs_gprs = rhs.src0
                assert (rhs.src0.src2 - rhs.src0.src1) == 1
                assert rhs.src1 in [1,2,4,8,16]
                shift_left = self.shift_bits(rhs.src1)
            inst = Instruction(bb,"v_lshl_add_u64")
            inst(dst_gprs, rhs_gprs, shift_left, lhs_gprs)
        return
    '''
    with the help of temp vars, complex expression can be recursively generated & appended into bb.
    '''
    def recursive_expr_gen(self, bb, dst_expr:GPRExpr,  expr:Union['GPRExpr',int,float], loc=""):
        assert dst_expr.op == "getitem"
        dst_gprs = dst_expr.src0
        dst_idx0 = dst_expr.src1
        dst_idx1 = dst_expr.src2

        # assign constant(float or int) to multiple VGPRs
        rtype = dst_gprs.rtype
        dtype = dst_gprs.dtype
        if isinstance(expr, float):
            expr = float_to_ieee754_bits_little(expr)
        if isinstance(expr, int):
            for dst in dst_expr:
                if rtype == "a":
                    new_inst = Instruction(bb, f"v_accvgpr_write_b32", loc=loc)
                else:
                    new_inst = Instruction(bb, f"{rtype}_mov_b32", loc=loc)
                new_inst(dst, expr)
            return

        if dst_idx1 - dst_idx0 + 1 > 1:
            self.special_expr_64bits(bb, dst_expr, expr, loc)
            return
        assert dst_idx1 == dst_idx0

        if isinstance(expr, GPRs):
            expr = expr.to_expr()
        assert isinstance(expr, GPRExpr)
        if expr.op == "getitem":
            # assign
            src_cnt = expr.src2 - expr.src1 + 1
            assert src_cnt == 1
            if rtype == "v" and expr.src0.rtype == "a":
                new_inst = Instruction(bb, f"v_accvgpr_read_b32", loc=loc)
            else:
                new_inst = Instruction(bb, f"{rtype}_mov_b32", loc=loc)
            new_inst(dst_expr, expr)
            return

        if dtype == "" or (dtype not in ["u32", "i32"]):
            if expr.op not in ["&","|","^"]:
                dtype = expr.find_dtype()
                self.log(f"infer dtype={dtype} by expr {expr}")
                assert dtype != ""

        if isinstance(expr.src0, GPRExpr):
            if expr.src0.op == "getitem":
                # "getitem" expr can be used as operand directly
                src0_operand = expr.src0
            else:
                src0_gprs = self.new_gpr(rtype, 1, dtype=dtype, align=1, name="src0_gprs")
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
                src1_gprs = self.new_gpr(rtype, 1, dtype=dtype, align=1, name="src1_gprs")
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
                    src1_operand = hex((-src1_operand) & 0xffffffff)
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
                    if isinstance(src0_operand, int) and (not self.is_iconst(src0_operand)):
                        assert src0_operand.bit_length() <= 24
                        new_inst = Instruction(bb, f"v_mul_u32_u24", loc=loc)
                    else:
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
                # src1 can only be vgpr, but src0 can be anything
                if (not isinstance(src1_operand, GPRExpr)) or (src1_operand.find_rtype() != "v"):
                    src1_operand, src0_operand = src0_operand, src1_operand
                    cmp_op_reverse = {"gt":"lt", "lt":"gt", "le":"ge", "ge":"le", "eq":"eq", "ne":"ne"}
                    op = cmp_op_reverse[op]
                cmp = Instruction(bb, f"v_cmp_{op}_{dtype}_e32", loc=loc)
                mov = Instruction(bb, f"v_cndmask_b32_e64", loc=loc)
                cmp("vcc", src0_operand, src1_operand)
                mov(dst_expr, 0, 1, "vcc")
            else:
                assert 0, f"unsupported rtype: {rtype}"
        elif expr.op == "floordiv":
            # from Hacker's Delight: use multiply+shift to calculate div
            assert dtype in ["u32","i32"]
            def find_min_k(d, n_max, allow_eq = False):
                k = 32
                while True:
                    mod = (1 << k) % d
                    if allow_eq and (1 << k) == n_max * (d - mod):
                        return k
                    elif (1 << k) > n_max * (d - mod):
                        return k
                    k += 1
            assert rtype == "s"
            if isinstance(src1_operand, int):
                d = src1_operand
                if dtype == "u32":
                    assert d > 0
                    """
                    a = n/d
                    b = n*[2**k/d + e]/2**k = n/d + (n*e/2**k) = a + (n*e/2**k)
                         e is error introduced by ceil(2**k/d): [d-(2**k % d)]/d

                    floor(b) = floor(a) when (e/2**k) < 1/d because a=n/d, floor(a+1/d) may > floor(a)
                    so we need 2**k > e*d = n*[d-(2**k % d)]
                    """
                    n_max = 2**32-1
                    min_k = find_min_k(d, n_max)
                    ceil_sd = (2**min_k + d - 1)//d
                    assert ceil_sd < 2**32, f"{d=} {min_k=} {ceil_sd=}"
                    Instruction(bb, f"s_mul_hi_u32", loc=loc)(dst_expr, src0_operand, ceil_sd)
                    Instruction(bb, f"s_lshr_b32", loc=loc)(dst_expr, dst_expr, min_k-32)
                else:
                    """
                    consider negative as a mirror of positive
                    """
                    #assert d > 0
                    abs_d = abs(d)
                    min_k = max(find_min_k(abs_d, 2**31 - 1), find_min_k(abs_d, 2**31, True))
                    ceil_sd = (2**min_k + abs_d - 1)//abs_d
                    assert ceil_sd < 2**32, f"{d=} {min_k=} {ceil_sd=} {hex(ceil_sd)=}"
                    need_compensation = 0 if ceil_sd < 2**31 else 1
                    if d < 0:
                        ceil_sd = (-ceil_sd) # restore sign of divisor

                    need_compensation = 0
                    if ceil_sd >= 2**31:
                        # sd will be interpreted as (sd - 2^32) by s_mul_hi_i32
                        # and get n*(sd - 2^32), need compensate it by adding n*2^32
                        need_compensation = 1
                    elif ceil_sd < -2**31:
                        # sd will be interpreted as (sd + 2^32) by s_mul_hi_i32
                        # and get n*(sd + 2^32), need compensate it by sub n*2^32
                        need_compensation = -1
                    ceil_sd = ceil_sd & 0xFFFFFFFF

                    temp = self.new_gpr(rtype, 1, dtype=dtype, align=1, name="temp")
                    Instruction(bb, f"s_mul_hi_i32", loc=loc)(dst_expr, src0_operand, ceil_sd)
                    if need_compensation == 1:
                        Instruction(bb, f"s_add_i32", loc=loc)(dst_expr, dst_expr, src0_operand)
                    if need_compensation == -1:
                        Instruction(bb, f"s_sub_i32", loc=loc)(dst_expr, dst_expr, src0_operand)
                    Instruction(bb, f"s_lshr_b32", loc=loc)(temp, dst_expr, 31)
                    Instruction(bb, f"s_ashr_i32", loc=loc)(dst_expr, dst_expr, min_k-32)
                    # when n/d < 0, arithematic shift right 1 bit is floor(n/2)
                    # so extra compensation is needed for C/C++'s rounding-toward-zero result
                    Instruction(bb, f"s_add_i32", loc=loc)(dst_expr, dst_expr, temp)
            else:
                # assert src1_operand.find_rtype() == "s"
                # copied from hip compiler's output
                s2 = src0_operand
                s3 = self.gpr("su32")
                s4 = self.gpr("su32")
                s5 = self.gpr("su32")
                s6 = self.gpr("su32")
                s7 = self.gpr("su32")
                v1 = self.gpr("vu32")
                self.s_abs_i32(s4, src1_operand)
                self.v_cvt_f32_u32_e32(v1, s4)
                self.s_sub_i32(s5, 0, s4)
                self.s_xor_b32(s3, s2, src1_operand)
                self.s_abs_i32(dst_expr, s2)
                self.v_rcp_iflag_f32_e32(v1, v1)
                self.s_ashr_i32(s3, s3, 31)
                self.v_mul_f32_e32(v1, 0x4f7ffffe, v1)
                self.v_cvt_u32_f32_e32(v1, v1)
                self.s_nop(mod="0")
                self.v_readfirstlane_b32(s6, v1)
                self.s_mul_i32(s5, s5, s6)
                self.s_mul_hi_u32(s5, s6, s5)
                self.s_add_i32(s6, s6, s5)
                self.s_mul_hi_u32(s5, dst_expr, s6)
                self.s_mul_i32(s6, s5, s4)
                self.s_sub_i32(dst_expr, dst_expr, s6)
                self.s_add_i32(s7, s5, 1)
                self.s_sub_i32(s6, dst_expr, s4)
                self.s_cmp_ge_u32(dst_expr, s4)
                self.s_cselect_b32(s5, s7, s5)
                self.s_cselect_b32(dst_expr, s6, dst_expr)
                self.s_add_i32(s6, s5, 1)
                self.s_cmp_ge_u32(dst_expr, s4)
                self.s_cselect_b32(dst_expr, s6, s5)
                self.s_xor_b32(dst_expr, dst_expr, s3)
                self.s_sub_i32(dst_expr, dst_expr, s3)
                # assert 0, f"TODO floordiv {rtype=} {dtype=} {dst_expr=} {src0_operand=} {src1_operand=}"
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

    def pass_a2v(self):
        # each operand must be gpr, not expression
        for bb in self.blocks:
            i = 0
            while i < len(bb.instructions):
                cur = bb.instructions[i]
                if not _utils_accept_accvgpr(cur.opcode, cur.mod):
                    for opid, op in enumerate(cur.operands):
                        if isinstance(op, GPRExpr) and op.find_rtype() == "a":
                            assert opid != 0, f"destination for {cur} cannot be accvgpr"
                            cnt = len(op)
                            vgpr = self.gpr(cnt, "vu32")
                            for j in range(cnt):
                                inst = Instruction(bb, "v_accvgpr_read_b32")
                                inst(vgpr[j], op[j], insert_bb_pos = i)
                            cur.operands[opid] = vgpr[...]
                            i += 1
                i += 1

    def pass_insert_nop(self):
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
                        if isinstance(op, GPRExpr) and vdst.overlap(op):
                            n_nops = 1
                            self.log(f"insert s_nop({n_nops}) at #{loc} : [VALU Trans op, Non-trans VALU op consumes result of that op]")
                            break

                if prev.opcode.startswith("v_") and cur.opcode.startswith("v_permlane"):
                    vdst = prev.operands[0]
                    for op in cur.operands:
                        if isinstance(op, GPRExpr) and vdst.overlap(op):
                            n_nops = 2
                            self.log(f"insert s_nop({n_nops}) at #{loc} : [VALU* writes vdst, V_PERMLANE* reads vdst]")
                            break

                if n_nops >= 0:
                    inst = Instruction(bb, "s_nop")
                    inst(n_nops, insert_bb_pos = i)

                i += 1

    def Interleave(self, *thread_jits):
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
                self.signal_table = signal_table

                self.is_mem_load = False
                self.is_mfma =  False
                self.is_dsrd =  False
                self.is_dswr =  False
                self.is_ds =  False
                self.next()  # initialize self.inst

            def progress(self):
                return self.index * 4

            def next(self):
                if self.finished or self.index >= len(self.bb.instructions):
                    self.finished = True
                    return None
                self.inst = None
                while self.index < len(self.bb.instructions):
                    inst = self.bb.instructions[self.index]
                    if inst.opcode == "signal":
                        signal_name = inst.mod
                        if signal_name in self.signal_table:
                            assert self.signal_table[signal_name][0] == 0
                            self.signal_table[signal_name][0] = 1
                            # change waitting thread's state into waitting
                            wait_thr = self.signal_table[signal_name][1]
                            # let waitting thread to move to next instruction
                            wait_thr.next()
                        else:
                            # signal a thread which is not in waitting yet
                            self.signal_table[signal_name] = [1, None]
                        self.index += 1
                        continue # signal thread keep fetching next instruction

                    if inst.opcode == "wait":
                        # go into wait status
                        signal_name = inst.mod
                        if signal_name in self.signal_table:
                            signaled, thr = self.signal_table[signal_name]
                            assert (thr is self) or (thr is None), "only 1 thread can wait on a signal only once"
                        else:
                            self.signal_table[signal_name] = [0, self] # so other thread can wake me up
                            signaled = 0
                        self.waitting = not signaled
                        if self.waitting:
                            # next() still going to wait
                            return
                        else:
                            # has signaled,just fetch next instruction
                            self.index += 1
                            continue
                    self.inst = inst
                    self.index += 1
                    break

                self.finished = self.inst is None

                if not self.finished:
                    self.is_mem_load = self.inst.opcode.startswith("global_load_") or self.inst.opcode.startswith("buffer_load_")
                    self.is_mfma = self.inst.opcode.startswith("v_mfma_")
                    self.is_dsrd = self.inst.opcode.startswith("ds_read")
                    self.is_dswr = self.inst.opcode.startswith("ds_write")
                    self.is_ds = self.inst.opcode.startswith("ds_")
                    return self.inst
                return None

        vthreads = []
        signal_table = {}
        for thr in thread_jits:
            bb = BasicBlock(self, thr.__name__)
            self.current_bb = bb
            thr()
            vthreads.append(VThread(bb, thr.__name__, signal_table))
        
        self.current_bb = oldbb
        # schedule instructions from bbs into self.current_bb
        ISSUE_INTERVAL = { "vmem": 32,"ds":32, "mfma": 16}
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
            for ithr, thr in  enumerate(vthreads):
                if thr.finished: continue
                is_all_finished = False
                if thr.waitting: continue
                # check issue-interval
                if thr.is_mem_load and (clock_cycle - last_global_load_cycle) < ISSUE_INTERVAL["vmem"]: continue
                if thr.is_ds and (clock_cycle - last_ds_cycle) < ISSUE_INTERVAL["ds"]: continue
                if thr.is_mfma and (clock_cycle - last_mfma_cycle) < ISSUE_INTERVAL["mfma"]: continue
                # select the candidate with minimal timeslice
                progress = thr.progress()
                if min_progress > progress or ithr == 0:
                    min_progress = progress
                    selected_thr = thr

            if selected_thr is not None:
                self.current_bb.add_instruction(selected_thr.inst)
                if selected_thr.is_mem_load: last_global_load_cycle = clock_cycle
                if selected_thr.is_ds: last_ds_cycle = clock_cycle
                if selected_thr.is_mfma: last_mfma_cycle = clock_cycle
                selected_thr.next()
            else:
                pass # just wait for some threads ready

            # assume each issue took 4 cycles
            clock_cycle += 4
        pass

    def shift_bits(self, imm):
        if imm <= 0 or (imm & (imm - 1)) != 0:
            assert 0, f"the imm{imm} is not power of 2"
        return imm.bit_length() - 1

    def float_bits(self, f):
        return float_to_ieee754_bits_little(f)

    def alloc_lds(self, num_bytes, align=4):
        offset = self.lds_allocator.malloc(num_bytes, align)
        return offset

    def free_lds(self, offset):
        self.lds_allocator.free(offset)

    def LDSTensor(self, shape, dtype):
        return LDSTensor(self, shape, dtype)

    def debug_setup(self, cond:GPRExpr):
        if self.debug_log_ptr is None:
            self.log("WARNNING: calling debug_setup w/o debug_log_ptr")
            return
        self.debug_cond_sgpr = self.gpr("su32")
        self.debug_cond_sgpr[0] = cond

    def debug_log(self, gprs:Union[GPRs, GPRExpr], torch_dtype, gpr_layout="", message=""):
        """
        cond : filter the kernel instance to enable log
        """
        if self.debug_log_ptr is None:
            self.log("WARNNING: calling debug_log w/o debug_log_ptr")
            return
        if self.debug_cond_sgpr is None:
            self.log("WARNNING: calling debug_log w/o debug_setup")
            return
        assert isinstance(self.debug_cond_sgpr, GPRs)
        log_index = len(self.debug_log_info)
        #self.Jump(f"debug_log_skip_{log_index}", cond, reverse=True)
        self.s_cmp_eq_u32(0, self.debug_cond_sgpr)
        self.s_cbranch_scc1(mod=f"debug_log_skip_{log_index}")
        caller_frame = inspect.stack()[1]
        name = "?"
        if caller_frame.code_context:
            src_line = caller_frame.code_context[0].strip()
            if ".debug_log(" in src_line:
                args = src_line.split(".debug_log(")[-1].split(",")
                if len(args) >= 1:
                    name = args[0].strip()
        if message:
            name = f'{name}("{message}")'
        # gprs
        gprs = gprs[...]
        if isinstance(gprs, GPRExpr):
            self.op == "getitem"
            rtype = gprs.src0.rtype
            count = gprs.src2 - gprs.src1 + 1
        elif isinstance(gprs, GPRs):
            rtype = gprs.rtype
            count = gprs.count

        if rtype == "s":
            vtemp = self.gpr(count, "vu32")
            for i in range(count):
                self.v_mov_b32(vtemp[i], gprs[i])
            gprs = vtemp

        vtemp = self.gpr("vu32")
        vtemp[0] = log_index
        self.global_store_dword((self.threadIdx.x[0] & 0), vtemp, self.debug_log_ptr)
        self.s_add_u32(self.debug_log_ptr[0], self.debug_log_ptr[0], 4)
        self.s_addc_u32(self.debug_log_ptr[1], self.debug_log_ptr[1], 0)

        # data from same lane will be stored as inner-most dimension
        # data between lanes has stride of (4*count)
        if callable(gpr_layout):
            shape = gpr_layout(self.debug_log_ptr, gprs)
        elif gpr_layout == "":
            # default layout, just put data in same lane together
            vaddr = self.gpr((self.threadIdx.x[0] & 63) * (count * 4))
            for i in range(count):
                self.global_store_dword(vaddr, gprs[i], self.debug_log_ptr)
                vaddr[0] = vaddr[0] + 4
            shape = (64, 4*count//torch_dtype.itemsize)
        else:
            # "4h.4h.16v.4h"
            # "4v1.4h.16v4.4h"   each segment has optional stride (rows in v-dir or num of dtypes in h-dir)
            layout = gpr_layout.split(".")
            assert len(layout) ==3 or len(layout) == 4
            lane_size = int(layout[-1][:-1])
            lane_bytes = lane_size * torch_dtype.itemsize
            lane_dir = layout[-1][-1]
            assert lane_dir== "h", "only lane along inner-most dim is supported"
            assert lane_bytes == 16 or lane_bytes == 4, f"{lane_bytes=} but only DWORDx1/x4 lane-size supported"
            lane_dim0, lane_dim1 = int(layout[-3][:-1]), int(layout[-2][:-1])
            lane_dir0, lane_dir1 = layout[-3][-1], layout[-2][-1]
            assert (lane_dim0 * lane_dim1) == 64, "64 lanes must be layout as 2D"
            assert (lane_dir0 == "v" or  lane_dir0 == "h")
            assert (lane_dir1 == "v" or  lane_dir1 == "h")
            assert (lane_dir0 != lane_dir1)
            if len(layout) == 4:
                dw4_count = int(layout[0][:-1])
                dw4_count_dir = layout[0][-1]
            else:
                dw4_count = 1
                dw4_count_dir = "h"
            assert dw4_count * (lane_bytes//4) == count, "GPR count must match"
            total_v = 1
            total_v *= lane_dim0 if lane_dir0 == "v" else 1
            total_v *= lane_dim1 if lane_dir1 == "v" else 1
            total_v *= dw4_count if dw4_count_dir == "v" else 1
            total_v *= lane_size if lane_dir == "v" else 1
            total_h = 1
            total_h *= lane_dim0 if lane_dir0 == "h" else 1
            total_h *= lane_dim1 if lane_dir1 == "h" else 1
            total_h *= dw4_count if dw4_count_dir == "h" else 1
            total_h *= lane_size if lane_dir == "h" else 1
            stride_bytes = total_h * torch_dtype.itemsize
            shape = (total_v, total_h)
            lane_id = self.gpr(self.threadIdx.x[0] % 64)
            if lane_dir1 == "v":
                vtemp[0] = (lane_id % lane_dim1)*stride_bytes + (lane_id//lane_dim1)*lane_bytes
                if dw4_count_dir == "v":
                    dw4_step = lane_dim1*stride_bytes
                else:
                    dw4_step = lane_dim0*lane_bytes
            else:
                vtemp[0] = (lane_id % lane_dim1)*lane_bytes + (lane_id//lane_dim1)*stride_bytes
                if dw4_count_dir == "v":
                    dw4_step = lane_dim0*stride_bytes
                else:
                    dw4_step = lane_dim1*lane_bytes

            for dc in range(dw4_count):
                if lane_bytes == 4:
                    self.global_store_dword(vtemp, gprs[dc], self.debug_log_ptr)
                if lane_bytes == 16:
                    self.global_store_dwordx4(vtemp, gprs[dc*4+0:dc*4+3], self.debug_log_ptr)
                vtemp[0] = vtemp[0] + dw4_step

        # record log info at compile time
        self.debug_log_info.append([
            name,
            rtype,
            count, # number of 32-bit regs
            torch_dtype,
            shape,
            gpr_layout
        ])
        self.s_add_u32(self.debug_log_ptr[0], self.debug_log_ptr[0], (64 * count * 4))
        self.s_addc_u32(self.debug_log_ptr[1], self.debug_log_ptr[1], 0)
        self.s_waitcnt(mod="vmcnt(0)")
        self.Label(f"debug_log_skip_{log_index}")
        # debug log function

    @staticmethod
    def parse_debug_logs(debug_log_info, debug_log_buff, verbose = True):
        global torch
        import torch
        from collections import OrderedDict

        buffer = debug_log_buff.cpu().numpy()
        offset = 0
        ret = {}
        cnt = 0
        color_id = 3
        color0 = f"\033[0;{30+(color_id % 8)}m"
        color1 = f"\033[0m"
        if verbose:
            print(f"{color0}==== {verbose} debug log ===={color1}")
        while True:
            index = struct.unpack_from('<I', buffer, offset)[0]
            offset += 4
            if index < 0 or index >= len(debug_log_info): break
            name, rtype, count, dtype, shape, layout = debug_log_info[index]
            dtype_size = torch.tensor([], dtype=dtype).element_size()

            tensor = torch.frombuffer(buffer, dtype=dtype, count=count*64*(4//dtype_size), offset=offset)
            if rtype == "s": tensor = tensor.reshape(64, (4*count//dtype_size))[0,:].tolist()
            offset += 64*4*count
            tag = f"[{cnt}]:  {name} ({rtype}gpr x {count})"
            if verbose:
                if rtype == "s":
                    print(f"{color0}{tag}{color1} {tensor}")
                else:
                    print(f"{color0}{tag} {shape=} {dtype=} {layout=} {color1}")
                    if layout != "":
                        tensor = tensor.reshape(*shape)
                        print(tensor)
                    else:
                        tensor = tensor.reshape(64, (4*count//dtype_size))
                        for lane_id in range(64):
                            print(f" lane {lane_id:2} : {tensor[lane_id, :]}")
            if name not in ret:
                ret[name] = []
            ret[name].append(tensor)
            cnt += 1
        return ret

    def pass_hide_dependency(self):
        # hide dependency between address calculation & ds_read/ds_write/global_load
        # by reorder instructions following the mem-access
        for bb in self.blocks:
            i = 1
            while i < len(bb.instructions):
                prev = bb.instructions[i-1]
                cur = bb.instructions[i]
                vaddr = None
                if cur.opcode.startswith("global_load_dword"):
                    vaddr = cur.operands[1]
                elif cur.opcode.startswith("ds_write_"):
                    vaddr = cur.operands[0]
                elif cur.opcode.startswith("ds_read_"):
                    vaddr = cur.operands[1]
                if (vaddr is not None) and len(prev.operands) and prev.opcode.startswith("v_") and prev.operands[0].overlap(vaddr):
                    # if next & next-next instructions is not s_waitcnt and another vmem/ds instruction,
                    # we can swap because any instruction try to use the result of load have to wait
                    swap_cnt = 0
                    while swap_cnt < 3:
                        if i + 1 >= len(bb.instructions):
                            break
                        next = bb.instructions[i + 1]
                        if next.opcode.startswith("v_") and isinstance(next.operands[0], GPRExpr) and (not next.operands[0].overlap(vaddr)):
                            bb.instructions[i], bb.instructions[i+1] = next, cur
                            i = i + 1
                            swap_cnt += 1
                        else:
                            break
                i = i + 1

    # 前移一些ALU指令到加载kargs的s_waitcnt之前, kernel编写者需要手工把无关的valu指令放在整个kernel的最开始处
    def pass_hide_karg_loads(self):
        if len(self.blocks) == 0: return

        sgpr_loading = []

        def can_move_forward(inst):
            if inst.opcode.startswith("v_") and len(inst.operands) > 0:
                for i in range(1,len(inst.operands)):
                    src = inst.operands[i]
                    if (not isinstance(src, GPRExpr)) and (not isinstance(src, GPRs)):
                        continue
                    for sgpr in sgpr_loading:
                        if src.overlap(sgpr): return False
                return True
            return False

        first_bb = self.blocks[0]
        for i in range(len(first_bb.instructions)):
            inst = first_bb.instructions[i]
            if inst.opcode.startswith("s_waitcnt"):
                # looking for v_ instructions(w/o using any sgpr) to move before this waitcnt
                k = i + 1
                i_s_waitcnt = i
                while k < len(first_bb.instructions):
                    inst_k = first_bb.instructions[k]
                    if not can_move_forward(inst_k): break
                    # move instruction forward
                    first_bb.instructions[i_s_waitcnt],first_bb.instructions[k] = inst_k, first_bb.instructions[i_s_waitcnt]
                    i_s_waitcnt = k
                    k = k + 1
                break
            if not inst.opcode.startswith("s_load_"): break
            if len(inst.operands):
                sgpr_loading.append(inst.operands[0])

    def pass_cse(self):
        if self.is_no_pass(): return
        debug_enabled = self.debug_print()

        def is_gpr(op):
            return isinstance(op, GPRExpr)

        # check GPRs, GPRs can be optimized by CSE pass
        #   - written only once
        #   - accessed by 1 BB only
        #   - being read after written
        #
        gpr_info = {}
        for bb_id, bb in enumerate(self.blocks):
            for t, inst in enumerate(bb.instructions):
                vdst_index = inst.dst_operand_id()
                for index, gpr in enumerate(inst.operands):
                    if isinstance(gpr, GPRExpr):
                        for i, r in enumerate(gpr):
                            key = repr(r)
                            if key not in gpr_info:
                                info = {"wr":"","location":list()}
                                gpr_info[key] = info
                            else:
                                info = gpr_info[key]
                            info["location"].append((bb_id, t, index))
                            if index == vdst_index:
                                info["wr"] += "w"
                            else:
                                info["wr"] += "r"
        ssa_gpr = []
        for k,info in gpr_info.items():
            wr = info["wr"]
            locs = info["location"]
            if debug_enabled:
                print(k, wr, locs)
            if len(wr) > 0 and \
                wr[0] == "w" and \
                wr.count("w") == 1:
                ssa_gpr.append(k)

        # TODO: if some ssa gpr also only depends on ssa gpr(like threadIdx.x),
        # it means their value can exist across BB boundary

        self.debug_print("pass_cse: ", f"{ssa_gpr=}")
        # we limited CSE to following instructions because they are used by recursive expression generation.
        # Result of these instructions can be reused w/o worry about side-effect.
        # but any other instructions, although cannot be elimited, may update registers holding the reusable value,
        # so any other instructions with it's 1st operand (most likely vdst)
        # holding a value will destroy this reuable value (maybe over-react, but it's for correctness/safety reason)
        #
        # TODO, support s_mul_i32,s_lshl_b32,s_lshr_b32,s_ashr_i32,s_and_b32,s_or_b32,s_xor_b32,s_sub_u32,s_sub_i32,s_add_i32,s_add_u32,s_addk_i32,s_cmp_
        cse_inst_list = [
            "v_mov_b32",
            "v_sub_",
            "v_add_",
            "v_mul_",
            "v_lshlrev_b32",
            "v_lshrrev_b32",
            "v_ashrrev_i32",
            "v_and_b32",
            "v_or_b32",
            "v_xor_b32",
            "v_readfirstlane_b32",
        ]
        def is_cse_inst(inst):
            for prefix in cse_inst_list:
                if inst.opcode.startswith(prefix):
                    return True
            return False

        reg_2_value_version = {}
        def get_value_version(reg):
            tag = repr(reg)
            if tag not in reg_2_value_version:
                reg_2_value_version[tag] = 0 # value-version
            return reg_2_value_version[tag]

        # any update to reg will create a new (versioned) value
        def inc_value_version(reg):
            tag = repr(reg)
            if tag not in reg_2_value_version:
                reg_2_value_version[tag] = 0 # value-version
            else:
                reg_2_value_version[tag] += 1 # increase version number
            return reg_2_value_version[tag]

        global replace_index
        CSE_LIMIT = int(os.getenv("CSE_LIMIT", "999999"))
        for bb in self.blocks:
            reg_2_value_version = {}
            value_table = {}

            # to make following algo easier, we record the instruction number of each
            # read & write to a register
            reg_accesses = {}
            def add_reg_access(r, time, rwtype):
                key = repr(r)
                if key not in reg_accesses:
                    reg_accesses[key] = []
                reg_accesses[key].append((time, rwtype))

            def reg_access_last_read_from(r, time):
                key = repr(r)
                last_time = time
                accesses = reg_accesses[key]
                for t, rw in accesses:
                    if t > time:
                        if "r" in rw:
                            last_time = t
                        if "w" in rw:
                            break
                return last_time

            def reg_access_next_write_to(r, time):
                key = repr(r)
                accesses = reg_accesses[key]
                for t, rw in accesses:
                    if t > time:
                        if "w" in rw:
                            return t
                # no write, the value is always valid, return the max time
                return len(bb.instructions)

            def replace_vdst_with_exist(t_first, t_last_read, vdst, vexist):
                key = repr(vdst)
                accesses = reg_accesses[key]
                for t, rw in accesses:
                    if t >= t_first and t <= t_last_read and "r" in rw:
                        inst = bb.instructions[t]
                        if debug_enabled: print(f"found  {inst}")
                        for i,op in enumerate(inst.operands):
                            if is_gpr(op):
                                if vdst.overlap(op):
                                    assert vdst.overlap(op, fully_match=True)
                                    inst.operands[i] = vexist
                                    if debug_enabled: print(f"\t {i} {op} {vexist}")

            def can_replace_vdst_with_exist_global(vdst):
                for bb_id, inst_id, op_id in gpr_info[repr(vdst)]["location"]:
                    inst = self.blocks[bb_id].instructions[inst_id]
                    op = inst.operands[op_id]
                    if vdst.overlap(op):
                        if not vdst.overlap(op, fully_match=True):
                            # some access pattern are not compatible, do not replace
                            return False
                return True

            def replace_vdst_with_exist_global(vdst, vexist):
                for bb_id, inst_id, op_id in gpr_info[repr(vdst)]["location"]:
                    inst = self.blocks[bb_id].instructions[inst_id]
                    op = inst.operands[op_id]
                    if vdst.overlap(op):
                        if debug_enabled: print(f"found  {inst}")
                        assert vdst.overlap(op, fully_match=True), f"{vdst}({vdst.src0.name}) overlap with {inst}"
                        inst.operands[op_id] = vexist
                        if debug_enabled: print(f"\t {i} {op} {vexist}")
                return True

            for t, inst in enumerate(bb.instructions):
                if len(inst.operands) == 0: continue
                # instruction may have multiple vdst/vsrc: v[4:7]
                dst = inst.operands[0]
                if isinstance(dst, GPRExpr):
                    # Why "rw" instead of just "w" ?
                    # we cannot be 100% sure that first operand is dst,
                    # it may be a src too, let's be conservative
                    for r in dst:
                        add_reg_access(r, t, "rw")

                for src in inst.operands[1:]:
                    if isinstance(src, GPRExpr):
                        for r in src:
                            add_reg_access(r, t, "r")

            removed_insts = []
            for t, inst in enumerate(bb.instructions):
                # live_value_numbering table:
                #  - update existing vdst generates a new value : 
                #       v2 = v1+v0;
                #       v1 = 8;     <===== v1 has updated, with value number: v1.1
                #       v3 = v1+v0; <===== this is not the same as v2
                # 
                if len(inst.operands) == 0: continue
                vdsts = inst.operands[0]
                if not isinstance(vdsts, GPRExpr):
                    continue
                
                # maintain a value version number for each vgpr instance
                is_valid_gpr = True
                for dst_gpr in vdsts:
                    is_valid_gpr = is_valid_gpr and (repr(dst_gpr) in ssa_gpr)
                    cur_version = inc_value_version(dst_gpr)
                    # invalidate previous value this vdst is holding
                    for k in list(value_table.keys()):
                        v, version = value_table[k]
                        if v is dst_gpr and version < cur_version:
                            del value_table[k]

                # all dst gprs must be valid (only written once)
                if is_valid_gpr and is_cse_inst(inst) :
                    # this key includes opcode & all input-values & modifiers
                    src_ops = []
                    for op in inst.operands[1:]:
                        if isinstance(op, GPRExpr):
                            op_name = ""
                            for v in op:
                                op_name += repr(v) + "." + str(get_value_version(v)) + ";"
                        else:
                            op_name = str(inst.op_repr(op))
                        src_ops.append(op_name)
                    # the dependent inputs are versioned, to avoid mismatches when some inputs
                    # may be updated with new value.
                    for idst,vdst in enumerate(vdsts):
                        key = f"{inst.opcode} {','.join(src_ops)} {inst.mod} [{idst}]"
                        if key not in value_table:
                            # first time a reusable value is generated in versioned vdst
                            value_table[key] = [vdst, cur_version]
                        elif len(vdsts) == 1:
                            vexist, version = value_table[key]
                            if repr(vexist) in ssa_gpr and replace_index < CSE_LIMIT:
                                if can_replace_vdst_with_exist_global(vdst):
                                    removed_insts.append(t)
                                    if debug_enabled:
                                        print(f"===========  {replace_index} / {CSE_LIMIT=} ")
                                        print(t, inst.loc, inst.debug_info, inst)
                                        print(f"{vdst} replaced with  {vexist} {key} globally")
                                    replace_vdst_with_exist_global(vdst, vexist)
                            else:
                                # can we safely use this existing value?
                                # we need to check life-cycle requirements:
                                #   the last read from vdst happens before the next update(write) to vexist
                                t_last_read = reg_access_last_read_from(vdst, t)
                                t_next_update = reg_access_next_write_to(vexist, t)
                                if debug_enabled:
                                    print(f"===========  {replace_index} / {CSE_LIMIT=} ", 
                                        (t_next_update > t_last_read) and (replace_index < CSE_LIMIT))
                                    print(t, inst.loc, inst.debug_info, inst)
                                    print(f"{vdst} replaced with  {vexist} {key} in range [{t+1}~{t_last_read}]")
                                    print(f"{reg_accesses[repr(vdst)]}")
                                if t_next_update > t_last_read and replace_index < CSE_LIMIT:
                                    removed_insts.append(t)
                                    replace_vdst_with_exist(t + 1, t_last_read, vdst, vexist)
                                else:
                                    # we cannot safely reuse vexist, optionally we can add another vexist
                                    # so others may benefit
                                    pass
                            replace_index += 1

            # Eliminate
            bb.instructions = [inst for t,inst in enumerate(bb.instructions) if t not in removed_insts]

    def show_code(self):
        self.log("===================== bb")
        for bb in self.blocks:
            self.log(bb.debug_str())
        if self.current_bb is not None:
            self.log("===================== current_bb")
            self.log(self.current_bb.debug_str())

    def dump_code(self, fname):
        global dump_serial_id
        if len(PYHIP_DUMP_DIR) == 0: return
        with open(f"{PYHIP_DUMP_DIR}/{dump_serial_id}-{fname}-{self.kernel_tag}.txt", 'w') as f:
            print("===================== bb", file=f)
            for bb in self.blocks:
                print(bb.debug_str(), file=f)
            print("===================== current_bb", file=f)
            if self.current_bb is not None:
                print(self.current_bb.debug_str(), file=f)
        dump_serial_id += 1

    def pass_remove_dead_bb(self):
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

    def build(self, kernel_name, signature, extra_compiler_options, cpp_src_fpath, dump_stat):
        self.asm_debug_info = ""
        self._finish_bb(self.current_bb)

        self.dump_code(f"after_nothing")

        self.pass_remove_dead_bb()
        self.dump_code(f"after_pass_remove_dead_bb")

        self.pass_a2v()
        self.dump_code(f"after_pass_a2v")

        # use following way only
        # if we want to do multi-expression optimization (like common expr extraction...)
        # self.compile_expressions()
        self.pass_insert_nop()
        self.dump_code(f"after_pass_insert_nop")

        # for bb in self.blocks: print(repr(bb))
        self.pass_hide_dependency()
        self.dump_code(f"after_pass_hide_dependency")

        # kernel args are loaded with s_waitcnt, many v_ instructions can be
        # moved into this wait cycles
        self.pass_hide_karg_loads()
        self.dump_code(f"after_pass_hide_karg_loads")

        # Common Subexpression Elimination，CSE
        #from viztracer import VizTracer
        #with VizTracer(output_file="optional.json") as tracer:
        self.pass_cse()
        self.dump_code(f"after_pass_cse")

        # Dead Store Elimination, DSE
        self.pass_dse()
        self.dump_code(f"after_pass_dse")

        # Dead Code Elimination，DCE
        self.pass_dce()
        self.dump_code(f"after_pass_dce")

        self.pass_break_down_gprs()
        self.dump_code(f"after_pass_break_down_gprs")

        # for bb in self.blocks: print(repr(bb))
        used_gprs = self.register_allocation_linear_scan()
        self.dump_code(f"after_reg_allocation")

        # generate asm: basic blocks are in natural order
        asm=""
        for bb in self.blocks:
            asm += repr(bb)

        if dump_stat:
            latencies = {
                'ds_read_b128': 8,
                'ds_write_b128': 20,
                'ds_write_b64': 12,
                'v_exp_f32': 16
            }
            for bb in self.blocks:
                ins_stat = {}
                print(f'\nblock "{bb.label_name}" has {len(bb.instructions)} instructions:')
                cycles = 0
                for ins in bb.instructions:
                    if ins.opcode not in ins_stat:
                        ins_stat[ins.opcode] = [0, 0]
                    latency = dict.get(latencies, ins.opcode, 4)
                    ins_stat[ins.opcode][0] += 1
                    ins_stat[ins.opcode][1] += latency
                    cycles += latency
                ops = sorted(ins_stat.keys())
                for op in ops:
                    print(f'\t{op:30}: {ins_stat[op][0]:5} times, {ins_stat[op][1]:8,} cycles')
                print(f'estimated issue cost(without stall) will be {cycles:,} cycles')
        # asm += self.asm_debug_info
        if 0:
            color0 = f"\033[0;{30+(2 % 8)}m"
            color1 = f"\033[0m"
            print(color0)
            print(self.asm_debug_info)
            print(color1)
        artifact = {"asm":[]}
        for a in asm.splitlines():
            if a[0] == ";": continue
            artifact["asm"].append(a)
            """
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
            """
        '''
        str_arg_c_del = ",".join([f"{a[0]} {a[1]}" for a in args])
        signature = f"{kernel_name}({str_arg_c_del})"
        '''
        with_debug = "-g" in extra_compiler_options
        inline_asm_lines = []
        line_no = 11
        asm_cnt = 0
        for line in asm.splitlines():
            if len(line):
                if asm_cnt > 0 and with_debug:
                    asm_cnt = 0
                    inline_asm_lines.append(f'"    .loc 2 {line_no} {5}\\n"')
                    line_no += 1
                inline_asm_lines.append(f'"{line}\\n"')
                line_no += 1
                asm_cnt += 1
            else:
                inline_asm_lines.append("")
        inline_asm = "\n".join(inline_asm_lines)
        str_used_gprs = ""
        if len(used_gprs):
            str_used_gprs = "," + ",".join([f'\"{s}\"' for s in used_gprs])
        self.log(f" kernel: {kernel_name}{signature}  used_gprs={used_gprs}")

        decl_lds = ""
        if self.lds_allocator.upper_bound() > 0:
            # (as3_uint32_ptr)(lds_buffer) is always 0
            # but this 2 lines of code has side effect: compiler may use s0 as `lds_buffer`` input 
            # which causes damage to kernal-arg pointer s[0:1], we can work-around it by putting
            # these 2 lines of codes at the end of the kernel's source code.
            decl_lds += f"    __shared__ uint lds_buffer[{self.lds_allocator.upper_bound()//4}];\n"
            decl_lds += f'    asm(" ; lds_buffer %0 "::"s"((as3_uint32_ptr)(lds_buffer)));\n'
            pass

        # embed revision into bin
        git_commit = [rf'".pushsection .rodata\n"']
        git_commit += [rf'"    .align 8\n"']
        git_commit += [rf'"    .asciz  \".git:{get_git_revision_hash()}\"\n"']
        git_commit += [rf'"    .popsection"']
        git_commit = '\n'.join(git_commit)
        git_commit = f'''    asm volatile({git_commit});\n'''
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
''' + decl_lds + git_commit + r'''
}
        '''

        with open(cpp_src_fpath, 'w', encoding='utf-8') as f:
            f.write(hip_src)

        artifact["used_gprs"] = used_gprs
        artifact["debug_log_info"] = self.debug_log_info

        return self.compile(kernel_name, cpp_src_fpath, extra_compiler_options), artifact

    def compile(self, kernel_name, cpp_src_fpath, extra_compiler_options):
        hip = module(cpp_src_fpath, extra_compiler_options)
        hip_func = getattr(hip, kernel_name)
        return hip_func

    def div(self, x, y):
        assert x % y == 0
        assert x >= y
        return x // y

    def div_up(self, x, y):
        return (x + y - 1) // y

    def round_up(self, x, y):
        return self.div_up(x, y) * y

    """
    def compute_generator():
        yield 16 # yield the cycles following instruction is going to take
        J.v_mfma
    
    gen = compute_generator()
    J.emit(gen, 32)         # emit instructions which consume 32 cycles

    """
    def emit(self, generators, cycles:int=99999999):
        # canonicalize generators a list/tuple of generators
        if isinstance(generators, types.GeneratorType):
            generators = (generators,)
        while cycles > 0:
            found = False
            for g in generators:
                yield_cycle = next(g, None)
                if yield_cycle is not None:
                    found = True
                    break
            if not found:
                return False
            cycles -= yield_cycle
        return True

    def emitter(self, yield_cycle = 1):
        def emit(generators:list, cycles:int=99999999):
            while cycles > 0:
                found = False
                for g in generators:
                    if next(g, None) is not None:
                        found = True
                        break
                if not found:
                    break
                cycles -= yield_cycle
        return emit

    @cache
    def get_sgpr_const(self, value):
        sgpr = self.gpr("su32", name=f"sgpr_const_{value}")
        sgpr[0] = value
        return sgpr
    '''
    reduce("v_max_f32", vinput)
    reduce("v_add_f32", vinput)
    '''
    def reduce(self, vinst, vinput):
        self.s_nop(mod="2")
        v1 = self.new_gpr('v',1,dtype="i32", align=1, name="reduce_v1")
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
        vaddr = self.new_gpr('v',1,dtype="i32", align=1, name="reduce_vaddr")
        vaddr[0] = 63*4 # broadcast last lane to all lanes
        self.ds_bpermute_b32(v1, vaddr, v1) # vdst,  vaddr,    vdata   offset
        self.s_waitcnt(mod=f"lgkmcnt({0})")
        return v1


    def transpose_per_lane(self, src_row, src_cols, dtype_bytes, src, dst):
        """
        jit is better at writting general logic,
        assume item in src&dst are row-major

        for example when N is 4:
            [x|x] is a b32 VGPR, x is a b16 (2bytes) with the VGPR
            both src & dst are 8 VGPRs contains 4x4 b16(16bits) array

                [0|0] [1|1]        [0|2] [4|6]
                [2|2] [3|3]  ====> [0|2] [4|6]
                [4|4] [5|5]        [1|3] [5|5]
                [6|6] [7|7]        [1|3] [7|7]
            trans2x2:
         src1   [01|23]  => [01|45]
         src0   [45|67]     [23|67]
        """
        def get_perm_pattern_16():
            return self.get_sgpr_const(0x01_00_05_04), self.get_sgpr_const(0x03_02_07_06)

        def get_perm_pattern_8():
            trans_low1 = self.get_sgpr_const(0x02_06_00_04)
            trans_high1 = self.get_sgpr_const(0x03_07_01_05)
            trans_low2 = self.get_sgpr_const(0x01_00_05_04)
            trans_high2 = self.get_sgpr_const(0x03_02_07_06)
            return trans_low1, trans_high1, trans_low2, trans_high2

        M, N = src_row, src_cols
        assert dtype_bytes in [1,2,4]
        items_per_GPR = 4 // dtype_bytes
        assert (M % items_per_GPR) == 0
        assert (N % items_per_GPR) == 0
        assert len(src)*items_per_GPR >= M*N, f"{len(src)=} {src=} {items_per_GPR=} {M=} {N=}"
        assert len(dst)*items_per_GPR >= N*M
        # these sgprs are generated ever time this function is called
        # not cached
        if dtype_bytes == 4:
            for src_row in range(0,M,items_per_GPR):
                for src_col in range(0,N,items_per_GPR):
                    dst_row, dst_col = src_col, src_row
                    src_idx = src_row * N//items_per_GPR + src_col//items_per_GPR
                    dst_idx = dst_row * M//items_per_GPR + dst_col//items_per_GPR
                    dst[dst_idx] = src[src_idx]
        if dtype_bytes == 2:
            trans_low, trans_high = get_perm_pattern_16()
            def trans2x2(s0, s1, d0, d1):
                self.v_perm_b32(d0, s0, s1, trans_low)
                self.v_perm_b32(d1, s0, s1, trans_high)

            for src_row in range(0,M,items_per_GPR):
                for src_col in range(0,N,items_per_GPR):
                    dst_row, dst_col = src_col, src_row # transpose
                    src_idx = src_row * N//items_per_GPR + src_col//items_per_GPR
                    dst_idx = dst_row * M//items_per_GPR + dst_col//items_per_GPR
                    #print(idx1, idx1+N//2, idx2, idx2+N//2)
                    trans2x2(src[src_idx], src[src_idx+N//items_per_GPR],
                            dst[dst_idx], dst[dst_idx+M//items_per_GPR])
        if dtype_bytes == 1:
            trans_low1, trans_high1, trans_low2, trans_high2 = get_perm_pattern_8()
            vtemp0 = self.gpr("vu32")
            vtemp1 = self.gpr("vu32")
            vtemp2 = self.gpr("vu32")
            vtemp3 = self.gpr("vu32")
            def trans4x4(s0, s1, s2, s3, d0, d1, d2, d3):
                self.v_perm_b32(vtemp0, s0, s1, trans_low1)
                self.v_perm_b32(vtemp1, s0, s1, trans_high1)
                self.v_perm_b32(vtemp2, s2, s3, trans_low1)
                self.v_perm_b32(vtemp3, s2, s3, trans_high1)
                self.v_perm_b32(d0, vtemp0, vtemp2, trans_low2)
                self.v_perm_b32(d2, vtemp0, vtemp2, trans_high2)
                self.v_perm_b32(d1, vtemp1, vtemp3, trans_low2)
                self.v_perm_b32(d3, vtemp1, vtemp3, trans_high2)

            for src_row in range(0,M,items_per_GPR):
                for src_col in range(0,N,items_per_GPR):
                    dst_row, dst_col = src_col, src_row # transpose
                    src_idx = src_row * N//items_per_GPR + src_col//items_per_GPR
                    dst_idx = dst_row * M//items_per_GPR + dst_col//items_per_GPR
                    trans4x4(
                        src[src_idx],
                        src[src_idx+N//items_per_GPR],
                        src[src_idx+N//items_per_GPR*2],
                        src[src_idx+N//items_per_GPR*3],
                        dst[dst_idx],
                        dst[dst_idx+M//items_per_GPR],
                        dst[dst_idx+M//items_per_GPR*2],
                        dst[dst_idx+M//items_per_GPR*3]
                    )

    def vmax(self, dtype, dst_vgpr, src_vgprs:list):
        others = []
        # put the dst at first place if it's one of the sources
        for s in src_vgprs:
            if s is dst_vgpr:
                others.append(s)
                break
        for s in src_vgprs:
            if s is not dst_vgpr:
                others.append(s)
        if len(others) >= 3:
            inst_vmax3 = getattr(self, f"v_max3_{dtype}")
            inst_vmax3(dst_vgpr, others.pop(0), others.pop(0), others.pop(0))
            while len(others) >= 2:
                inst_vmax3 = getattr(self, f"v_max3_{dtype}")
                inst_vmax3(dst_vgpr, dst_vgpr, others.pop(0), others.pop(0))
            if len(others):
                inst_vmax2 = getattr(self, f"v_max_{dtype}")
                inst_vmax2(dst_vgpr, dst_vgpr, others.pop(0))
        else:
            assert len(others) == 2
            inst_vmax2 = getattr(self, f"v_max_{dtype}")
            inst_vmax2(dst_vgpr, others.pop(0), others.pop(0))

    def sigmoid(self, vgpr_src):
        temp0 = self.gpr("vf32")
        self.v_exp_f32(temp0, vgpr_src * (-math.log2(math.exp(1))))
        out = self.gpr("vf32")
        self.v_rcp_f32(out[0], temp0[0] + 1.0)
        return out

    def silu(self, vgpr_src):
        return self.gpr(self.sigmoid(vgpr_src) * vgpr_src)


    # a unified instruction for f32=>bf16 conversion
    def uni_cvt_pk_bf16_f32(self, vdst, vsrc0, vsrc1):
        if self.cdna >= 4:
            return self.v_cvt_pk_bf16_f32(vdst, vsrc0, vsrc1)
        else:
            # this is simple but less accurate, no round to even
            s_cvt_bf16_bias = self.get_sgpr_const(0x00008000)
            return self.v_perm_b32(vdst,
                                   vsrc0 + s_cvt_bf16_bias,
                                   vsrc1 + s_cvt_bf16_bias,
                                   self.get_sgpr_const(0x03_02_07_06))

    def pk_f32_to_bf16(self, vdst, vsrc0, vsrc1):
        self.v_perm_b32(vdst, vsrc0, vsrc1, self.get_sgpr_const(0x03_02_07_06))

    def tb_swizzle(self, block_1d_id:"sgpr", M:"sgpr", wg_M:int, wg_N:int, N:int, M01:int, GroupNum:int):
        J = self
        if GroupNum <= 1 and M01 <= 1:
            N0 = J.div_up(N, wg_N)
            blk_m = J.gpr(block_1d_id // N0)
            blk_n = J.gpr(block_1d_id - blk_m*N0)
            return blk_m, blk_n

        M0 = J.gpr(J.div_up(M, wg_M))
        N0 = J.div_up(N, wg_N)
        group_size    = J.div_up(M0 * N0, GroupNum)
        big_group_num = J.gpr(GroupNum - (group_size * GroupNum - M0 * N0))
        group_id_y    = J.gpr(block_1d_id // GroupNum)
        group_id_x    = J.gpr(block_1d_id - group_id_y * GroupNum) 

        remap_block_1d_id = J.gpr(group_id_x * group_size + group_id_y)

        with J.If(group_id_x > big_group_num):
            remap_block_1d_id[0] += (big_group_num - group_id_x)

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

    @classmethod
    def show_mfma_in_lds(cls, mfma_MN, num_mfmas, swizzle_1=0, swizzle_2=0):
        '''
            visualize mfma lanes inside LDS
        '''
        if cls.cdna == 4:
            lane_groups = [
                [0,1,2,3, 12,13,14,15, 20,21,22,23, 24,25,26,27],
                [4,5,6,7, 8,9,10,11,  16,17,18,19, 28,29,30,31],
                [32,33,34,35, 44,45,46,47, 52,53,54,55, 56,57,58,59],
                [36,37,38,39, 40,41,42,43, 48,49,50,51, 60,61,62,63]
            ]
            num_LDS_banks = 64
        else:
            assert cls.cdna == 3
            lane_groups = [
                [0, 1, 2, 3, 20, 21, 22, 23],
                [4, 5, 6, 7, 16, 17, 18, 19],
                [8, 9, 10, 11, 28, 29, 30, 31],
                [12, 13, 14, 15, 24, 25, 26, 27],
                [32, 33, 34, 35, 52, 53, 54, 55],
                [36, 37, 38, 39, 48, 49, 50, 51],
                [40, 41, 42, 43, 60, 61, 62, 63],
                [44, 45, 46, 47, 56, 57, 58, 59]
            ]
            num_LDS_banks = 32
        def lane_group_id(i):
            for gid,lg in enumerate(lane_groups):
                if i in lg:
                    return gid
            return -1
        def lane_str(i):
            group_id = lane_group_id(i)
            if group_id < 0:
                i = (i % 64)
                group_id = -53 # use a special color
            color0 = f"\033[0;{100+(group_id)}m"
            color1 = f"\033[0m"
            return f"{color0} {i:03} {color1}"

        num_LDS_DW4_banks = num_LDS_banks // 4

        assert cls.warp_size % mfma_MN == 0
        mfma_K_lanes = cls.warp_size // mfma_MN

        k_lanes = mfma_K_lanes * num_mfmas

        # assume the size of mfma lane is also DW4
        assert num_LDS_DW4_banks % k_lanes == 0
        mfma_rows_per_banks = num_LDS_DW4_banks // k_lanes
        assert mfma_MN % mfma_rows_per_banks == 0
        num_bank_rows = mfma_MN // mfma_rows_per_banks

        print(f"CDNA{cls.cdna} MFMA:{mfma_MN}x{mfma_K_lanes} 1x{num_mfmas} {num_LDS_banks=} {num_LDS_DW4_banks=} {k_lanes=} {swizzle_1=} {swizzle_2=}")
        if swizzle_2:
            print(f"\t  logical_col = (logical_col ^ (logical_row // {swizzle_2})) % {k_lanes}")
        bank_lg = [list() for bank_col in range(num_LDS_DW4_banks)]
        for bank_row in range(num_bank_rows):
            for bank_col in range(num_LDS_DW4_banks):
                row = bank_row
                # swizzle is done in term of LDS bank rows & cols
                if swizzle_1:
                    col = (bank_col ^ row) % num_LDS_DW4_banks
                else:
                    col = bank_col

                logical_row = (row * mfma_rows_per_banks + col//k_lanes)
                logical_col = col % k_lanes
                if swizzle_2:
                    logical_col = (logical_col ^ (logical_row // swizzle_2)) % k_lanes
                # at physical location (r0,c0)
                # 
                i0 = logical_col*mfma_MN + logical_row
                if i0 < cls.warp_size:
                    lg = lane_group_id(i0)
                    bank_lg[bank_col].append(lg)
                print(lane_str(i0),end="")
            print("")

        for bank in range(num_LDS_DW4_banks):
            conflict = 0
            for lg in set(bank_lg[bank]):
                c = bank_lg[bank].count(lg)
                conflict += c - 1 if c > 1 else 0
            if conflict:
                print(f"bank{bank}: {conflict} conflicts : lane-groups {bank_lg[bank]} ")
        print()

    @property
    def warp_id(self):
        if "warp_id" not in self.special_vars:
            _warp_id = self.gpr("su32")
            self.v_readfirstlane_b32(_warp_id, self.threadIdx.x[0] // 64)
            self.special_vars["warp_id"] = _warp_id
        return self.special_vars["warp_id"]

    @property
    def lane_id(self):
        if "lane_id" not in self.special_vars:
            _lane_id = self.gpr(self.threadIdx.x[0] % 64)
            self.special_vars["lane_id"] = _lane_id
        return self.special_vars["lane_id"]


gen_hip_file_unique_id = 0

class Idx3D:
    def __init__(self):
        pass

_jit_kernel_unique_id = {}
class jit_kernel:
    def __init__(self, gen_func, extra_compiler_options, with_debug_log = False, dump_stat = False, force_recompile = False, no_pass = None):
        assert callable(gen_func)
        self.extra_compiler_options = extra_compiler_options
        self.dump_stat = dump_stat
        # with_debug_log needs extra internal debug-log buffer, so it's always recompiled
        self.force_recompile = force_recompile
        self.with_debug_log = with_debug_log
        self.gen_func = gen_func
        self.func_name = gen_func.__name__
        self.no_pass = no_pass
        argspec = inspect.getfullargspec(gen_func)
        argtypes = gen_func.__annotations__
        compile_time_args = []
        runtime_time_args = []
        for arg_id, arg_name in enumerate(argspec.args[1:]):
            if arg_name in argtypes:
                atype = argtypes[arg_name].strip()
                if isinstance(atype, str):
                    # runtime args (and only runtime args) must have 
                    # C-type string annotations
                    runtime_time_args.append((arg_id, arg_name, atype))
                else:
                    # other args w/o string annotation are compile-time args
                    compile_time_args.append((arg_id, arg_name, atype))
            else:
                compile_time_args.append((arg_id, arg_name, None))

        self.compile_time_arg_info = compile_time_args
        self.runtime_time_arg_info = runtime_time_args
        self.gen_total_args = len(argspec.args) - 1 # except first arg J
        self.kernel_cache = {}

        self.gen_src_file = inspect.getfile(gen_func)
        self.gen_src_fname = os.path.basename(self.gen_src_file)
        self.line_no = inspect.getsourcelines(gen_func)[1]
        global _jit_kernel_unique_id
        if self.func_name not in _jit_kernel_unique_id:
            _jit_kernel_unique_id[self.func_name] = 0
        _jit_kernel_unique_id[self.func_name] += 1

        # same gen function may constructed many times
        self.gen_construct_id = _jit_kernel_unique_id[self.func_name]

        # self.gen_func_unique_id = hashlib.sha256(f"{self.gen_src_file}-{self.line_no}-{_jit_kernel_unique_id}".encode('utf-8')).hexdigest()
        self.gen_func_unique_id = f"{self.gen_src_file}-{self.line_no}".replace("/","-")

    def split_args(self, args:tuple):
        compile_args = []
        runtime_args = []
        kernel_key = []

        # gridDims, blockDims
        runtime_args.append(args[0])
        runtime_args.append(args[1])

        for c in self.compile_time_arg_info:
            value = args[c[0] + 2]
            name = c[1]
            compile_args.append(value)
            tag = f"{name}={value}".replace("[","").replace("]","").replace(" ","").replace(",","_")
            kernel_key.append(tag)

        for c in self.runtime_time_arg_info:
            runtime_args.append(args[c[0] + 2])

        return compile_args, runtime_args, "-".join(kernel_key)

    def build(self, compile_args, kernel_key):
        # put all generated files under $HOME/.pyhip
        cpp_src_fpath = f"{PYHIP_CACHE_DIR}/{self.func_name}-{self.gen_construct_id}-{kernel_key}-{self.gen_func_unique_id}.cpp"

        J = JIT(f"{self.func_name}-{self.gen_construct_id}-{kernel_key}-{self.gen_func_unique_id}", self.no_pass)

        with filelock.FileLock('.compile.lock'):
            # skip compilation process when target file already exists
            # note the `cpp_src_fpath` is supposed to be generated in previous run(with same compile_args)

            if not self.force_recompile and \
                not self.with_debug_log and \
                os.path.isfile(cpp_src_fpath) and \
                os.path.getmtime(cpp_src_fpath) > os.path.getmtime(self.gen_src_file):
                # if user want non-empty artifact, need to set force_recompile=True
                return J.compile(self.func_name, cpp_src_fpath, self.extra_compiler_options), {}

            # create special sgpr for args
            # and generate codes to load these args
            signatures = []
            sgpr_args = []
            J.kargs = J.new_gpr('s',[0,1],dtype="u32",name="kargs")
            J.threadIdx = Idx3D()
            J.blockIdx = Idx3D()
            J.threadIdx.x = J.new_gpr('v',[0,0], dtype="u32", name="threadIdx.x")
            J.blockIdx.x = J.new_gpr('s',[2,2], dtype="u32", name="blockIdx.x")
            J.blockIdx.y = J.new_gpr('s',[3,3], dtype="u32", name="blockIdx.y")
            J.blockIdx.z = J.new_gpr('s',[4,4], dtype="u32", name="blockIdx.z")
            gen_time_args = [None for _ in range(self.gen_total_args)]

            # fill compile-time args
            for i, (arg_id, arg_name, atype) in enumerate(self.compile_time_arg_info):
                gen_time_args[arg_id] = compile_args[i]

            # fill generation-time args
            arg_offset = 0
            for arg_id, arg_name, atype in self.runtime_time_arg_info:
                signatures.append(f"{atype} {arg_name}")
                if atype.endswith("*"):
                    arg_offset = ((arg_offset + 7) // 8) * 8
                    sgpr = J.new_gpr('s',2, dtype="u32", align=2, name=arg_name)
                    J.s_load_dwordx2(sgpr, J.kargs, arg_offset)
                    gen_time_args[arg_id] = sgpr
                    arg_offset += 8
                    continue
                if atype in ["int","uint","unsigned int"]:
                    arg_offset = ((arg_offset + 3) // 4) * 4
                    sgpr = J.new_gpr('s',1, dtype=f"{atype[0]}32", align=1, name=arg_name)
                    J.s_load_dword(sgpr, J.kargs, arg_offset)
                    gen_time_args[arg_id] = sgpr
                    arg_offset += 4
                    continue
                assert 0, f"unsupported runtime arg type {arg_id=} {arg_name=} {atype=}"

            if self.with_debug_log:
                # this extra debug log ptr will be provided by self
                signatures.append(f"void* debug_log_ptr")
                arg_offset = ((arg_offset + 7) // 8) * 8
                J.debug_log_ptr = J.new_gpr('s',2, dtype="u32", align=2, name="debug_log_ptr")
                J.s_load_dwordx2(J.debug_log_ptr, J.kargs, arg_offset)
                arg_offset += 8

            # initialize warp_id/lane_id vgpr here to prevent ExecMask issue
            # (dce will remove them if no one use it)
            J.warp_id
            J.lane_id

            if arg_offset > 0:
                J.s_waitcnt(mod=f"lgkmcnt(0)")

            # now generate your kernel code
            self.gen_func(J, *gen_time_args)

            return J.build(self.func_name, f"({','.join(signatures)})", self.extra_compiler_options,
                        cpp_src_fpath = cpp_src_fpath,
                        dump_stat=self.dump_stat)

    def __call__(self, *args):
        compile_args, runtime_args, kernel_key = self.split_args(args)

        if self.with_debug_log:
            global torch
            import torch
            debug_log_buff = torch.full([1024*1024//4], -1, dtype=torch.int32)
            runtime_args.append(debug_log_buff.data_ptr())
            kernel, artifact = self.build(compile_args, kernel_key)
            kernel(*runtime_args)
            artifact["debug_log"] = JIT.parse_debug_logs(artifact["debug_log_info"], debug_log_buff, verbose=f"{self.func_name}-{kernel_key}")
            return artifact

        # check cache to see if such object already exist
        if kernel_key in self.kernel_cache:
            # invoke the kernel binary directly with runtime-args only
            kernel, artifact = self.kernel_cache[kernel_key]
        else:
            kernel, artifact = self.build(compile_args, kernel_key)
            self.kernel_cache[kernel_key] = (kernel, artifact)

        kernel(*runtime_args)

        return artifact

class jit:
    def __init__(self, extra_compiler_options = "", with_debug_log=False, dump_stat = False, force_recompile = False, no_pass = None):
        self.extra_compiler_options = extra_compiler_options
        self.with_debug_log = with_debug_log
        self.dump_stat = dump_stat
        self.force_recompile = force_recompile
        self.no_pass = no_pass

    def __call__(self, gen_func):
        return jit_kernel(gen_func, self.extra_compiler_options, self.with_debug_log, self.dump_stat, self.force_recompile, self.no_pass)

class Addr2D:
    def __init__(self, J:JIT, base, row_init, col_init, stride):
        self.vaddr = J.gpr('vu32')
        # TODO: use shift to simplify
        # TODO: remove zero add/mul
        if isinstance(base, int) and base == 0:
            self.vaddr[0] = row_init * stride + col_init
        else:
            self.vaddr[0] = base + row_init * stride + col_init

    def get_addr(self):
        return self.vaddr