from functools import cache
from typing import List, Optional, Set, Union

import filelock
from .hiptools import module
import inspect
import os
from contextlib import contextmanager
import math

from .mem_allocator import SimpleMemoryAllocator

import hashlib
import subprocess

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
                jit.recursive_expr_gen(self.parent_bb, dst_expr, op, loc=self.loc)
                self.operands[i] = dst_expr

            assert isinstance(op, GPRExpr) or \
                   isinstance(op, int) or isinstance(op, float) or \
                   (isinstance(op, str) and (op in ["scc", "vcc", "exec", "off", "execz", "m0"])) or \
                   (isinstance(op, str) and op.startswith("0x")) or \
                   isinstance(op, GPRs), \
            f"arg {i} : {type(op)} {op}"
        self.parent_bb.add_instruction(self, insert_bb_pos)
        return self

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
        self.J.buffer_load_dwordx4(vdst, voffset, self.desc, soffset, mod=mod)

    def load_dwordx2(self, vdst, voffset, soffset, offset12=0):
        # vdst,     vaddr,           srsrc, soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        self.J.buffer_load_dwordx2(vdst, voffset, self.desc, soffset, mod=mod)

    def load_dword(self, vdst, voffset, soffset, offset12=0):
        # vdst,     vaddr,           srsrc, soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset:{offset12}"
        self.J.buffer_load_dword(vdst, voffset, self.desc, soffset, mod=mod)

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

@cache
def get_perm_pattern_16(J):
    trans_low = J.gpr("su32")
    trans_low[0] = 0x01_00_05_04
    trans_high = J.gpr("su32")
    trans_high[0] = 0x03_02_07_06
    return trans_low, trans_high

@cache
def get_perm_pattern_8(J):
    trans_low1 = J.gpr("su32")
    trans_low1[0] = 0x02_06_00_04
    trans_high1 = J.gpr("su32")
    trans_high1[0] = 0x03_07_01_05

    trans_low2 = J.gpr("su32")
    trans_low2[0] = 0x01_00_05_04
    trans_high2 = J.gpr("su32")
    trans_high2[0] = 0x03_02_07_06
    return trans_low1, trans_high1, trans_low2, trans_high2

class LDSTensor:
    def __init__(self, J, shape, dtype):
        self.J = J
        self.shape = shape
        stride_bytes = [dtype.itemsize]
        for i,dim in enumerate(reversed(shape)):
            cur = dim * stride_bytes[-1]
            stride_bytes.append(cur)
        self.size_bytes = stride_bytes[-1]
        self.stride_bytes = list(reversed(stride_bytes))[1:]
        self.dtype = dtype
        self.lds_base = self.J.alloc_lds(self.size_bytes)

    def __del__(self):
        self.J.free_lds(self.lds_base)

    def read(self, dtype:str, vdst, *coord_exprs, offset1=None):
        self._access("read", dtype, vdst, *coord_exprs, offset1=offset1)

    def write(self, dtype:str, vdst, *coord_exprs, offset1=None):
        self._access("write", dtype, vdst, *coord_exprs, offset1=offset1)

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
            assert offset0 >= 0 and offset0 < 64*1024
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
class JIT:
    def __init__(self):
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
        self.lds_allocator = SimpleMemoryAllocator(160*1024)

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
        PYHIP_JIT_LOG = int(os.getenv("PYHIP_JIT_LOG", "1"))
        PYHIP_DEBUG_LOG = os.getenv("PYHIP_DEBUG_LOG", "")
        if PYHIP_JIT_LOG or len(PYHIP_DEBUG_LOG):
            color_id = 3
            color0 = f"\033[0;{30+(color_id % 8)}m"
            color1 = f"\033[0m"
            print(color0, f"[{PYHIP_JIT_LOG=}] ", *args, color1, **kwargs)

    def debug_print(self, *args, **kwargs):
        # caller's function name must be
        caller_name = inspect.currentframe().f_back.f_code.co_name
        PYHIP_DEBUG_LOG = os.getenv("PYHIP_DEBUG_LOG", "")
        for item in PYHIP_DEBUG_LOG.split(":"):
            if item == "":continue
            if item in caller_name or item == "*":
                color_id = 3
                color0 = f"\033[0;{30+(color_id % 8)}m"
                color1 = f"\033[0m"
                print(color0, f"[PYHIP_DEBUG_LOG: {caller_name}] ", *args, color1, **kwargs)
                return True
        return False

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
            dst_gprs = self.new_gpr("s", 1, dtype=dtype, align=1, name="")
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
        dst_gprs = self.new_gpr(rtype, 1, dtype=dtype, align=1)
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
    def ExecMask(self, cond:GPRExpr = None):
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back.f_back
        lineno = caller_frame.f_lineno
        label_begin = f"_execmask_begin_{lineno}_{self.mark_idx}"
        label_end = f"_execmask_end_{lineno}_{self.mark_idx}"
        self.mark_idx += 1
        
        self.SetMask("vcc", cond)
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
        label_begin = f"_simtwhile_begin_{lineno}_{self.mark_idx}"
        label_end = f"_simtwhile_end_{lineno}_{self.mark_idx}"
        self.mark_idx += 1
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
                last_inst.operands[0] is dst_expr and \
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
    a = J.gpr(4, 4, 4,"abf16x2"): alloc 4*4*4 AccVGPRs, each 32-bit gpr has bf16x2 type
    a[2,0,0]   : referencing single gpr
    a[2,0,0:3] : referencing gpr groups (continous, no more than instruction needed)
    a[2,0]     : same as above
    a[2]       : referencing
    '''
    def gpr(self, *desc, align=0, name=""):
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

        if len(desc) == 1 and isinstance(desc[0], GPRExpr):
            return self.auto_gpr(desc[0], name=name, loc=inspect.currentframe().f_back.f_lineno)

        # allocate GPRs
        dtype = desc[-1]
        assert isinstance(dtype, str), f"{dtype=}"

        num_gprs = 1
        shape = []
        for i in range(len(desc)-1):
            dim = desc[i]
            assert isinstance(dim, int)
            shape.append(dim)
            num_gprs *= dim

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
        return gprs


    def new_gpr(self, reg_type, count_range, dtype="u32", align=1, name=""):
        if name == "":
            # try to reover python var's name from code_context
            stack = inspect.stack()
            caller_frame = stack[1]
            if caller_frame.code_context:
                src_line = caller_frame.code_context[0].strip()
                name = src_line.split("=")[0].strip()

        dtype = dtype.replace("fp32","f32")
        assert dtype in ["u32","i32","f32","bf16x2","fp16x2","bf8x4","fp8x4"]
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
        inst_list = []
        serial_index = 0
        for bb in self.blocks:
            for inst in bb.instructions:
                inst.sid = serial_index
                inst_list.append(inst)
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
        possible_dead_variables = {}
        for gprs in live_intervals:
            first_sid, last_sid = live_intervals[gprs]
            self.asm_debug_info += f";'{gprs.name}'{repr(gprs):10s}  {first_sid} ~ {last_sid}\n"

            if first_sid == last_sid:
                if first_sid not in possible_dead_variables:
                    possible_dead_variables[first_sid] = []
                possible_dead_variables[first_sid].append(gprs)

        # mark-up dead-loads
        for bb in self.blocks:
            for inst in bb.instructions:
                if inst.sid in possible_dead_variables:
                    if inst.opcode.startswith("s_load_"):
                        dst_gpr = inst.operands[0]
                        is_dead_load = False
                        for gpr in possible_dead_variables[inst.sid]:
                            if dst_gpr.overlap(gpr,fully_match=True):
                                is_dead_load = True
                                break
                        inst.is_dead = is_dead_load

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
                    return
            # summary for diagnose GPR overflow issue
            summary = gpr_usage() + "\n"
            for g in alive_gprs:
                summary += f"\t{str(g.count):5s} {g.rtype}GPRs  {repr(g):15s} {g.name:20s}  {inst_list[g.sid].loc}\n"
            assert 0, f"cannot allocate '{gprs.name}'  {rtype}GPRs x {count} {align=}, not enough resource:\n {summary}"

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
                        self.asm_debug_info += f";free '{gprs.name}'{gprs} at #{sid}\n"
                    else:
                        unalloc_gprs.append(gprs)
            # allocation 
            for ev,gprs in events:
                if ev == 0: # first use(as dst)
                    alloc_gpr(gprs, sid)
                    self.asm_debug_info += f";alloc '{gprs.name}'{gprs} at #{sid}      {gpr_usage()}\n"

            # in case some gpr's last & first sids are the same
            for gprs in unalloc_gprs:
                free_gpr(gprs)

        # remove dead-loads
        for bb in self.blocks:
            # delete dead instructions
            for i in range(len(bb.instructions)-1, -1, -1):
                if bb.instructions[i].is_dead:
                    bb.instructions.pop(i)

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
                src1_gprs = self.new_gpr("s", 1, dtype="u32", align=1, name="")
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
                    assert ceil_sd < 2**32
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

                    temp = self.new_gpr(rtype, 1, dtype=dtype, align=1, name="")
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
                        if (isinstance(op, GPRExpr) or isinstance(op, GPRs)) and vdst.overlap(op):
                            n_nops = 1
                            self.log(f"insert s_nop({n_nops}) at #{loc} : [VALU Trans op, Non-trans VALU op consumes result of that op]")
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

    def debug_log(self, gprs:Union[GPRs, GPRExpr], torch_dtype, gpr_layout=""):
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
        # gprs
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
                        if next.opcode.startswith("v_") and (not next.operands[0].overlap(vaddr)):
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
        debug_enabled = self.debug_print()
        if debug_enabled:
            self.show_code()

        def is_gpr(op):
            return isinstance(op, GPRs) or isinstance(op, GPRExpr)

        # check GPRs, GPRs can be optimized by CSE pass
        #   - written only once
        #   - accessed by 1 BB only
        #   - being read after written
        #
        def vdst_operand_id(inst):
            opcode = inst.opcode
            mod = inst.mod
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

        gpr_info = {}
        for bb_id, bb in enumerate(self.blocks):
            for t, inst in enumerate(bb.instructions):
                vdst_index = vdst_operand_id(inst)
                for index, gpr in enumerate(inst.operands):
                    if is_gpr(gpr):
                        for i, r in enumerate(gpr):
                            key = repr(r)
                            if key not in gpr_info:
                                gpr_info[key] = {"wr":"","location":list()}
                            gpr_info[key]["location"].append((bb_id, t, index))
                            if index == vdst_index:
                                gpr_info[key]["wr"] += "w"
                            else:
                                gpr_info[key]["wr"] += "r"
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

        self.log("pass_cse: ", f"{ssa_gpr=}")
        # we limited CSE to following instructions because they are used by recursive expression generation.
        # Result of these instructions can be reused w/o worry about side-effect.
        # but any other instructions, although cannot be elimited, may update registers holding the reusable value,
        # so any other instructions with it's 1st operand (most likely vdst)
        # holding a value will destroy this reuable value (maybe over-react, but it's for correctness/safety reason)
        #
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

        replace_index = 0
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

            def replace_vdst_with_exist_global(vdst, vexist):
                for bb_id, inst_id, op_id in gpr_info[repr(vdst)]["location"]:
                    inst = self.blocks[bb_id].instructions[inst_id]
                    op = inst.operands[op_id]
                    if vdst.overlap(op):
                        if debug_enabled: print(f"found  {inst}")
                        assert vdst.overlap(op, fully_match=True)
                        inst.operands[op_id] = vexist
                        if debug_enabled: print(f"\t {i} {op} {vexist}")

            for t, inst in enumerate(bb.instructions):
                if len(inst.operands) == 0: continue
                # instruction may have multiple vdst/vsrc: v[4:7]
                dst = inst.operands[0]
                if isinstance(dst, GPRExpr) or isinstance(dst, GPRs):
                    # Why "rw" instead of just "w" ?
                    # we cannot be 100% sure that first operand is dst,
                    # it may be a src too, let's be conservative
                    for r in dst:
                        add_reg_access(r, t, "rw")

                for src in inst.operands[1:]:
                    if isinstance(src, GPRExpr) or isinstance(src, GPRs):
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
                vdst = inst.operands[0]
                if not is_gpr(vdst):
                    continue

                # invalidate previous value this vdst is holding
                for k in list(value_table.keys()):
                    v, version = value_table[k]
                    if v is vdst and version < cur_version:
                        del value_table[k]

                # all dst gprs must be valid (only written once)
                is_valid_gpr = all([repr(r) in ssa_gpr for r in vdst])
                if is_cse_inst(inst) and is_valid_gpr:
                    assert is_gpr(vdst)
                    cur_version = inc_value_version(vdst)

                    # this key includes opcode & all input-values & modifiers
                    src_ops = []
                    for op in inst.operands[1:]:
                        op_name = str(inst.op_repr(op))
                        if is_gpr(op):
                            op_name += "." + str(get_value_version(op))
                        src_ops.append(op_name)
                    # the dependent inputs are versioned, to avoid mismatches when some inputs
                    # may be updated with new value.
                    key = f"{inst.opcode} {','.join(src_ops)} {inst.mod}"
                    if key not in value_table:
                        # first time a reusable value is generated in versioned vdst
                        value_table[key] = [vdst, cur_version]
                    else:
                        vexist, version = value_table[key]
                        if repr(vexist) in ssa_gpr and replace_index < CSE_LIMIT:
                            removed_insts.append(t)
                            replace_vdst_with_exist_global(vdst, vexist)
                            if debug_enabled:
                                print(f"===========  {replace_index} / {CSE_LIMIT=} ")
                                print(t, inst.loc, inst.debug_info, inst)
                                print(f"{vdst} replaced with  {vexist} {key} globally")
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
        if debug_enabled:
            self.show_code()

    def show_code(self):
        self.log("===================== bb")
        for bb in self.blocks:
            self.log(bb.debug_str())
        if self.current_bb is not None:
            self.log("===================== current_bb")
            self.log(self.current_bb.debug_str())
        

    def build(self, kernel_name, signature, extra_compiler_options, cpp_src_fpath, dump_stat):
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
        self.pass_insert_nop()

        # for bb in self.blocks: print(repr(bb))
        self.pass_hide_dependency()

        # kernel args are loaded with s_waitcnt, many v_ instructions can be
        # moved into this wait cycles
        self.pass_hide_karg_loads()

        # Common Subexpression Elimination，CSE
        self.pass_cse()

        # for bb in self.blocks: print(repr(bb))
        self.register_allocation_linear_scan()

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
        used_gprs = []
        artifact = {"asm":[]}
        for a in asm.splitlines():
            if a[0] == ";": continue
            artifact["asm"].append(a)
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

    def div_up(self, x, y):
        return (x + y - 1) // y

    def round_up(self, x, y):
        return self.div_up(x, y) * y

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
            trans_low, trans_high = get_perm_pattern_16(self)
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
            trans_low1, trans_high1, trans_low2, trans_high2 = get_perm_pattern_8(self)
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


gen_hip_file_unique_id = 0

class Idx3D:
    def __init__(self):
        pass

_jit_kernel_unique_id = {}
class jit_kernel:
    def __init__(self, gen_func, extra_compiler_options, with_debug_log = False, dump_stat = False, force_recompile = False):
        assert callable(gen_func)
        self.extra_compiler_options = extra_compiler_options
        self.dump_stat = dump_stat
        # with_debug_log needs extra internal debug-log buffer, so it's always recompiled
        self.force_recompile = force_recompile
        self.with_debug_log = with_debug_log
        self.gen_func = gen_func
        self.func_name = gen_func.__name__
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
        os.makedirs(os.path.expanduser(f"~/.pyhip"), exist_ok=True)
        cpp_src_fpath = os.path.expanduser(f"~/.pyhip/{self.func_name}-{self.gen_construct_id}-{kernel_key}-{self.gen_func_unique_id}.cpp")

        J = JIT()

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
    def __init__(self, extra_compiler_options = "", with_debug_log=False, dump_stat = False, force_recompile = False):
        self.extra_compiler_options = extra_compiler_options
        self.with_debug_log = with_debug_log
        self.dump_stat = dump_stat
        self.force_recompile = force_recompile

    def __call__(self, gen_func):
        return jit_kernel(gen_func, self.extra_compiler_options, self.with_debug_log, self.dump_stat, self.force_recompile)

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