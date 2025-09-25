import subprocess
import argparse
import re
from functools import lru_cache

@lru_cache(maxsize=None)
def get_file_lines(fname):
    with open(f"{fname}", 'r') as file:
        lines = file.readlines()
    return lines


inst_msg = {}

'''
usage 1 : w/o parameter, passing func directly into gfx_inst decorator
    @gfx_inst
    def func()

usage 2 : with keyword parameter but no func, these parameter was registered along with func
    @gfx_inst(opt="glc")
    def func()
'''
def gfx_inst(func=None, **kwargs):
    global inst_msg

    if callable(func):
        prefix = func.__name__.lower()
        inst_msg[prefix] = (func, None)
        return func
    # func is none, return a decorator
    def decorator(func):
        prefix = func.__name__.lower()
        inst_msg[prefix] = (func, kwargs)
        return func

    return decorator


def add_gfx_msg(asm_line, line_no):
    instops = asm_line.replace(","," ").split()
    if len(instops) == 0: return asm_line
    
    # longer prefix gets priority
    prefix_max_len = 0
    prefix_k = ""
    for k in inst_msg:
        if instops[0].startswith(k) and len(k) > prefix_max_len:
            prefix_max_len = len(k)
            prefix_k = k
    if prefix_max_len == 0: return asm_line

    opcodes = instops[0][len(prefix_k):].split('_')
    # extract keywords args
    kwargs = {}
    list_args = []
    for a in instops[1:]:
        if a == ";": break  # comment
        if a.startswith("offset:"):
            kwargs["offset"] = int(a[7:], 0)
        elif a.startswith("format:"):
            kwargs["format"] = int(a[7:], 0)
        elif a in ["glc","slc","sc0","sc1","nt","idxen", "offen","lds"]:
            kwargs[a] = 1
        else:
            list_args.append(a)

    func, dec_args = inst_msg[prefix_k]

    if dec_args and "opt" in dec_args:
        # this means the first operand of func is optional
        opt_kw = dec_args["opt"]
        if opt_kw[0] == "!":
            # existing only when opt_kw is omitted
            opt_not_exist = opt_kw[1:] in kwargs
        else:
            # existing only when opt_kw is presented
            opt_not_exist = opt_kw not in kwargs

        if opt_not_exist:
            # when opt is not presented, we append it manually 
            # to align instruction syntax with python function call syntax
            list_args.insert(0, None)
    try:
        msg = func(opcodes, *list_args, **kwargs)
    except:
        print("line      :\t",line_no)
        print("source    :\t",asm_line.strip())
        print("prefix_k  :\t",prefix_k)
        print("opcodes   :\t",opcodes)
        print("list_args :\t",list_args)
        print("kwargs    :\t",kwargs)
        raise
    asm_line = asm_line.rstrip()
    cols = len(asm_line)
    align_to_col = 50
    if cols < align_to_col:
        asm_line += " " * (align_to_col-cols)
    asm_line += ";\t" + msg + "\n"
    return asm_line

# vdst, vaddr, srsrc, soffset          fmt idxen offen offset12 sc0 nt sc1

##@gfx_inst
#def buffer_load_(opcodes, vdst, src0, src1, src2):

@gfx_inst
def ds_read_addtid_b32(opcodes, vdst, offset=0, gds=0):
    mtype = "GDS" if gds else "LDS"
    return f"{vdst}.u32 = {mtype}_MEM[32'I({offset} + M0[15:0]) + laneID.i32 * 4].u32"

@gfx_inst
def ds_write_addtid_b32(opcodes, vdata, offset=0, gds=0):
    mtype = "GDS" if gds else "LDS"
    return f"{mtype}_MEM[32'I({offset} + M0[15:0]) + laneID.i32 * 4].u32 = {vdata}.u32"



@gfx_inst
def ds_read_(opcodes, vdst, vaddr, offset=0, gds=0):
    mtype = "GDS" if gds else "LDS"
    m_dtype = opcodes[0]
    mem_data = f"{mtype}_MEM[{vaddr} + {offset}].{m_dtype}"
    if len(opcodes) == 1 and m_dtype[0] == "b":
        assert m_dtype == "b32" or m_dtype == "b64" or  m_dtype == "b96" or   m_dtype == "b128"
        return f"{vdst} = {mem_data}; // read w/o any type convertion"
    elif len(opcodes) == 1:
        assert m_dtype == "u8" or m_dtype == "i8" or m_dtype == "u16" or m_dtype == "i16"
        return f"{vdst} = extend_to_32bits({mem_data});"
    elif len(opcodes) == 2:
        t_type = opcodes[1]
        assert t_type == "d16"
        return f"{vdst}[15:0] = extend_to_16bits({mem_data});"
    else:
        assert len(opcodes) == 3
        assert opcodes[2] == "hi"
        return f"{vdst}[31:16] = extend_to_16bits({mem_data});"

@gfx_inst
def ds_write_(opcodes, vaddr, vdata, offset=0, gds=0):
    m_dtype = opcodes[0]
    assert m_dtype[0] == 'b'
    mtype = "GDS" if gds else "LDS"
    if len(opcodes) == 1:
        return f"{mtype}_MEM[{vaddr} + {offset}].{m_dtype} = {vdata}.{m_dtype}"
    else:
        assert len(opcodes) == 3
        assert opcodes[1] == "d16" and opcodes[2] == "hi"
        return f"{mtype}_MEM[{vaddr} + {offset}].{m_dtype} = {vdata}.hi_16bits"

@gfx_inst
def v_lshl_add_(opcodes, vdst, src0, src1, src2):
    dtype = opcodes[-1]
    return f"{vdst}.{dtype} = ({src0}.{dtype} << {src1}.u32[2 : 0].u32) + {src2}.{dtype}"

@gfx_inst
def v_add_co_(opcodes, vdst, sdst, src0, src1, clamp=0):
    dtype = opcodes[0]
    return f"tmp=64'U({src0})+64'U({src1}); {sdst}[laneId]=overflow_carry; {vdst}=tmp.{dtype}"

@gfx_inst
def v_addc_co_(opcodes,  vdst, vcc0, src0, vsrc1, vcc1, *rest):
    dtype = opcodes[0]
    return f"tmp = 64'U({src0}) + 64'U({vsrc1}) + {vcc0}.u64[laneId]; {vcc1}.u64[laneId]=overflow_carry;  {vdst}=tmp"

@gfx_inst
def v_add_(opcodes, vdst, src0, src1, clamp=0, omod=0):
    dtype = opcodes[0]
    return f"{vdst} = {src0} + {src1}"

@gfx_inst
def s_and_(opcodes, vdst, src0, src1):
    return f"{vdst} = {src0} & {src1}"

@gfx_inst
def s_mul_(opcodes, vdst, src0, src1):
    return f"{vdst} = {src0} * {src1}"

@gfx_inst
def global_load_(opcodes, vdst, vaddr, saddr, offset=0, glc=0, slc=0, sc0=0, sc1=0):
    dtype = opcodes[-1]
    msg = f"{vdst} = load_{dtype}_from_addr({vaddr}"
    if saddr != "off": msg += f" + {saddr}"
    if offset > 0: msg += f" + {offset}"
    if glc: msg += f", glc"
    if slc: msg += f", slc"
    msg += ")"
    return msg

@gfx_inst
def global_store_(opcodes, vaddr, vdata, saddr, offset=0, nt=0, sc0=0, sc1=0):
    dtype = opcodes[-1]
    msg = f"save_{dtype}_to_addr({vdata}, addr={vaddr}"
    if saddr != "off": msg += f" + {saddr}"
    if offset > 0: msg += f" + {offset}"
    msg += ")"
    return msg

@gfx_inst
def s_load_(opcodes, sdst, sbase, soffset=0, offset=0, glc=0):
    addr = f"{sbase}"
    if soffset != 0: addr += f" + {soffset}"
    if offset != 0: addr += f" + {offset}"
    return f"{sdst} = load_{opcodes[0]}_from({addr}, glc={glc});  // 8.2.1.1. Scalar Memory Addressing"


@gfx_inst
def v_cmp_(opcodes, vdst, src0, src1, clamp=None):
    op_to_sign = {"gt":">", "lt":"<", "ge":">=", "le":"<=", "eq":"==", "ne":"<>", "u":"not-orderable"}
    cmp_op = opcodes[0]
    dtype = opcodes[1]
    msg = f"{vdst}.u64[laneId] = "
    if cmp_op[0] == 'n' and len(cmp_op) == 3:
        cmp_op = cmp_op[1:]
        msg += " ! "
    msg += f"({src0}.{dtype} {op_to_sign[cmp_op]} {src1}.{dtype})"
    return msg

@gfx_inst
def s_cmp_(opcodes, ssrc0, ssrc1):
    op_to_sign = {"gt":">", "lt":"<", "ge":">=", "le":"<=", "eq":"==", "ne":"<>", "lg":"<>"}
    cmp_op = opcodes[0]
    dtype = opcodes[1]
    msg = f"scc = ({ssrc0}.{dtype} {op_to_sign[cmp_op]} {ssrc1}.{dtype})"
    return msg

@gfx_inst
def v_cndmask_(opcodes, vdst, src0, vsrc1, vcc, *rest):
    dtype = opcodes[0]
    return f"{vdst}.{dtype} = {vcc}.u64[laneId] ? {vsrc1}.u32 : {src0}.u32"


@gfx_inst
def s_and_saveexec_b64(op, sdst, ssrc):
    return f"exec={ssrc}&exec; {sdst}=old_exec; scc=(exec!=0)"

@gfx_inst
def s_or_saveexec_b64(op, sdst, ssrc):
    return f"exec={ssrc}|exec; {sdst}=old_exec; scc=(exec!=0)"

@gfx_inst
def s_cbranch_(opcodes, label):
    cond = opcodes[-1]
    return f"jump to {label} if {cond}"

@gfx_inst
def s_mov_(op, dst, src):
    return f"{dst} = {src}"

@gfx_inst
def s_ff1_(op, dst, src):
    return f"{dst} = number of trailing 0 bits before the first 1 in {src}"

@gfx_inst
def s_ff0_(op, dst, src):
    return f"{dst} = number of trailing 1 bits before the first 0 in {src}"

@gfx_inst
def S_BCNT1_(op, dst, src):
    return f"{dst} = number of 1 bits in {src}"

@gfx_inst
def S_BCNT0_(op, dst, src):
    return f"{dst} = number of 0 bits in {src}"

@gfx_inst
def v_readlane_(op, sdst, src0, ssrc1):
    return f"{sdst} = {src0}.lane[{ssrc1}[5:0]]"

@gfx_inst
def s_lshl_(opcodes, sdst, src0, ssrc1):
    bits = 5 if opcodes[0] == "b64" else 4
    return f"{sdst} = {src0} << {ssrc1}[{bits}:0]; scc=({sdst}!=0);"

@gfx_inst
def s_add_(opcodes, sdst, src0, ssrc1):
    dtype = opcodes[0]
    return f"{sdst}.{dtype} = {src0} + {ssrc1}; scc=overflow_or_carry"

@gfx_inst
def s_addc_(opcodes, sdst, src0, ssrc1):
    dtype = opcodes[0]
    return f"{sdst}.{dtype} = {src0} + {ssrc1} + scc; scc=overflow_or_carry"

@gfx_inst
def s_cmpk_(opcodes, ssrc, imm16):
    op = opcodes[0]
    dtype = opcodes[1]
    op_to_sign = {"gt":">", "lt":"<", "ge":">=", "le":"<=", "eq":"==", "ne":"<>", "lg":"<>"}
    return f"scc = ({ssrc}.{dtype} {op_to_sign[op]} extend_as_{dtype}({imm16}))"

@gfx_inst
def s_cselect_(opcodes,  sdst, ssrc0, ssrc1):
    dtype = opcodes[0]
    return f"{sdst} = scc ? {ssrc0} : {ssrc1}"

@gfx_inst
def s_andn2_(opcodes, sdst, src0, ssrc1):
    return f"{sdst} = ({src0} & ~{ssrc1}); scc=({sdst}!=0);"

@gfx_inst
def s_andn2_saveexec_(opcodes, sdst, ssrc):
    return f"exec=({ssrc} & ~exec);  {sdst}=old_exec;  scc=(exec != 0)"

@gfx_inst
def v_mov_(opcodes,  vdst, src, *rest):
    return f"{vdst} = {src};"

@gfx_inst
def v_accvgpr_read_(opcodes,  vdst, src):
    return f"{vdst} = {src};"

@gfx_inst
def v_accvgpr_write_(opcodes,  vdst, src):
    return f"{vdst} = {src};"

@gfx_inst(opt="sc0") # first arg is opt and exists when sc0 is set
def global_atomic_(opcodes, vdst, vaddr, vdata, saddr, offset=0, glc=0, slc=0, sc0=0, sc1=0):
    op = opcodes[0]
    if vdst is None:
        vdst = "_"
    addr = f"{vaddr}"
    if saddr != "off":
        addr += f" + {saddr}"
    return f"{vdst} = atomic_{op}(addr={addr}, data={vdata})"

@gfx_inst
def s_or_(opcodes,  dst, src0, src1):
    return f"{dst} = {src0} | {src1};  scc=({dst}!=0);"

@gfx_inst
def s_xor_(opcodes,  dst, src0, src1):
    return f"{dst} = {src0} ^ {src1};  scc=({dst}!=0);"

@gfx_inst
def s_xor_saveexec_(opcodes,  dst, src0):
    return f"exec={src0}^exec;  {dst}=old_exec;  scc=(exec!=0);"

@gfx_inst
def v_lshlrev_(opcodes, vdst, vsrc0, vsrc1, *rest):
    dtype = opcodes[0]
    bitrng = ""
    if not vsrc0.isdigit(): 
        if dtype == "b64" : bitrng = "[5:0]"
        if dtype == "b32" : bitrng = "[4:0]"
        if dtype == "b16" : bitrng = "[3:0]"
    return f"{vdst}.{dtype} = {vsrc1} << {vsrc0}{bitrng};"

@gfx_inst
def v_lshrrev_(opcodes, vdst, vsrc0, vsrc1, *rest):
    dtype = opcodes[0]
    bitrng = ""
    if not vsrc0.isdigit(): 
        if dtype == "b64" : bitrng = "[5:0]"
        if dtype == "b32" : bitrng = "[4:0]"
        if dtype == "b16" : bitrng = "[3:0]"
    return f"{vdst}.{dtype} = {vsrc1} >> {vsrc0}{bitrng};"

@gfx_inst
def ds_bpermute_b32(opcodes, vdst, vaddr, vdata, offset=0):
    str_off = ""
    if offset > 0: str_off = f" + {offset}"
    return f"{vdst} = {vdata}.lane[ ({vaddr}{str_off})/4 % 64 ];   select source lane with {vaddr}"

@gfx_inst
def ds_permute_b32(opcodes, vdst, vaddr, vdata, offset=0):
    str_off = ""
    if offset > 0: str_off = f" + {offset}"
    return f"{vdst}.lane[ ({vaddr}{str_off})/4 % 64 ] = {vdata};   select dst lane with {vaddr}, "

@gfx_inst
def v_and_or_b32(opcodes, vdst, src0, src1, src2):
    return f"{vdst}.u32 = ({src0}.u32 & {src1}.u32) | {src2}.u32;"

@gfx_inst
def v_lshl_or_b32(opcodes, vdst, src0, src1, src2):
    return f"{vdst}.u32 = ({src0}.u32 << {src1}[4:0].u32) | {src2}.u32;"

@gfx_inst
def v_and_b32(opcodes, vdst, src0, vsrc1):
    return f"{vdst}.u32 = ({src0} & {vsrc1}.u32)"

@gfx_inst
def v_xor_b32(opcodes, vdst, src0, vsrc1):
    return f"{vdst}.u32 = ({src0} ^ {vsrc1}.u32)"

@gfx_inst
def s_memtime(opcodes, sdst):
    return f"{sdst} = value of 64-bit clock counter"

@gfx_inst
def s_memrealtime(opcodes, sdst):
    return f"{sdst} = value of 64-bit 100MHz real-time clock counter"

@gfx_inst
def s_sub_(opcodes, vdst, src0, vsrc1):
    dtype = opcodes[0]
    return f"{vdst}.{dtype} = ({src0}.{dtype} - {vsrc1}.{dtype}); scc=(overflow or carry-out of last arith);"

@gfx_inst
def s_subb_(opcodes, vdst, src0, vsrc1):
    dtype = opcodes[0]
    return f"{vdst}.{dtype} = ({src0}.{dtype} - {vsrc1}.{dtype} - scc.{dtype}); scc=(overflow or carry-out of last arith);"

@gfx_inst
def v_mbcnt_(opcodes, vdst, src0, src1):
    if opcodes[0] == 'lo':
        lanes = f"laneId:0"
    else:
        lanes = f"(laneId-32):0"
    return f"{vdst} = {src1} + number_of_1bits_for({src0}[{lanes}])"

def prettify(asm_file_in, asm_file_out, do_demangle = False):

    with open(asm_file_in, 'r') as file:
        lines = file.readlines()

    demangled = {}
    if do_demangle:
        for line in lines:
            if match := re.search(r'\.globl\s*(.*)', line):
                key = match.group(1)
                result = subprocess.run(["c++filt", key], capture_output=True,text=True,check=True)
                demangled[key] = result.stdout.strip()

    with open(asm_file_out, 'w') as file:
        new_lines = []
        for lineno, line in enumerate(lines):
            lineno += 1
            if not line.strip().startswith("."):
                line = line.split("//")[0]
                if not line.endswith("\n"): line += "\n"
                if match := re.search(r"\.loc\s.*\;\s*(.*)\:(\d*)\:(\d*)", line):
                    # embedding source code
                    line = line.replace(".loc", "; .loc")
                    src_line = get_file_lines(match.group(1))[int(match.group(2))]
                    line = line.rstrip() + src_line
                else:
                    # demangle
                    for key in demangled:
                        if key in line:
                            line = line.rstrip() + f" ; {demangled[key]}\n"
                            break
                    # instruction comment
                    line = add_gfx_msg(line, lineno)
            new_lines.append(line)
        file.write("".join(new_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mb', action="store_true")
    parser.add_argument('-cb', action="store_true")
    parser.add_argument('-c',"--config", type=int, nargs=4, default=[1,4096,40960, 0])
    parser.add_argument('-d',"--dtype", type=str, default="fp16")
    parser.add_argument('--prof', action="store_true")
    parser.add_argument('-g', '--debug', action="store_true")
    parser.add_argument('fname')
    args = parser.parse_args()

    if args.fname.endswith(".s"):
        asm_file_name = args.fname
        cmd_line = None
    else:
        asm_file_name = f"{args.fname}.s"
        extra_options = ""
        if args.debug:
            extra_options += "-g "
        cmd_line = f"hipcc -x hip --offload-arch=native --offload-device-only -O2 {extra_options} -S {args.fname} -o {asm_file_name}"
        output = subprocess.check_output(cmd_line.split(' '), text=True)

    prettify(asm_file_name, f"{asm_file_name}.s")

    print("==========================")
    if cmd_line: print(f"cmd_line       : {cmd_line}")
    print(f"raw asm       : {asm_file_name}")
    print(f"commented asm : {asm_file_name}.s")
