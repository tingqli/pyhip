import ctypes
from ctypes.util import find_library
import re
import subprocess
import functools
import os, sys
import inspect
from .asmtools import prettify
import torch
from typing import List, Optional, Tuple

@functools.cache
def get_lib():
    try:
        lib = ctypes.CDLL(find_library("amdhip64"))
    except Exception as e:
        print(e)
        import torch
        torch_amdhip64 = os.path.join(torch.__path__[0], "lib", "libamdhip64.so")
        print(f"Try {torch_amdhip64} instead...")
        lib = ctypes.CDLL(torch_amdhip64)
    lib.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.hipModuleLoad.restype = ctypes.c_int32
    lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.hipModuleGetFunction.restype = ctypes.c_int32
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p, 
                                    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                    ctypes.c_uint32, # unsigned int sharedMemBytes
                                    ctypes.c_void_p, # hipStream_t stream
                                    ctypes.c_void_p, # void **kernelParams
                                    ctypes.c_void_p, # void **extra
                                    ]
    lib.hipModuleLaunchKernel.restype = ctypes.c_int32
    lib.hipGetErrorString.argtypes = [ctypes.c_int32]
    lib.hipGetErrorString.restype = ctypes.c_char_p
    return lib

def hip_check_error(err):
    if err != 0:
        raise Exception("HIP error:" + get_lib().hipGetErrorString(err).decode("utf-8"))

@functools.cache
def hipModuleLoad(module_fpath):
    p_module = ctypes.c_void_p()
    hip_check_error(get_lib().hipModuleLoad(ctypes.byref(p_module), module_fpath.encode('utf-8')))
    # print(f"hipModuleLoad({module_fpath}) ... success.")
    return p_module

def hipModuleGetFunction(p_module, func_name):
    p_func = ctypes.c_void_p()
    hip_check_error(get_lib().hipModuleGetFunction(ctypes.byref(p_func), p_module, func_name.encode('utf-8')))
    # print(f"hipModuleGetFunction({func_name}) ... success.")
    return p_func

class amdhip_func:
    def __init__(self, module_fpath, sym_name, kname, kargs):
        self.module_fpath = module_fpath
        self.sym_name = sym_name
        self.kname = kname
        self.p_func = None
        fields = []
        for i,arg_type in enumerate(kargs):
            if arg_type.endswith("*"):
                fields.append((f"arg_{i}", ctypes.c_void_p))
            elif arg_type == "int":
                fields.append((f"arg_{i}", ctypes.c_int))
            elif arg_type == "unsigned long":
                fields.append((f"arg_{i}", ctypes.c_ulong))
            elif arg_type == "unsigned int":
                fields.append((f"arg_{i}", ctypes.c_uint))
            elif arg_type == "float" or arg_type == "float32_t": 
                fields.append((f"arg_{i}", ctypes.c_float))
            elif arg_type == "double":
                fields.append((f"arg_{i}", ctypes.c_double))
            else:
                raise Exception(f"Unsupported arg type: {arg_type}")
        class Args(ctypes.Structure):
            _fields_ = fields
        self.args = Args()
        ExtraType = ctypes.c_void_p * 5
        self.arg_size = ctypes.c_uint64(ctypes.sizeof(self.args))
        self.config = ExtraType(1, ctypes.addressof(self.args), 2, ctypes.addressof(self.arg_size), 3)
        self.fun_loaded = False

    def lazy_load_func(self):
        if self.p_func is None:
            self.fun_loaded = True
            p_module = hipModuleLoad(self.module_fpath)
            self.p_func = hipModuleGetFunction(p_module, self.sym_name)

    def __call__(self, gridDims:list[int], blockDims:list[int], *args, sharedMemBytes = 0, force_occupancy = 0):
        self.lazy_load_func()
        if force_occupancy > 0:
            LDS_bytes = 64*1024
            sharedMemBytes = LDS_bytes//force_occupancy
        for i,a in enumerate(args):
            setattr(self.args, f"arg_{i}", a)
        while len(gridDims) < 3:
            gridDims.append(1)
        while len(blockDims) < 3:
            blockDims.append(1)
        stream = ctypes.cast(torch.cuda.current_stream(), ctypes.c_void_p)
        hip_check_error(get_lib().hipModuleLaunchKernel(self.p_func, *gridDims, *blockDims, sharedMemBytes, stream, 0, ctypes.byref(self.config)))

@functools.cache
def amdgpu_arch():
    # 优先用环境变量指定 ISA；否则调用 amdgpu-arch 列出本机 GPU
    # 方便交叉编译
    override = os.environ.get("PYHIP_AMDGPU_ARCH", "").strip()
    if override:
        return override
    gfx_archs = subprocess.check_output(["/opt/rocm/llvm/bin/amdgpu-arch"]).decode("utf-8")
    # index = torch.cuda.current_device()
    index = 0
    return gfx_archs.splitlines()[index].strip()

PYHIP_CACHE_DIR = os.getenv("PYHIP_CACHE_DIR", os.path.expanduser("~/.pyhip"))
os.makedirs(PYHIP_CACHE_DIR, exist_ok=True)

# 匹配 #include "relative/path.hpp"（不含系统头 <...>）
_RE_INCLUDE_QUOTED = re.compile(r'^\s*#\s*include\s+"([^"]+)"')

def _device_src_mtime_with_local_includes(src_path: str) -> float:
    """
    主源文件 **最后修改时间**（POSIX 里常记作 mtime，即 ``os.path.getmtime``）与其中
    ``#include "..."`` 所引用且存在于磁盘上的文件的 mtime 的**最大值**。
    路径相对于 **主源文件所在目录** 解析（与编译器 ``-I`` 同目录时行为一致）。
    不递归展开被包含文件里的 include（避免循环与成本；需要时可改环境变量补依赖）。
    """
    src_path = os.path.abspath(src_path)
    mt = os.path.getmtime(src_path)
    src_folder = os.path.dirname(src_path)
    try:
        with open(src_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = _RE_INCLUDE_QUOTED.match(line)
                if not m:
                    continue
                rel = m.group(1).strip()
                if not rel or rel.startswith("<"):
                    continue
                inc_path = os.path.normpath(os.path.join(src_folder, rel))
                if os.path.isfile(inc_path):
                    mt = max(mt, os.path.getmtime(inc_path))
    except OSError:
        pass
    # 可选：``PYHIP_DEVICE_COMPILE_EXTRA_DEPS=/a/x.hip,/b/y.hip`` 额外纳入 mtime（绝对路径或相对 cwd）
    extra = os.environ.get("PYHIP_DEVICE_COMPILE_EXTRA_DEPS", "").strip()
    if extra:
        for part in extra.split(os.pathsep if os.pathsep in extra else ","):
            p = part.strip()
            if not p:
                continue
            p = os.path.abspath(os.path.expanduser(p))
            if os.path.isfile(p):
                mt = max(mt, os.path.getmtime(p))
    return mt


def compile_hip_device_only(src_path, extra_compiler_options, macros=None):
    src_mtime = _device_src_mtime_with_local_includes(src_path)

    src_folder, src_filename = os.path.split(src_path)
    pre, ext = os.path.splitext(src_filename)
    # 使与源文件同目录的头文件（如显式实例化）可被 #include ""
    inc_dir = f"-I{src_folder}" if src_folder else "-I."

    # put all intermedite file under cache
    pre = f"{PYHIP_CACHE_DIR}/{pre}"

    if macros is not None:
        assert isinstance(macros, tuple)
        for k,v in macros:
            extra_compiler_options += f" -D{k}={v}"
            pre += f"-{k}={v}"

    gfx_arch = amdgpu_arch()
    # 不同 offload-arch 必须分文件缓存，否则 .co 会错用其它 ISA 的汇编
    pre = f"{pre}-{gfx_arch}"

    assert ext in (".cpp", ".s", ".hip")
    ll_path = pre + ".ll"
    asm_path = pre + ".s"
    co_path = pre + ".co"

    if os.getenv("DUMP_LL", None) is not None:
        cmd1 = f"hipcc -x hip --offload-device-only --offload-arch={gfx_arch} -std=c++20 {inc_dir} -O2 {src_path} {extra_compiler_options} -S -emit-llvm -o {ll_path}"
        print(cmd1)
        if os.system(cmd1) != 0:
            raise Exception("compilation 0 failed")
        print(f"\033[0;32m LLVM IR {ll_path} was generated. \033[0m")

    if ext in (".cpp", ".hip") and (not os.path.isfile(asm_path) or src_mtime > os.path.getmtime(asm_path)):
        # (re)compile into asm
        cmd1 = f"hipcc -x hip --offload-device-only --offload-arch={gfx_arch} -std=c++20 {inc_dir} -O2 {extra_compiler_options} -Rpass-analysis=kernel-resource-usage {src_path} -S -o {asm_path}"
        print(cmd1)
        if os.system(cmd1) != 0:
            raise Exception("compilation 1 failed")
        prettify(asm_path, asm_path)

    if not os.path.isfile(co_path) or src_mtime > os.path.getmtime(co_path):
        # (re)compile into co
        cmd2 = f"/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu={gfx_arch} {asm_path}  -o {co_path}"
        print(cmd2)
        if os.system(cmd2) != 0:
            raise Exception("compilation 2 failed")
    return co_path


def _parse_kernel_demangle_func_sig(func_sig: str) -> Optional[Tuple[str, List[str]]]:
    """
    从 llvm-objdump --demangle 拼出的 ``func_sig``（``join(ls[6:])``）解析出
    ``(fname, [arg types...])``；无法解析（无括号）时返回 None。
    供 ``get_all_kernel_args`` 与单测共用。
    """
    func_sig = func_sig.strip().rstrip(")")
    if "(" not in func_sig:
        return None
    fname, args_str = func_sig.split("(", 1)
    fname = fname.strip()
    if fname.startswith("void "):
        fname = fname[5:].strip()
    args_str = args_str.rstrip(")").strip()
    if not args_str:
        args = []
    else:
        args = [a.strip() for a in args_str.split(",")]
    return fname, args


@functools.cache
def get_all_kernel_args(co_path):
    """
    从 .co 的 ELF 动态符号表解析每个 kernel 的「Python 侧函数名」与「参数类型」。
    hipModuleGetFunction 必须用 **未 demangle 的符号名**（raw），而参数类型串只在
    ``--demangle`` 输出里可读，因此跑两次 llvm-objdump。
    """
    dynamic_syms = subprocess.check_output(
        ["/opt/rocm/llvm/bin/llvm-objdump", "--dynamic-syms", "--demangle", co_path]
    ).decode("utf-8")
    dynamic_syms_raw = subprocess.check_output(
        ["/opt/rocm/llvm/bin/llvm-objdump", "--dynamic-syms", co_path]
    ).decode("utf-8")
    kernel_args = {}
    for line, line_raw in zip(dynamic_syms.splitlines(), dynamic_syms_raw.splitlines()):
        ls = line.split()
        if len(ls) < 7:
            continue
        if ls[3] != ".text":
            continue
        # 与 hipModuleLoad 后 GetFunction 传入的字符串一致，必须用 raw 列里的名字（常为 _Z 修饰名）
        symbol_name = line_raw.split()[6]
        func_sig = " ".join(ls[6:]).strip().rstrip(")")
        parsed = _parse_kernel_demangle_func_sig(func_sig)
        if parsed is None:
            continue
        fname, args = parsed
        kernel_args[fname] = (symbol_name, args)
    return kernel_args

'''
Compile .cpp into .s and .co, then exposes test_kernel as a python function

    @pyhip.module("mla-hip.cpp")
    def test_kernel(q, k, v, n, sm): ...

Compile .s into .co and expose test_kernel as a python function

    @pyhip.module("mla-hip.s")
    def test_kernel(q, k, v, n, sm): ...

'''

@functools.cache
def _build_hip_src(src_fpath, extra_compiler_options, macros):
    if src_fpath.endswith((".cpp", ".hip", ".s")):
        module_fpath = compile_hip_device_only(src_fpath, extra_compiler_options, macros)
    else:
        assert src_fpath.endswith(".co")
        module_fpath = src_fpath

    kargs = get_all_kernel_args(module_fpath)
    return module_fpath, kargs

@functools.cache
def _build_hip_func(src_fpath, extra_compiler_options, macros, func_name):
    module_fpath, kargs = _build_hip_src(src_fpath, extra_compiler_options, macros)
    sym_name, arg_types = kargs[func_name]
    wrapper = amdhip_func(module_fpath, sym_name, func_name, arg_types)
    wrapper.__name__ = func_name
    return wrapper

class callable_kernel:
    def __init__(self, src_fpath, extra_compiler_options, func_name):
        self.src_fpath = src_fpath
        self.extra_compiler_options = extra_compiler_options
        self.func_name = func_name

    def build(self, kwargs = None):
        macros = []
        if kwargs is not None:
            for key in list(kwargs.keys()):
                if key[0].isupper():
                    macros.append((key, kwargs.pop(key)))
        return _build_hip_func(self.src_fpath, self.extra_compiler_options, tuple(macros), self.func_name)

    def __call__(self, *args, **kwargs):
        func = self.build(kwargs)
        func(*args, **kwargs)

class module:
    def __init__(self, src_file_path, extra_compiler_options = ""):
        if os.path.isabs(src_file_path):
            src_fpath = src_file_path
        else:
            base_dir = os.path.dirname(inspect.getframeinfo(sys._getframe(1)).filename)
            src_fpath = os.path.join(base_dir, src_file_path)

        self.src_fpath = src_fpath
        self.extra_compiler_options = extra_compiler_options

    def __call__(self, func):
        '''
        decorator on func
        '''
        assert callable(func)
        func_name = func.__name__
        return callable_kernel(self.src_fpath, self.extra_compiler_options, func_name)

    def __getattr__(self, func_name):
        '''
        w/o decorator on func, directly build wrapper from name
        '''
        return callable_kernel(self.src_fpath, self.extra_compiler_options, func_name)
