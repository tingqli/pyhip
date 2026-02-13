import ctypes
from ctypes.util import find_library
import subprocess
import functools
import os, sys
import inspect
from .asmtools import prettify
import torch

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
    gfx_archs = subprocess.check_output(["/opt/rocm/llvm/bin/amdgpu-arch"]).decode('utf-8')
    # index = torch.cuda.current_device()
    index = 0
    return gfx_archs.splitlines()[index].strip()

def compile_hip_device_only(src_path, extra_compiler_options):
    src_mtime = os.path.getmtime(src_path)
    pre, ext = os.path.splitext(src_path)
    assert ext == ".cpp" or ext == ".s"
    ll_path = pre + ".ll"
    asm_path = pre + ".s"
    co_path = pre + ".co"
    gfx_arch = amdgpu_arch()

    if os.getenv("DUMP_LL", None) is not None:
        cmd1 = f"hipcc -x hip --offload-device-only --offload-arch={gfx_arch} -std=c++20 -I. -O2 {src_path} {extra_compiler_options} -S -emit-llvm -o {ll_path}"
        print(cmd1)
        if os.system(cmd1) != 0:
            raise Exception("compilation 0 failed")
        print(f"\033[0;32m LLVM IR {ll_path} was generated. \033[0m")

    if ext == ".cpp" and (not os.path.isfile(asm_path) or src_mtime > os.path.getmtime(asm_path)):
        # (re)compile into asm
        cmd1 = f"hipcc -x hip --offload-device-only --offload-arch={gfx_arch} -std=c++20 -I. -O2 {extra_compiler_options} -Rpass-analysis=kernel-resource-usage {src_path} -S -o {asm_path}"
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

@functools.cache
def get_all_kernel_args(co_path):
    # we need both demangle & symbol name for loading & argtype parsing
    dynamic_syms = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", "--dynamic-syms", "--demangle", co_path]).decode('utf-8')
    dynamic_syms_raw = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", "--dynamic-syms", co_path]).decode('utf-8')
    kernel_args = {}
    for line, line_raw in zip(dynamic_syms.splitlines(), dynamic_syms_raw.splitlines()):
        ls = line.split()
        if len(ls) < 7: continue
        if ls[3] != ".text" : continue
        symbol_name = line_raw.split()[6]
        func_sig = " ".join(ls[6:]).strip().rstrip(")")
        fname, args = func_sig.split("(")
        if len(args) == 0:
            args = []
        else:
            args = [a.strip() for a in args.split(",")]
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

class module:
    def __init__(self, src_file_path, extra_compiler_options = ""):
        if os.path.isabs(src_file_path):
            src_fpath = src_file_path
        else:
            base_dir = os.path.dirname(inspect.getframeinfo(sys._getframe(1)).filename)
            src_fpath = os.path.join(base_dir, src_file_path)

        if src_fpath.endswith(".cpp") or src_fpath.endswith(".s"):
            module_fpath = compile_hip_device_only(src_fpath, extra_compiler_options)
        else:
            module_fpath = src_fpath

        self.src_fpath = src_fpath
        self.module_fpath = module_fpath
        self.kargs = get_all_kernel_args(module_fpath)
        self.funcs = {}
    
    def __call__(self, func):
        '''
        decorator on func
        '''
        assert callable(func)
        func_name = func.__name__

        if func_name not in self.kargs:
            raise Exception(f"Cannot find {func_name} in {self.module_fpath}, only found {list(self.kargs.keys())}")

        sym_name, arg_types = self.kargs[func_name]

        print(f"\033[0;32m Kernel {func_name}({arg_types})  {self.src_fpath} : {self.module_fpath} : {sym_name} \033[0m")

        wrapper = amdhip_func(self.module_fpath, sym_name, func_name, arg_types)
        functools.update_wrapper(wrapper, func) 
        return wrapper

    def __getattr__(self, func_name):
        '''
        w/o decorator on func, directly build wrapper from name
        '''
        if func_name in self.funcs:
            return self.funcs[func_name]

        if func_name not in self.kargs:
            raise Exception(f"Cannot find {func_name} in {self.module_fpath}, only found {list(self.kargs.keys())}")

        sym_name, arg_types = self.kargs[func_name]

        print(f"\033[0;32m Kernel {func_name}({arg_types})  {self.src_fpath} : {self.module_fpath} : {sym_name} \033[0m")

        wrapper = amdhip_func(self.module_fpath, sym_name, func_name, arg_types)
        wrapper.__name__ = func_name

        self.funcs[func_name] = wrapper
        return wrapper
