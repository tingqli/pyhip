"""
CLI tools to compile MLIR to binary (co) for AMD GPU.

 #1. MLIR lowering/optimization
  mlir-opt input.mlir \
  --convert-math-to-rocdl \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-vector-to-llvm \
  --expand-strided-metadata \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o opt.mlir

 #2. MLIR to LLVM IR
  mlir-translate -mlir-to-llvmir opt.mlir -o opt.ll

 #3. Link external libraries (e.g., when using gpu.printf, link ROCDL runtime library)

  /opt/rocm/llvm/bin/llvm-link opt.ll /opt/rocm/amdgcn/bitcode/ockl.bc /opt/rocm/amdgcn/bitcode/oclc_isa_version_942.bc  /opt/rocm/amdgcn/bitcode/oclc_wavefrontsize64_on.bc /opt/rocm/amdgcn/bitcode/oclc_abi_version_600.bc  -S -o combined.ll

 #4. LLVM IR to ISA assembly
  llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=asm combined.ll -o final.s

 #5. Assemble to binary
  /opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 final.s -o final.hsaco

从jit角度来看，直接从MLIR编译出co不太顺畅，步骤2,3,4,5没有单独使用python binding暴露出来
但是有个非常强大的内置 pass  "gpu-module-to-binary{format=}" 可以完成步骤2,3,4,5的所有功能：
 - format=offloading/llvm：输出 LLVM 级表示
 - format=assembly：输出 ISA 汇编(内部会调用 translateModuleToISA)
 - format=binary：输出二进制(内部会调用ROCDL::assembleIsa/ROCDL::linkObjectCode)
 - format=fatbin：输出binary(最终链接调用ld.lld)

这个 pass 不会负责把高层 GPU dialect 自动降到 LLVM dialect；它接收的应该已经是“足够低”的 GPU module。
从官方 pipeline 也能看出来，gpu-module-to-binary 是放在 convert-gpu-to-rocdl / gpu-to-llvm 之后的：

它包含：LLVM IR -> AMDGPU ISA -> object -> HSACO(binary)
它不包含：高层 MLIR GPU ops -> LLVM dialect/LLVM IR 的 lowering

"""

import sys
import os

# Setup MLIR Python path
MLIR_INSTALL = "/root/tingqli/llvm-project/mlir_install"
sys.path.insert(0, os.path.join(MLIR_INSTALL, "python_packages", "mlir_core"))

from mlir import ir
from mlir.passmanager import PassManager

def save_rocdl_binaries(module: ir.Module, out_dir: str = "."):
    for op in module.body.operations:
        if op.operation.name != "gpu.binary":
            continue

        sym_name_attr = op.attributes["sym_name"]
        sym_name = str(sym_name_attr).strip('"')

        objects_attr = op.attributes["objects"]

        for i, obj_attr in enumerate(objects_attr):
            # downcast 到 gpu.ObjectAttr
            gpu_obj = gpu.ObjectAttr(obj_attr)

            target = gpu_obj.target
            fmt = gpu_obj.format
            blob = gpu_obj.object  # Python bytes

            # 简单判断是不是 rocdl target
            target_text = str(target)
            if "#rocdl.target" not in target_text:
                continue

            filename = f"{out_dir}/{sym_name}.{i}.co"
            with open(filename, "wb") as f:
                f.write(blob)

            print(f"wrote {filename}, {len(blob)} bytes, format={fmt}, target={target_text}")


"""
FlyDSL/python/flydsl/compiler/backends/rocm.py
"""
def apply_gpu_pipeline(module, chip_type="gfx942"):
    """Applies the GPU compilation pipeline to the MLIR module."""
    fast_fp_math = False
    unsafe_fp_math = False
    is_rdna_arch = False
    enable_debug_info = False
    waves_per_eu = None
    maxnreg = None
    enable_ir_printing = False

    rocdl_opts = {
        "O": 2,
        "abi": 600,
        "chip": chip_type,
        "correct-sqrt": "true",
        "daz": "false",
        "fast": "true" if fast_fp_math else "false",
        "features": "",
        "finite-only": "false",
        "module": "",
        "triple": "amdgcn-amd-amdhsa",
        "unsafe-math": "true" if unsafe_fp_math else "false",
        "wave64": "false" if is_rdna_arch else "true",
    }
    rocdl_opts_str = " ".join(f"{k}={v}" for k, v in rocdl_opts.items())
    
    pm = PassManager()
    if enable_ir_printing:
        pm.enable_ir_printing(print_after_change=True)
    pm.add("canonicalize")
    pm.add(
        "one-shot-bufferize{ bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map }"
    )
    pm.add("canonicalize")
    pm.add("convert-linalg-to-affine-loops")
    pm.add("func.func(affine-loop-invariant-code-motion)")
    pm.add("func.func(convert-affine-for-to-gpu)")
    pm.add("gpu-kernel-outlining")
    pm.add("lower-affine")
    pm.add("gpu-decompose-memrefs")
    pm.add("expand-strided-metadata")
    pm.add("normalize-memrefs")
    pm.add(
            f"gpu.module(convert-scf-to-cf,cse,"
            f"convert-gpu-to-rocdl{{chipset={chip_type} index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true}})",
    )
    pm.add(f"rocdl-attach-target{{{rocdl_opts_str}}}")
    pm.add("convert-scf-to-cf")
    pm.add("convert-cf-to-llvm")
    pm.add("gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}")
    pm.add("convert-vector-to-llvm")
    pm.add("convert-arith-to-llvm")
    pm.add("convert-func-to-llvm")
    pm.add("reconcile-unrealized-casts")

    bin_cli_opts = []
    if enable_debug_info:
        bin_cli_opts.append("-g")
    if waves_per_eu:
        bin_cli_opts.append(f"--amdgpu-waves-per-eu={waves_per_eu}")
    if maxnreg:
        bin_cli_opts.append(f"--amdgpu-num-vgpr={maxnreg}")

    pm.add(f'gpu-module-to-binary{{format=fatbin opts="{" ".join(bin_cli_opts)}"}}')

    pm.run(module.operation)

    return module


input_mlir=r"""
module attributes {
  gpu.container_module
} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @vec_cos_kernel(%arg0: memref<1024xf32, 1>,
                             %arg1: memref<1024xf32, 1>)
        kernel 
        attributes {
            passthrough = ["amdgpu-no-implicitarg-ptr"]
        }        {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c4 = arith.constant 4 : index

      scf.for %i = %c0 to %c1024 step %c4 {
        %v = vector.load %arg0[%i] : memref<1024xf32, 1>, vector<4xf32>

        %f0 = vector.extract %v[0] : f32 from vector<4xf32>
        %f1 = vector.extract %v[1] : f32 from vector<4xf32>
        %f2 = vector.extract %v[2] : f32 from vector<4xf32>
        %f3 = vector.extract %v[3] : f32 from vector<4xf32>

        %c0v = rocdl.cos %f0 f32 -> f32
        %c1v = rocdl.cos %f1 f32 -> f32
        %c2v = rocdl.cos %f2 f32 -> f32
        %c3v = rocdl.cos %f3 f32 -> f32

        %r0 = vector.insert %c0v, %v[0] : f32 into vector<4xf32>
        %r1 = vector.insert %c1v, %r0[1] : f32 into vector<4xf32>
        %r2 = vector.insert %c2v, %r1[2] : f32 into vector<4xf32>
        %r3 = vector.insert %c3v, %r2[3] : f32 into vector<4xf32>

        vector.store %r3, %arg1[%i] : memref<1024xf32, 1>, vector<4xf32>
      }

      gpu.return
    }
  }
}
"""




from mlir.dialects import llvm, builtin, arith, gpu, func

def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return llvm.LoadOp(result_type, ptr).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return llvm.StoreOp(value, ptr)

def build_module(chip = "gfx942"):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    ctx.enable_multithreading(False)

    #target_attrs = [ir.Attribute.parse(f'#rocdl.target<chip = "{chip}">')]

    module = None
    with ctx, ir.Location.unknown():
        module = ir.Module.create()
        module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()
        #module.operation.attributes["llvm.target_triple"] = ir.StringAttr.get(
        #    "amdgcn-amd-amdhsa"
        #)

        with ir.InsertionPoint(module.body):
            module_op = gpu.GPUModuleOp("kernels") # targets=ir.ArrayAttr.get(target_attrs)
            module_op.regions[0].blocks.append()
            with ir.InsertionPoint(module_op.regions[0].blocks[0]):
                ptr_as1 = ir.Type.parse("!llvm.ptr<1>")
                f32 = ir.F32Type.get()
                kernel_arg_types = [ptr_as1, f32]
                gpu_func = gpu.GPUFuncOp(
                    ir.FunctionType.get(kernel_arg_types, []),
                    sym_name="my_kernel",
                    kernel=True,
                )
                if 0:
                    passthrough_attr = ir.ArrayAttr.get([
                        ir.StringAttr.get("amdgpu-no-implicitarg-ptr")
                    ])
                    gpu_func.attributes["passthrough"] = passthrough_attr

                gpu_func.regions[0].blocks.append(*kernel_arg_types)
                entry_block = gpu_func.regions[0].blocks[0]

                with ir.InsertionPoint(gpu_func.entry_block):
                    i32 = ir.IntegerType.get_signless(32)
                    const = arith.ConstantOp(i32, ir.IntegerAttr.get(i32, 42)).result
                    gpu.PrintfOp(format="Hello,World %d\n", args=[const])
                    #val = llvm.LoadOp(i32, func_op.arguments[0]).result
                    gpu.ReturnOp([])
    #print("<<<<<<<<<<")
    #print(str(module))
    #print(">>>>>>>>>>>")
    #assert 0
    return module

mlir_src = r"""
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @my_kernel(%arg0: !llvm.ptr<1>, %arg1: f32) kernel {
      %c42_i32 = arith.constant 42 : i32
      gpu.printf "Hello,World %d\0A", %c42_i32 : i32
      gpu.return
    }
  }
}
"""

def compile_mlir_to_binary(mlir_module_str: str, chip_type="gfx942"):
    """Compiles MLIR module string to binary code."""
    with ir.Context():
        # Parse the input module
        #module = ir.Module.parse(mlir_module_str)
        #module = ir.Module.parse(mlir_src)
        module = build_module()

        module = apply_gpu_pipeline(module, chip_type)

        save_rocdl_binaries(module)



compile_mlir_to_binary(input_mlir, chip_type="gfx942")


from pyhip.hsaco_tools import get_lib
import torch
torch.set_default_device("cuda")
a = torch.randn(1024, device="cuda")
lib = get_lib("./kernels.0.co")

lib.my_kernel([1],[64], a, 6)
