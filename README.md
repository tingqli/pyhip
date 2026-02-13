# pyhib

PyHIP provides a one-stop toolkit for Python-based CDNA Assembly kernel development.

 - Following Triton's design philosophy, kernels are described in Python with jit decorators triggering the compilation pipeline. The resulting kernels can be called directly and interact seamlessly with PyTorch.
 - No automatic spilling; any variable allocated with J.gpr is guaranteed to be mapped to physical registers.
 - SGPR/VGPR register lifetimes are automatically managed and allocated.
 - SGPR/VGPR expressions support basic arithmetic (add, subtract, multiply, divide, modulo), bitwise shifts, AND, OR, NOT — clean, intuitive, and maintainable.
 - Python context managers enable control flow code generation for While/If constructs, significantly improving code readability.
 - Direct invocation of arbitrary instructions with explicit s_waitcnt, enabling fine-grained instruction pipelining.

# Install

Create [prebuilt Docker image with PyTorch pre-installed](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html), then run a container and install this package inside container.

```bash
pip install git+https://github.com/tingqli/pyhip.git
# or clone the repo and run editable install with "pip install -e ."
```

# Usage - Assembly kernels

 - use `pyhip.jit` decorator to declare a jit asm kernel
 - the first arg J of type `pyhip.JIT` provides key methods to:
   - allocate SGPR/VGPR/AccVGPRs explictly : `J.gpr(*shape, type, [initializers])`
   - call **any** assembly instructions directly : `J.s_store_dword/J.s_waitcnt ...`
   - build runtime control flow : `J.While/J.If/...`
 - kernel args with string-type annotations are runtime-args, the annotation is it's HIP type; all other args w/o string-type annotations are compile-time args.
 - directly invoke the kernel with args of torch tensors, int, float ...

```python
@pyhip.jit()
def kernel(J, N, s_pout:"int*"):
    s_i = J.gpr('si32')
    s_cnt = J.gpr('si32')
    s_i[0] = 0
    s_cnt[0] = 0
    with J.While(s_i[0] < N) as loop:
        with J.If((s_i[0] & 1)==0):
            s_cnt[0] = s_cnt + s_i
        s_i[0] = s_i[0] + 1

    J.s_store_dword(s_cnt, s_pout, 0, mod="glc")
    J.s_waitcnt(mod=f"lgkmcnt({0})")

OUT = torch.arange(0,32, dtype=torch.int)
kernel([1],[64], 32, OUT.data_ptr())
assert OUT[0] == sum([i if (i & 1) == 0 else 0 for i in range(32)]), f"{OUT=}"
```

Internally, `pyhip.jit` wraps the jit kernel, on invokation, it checks kernel binary file-cache under `~/.pyhip` using compile-time args as the key, and only do re-build on cache miss.

The build process will call jit kernel to generate IR (which is just a structured assembly), and apply following passes and then convert it into inline-asm into a HIP kernel and invokes ROCM to compile it into `.co` binary.

 - pass_remove_dead_bb: remove unreachable BB
 - pass_a2v: convert AccVGPRs to VGPRs for unsuitable instruction
 - pass_insert_nop: insert s_nop according to CDNA ISA
 - pass_hide_dependency: reorder nearby instructions to hide inst-to-inst dependency stall
 - pass_hide_karg_loads: reorder prelog instructions to hide kernel-arg load latency
 - pass_cse: common sub-expression elimination
 - pass_dse: dead store elimination
 - pass_dce: deac code elimination
 - pass_break_down_gprs: break logical VGPR array into smaller blocks according to instructions actually accessing needs
 - register_allocation_linear_scan: allocate physical register to logic registers


# Usage - HIP kernels

It can also invoke ROCM tools to compile HIP source into co file and then using `ctypes` to call HIP runtime API (from libamdhip64) to load & launch kernels from .co file.

specifically, following tools from ROCm is used:

 - `/opt/rocm/llvm/bin/amdgpu-arch` : detect GPU arch
 - `hipcc -x hip --offload-device-only -S` : compile HIP source into .s
 - `/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa` : assemble .s into .co

First implement your device kernel using HIP language in a .cpp or .s file, and then writting following code on python side to compile & launch them.

```python
# import torch before pyhip to load correct version of runtime (libamdhip64.so)
import torch
import pyhip

# this decorator will do recompilation if source file is modified after .co file
#   - compile mla.cpp file into mla.s using hipcc -x hip
#   - compile mla.s file into mla.co using clang++ -x assembler
# and then it loads mla.co and kernel function test_kernel, wrap it as a normal python callable
# you can call it with gridDim & blockDim & torch cuda tensors like this:
#     test_kernel([gridx,gridy,gridz],[blkx,blky,blkz], q.data_ptr(), ....)
@pyhip.module("mla.cpp")
def test_kernel(q, k, v, n, sm): ...

# when we meet performance issue, we can make a copy of mla.s and hack it by hand, this allows us
# to root-cause performance issue much easier since we can avoid involving complex HIP compiler.
@pyhip.module("mla-copy.s")
def test_kernel(q, k, v, n, sm): ...

# launch kernel on torch cuda tensor
test_kernel([gridDim_x, gridDim_y, gridDim_z],    # grid dimensions
            [blockDim_x, blockDim_y, blockDim_z], # block dimensions
            q.data_ptr(), # q is a torch cuda tensor, data_ptr() get it's device side pointer
            k.data_ptr(), # k is a torch cuda tensor, data_ptr() get it's device side pointer
            v.data_ptr(), # v is a torch cuda tensor, data_ptr() get it's device side pointer
            n             # normal argument
            )

```



