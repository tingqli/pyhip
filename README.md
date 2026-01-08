# pyhib

a simple python tools which invokes ROCM tools to compile HIP source into co file and then using `ctypes` to call HIP runtime API (from libamdhip64) to load & launch kernels from .co file.

specifically, following tools from ROCm is used:

 - `/opt/rocm/llvm/bin/amdgpu-arch` : detect GPU arch
 - `hipcc -x hip --offload-device-only -S` : compile HIP source into .s
 - `/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa` : assemble .s into .co

# Install

```bash
pip install git+https://github.com/tingqli/pyhip.git
```

Development setup,

```bash
# create a MI300 GPU Droplet with ROCM (7.1) on https://amd.digitalocean.com/
git clone https://github.com/tingqli/pyhip.git
git config --global user.name
git config --global user.email
apt install python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
pip3 install numpy pytest
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.1
```

# Usage

install torch rocm before using it in your script
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
```

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


