# setup
```
git clone https://github.com/ROCm/triton -b gluon_ext
cd triton
pip install -r python/requirements.txt
pip install .
```

# debug
dump asm code:
```
#MLIR_ENABLE_DUMP=1 MLIR_DUMP_PATH=. TRITON_DISABLE_LINE_INFO=1
AMDGCN_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python moe.py > a.asm
```

check if there is bankconflict:
```
ROCPROF=/opt/rocm/bin/rocprofv3 rocprof-compute profile --no-roof  --name mytest -b 12 -k moe_up -- python moe.py
ROCPROF=/opt/rocm/bin/rocprofv3 rocprof-compute analyze -p workloads/mytest/MI355/
```

profile timing:
```
rocprofv3 --hip-trace --kernel-trace --output-format pftrace -- python moe.py
```

profile micro-arch:
```
# curl -Lk https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --output rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
rocprofv3 -i v3input.yaml --att-library-path=./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux -- python moe.py
```