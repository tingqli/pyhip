# Build LLDB family

```bash

# install dependencies (for lldb support python's formatter file)
apt install python3-dev swig

# under cloned llvm-project
mkdir build-lldb
cd build-lldb
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="lldb;" -DLLVM_TARGETS_TO_BUILD=X86 -DLLDB_ENABLE_PYTHON=ON -DLLDB_ENABLE_SWIG=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 ../llvm
ninja lldb lldb-server lldb-dap

# add /prefix/llvm-project/build-lldb/bin to PATH (in ~/.bashrc or /etc/profile)
echo 'export PATH="$PATH:/root/tingqli/llvm-project/build-lldb/bin"' >> ~/.bashrc

```

# Config VSCode's launch.json

```json
        {
            "name": "lldb-dap-Attach",
            "type": "lldb-dap",
            "request": "attach",
            "program": "/usr/bin/python3",
            "attachCommands": ["gdb-remote 1234"],
            "initCommands": [
                "command script import /root/tingqli/llvm-project/llvm/utils/lldbDataFormatters.py",
                "command script import /root/tingqli/llvm-project/mlir/utils/lldb-scripts/mlirDataFormatters.py",
            ]
        },
        {
            "name": "Python Debugger: Attach",
            //  -m debugpy --listen localhost:5678 --wait-for-client 
            "type": "debugpy",
            "justMyCode": false, // important
            "request": "attach",
                "connect": {
                    "host": "localhost",
                    "port": 5679
                }
        },
```

# Start debug servers

```bash

export CPYDEBUG='lldb-server g :1234 -- /usr/bin/python3 -m debugpy --listen localhost:5678 --wait-for-client'

$CPYDEBUG /your/python/script.py

# `lldb-dap-Attach` first, then `Python Debugger: Attach`
```

# Enjoy the runtime evaluation

Inside VsCode's `DEBUG CONSOLE`, type any C++ expression to evaluate it in runtime, including call functions of object variable.
for example, most MLIR/LLVM C++ object support dump() method, call directly by `t.dump()` (t is a varible in my C++)

