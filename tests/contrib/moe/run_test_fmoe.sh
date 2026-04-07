#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ulimit -c 0
rm ~/.pyhip -rf
pytest $SCRIPT_DIR/test_fused_moe.py