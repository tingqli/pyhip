#!/bin/bash
# The 4-wave small-M benchmarks (bench_ab.py: the repository benchmark plus the old 8-wave path), each run once
# after an idle window of GPU $GPU (activity <= 3 % for 3 checks 10 s apart). Blocks during which another process
# ran kernels are redone inside the benchmark (gpu_contention.ExternalLoadMonitor); the gate records stay in the logs.
GPU=${GPU:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$(dirname "$HERE")
# pyhip must be importable as a package (e.g. PYTHONPATH holding a pyhip -> src link).
export HIP_VISIBLE_DEVICES=$GPU
cd "$D/../../../../tests/contrib/moe"
ASM="auto vs_ps_32x256 vs_psx_64x256 vs_64x256 vs_32x256 novs_ps_32x256"
LOG=$D/run_4w.wait
python $HERE/make_bench_ab.py
B=$HERE/bench_ab.py

wait_idle() {
  local idle=0
  while [ $idle -lt 3 ]; do
    act=$(amd-smi metric -g $GPU -u 2>/dev/null | awk '/GFX_ACTIVITY/ {a=$2} END {print a+0}')
    if [ "$act" -le 3 ]; then idle=$((idle + 1)); else idle=0; fi
    sleep 10
  done
}

run() {  # name, args...
  local name=$1; shift
  wait_idle
  echo "$(date +%T) $name start" >> $LOG
  python "$@" > $D/$name.log 2>&1
  echo "$(date +%T) $name done ($(grep -c 'contaminated blocks redone' $D/$name.log) lines with redone blocks)" >> $LOG
}

run final_ep_uniform $B --tokens 256 512 1024 2048 4096 --asm $ASM --paths asm bm256 small old8 --kernels --no-abort --json $D/final_ep_uniform.json
run final_random $B --tokens 256 512 1024 2048 4096 --routing random --asm $ASM --paths asm bm256 small old8 --kernels --no-abort --json $D/final_random.json
run burst_ep_uniform $B --tokens 256 512 1024 2048 4096 --asm $ASM --paths asm small old8 --warmup-ms 0 --no-abort --json $D/burst_ep_uniform.json
run burst_random $B --tokens 256 512 1024 2048 4096 --routing random --asm $ASM --paths asm small old8 --warmup-ms 0 --no-abort --json $D/burst_random.json
run final_large $B --tokens 6144 8192 --routing random --asm $ASM --paths asm bm256 small --no-abort --json $D/final_large.json
echo "$(date +%T) all done" >> $LOG
