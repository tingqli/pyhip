#!/bin/bash
# The 4-wave 128-column benchmarks (bench_ab.py: the repository benchmark plus the old 8-wave path "old8" and the
# 64-column 4-wave path "c64"), results in ../../4wave_128. Each run starts after GPU $GPU has been idle (activity
# <= 3 % and VRAM <= $MAX_VRAM_PCT %, i.e. no other job's buffers, for 3 checks 10 s apart) and is redone until all
# its amdsmi gates pass: KFD does not report the CU occupancy of every process, so gpu_contention.ExternalLoadMonitor
# alone can miss another job. Failed attempts stay in the logs (<name>.attempt<k>.log). MAX_VRAM_PCT=100 when an idle
# job keeps its buffers allocated (activity gates only).
GPU=${GPU:-1}
MAX_VRAM_PCT=${MAX_VRAM_PCT:-20}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$(cd "$HERE/../.." && pwd)/4wave_128
mkdir -p "$D"
# pyhip must be importable as a package (e.g. PYTHONPATH holding a pyhip -> src link).
export HIP_VISIBLE_DEVICES=$GPU
cd "$HERE/../../../../../tests/contrib/moe"
ASM="auto vs_ps_32x256 vs_psx_64x256 vs_64x256 vs_32x256 novs_ps_32x256"
LOG=$D/run_128.wait
python $HERE/make_bench_ab.py
B=$HERE/bench_ab.py

wait_idle() {
  local idle=0
  while [ $idle -lt 3 ]; do
    m=$(amd-smi metric -g $GPU -u -m 2>/dev/null)
    act=$(echo "$m" | awk '/GFX_ACTIVITY/ {a=$2} END {print a+0}')
    used=$(echo "$m" | awk '/ USED_VRAM/ {u=$2} END {print u+0}')
    total=$(echo "$m" | awk '/ TOTAL_VRAM/ {t=$2} END {print t+0}')
    if [ "$act" -le 3 ] && [ $((used * 100)) -le $((total * MAX_VRAM_PCT)) ]; then idle=$((idle + 1)); else idle=0; fi
    sleep 10
  done
}

run() {  # name, args...
  local name=$1; shift
  for attempt in $(seq 1 60); do
    wait_idle
    echo "$(date +%T) $name attempt $attempt start" >> $LOG
    python "$@" --max-vram-pct $MAX_VRAM_PCT > $D/$name.log 2>&1
    if tail -1 $D/$name.log | grep -q "gates PASS"; then
      echo "$(date +%T) $name done ($(grep -c 'contaminated blocks redone' $D/$name.log) lines with redone blocks)" >> $LOG
      return
    fi
    mv $D/$name.log $D/$name.attempt$attempt.log
    echo "$(date +%T) $name attempt $attempt: gates failed" >> $LOG
  done
}

run final_ep_uniform $B --tokens 256 512 1024 2048 4096 --asm $ASM --paths asm bm256 small old8 c64 --kernels --no-abort --json $D/final_ep_uniform.json
run final_random $B --tokens 256 512 1024 2048 4096 --routing random --asm $ASM --paths asm bm256 small old8 c64 --kernels --no-abort --json $D/final_random.json
run burst_ep_uniform $B --tokens 256 512 1024 2048 4096 --asm $ASM --paths asm small old8 c64 --warmup-ms 0 --no-abort --json $D/burst_ep_uniform.json
run burst_random $B --tokens 256 512 1024 2048 4096 --routing random --asm $ASM --paths asm small old8 c64 --warmup-ms 0 --no-abort --json $D/burst_random.json
run final_large $B --tokens 6144 8192 --routing random --asm $ASM --paths asm bm256 small c64 --no-abort --json $D/final_large.json
echo "$(date +%T) all done" >> $LOG
