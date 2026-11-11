#!/bin/bash
# The 128-row branch inside the BM256 kernels ("bm128": fused_moe_a8w8_blockscale_gfx942 with bm128=True, same grids
# as the aiter path) against the BM256 path ("bm256", bm128=False), the merged path (option A), option B ("split"),
# the two-launch path ("small") and the tuned asm, one session per run of bench_moe_small_m_gfx942.py, results in ..
# Protocol of ../../merged/tools/run_ab.sh: each run starts after GPU $GPU has been idle (activity <= 3 % and VRAM <=
# $MAX_VRAM_PCT %, 3 checks 10 s apart) and is redone until all its amdsmi gates pass; failed attempts stay as
# <name>.attempt<k>.log.
GPU=${GPU:-1}
MAX_VRAM_PCT=${MAX_VRAM_PCT:-20}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$(cd "$HERE/.." && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
# pyhip must be importable as a package (e.g. PYTHONPATH holding a pyhip -> src link).
export HIP_VISIBLE_DEVICES=$GPU
cd "$REPO/tests/contrib/moe"
ASM="auto vs_ps_32x256 vs_psx_64x256 vs_64x256 vs_32x256 novs_ps_32x256"
LOG=$D/run_bm128.wait
B=bench_moe_small_m_gfx942.py
(cd "$REPO" && sha256sum src/contrib/flydsl/moe_a8w8_blockscale_gfx942.py src/contrib/flydsl/moe_gateup_a8w8_blockscale_gfx942_lds.py \
  src/contrib/flydsl/moe_down_a8w8_blockscale_gfx942_lds.py src/contrib/flydsl/moe_small_m_a8w8_blockscale_gfx942.py \
  src/contrib/flydsl/moe_merged_a8w8_blockscale_gfx942.py tests/contrib/moe/bench_moe_small_m_gfx942.py) > $D/sources.sha256

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

T_ALL="256 512 1024 2048 4096 6144 8192"
P="asm bm256 bm128 small split merged"
run final_ep_uniform $B --tokens $T_ALL --asm $ASM --paths $P --kernels --no-abort --json $D/final_ep_uniform.json
run final_random $B --tokens $T_ALL --routing random --asm $ASM --paths $P --kernels --no-abort --json $D/final_random.json
run burst_ep_uniform $B --tokens $T_ALL --asm $ASM --paths $P --warmup-ms 0 --no-abort --json $D/burst_ep_uniform.json
run burst_random $B --tokens $T_ALL --routing random --asm $ASM --paths $P --warmup-ms 0 --no-abort --json $D/burst_random.json
run final_large_random $B --tokens 12288 16384 32768 --routing random --asm $ASM --paths $P --kernels --no-abort --json $D/final_large_random.json
run final_large_ep_uniform $B --tokens 12288 16384 32768 --asm $ASM --paths $P --kernels --no-abort --json $D/final_large_ep_uniform.json
echo "$(date +%T) all done" >> $LOG
