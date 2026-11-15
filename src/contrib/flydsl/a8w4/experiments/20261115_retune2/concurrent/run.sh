#!/bin/bash
# Do 8 concurrent single-GPU processes (as in the table generation: one tuner per GPU)
# change the timings? GPU 0 runs `tune verify` (rotated weights, the table's configs)
# alone, then while GPUs 1-7 run the EP8 tuner on large batches; twice.
#   bash a8w4/experiments/20261115_retune2/concurrent/run.sh     (from src/contrib/flydsl)
cd "$(dirname "$0")/../../../.."   # src/contrib/flydsl
D=a8w4/experiments/20261115_retune2/concurrent
V="python -m a8w4.tune verify --case mimo-v2.6-flash-ep8 -t 2048 4096 8192 16384 32768"
LOAD="python -m a8w4.tune tune --case mimo-v2.6-flash-ep8 --rounds 100 -t 32768 16384"
# compile every candidate once, so the loads below are timing (not compiling) during verify
HIP_VISIBLE_DEVICES=1 python -m a8w4.tune tune --case mimo-v2.6-flash-ep8 --rounds 1 \
    --iters 2 -t 8192 --out /tmp/a8w4_conc_warm.csv > /dev/null 2>&1
for rep in 1 2; do
  HIP_VISIBLE_DEVICES=0 $V > $D/alone_$rep.txt 2>&1
  pids=()
  for g in 1 2 3 4 5 6 7; do
    HIP_VISIBLE_DEVICES=$g $LOAD --out /tmp/a8w4_conc_$g.csv > $D/load_${rep}_$g.log 2>&1 &
    pids+=($!)
  done
  until [ "$(grep -l candidates $D/load_${rep}_*.log 2>/dev/null | wc -l)" = 7 ]; do sleep 2; done
  sleep 10
  rocm-smi --showuse | grep "GPU use" > $D/loaded_$rep.txt
  HIP_VISIBLE_DEVICES=0 $V >> $D/loaded_$rep.txt 2>&1
  alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=$((alive + 1)); done
  echo "loads still running at the end of verify: $alive / 7" >> $D/loaded_$rep.txt
  kill "${pids[@]}" 2>/dev/null; wait
  rm -f $D/load_${rep}_*.log
done
rm -f /tmp/a8w4_conc_*.csv
