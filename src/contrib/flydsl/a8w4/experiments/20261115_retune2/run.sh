#!/bin/bash
# Full table regeneration with the weight-copy timing (configs/README.md procedure).
cd "$(dirname "$0")/../../.."   # src/contrib/flydsl
T="1 2 3 4 6 8 12 16 24 32 48 64 96 128 192 256 320 384 448 512 640 768 1024 1280 1536 2048 2560 3072 4096 6144 8192 12288 16384 24576 32768"
OUT=a8w4/experiments/20261115_retune2
tune4() {
  for g in 0 1 2 3; do
    HIP_VISIBLE_DEVICES=$(($3 + g)) python -m a8w4.tune tune --act $1 --case $2 \
        -t $(echo $T | tr ' ' '\n' | awk -v g=$g 'NR % 4 == g') \
        --out $OUT/$1_$2_$g.csv > $OUT/$1_$2_$g.log 2>&1 &
  done
}
tune4 situv2 mimo-v2.6-flash-tp8 0; tune4 situv2 mimo-v2.6-flash-ep8 4; wait
tune4 silu mimo-v2.6-flash-tp8 0; tune4 silu mimo-v2.6-flash-ep8 4; wait
tune4 silu mimo-v2.6-flash-tp4 0; wait
grep -h -E "BAD|FAIL" $OUT/*.log
python -m a8w4.tune merge $OUT/*_[0-3].csv --csv $OUT/a8w4_moe_tuned.new.csv
