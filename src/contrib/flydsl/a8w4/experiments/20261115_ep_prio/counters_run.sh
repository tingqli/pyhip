#!/bin/bash
# rocprofv3 counter passes of counters.py (see its docstring); one T per output dir.
D=a8w4/experiments/20261115_ep_prio
for tb in "16 t16" "256 s16" "1024 m64" "16384 l128"; do
  set -- $tb
  for g in 1 2; do
    if [ $g = 1 ]; then P="TCC_HIT_sum TCC_MISS_sum TCC_READ_sum TCC_STREAMING_REQ_sum"
    else P="TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_RDREQ_DRAM_sum"; fi
    rocprofv3 --pmc $P -d $D/counters/t$1_$2_g$g -o c --output-format csv -- \
      python $D/counters.py $1 --base $2 > $D/counters/t$1_$2_g$g.log 2>&1
  done
done
