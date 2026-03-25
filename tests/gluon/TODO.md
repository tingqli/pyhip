# 8 waves
- base: 1.37 Pflops
    - 4 wave(2x2), TILE: 256x256x64
    - 3 stage: global prefetch + local prefetch + compute
    - 2 pinpong lds
    - N tile splits to top+bot
- xcd/l2 aware: 1.34 Pflops
    - get_pids
    - remove '.cg'