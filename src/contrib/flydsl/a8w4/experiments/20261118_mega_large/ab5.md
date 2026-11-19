
> 2026-11-19T01:36:41 | git 5312964+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 4096 8192 16384 32768 -v 'mega_large p16 bcm01' 'lag32=mega_large p16 bcm01:lag=32' 'lag48=mega_large p16 bcm01:lag=48' 'lag64=mega_large p16 bcm01:lag=64' 'lag96=mega_large p16 bcm01:lag=96' --out a8w4/experiments/20261118_mega_large --name ab5`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_large p16 bcm01#2 delta|, half the base's round spread, 0.3%))

#### 4096 tokens (seed 4103)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 519.9 | +0.0 | 515.1-521.2 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 519.8 | -0.0 | 517.1-524.0 | noise floor | 1.8e-03 | 5.1e-06 |
| lag32 | `mega_large p16 bcm01:lag=32` | 498.6 | -4.1 | 497.3-500.0 | faster | 1.8e-03 | 5.9e-06 |
| lag48 | `mega_large p16 bcm01:lag=48` | 519.0 | -0.2 | 517.3-526.2 | ~ | 1.8e-03 | 6.4e-06 |
| lag64 | `mega_large p16 bcm01:lag=64` | 518.8 | -0.2 | 516.0-522.9 | ~ | 1.8e-03 | 6.3e-06 |
| lag96 | `mega_large p16 bcm01:lag=96` | 518.4 | -0.3 | 516.7-520.6 | ~ | 1.8e-03 | 5.4e-06 |
| *noise band +-0.6%* |  |  |  |  |  |  |  |

#### 8192 tokens (seed 8199)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 822.0 | +0.0 | 820.8-823.3 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 822.7 | +0.1 | 821.0-824.8 | noise floor | 1.8e-03 | 1.9e-06 |
| lag32 | `mega_large p16 bcm01:lag=32` | 815.2 | -0.8 | 813.4-818.8 | faster | 1.8e-03 | 3.3e-06 |
| lag48 | `mega_large p16 bcm01:lag=48` | 818.4 | -0.4 | 818.1-823.2 | faster | 1.8e-03 | 2.8e-06 |
| lag64 | `mega_large p16 bcm01:lag=64` | 820.0 | -0.2 | 819.7-824.8 | ~ | 1.8e-03 | 2.6e-06 |
| lag96 | `mega_large p16 bcm01:lag=96` | 822.3 | +0.0 | 820.1-824.6 | ~ | 1.8e-03 | 1.8e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 16384 tokens (seed 16391)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 1517.5 | +0.0 | 1472.2-1557.3 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 1542.1 | +1.6 | 1491.4-1591.3 | noise floor | 1.8e-03 | 1.9e-06 |
| lag32 | `mega_large p16 bcm01:lag=32` | 1511.3 | -0.4 | 1460.3-1588.5 | ~ | 1.8e-03 | 3.2e-06 |
| lag48 | `mega_large p16 bcm01:lag=48` | 1479.6 | -2.5 | 1453.3-1559.8 | ~ | 1.8e-03 | 2.9e-06 |
| lag64 | `mega_large p16 bcm01:lag=64` | 1470.0 | -3.1 | 1455.3-1547.9 | faster | 1.8e-03 | 2.7e-06 |
| lag96 | `mega_large p16 bcm01:lag=96` | 1465.7 | -3.4 | 1459.8-1529.3 | faster | 1.8e-03 | 2.3e-06 |
| *noise band +-2.8%* |  |  |  |  |  |  |  |

#### 32768 tokens (seed 32775)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 2954.2 | +0.0 | 2845.3-3139.3 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 3045.3 | +3.1 | 2909.8-3130.8 | noise floor | 1.8e-03 | 3.5e-06 |
| lag32 | `mega_large p16 bcm01:lag=32` | 2902.8 | -1.7 | 2788.3-2946.5 | ~ | 1.8e-03 | 3.6e-06 |
| lag48 | `mega_large p16 bcm01:lag=48` | 2897.7 | -1.9 | 2794.7-2931.6 | ~ | 1.8e-03 | 3.5e-06 |
| lag64 | `mega_large p16 bcm01:lag=64` | 2921.3 | -1.1 | 2777.9-2975.3 | ~ | 1.8e-03 | 3.7e-06 |
| lag96 | `mega_large p16 bcm01:lag=96` | 2906.4 | -1.6 | 2867.5-2930.9 | ~ | 1.8e-03 | 4.0e-06 |
| *noise band +-5.0%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 4096 | 8192 | 16384 | 32768 |
|---|---|---:|---:|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 519.9 us | 822.0 us | 1517.5 us | 2954.2 us |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | -0.0%~ | +0.1%~ | +1.6%~ | +3.1%~ |
| lag32 | `mega_large p16 bcm01:lag=32` | -4.1% | -0.8% | -0.4%~ | -1.7%~ |
| lag48 | `mega_large p16 bcm01:lag=48` | -0.2%~ | -0.4% | -2.5%~ | -1.9%~ |
| lag64 | `mega_large p16 bcm01:lag=64` | -0.2%~ | -0.2%~ | -3.1% | -1.1%~ |
| lag96 | `mega_large p16 bcm01:lag=96` | -0.3%~ | +0.0%~ | -3.4% | -1.6%~ |

