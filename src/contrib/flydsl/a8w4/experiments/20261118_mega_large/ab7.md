
> 2026-11-19T01:57:17 | git 5312964+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 2048 4096 8192 16384 32768 -v 'mega_large p16 bcm01' 'cs8=mega_large p16 bcm01:comb=2' 'cs8_lag32=mega_large p16 bcm01:comb=2,lag=32' 'cs8_bcm22=mega_large p16 bcm22:comb=2' --out a8w4/experiments/20261118_mega_large --name ab7`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_large p16 bcm01#2 delta|, half the base's round spread, 0.3%))

#### 2048 tokens (seed 2055)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 336.1 | +0.0 | 334.5-337.1 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 337.1 | +0.3 | 335.7-337.9 | noise floor | 1.8e-03 | 6.0e-06 |
| cs8 | `mega_large p16 bcm01:comb=2` | 340.5 | +1.3 | 339.6-340.6 | slower | 2.2e-03 | 3.3e-04 |
| cs8_lag32 | `mega_large p16 bcm01:comb=2,lag=32` | 339.9 | +1.1 | 338.9-340.7 | slower | 2.2e-03 | 3.3e-04 |
| cs8_bcm22 | `mega_large p16 bcm22:comb=2` | 334.7 | -0.4 | 332.9-335.7 | faster | 2.2e-03 | 3.3e-04 |
| *noise band +-0.4%* |  |  |  |  |  |  |  |

#### 4096 tokens (seed 4103)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 518.5 | +0.0 | 517.0-521.8 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 519.9 | +0.3 | 517.8-520.4 | noise floor | 1.8e-03 | 3.9e-06 |
| cs8 | `mega_large p16 bcm01:comb=2` | 501.1 | -3.3 | 499.4-501.7 | faster | 2.2e-03 | 3.3e-04 |
| cs8_lag32 | `mega_large p16 bcm01:comb=2,lag=32` | 504.1 | -2.8 | 503.7-505.1 | faster | 2.2e-03 | 3.3e-04 |
| cs8_bcm22 | `mega_large p16 bcm22:comb=2` | 515.4 | -0.6 | 515.0-516.8 | faster | 2.2e-03 | 3.3e-04 |
| *noise band +-0.5%* |  |  |  |  |  |  |  |

#### 8192 tokens (seed 8199)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 819.0 | +0.0 | 817.8-820.5 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 819.9 | +0.1 | 816.1-821.0 | noise floor | 1.8e-03 | 1.8e-06 |
| cs8 | `mega_large p16 bcm01:comb=2` | 815.2 | -0.5 | 813.5-817.8 | faster | 2.2e-03 | 3.3e-04 |
| cs8_lag32 | `mega_large p16 bcm01:comb=2,lag=32` | 812.1 | -0.8 | 810.9-813.5 | faster | 2.2e-03 | 3.3e-04 |
| cs8_bcm22 | `mega_large p16 bcm22:comb=2` | 864.2 | +5.5 | 863.7-866.1 | slower | 2.2e-03 | 3.3e-04 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 16384 tokens (seed 16391)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 1452.4 | +0.0 | 1451.3-1455.0 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 1451.3 | -0.1 | 1450.3-1454.9 | noise floor | 1.8e-03 | 2.0e-06 |
| cs8 | `mega_large p16 bcm01:comb=2` | 1467.5 | +1.0 | 1466.5-1468.6 | slower | 2.2e-03 | 3.3e-04 |
| cs8_lag32 | `mega_large p16 bcm01:comb=2,lag=32` | 1467.0 | +1.0 | 1463.4-1468.0 | slower | 2.2e-03 | 3.3e-04 |
| cs8_bcm22 | `mega_large p16 bcm22:comb=2` | 1571.2 | +8.2 | 1569.8-1572.3 | slower | 2.2e-03 | 3.3e-04 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 32768 tokens (seed 32775)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 2873.8 | +0.0 | 2845.5-2882.9 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 2832.6 | -1.4 | 2757.8-2883.6 | noise floor | 1.8e-03 | 3.4e-06 |
| cs8 | `mega_large p16 bcm01:comb=2` | 2653.7 | -7.7 | 2648.2-2738.9 | faster | 2.2e-03 | 3.3e-04 |
| cs8_lag32 | `mega_large p16 bcm01:comb=2,lag=32` | 2700.2 | -6.0 | 2688.9-2770.5 | faster | 2.2e-03 | 3.3e-04 |
| cs8_bcm22 | `mega_large p16 bcm22:comb=2` | 2833.9 | -1.4 | 2817.9-2864.3 | ~ | 2.2e-03 | 3.3e-04 |
| *noise band +-1.4%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---:|---:|---:|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 336.1 us | 518.5 us | 819.0 us | 1452.4 us | 2873.8 us |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | +0.3%~ | +0.3%~ | +0.1%~ | -0.1%~ | -1.4%~ |
| cs8 | `mega_large p16 bcm01:comb=2` | +1.3% | -3.3% | -0.5% | +1.0% | -7.7% |
| cs8_lag32 | `mega_large p16 bcm01:comb=2,lag=32` | +1.1% | -2.8% | -0.8% | +1.0% | -6.0% |
| cs8_bcm22 | `mega_large p16 bcm22:comb=2` | -0.4% | -0.6% | +5.5% | +8.2% | -1.4%~ |

