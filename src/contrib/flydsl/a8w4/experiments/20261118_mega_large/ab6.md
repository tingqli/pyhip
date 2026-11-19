
> 2026-11-19T01:45:28 | git 5312964+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 2048 4096 8192 16384 32768 -v 'mega_large p16 bcm01' 'cs=mega_large p16 bcm01:comb=1' 'cs_lag32=mega_large p16 bcm01:comb=1,lag=32' 'cs_lag96=mega_large p16 bcm01:comb=1,lag=96' 'cs_bcm22=mega_large p16 bcm22:comb=1' --out a8w4/experiments/20261118_mega_large --name ab6`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_large p16 bcm01#2 delta|, half the base's round spread, 0.3%))

#### 2048 tokens (seed 2055)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 337.9 | +0.0 | 336.8-338.3 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 338.0 | +0.0 | 337.2-339.6 | noise floor | 1.8e-03 | 6.1e-06 |
| cs | `mega_large p16 bcm01:comb=1` | 344.5 | +2.0 | 342.6-347.1 | slower | 1.8e-03 | 8.0e-06 |
| cs_lag32 | `mega_large p16 bcm01:comb=1,lag=32` | 344.7 | +2.0 | 343.9-345.4 | slower | 1.8e-03 | 8.0e-06 |
| cs_lag96 | `mega_large p16 bcm01:comb=1,lag=96` | 344.6 | +2.0 | 343.5-345.6 | slower | 1.8e-03 | 8.0e-06 |
| cs_bcm22 | `mega_large p16 bcm22:comb=1` | 338.8 | +0.3 | 337.2-339.7 | ~ | 1.8e-03 | 8.0e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 4096 tokens (seed 4103)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 522.2 | +0.0 | 518.9-523.1 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 518.6 | -0.7 | 517.5-520.2 | noise floor | 1.8e-03 | 3.9e-06 |
| cs | `mega_large p16 bcm01:comb=1` | 528.2 | +1.1 | 526.2-531.8 | slower | 1.8e-03 | 8.0e-06 |
| cs_lag32 | `mega_large p16 bcm01:comb=1,lag=32` | 514.4 | -1.5 | 513.5-514.7 | faster | 1.8e-03 | 8.0e-06 |
| cs_lag96 | `mega_large p16 bcm01:comb=1,lag=96` | 527.9 | +1.1 | 525.2-530.1 | slower | 1.8e-03 | 8.0e-06 |
| cs_bcm22 | `mega_large p16 bcm22:comb=1` | 553.4 | +6.0 | 547.7-557.1 | slower | 1.8e-03 | 8.0e-06 |
| *noise band +-0.7%* |  |  |  |  |  |  |  |

#### 8192 tokens (seed 8199)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 820.9 | +0.0 | 818.3-821.4 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 820.4 | -0.1 | 818.9-820.9 | noise floor | 1.8e-03 | 2.0e-06 |
| cs | `mega_large p16 bcm01:comb=1` | 849.2 | +3.5 | 848.8-851.0 | slower | 1.8e-03 | 8.0e-06 |
| cs_lag32 | `mega_large p16 bcm01:comb=1,lag=32` | 842.9 | +2.7 | 841.4-843.4 | slower | 1.8e-03 | 8.0e-06 |
| cs_lag96 | `mega_large p16 bcm01:comb=1,lag=96` | 848.8 | +3.4 | 846.8-850.4 | slower | 1.8e-03 | 8.0e-06 |
| cs_bcm22 | `mega_large p16 bcm22:comb=1` | 900.1 | +9.7 | 898.6-902.9 | slower | 1.8e-03 | 8.0e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 16384 tokens (seed 16391)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 1455.7 | +0.0 | 1452.5-1459.6 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 1455.5 | -0.0 | 1454.9-1461.5 | noise floor | 1.8e-03 | 1.7e-06 |
| cs | `mega_large p16 bcm01:comb=1` | 1485.9 | +2.1 | 1485.7-1487.1 | slower | 1.8e-03 | 7.9e-06 |
| cs_lag32 | `mega_large p16 bcm01:comb=1,lag=32` | 1486.2 | +2.1 | 1484.6-1486.8 | slower | 1.8e-03 | 7.9e-06 |
| cs_lag96 | `mega_large p16 bcm01:comb=1,lag=96` | 1482.7 | +1.9 | 1481.8-1484.2 | slower | 1.8e-03 | 7.9e-06 |
| cs_bcm22 | `mega_large p16 bcm22:comb=1` | 1590.6 | +9.3 | 1589.3-1601.4 | slower | 1.8e-03 | 7.9e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 32768 tokens (seed 32775)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 2872.0 | +0.0 | 2775.2-3100.4 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 2807.2 | -2.3 | 2761.4-2928.9 | noise floor | 1.8e-03 | 3.6e-06 |
| cs | `mega_large p16 bcm01:comb=1` | 2838.7 | -1.2 | 2695.1-2891.8 | ~ | 1.8e-03 | 8.0e-06 |
| cs_lag32 | `mega_large p16 bcm01:comb=1,lag=32` | 2798.5 | -2.6 | 2707.6-3027.9 | ~ | 1.8e-03 | 8.0e-06 |
| cs_lag96 | `mega_large p16 bcm01:comb=1,lag=96` | 2802.1 | -2.4 | 2732.0-2857.8 | ~ | 1.8e-03 | 8.0e-06 |
| cs_bcm22 | `mega_large p16 bcm22:comb=1` | 3023.5 | +5.3 | 2854.6-3202.8 | ~ | 1.8e-03 | 8.0e-06 |
| *noise band +-5.7%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---:|---:|---:|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 337.9 us | 522.2 us | 820.9 us | 1455.7 us | 2872.0 us |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | +0.0%~ | -0.7%~ | -0.1%~ | -0.0%~ | -2.3%~ |
| cs | `mega_large p16 bcm01:comb=1` | +2.0% | +1.1% | +3.5% | +2.1% | -1.2%~ |
| cs_lag32 | `mega_large p16 bcm01:comb=1,lag=32` | +2.0% | -1.5% | +2.7% | +2.1% | -2.6%~ |
| cs_lag96 | `mega_large p16 bcm01:comb=1,lag=96` | +2.0% | +1.1% | +3.4% | +1.9% | -2.4%~ |
| cs_bcm22 | `mega_large p16 bcm22:comb=1` | +0.3%~ | +6.0% | +9.7% | +9.3% | +5.3%~ |

