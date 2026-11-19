
> 2026-11-19T01:15:44 | git 937370e+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 2048 8192 32768 -v 'mega_large p16 bcm01' 'qx128=mega_large p16 bcm01:qx=1' 'mega p16 l4 bcm01' --out a8w4/experiments/20261118_mega_large --name ab3`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_large p16 bcm01#2 delta|, half the base's round spread, 0.3%))

#### 2048 tokens (seed 2055)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 338.3 | +0.0 | 336.4-340.1 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 338.5 | +0.1 | 336.8-338.8 | noise floor | 1.8e-03 | 4.4e-06 |
| qx128 | `mega_large p16 bcm01:qx=1` | 356.6 | +5.4 | 356.1-357.7 | slower | 1.6e-04 | 1.8e-03 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 374.4 | +10.7 | 374.3-375.8 | slower | 1.6e-04 | 1.8e-03 |
| *noise band +-0.5%* |  |  |  |  |  |  |  |

#### 8192 tokens (seed 8199)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 819.9 | +0.0 | 817.2-821.9 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 820.7 | +0.1 | 818.6-824.5 | noise floor | 1.8e-03 | 2.1e-06 |
| qx128 | `mega_large p16 bcm01:qx=1` | 888.8 | +8.4 | 877.5-891.0 | slower | 1.5e-04 | 1.8e-03 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 905.1 | +10.4 | 903.2-907.0 | slower | 1.5e-04 | 1.8e-03 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 32768 tokens (seed 32775)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 3005.2 | +0.0 | 2899.0-3015.9 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 3006.6 | +0.0 | 2878.7-3067.3 | noise floor | 1.8e-03 | 2.7e-06 |
| qx128 | `mega_large p16 bcm01:qx=1` | 3095.4 | +3.0 | 3061.0-3260.1 | slower | 1.5e-04 | 1.8e-03 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 3186.8 | +6.0 | 3100.3-3351.9 | slower | 1.5e-04 | 1.8e-03 |
| *noise band +-1.9%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 2048 | 8192 | 32768 |
|---|---|---:|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 338.3 us | 819.9 us | 3005.2 us |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | +0.1%~ | +0.1%~ | +0.0%~ |
| qx128 | `mega_large p16 bcm01:qx=1` | +5.4% | +8.4% | +3.0% |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | +10.7% | +10.4% | +6.0% |

