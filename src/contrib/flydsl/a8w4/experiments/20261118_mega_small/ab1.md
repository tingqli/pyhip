
> 2026-11-18T16:39:53 | git bdd7506+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 64 256 512 1024 -v 'mega_small bm32' 'mega_small bm16' 'mori s16' 'mega p32 l2 bcm22' --out /tmp/small_ab --name ab1`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_small bm32#2 delta|, half the base's round spread, 0.3%))

#### 64 tokens (seed 71)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 155.2 | +0.0 | 154.2-155.9 | base | 1.9e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 155.3 | +0.1 | 154.6-156.2 | noise floor | 1.9e-04 | 0 |
| mega_small bm16 | `mega_small bm16` | 155.7 | +0.3 | 155.2-156.5 | ~ | 1.9e-04 | 0 |
| mori s16 | `mori s16` | 178.8 | +15.2 | 177.9-179.1 | slower | 9.6e-05 | 2.0e-04 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 191.0 | +23.1 | 190.5-191.2 | slower | 2.0e-04 | 8.7e-06 |
| *noise band +-0.6%* |  |  |  |  |  |  |  |

#### 256 tokens (seed 263)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 162.0 | +0.0 | 161.0-163.1 | base | 1.7e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 161.4 | -0.4 | 160.7-162.5 | noise floor | 1.7e-04 | 0 |
| mega_small bm16 | `mega_small bm16` | 162.0 | +0.0 | 161.9-162.9 | ~ | 1.7e-04 | 0 |
| mori s16 | `mori s16` | 187.4 | +15.7 | 187.0-187.5 | slower | 8.9e-05 | 1.8e-04 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 200.3 | +23.7 | 199.5-200.8 | slower | 1.7e-04 | 8.2e-06 |
| *noise band +-0.6%* |  |  |  |  |  |  |  |

#### 512 tokens (seed 519)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 179.2 | +0.0 | 178.8-180.0 | base | 1.6e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 179.7 | +0.3 | 179.0-179.8 | noise floor | 1.6e-04 | 0 |
| mega_small bm16 | `mega_small bm16` | 224.5 | +25.3 | 223.8-224.6 | slower | 1.6e-04 | 0 |
| mori s16 | `mori s16` | 245.6 | +37.0 | 245.5-245.8 | slower | 7.7e-05 | 1.7e-04 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 208.5 | +16.3 | 207.4-209.7 | slower | 1.6e-04 | 8.1e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 1024 tokens (seed 1031)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 266.5 | +0.0 | 266.1-269.2 | base | 1.5e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 267.1 | +0.2 | 266.7-267.7 | noise floor | 1.5e-04 | 0 |
| mega_small bm16 | `mega_small bm16` | 343.3 | +28.8 | 341.9-343.5 | slower | 1.5e-04 | 0 |
| mori s16 | `mori s16` | 360.2 | +35.2 | 360.0-360.7 | slower | 7.9e-05 | 1.7e-04 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 240.6 | -9.7 | 239.2-241.6 | faster | 1.6e-04 | 8.1e-06 |
| *noise band +-0.6%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 64 | 256 | 512 | 1024 |
|---|---|---:|---:|---:|---:|
| mega_small bm32 | `mega_small bm32` | 155.2 us | 162.0 us | 179.2 us | 266.5 us |
| mega_small bm32#2 | `mega_small bm32` | +0.1%~ | -0.4%~ | +0.3%~ | +0.2%~ |
| mega_small bm16 | `mega_small bm16` | +0.3%~ | +0.0%~ | +25.3% | +28.8% |
| mori s16 | `mori s16` | +15.2% | +15.7% | +37.0% | +35.2% |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | +23.1% | +23.7% | +16.3% | -9.7% |

