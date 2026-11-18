
> 2026-11-18T17:59:09 | git bdd7506+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 512 1024 2048 -v 'mega_small bm32' 'b64=mega_small bm32:bm=64' 'mega p32 l2 bcm22' 'mega p16 l1 bcm22' 'mega p16 l4 bcm01' --out /tmp/small_ab --name ab4`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_small bm32#2 delta|, half the base's round spread, 0.3%))

#### 512 tokens (seed 519)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 174.4 | +0.0 | 173.3-175.4 | base | 1.6e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 174.7 | +0.1 | 174.2-174.9 | noise floor | 1.6e-04 | 0 |
| b64 | `mega_small bm64` | 174.6 | +0.1 | 174.2-175.4 | ~ | 1.6e-04 | 0 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 208.9 | +19.8 | 207.9-209.2 | slower | 1.6e-04 | 8.0e-06 |
| mega p16 l1 bcm22 | `mega p16 l1 bcm22` | 208.9 | +19.8 | 208.1-210.7 | slower | 1.6e-04 | 8.1e-06 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 222.0 | +27.2 | 221.6-222.5 | slower | 1.6e-04 | 8.2e-06 |
| *noise band +-0.6%* |  |  |  |  |  |  |  |

#### 1024 tokens (seed 1031)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 258.3 | +0.0 | 257.8-258.7 | base | 1.5e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 257.8 | -0.2 | 257.1-258.9 | noise floor | 1.5e-04 | 0 |
| b64 | `mega_small bm64` | 227.3 | -12.0 | 226.5-228.0 | faster | 1.5e-04 | 0 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 240.1 | -7.0 | 239.6-240.9 | faster | 1.6e-04 | 8.1e-06 |
| mega p16 l1 bcm22 | `mega p16 l1 bcm22` | 240.5 | -6.9 | 239.9-241.1 | faster | 1.6e-04 | 8.1e-06 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 248.7 | -3.7 | 248.0-249.2 | faster | 1.6e-04 | 8.1e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 2048 tokens (seed 2055)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 437.2 | +0.0 | 436.4-437.7 | base | 1.5e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 435.9 | -0.3 | 435.0-436.8 | noise floor | 1.5e-04 | 0 |
| b64 | `mega_small bm64` | 364.4 | -16.6 | 363.0-365.0 | faster | 1.5e-04 | 0 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 369.6 | -15.5 | 369.2-370.4 | faster | 1.6e-04 | 8.0e-06 |
| mega p16 l1 bcm22 | `mega p16 l1 bcm22` | 365.8 | -16.3 | 364.9-367.0 | faster | 1.6e-04 | 8.0e-06 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 371.7 | -15.0 | 371.6-372.3 | faster | 1.6e-04 | 8.0e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 512 | 1024 | 2048 |
|---|---|---:|---:|---:|
| mega_small bm32 | `mega_small bm32` | 174.4 us | 258.3 us | 437.2 us |
| mega_small bm32#2 | `mega_small bm32` | +0.1%~ | -0.2%~ | -0.3%~ |
| b64 | `mega_small bm64` | +0.1%~ | -12.0% | -16.6% |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | +19.8% | -7.0% | -15.5% |
| mega p16 l1 bcm22 | `mega p16 l1 bcm22` | +19.8% | -6.9% | -16.3% |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | +27.2% | -3.7% | -15.0% |

