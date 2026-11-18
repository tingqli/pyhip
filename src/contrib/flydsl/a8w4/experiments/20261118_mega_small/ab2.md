
> 2026-11-18T16:47:59 | git bdd7506+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 64 512 1024 -v 'mega_small bm32' 'b16w2=mega_small bm16:arb=128,wpc=2' 'b32w2=mega_small bm32:arb=128,wpc=2' 'b32a128=mega_small bm32:arb=128' 'b64a128=mega_small bm32:bm=64,arb=128' 'no128=mega_small bm32:no=128' --out /tmp/small_ab --name ab2`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_small bm32#2 delta|, half the base's round spread, 0.3%))

#### 64 tokens (seed 71)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 155.5 | +0.0 | 155.1-158.2 | base | 1.9e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 155.5 | -0.0 | 154.8-156.0 | noise floor | 1.9e-04 | 0 |
| b16w2 | `mega_small bm16:arb=128,wpc=2` | 148.7 | -4.4 | 148.3-149.3 | faster | 1.9e-04 | 0 |
| b32w2 | `mega_small bm32:arb=128,wpc=2` | 153.2 | -1.5 | 152.2-154.2 | faster | 1.9e-04 | 0 |
| b32a128 | `mega_small bm32:arb=128` | 157.9 | +1.6 | 157.5-159.0 | slower | 1.9e-04 | 0 |
| b64a128 | `mega_small bm64:arb=128` | 157.8 | +1.5 | 157.6-158.8 | slower | 1.9e-04 | 0 |
| no128 | `mega_small bm32:no=128` | 158.1 | +1.7 | 157.4-158.7 | slower | 1.9e-04 | 0 |
| *noise band +-1.0%* |  |  |  |  |  |  |  |

#### 512 tokens (seed 519)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 179.3 | +0.0 | 178.6-179.5 | base | 1.6e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 179.2 | -0.1 | 179.0-180.0 | noise floor | 1.6e-04 | 0 |
| b16w2 | `mega_small bm16:arb=128,wpc=2` | 212.3 | +18.4 | 211.4-213.1 | slower | 1.6e-04 | 0 |
| b32w2 | `mega_small bm32:arb=128,wpc=2` | 181.5 | +1.2 | 179.9-182.1 | slower | 1.6e-04 | 0 |
| b32a128 | `mega_small bm32:arb=128` | 183.0 | +2.0 | 182.9-183.4 | slower | 1.6e-04 | 0 |
| b64a128 | `mega_small bm64:arb=128` | 183.3 | +2.2 | 183.3-183.5 | slower | 1.6e-04 | 0 |
| no128 | `mega_small bm32:no=128` | 183.7 | +2.4 | 182.1-184.1 | slower | 1.6e-04 | 0 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 1024 tokens (seed 1031)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 265.6 | +0.0 | 265.2-268.7 | base | 1.5e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 264.9 | -0.3 | 264.4-266.0 | noise floor | 1.5e-04 | 0 |
| b16w2 | `mega_small bm16:arb=128,wpc=2` | 321.2 | +20.9 | 320.8-321.8 | slower | 1.5e-04 | 0 |
| b32w2 | `mega_small bm32:arb=128,wpc=2` | 263.7 | -0.7 | 263.4-265.9 | faster | 1.5e-04 | 0 |
| b32a128 | `mega_small bm32:arb=128` | 270.6 | +1.9 | 270.3-271.2 | slower | 1.5e-04 | 0 |
| b64a128 | `mega_small bm64:arb=128` | 246.1 | -7.4 | 244.7-246.2 | faster | 1.5e-04 | 0 |
| no128 | `mega_small bm32:no=128` | 276.4 | +4.0 | 275.7-276.5 | slower | 1.5e-04 | 0 |
| *noise band +-0.7%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 64 | 512 | 1024 |
|---|---|---:|---:|---:|
| mega_small bm32 | `mega_small bm32` | 155.5 us | 179.3 us | 265.6 us |
| mega_small bm32#2 | `mega_small bm32` | -0.0%~ | -0.1%~ | -0.3%~ |
| b16w2 | `mega_small bm16:arb=128,wpc=2` | -4.4% | +18.4% | +20.9% |
| b32w2 | `mega_small bm32:arb=128,wpc=2` | -1.5% | +1.2% | -0.7% |
| b32a128 | `mega_small bm32:arb=128` | +1.6% | +2.0% | +1.9% |
| b64a128 | `mega_small bm64:arb=128` | +1.5% | +2.2% | -7.4% |
| no128 | `mega_small bm32:no=128` | +1.7% | +2.4% | +4.0% |

