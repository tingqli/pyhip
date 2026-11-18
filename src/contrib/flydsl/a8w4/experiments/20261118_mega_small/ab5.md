
> 2026-11-18T18:17:50 | git bdd7506+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 64 512 1024 -v 'mega_small bm64' 'p32=mega_small bm64:producers=32' 'p8=mega_small bm64:producers=8' 'a128=mega_small bm64:arb=128' --out /tmp/small_ab --name ab5`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_small bm64#2 delta|, half the base's round spread, 0.3%))

#### 64 tokens (seed 71)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm64 | `mega_small bm64` | 151.8 | +0.0 | 151.4-153.7 | base | 1.9e-04 | 0 |
| mega_small bm64#2 | `mega_small bm64` | 151.7 | -0.1 | 151.3-153.1 | noise floor | 1.9e-04 | 0 |
| p32 | `mega_small bm64:producers=32` | 152.2 | +0.3 | 151.2-153.8 | ~ | 1.9e-04 | 0 |
| p8 | `mega_small bm64:producers=8` | 151.7 | -0.1 | 150.7-152.9 | ~ | 1.9e-04 | 0 |
| a128 | `mega_small bm64:arb=128` | 154.4 | +1.7 | 154.1-155.9 | slower | 1.9e-04 | 0 |
| *noise band +-0.8%* |  |  |  |  |  |  |  |

#### 512 tokens (seed 519)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm64 | `mega_small bm64` | 174.0 | +0.0 | 172.9-174.2 | base | 1.6e-04 | 0 |
| mega_small bm64#2 | `mega_small bm64` | 173.7 | -0.1 | 173.5-174.1 | noise floor | 1.6e-04 | 0 |
| p32 | `mega_small bm64:producers=32` | 173.0 | -0.6 | 172.7-173.6 | faster | 1.6e-04 | 0 |
| p8 | `mega_small bm64:producers=8` | 175.5 | +0.9 | 174.9-175.6 | slower | 1.6e-04 | 0 |
| a128 | `mega_small bm64:arb=128` | 177.0 | +1.8 | 176.6-177.9 | slower | 1.6e-04 | 0 |
| *noise band +-0.4%* |  |  |  |  |  |  |  |

#### 1024 tokens (seed 1031)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm64 | `mega_small bm64` | 224.1 | +0.0 | 223.4-225.0 | base | 1.5e-04 | 0 |
| mega_small bm64#2 | `mega_small bm64` | 224.2 | +0.0 | 224.0-225.1 | noise floor | 1.5e-04 | 0 |
| p32 | `mega_small bm64:producers=32` | 223.9 | -0.1 | 223.3-224.9 | ~ | 1.5e-04 | 0 |
| p8 | `mega_small bm64:producers=8` | 229.5 | +2.4 | 229.3-229.8 | slower | 1.5e-04 | 0 |
| a128 | `mega_small bm64:arb=128` | 226.2 | +0.9 | 225.1-226.5 | slower | 1.5e-04 | 0 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 64 | 512 | 1024 |
|---|---|---:|---:|---:|
| mega_small bm64 | `mega_small bm64` | 151.8 us | 174.0 us | 224.1 us |
| mega_small bm64#2 | `mega_small bm64` | -0.1%~ | -0.1%~ | +0.0%~ |
| p32 | `mega_small bm64:producers=32` | +0.3%~ | -0.6% | -0.1%~ |
| p8 | `mega_small bm64:producers=8` | -0.1%~ | +0.9% | +2.4% |
| a128 | `mega_small bm64:arb=128` | +1.7% | +1.8% | +0.9% |

