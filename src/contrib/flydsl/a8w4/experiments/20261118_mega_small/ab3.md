
> 2026-11-18T17:56:10 | git bdd7506+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 64 256 512 1024 -v 'mega_small bm32' 'mega_small bm16' 'b64=mega_small bm32:bm=64' 'b16w2=mega_small bm16:wpc=2' 'b32c=mega_small bm32:bcm1=0,bcm2=0' 'b64c=mega_small bm32:bm=64,bcm1=0,bcm2=0' 'mega p32 l2 bcm22' --out /tmp/small_ab --name ab3`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_small bm32#2 delta|, half the base's round spread, 0.3%))

#### 64 tokens (seed 71)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 152.1 | +0.0 | 151.6-153.1 | base | 1.9e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 151.8 | -0.2 | 150.8-152.4 | noise floor | 1.9e-04 | 0 |
| mega_small bm16 | `mega_small bm16` | 152.9 | +0.5 | 152.4-153.4 | ~ | 1.9e-04 | 0 |
| b64 | `mega_small bm64` | 151.5 | -0.4 | 151.4-153.4 | ~ | 1.9e-04 | 0 |
| b16w2 | `mega_small bm16:wpc=2` | 155.0 | +1.9 | 154.4-156.6 | slower | 1.9e-04 | 0 |
| b32c | `mega_small bm32:bcm1=0,bcm2=0` | 165.3 | +8.6 | 164.0-165.7 | slower | 1.9e-04 | 0 |
| b64c | `mega_small bm64:bcm1=0,bcm2=0` | 164.6 | +8.2 | 163.9-165.4 | slower | 1.9e-04 | 0 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 190.8 | +25.4 | 189.9-191.6 | slower | 2.0e-04 | 8.6e-06 |
| *noise band +-0.5%* |  |  |  |  |  |  |  |

#### 256 tokens (seed 263)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 159.4 | +0.0 | 158.3-159.7 | base | 1.7e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 158.8 | -0.4 | 158.6-160.1 | noise floor | 1.7e-04 | 0 |
| mega_small bm16 | `mega_small bm16` | 159.4 | -0.0 | 159.0-159.6 | ~ | 1.7e-04 | 0 |
| b64 | `mega_small bm64` | 158.9 | -0.3 | 158.1-159.7 | ~ | 1.7e-04 | 0 |
| b16w2 | `mega_small bm16:wpc=2` | 160.8 | +0.8 | 160.5-161.2 | slower | 1.7e-04 | 0 |
| b32c | `mega_small bm32:bcm1=0,bcm2=0` | 173.2 | +8.6 | 172.3-173.9 | slower | 1.7e-04 | 0 |
| b64c | `mega_small bm64:bcm1=0,bcm2=0` | 172.5 | +8.2 | 172.0-173.5 | slower | 1.7e-04 | 0 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 200.5 | +25.7 | 199.7-202.6 | slower | 1.7e-04 | 8.1e-06 |
| *noise band +-0.5%* |  |  |  |  |  |  |  |

#### 512 tokens (seed 519)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 174.4 | +0.0 | 174.1-174.7 | base | 1.6e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 174.2 | -0.1 | 174.0-175.0 | noise floor | 1.6e-04 | 0 |
| mega_small bm16 | `mega_small bm16` | 224.3 | +28.6 | 223.7-224.8 | slower | 1.6e-04 | 0 |
| b64 | `mega_small bm64` | 174.8 | +0.2 | 173.6-175.3 | ~ | 1.6e-04 | 0 |
| b16w2 | `mega_small bm16:wpc=2` | 218.6 | +25.4 | 217.6-219.4 | slower | 1.6e-04 | 0 |
| b32c | `mega_small bm32:bcm1=0,bcm2=0` | 186.7 | +7.1 | 186.5-186.9 | slower | 1.6e-04 | 0 |
| b64c | `mega_small bm64:bcm1=0,bcm2=0` | 186.1 | +6.7 | 185.6-186.4 | slower | 1.6e-04 | 0 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 210.0 | +20.4 | 209.0-212.0 | slower | 1.6e-04 | 8.1e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 1024 tokens (seed 1031)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_small bm32 | `mega_small bm32` | 257.7 | +0.0 | 257.0-257.9 | base | 1.5e-04 | 0 |
| mega_small bm32#2 | `mega_small bm32` | 257.0 | -0.3 | 256.4-258.3 | noise floor | 1.5e-04 | 0 |
| mega_small bm16 | `mega_small bm16` | 342.0 | +32.7 | 341.5-342.3 | slower | 1.5e-04 | 0 |
| b64 | `mega_small bm64` | 226.3 | -12.2 | 225.8-226.8 | faster | 1.5e-04 | 0 |
| b16w2 | `mega_small bm16:wpc=2` | 334.5 | +29.8 | 334.2-334.5 | slower | 1.5e-04 | 0 |
| b32c | `mega_small bm32:bcm1=0,bcm2=0` | 247.9 | -3.8 | 247.5-248.3 | faster | 1.5e-04 | 0 |
| b64c | `mega_small bm64:bcm1=0,bcm2=0` | 233.8 | -9.3 | 233.4-234.6 | faster | 1.5e-04 | 0 |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | 240.5 | -6.7 | 240.2-240.7 | faster | 1.6e-04 | 8.1e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 64 | 256 | 512 | 1024 |
|---|---|---:|---:|---:|---:|
| mega_small bm32 | `mega_small bm32` | 152.1 us | 159.4 us | 174.4 us | 257.7 us |
| mega_small bm32#2 | `mega_small bm32` | -0.2%~ | -0.4%~ | -0.1%~ | -0.3%~ |
| mega_small bm16 | `mega_small bm16` | +0.5%~ | -0.0%~ | +28.6% | +32.7% |
| b64 | `mega_small bm64` | -0.4%~ | -0.3%~ | +0.2%~ | -12.2% |
| b16w2 | `mega_small bm16:wpc=2` | +1.9% | +0.8% | +25.4% | +29.8% |
| b32c | `mega_small bm32:bcm1=0,bcm2=0` | +8.6% | +8.6% | +7.1% | -3.8% |
| b64c | `mega_small bm64:bcm1=0,bcm2=0` | +8.2% | +8.2% | +6.7% | -9.3% |
| mega p32 l2 bcm22 | `mega p32 l2 bcm22` | +25.4% | +25.7% | +20.4% | -6.7% |

