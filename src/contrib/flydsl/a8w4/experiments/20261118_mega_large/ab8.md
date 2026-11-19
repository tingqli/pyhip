
> 2026-11-19T02:53:18 | git 5312964+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 2048 8192 32768 -v 'mega_large p16 cadd bcm01' 'mega_large p16 cfp8 bcm01' 'mega_large p16 cfp8 bcm22' 'mega p16 l4 bcm01' --out a8w4/experiments/20261118_mega_large --name ab8`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_large p16 cadd bcm01#2 delta|, half the base's round spread, 0.3%))

#### 2048 tokens (seed 2055)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 cadd bcm01 | `mega_large p16 cadd bcm01` | 337.3 | +0.0 | 336.3-337.5 | base | 1.8e-03 | 0 |
| mega_large p16 cadd bcm01#2 | `mega_large p16 cadd bcm01` | 337.5 | +0.1 | 336.4-337.6 | noise floor | 1.8e-03 | 5.8e-06 |
| mega_large p16 cfp8 bcm01 | `mega_large p16 cfp8 bcm01` | 339.6 | +0.7 | 338.7-340.1 | slower | 2.2e-03 | 3.3e-04 |
| mega_large p16 cfp8 bcm22 | `mega_large p16 cfp8 bcm22` | 333.9 | -1.0 | 333.1-334.8 | faster | 2.2e-03 | 3.3e-04 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 375.1 | +11.2 | 374.1-376.1 | slower | 1.6e-04 | 1.8e-03 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 8192 tokens (seed 8199)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 cadd bcm01 | `mega_large p16 cadd bcm01` | 818.7 | +0.0 | 816.3-821.2 | base | 1.8e-03 | 0 |
| mega_large p16 cadd bcm01#2 | `mega_large p16 cadd bcm01` | 818.7 | -0.0 | 816.7-824.8 | noise floor | 1.8e-03 | 2.1e-06 |
| mega_large p16 cfp8 bcm01 | `mega_large p16 cfp8 bcm01` | 816.3 | -0.3 | 813.1-829.2 | ~ | 2.2e-03 | 3.3e-04 |
| mega_large p16 cfp8 bcm22 | `mega_large p16 cfp8 bcm22` | 866.2 | +5.8 | 864.7-868.7 | slower | 2.2e-03 | 3.3e-04 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 904.8 | +10.5 | 901.4-909.7 | slower | 1.5e-04 | 1.8e-03 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 32768 tokens (seed 32775)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 cadd bcm01 | `mega_large p16 cadd bcm01` | 2943.9 | +0.0 | 2885.7-3039.4 | base | 1.8e-03 | 0 |
| mega_large p16 cadd bcm01#2 | `mega_large p16 cadd bcm01` | 2927.8 | -0.5 | 2838.2-3020.3 | noise floor | 1.8e-03 | 4.2e-06 |
| mega_large p16 cfp8 bcm01 | `mega_large p16 cfp8 bcm01` | 2835.8 | -3.7 | 2785.8-2850.5 | faster | 2.2e-03 | 3.3e-04 |
| mega_large p16 cfp8 bcm22 | `mega_large p16 cfp8 bcm22` | 3148.6 | +7.0 | 3042.8-3179.8 | slower | 2.2e-03 | 3.3e-04 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 3159.0 | +7.3 | 3089.7-3238.1 | slower | 1.5e-04 | 1.8e-03 |
| *noise band +-2.6%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 2048 | 8192 | 32768 |
|---|---|---:|---:|---:|
| mega_large p16 cadd bcm01 | `mega_large p16 cadd bcm01` | 337.3 us | 818.7 us | 2943.9 us |
| mega_large p16 cadd bcm01#2 | `mega_large p16 cadd bcm01` | +0.1%~ | -0.0%~ | -0.5%~ |
| mega_large p16 cfp8 bcm01 | `mega_large p16 cfp8 bcm01` | +0.7% | -0.3%~ | -3.7% |
| mega_large p16 cfp8 bcm22 | `mega_large p16 cfp8 bcm22` | -1.0% | +5.8% | +7.0% |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | +11.2% | +10.5% | +7.3% |

