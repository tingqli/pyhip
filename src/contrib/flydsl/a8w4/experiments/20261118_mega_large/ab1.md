
> 2026-11-18T22:54:58 | git 937370e+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 2048 4096 8192 32768 -v 'mega_large p16 bcm22' 'mega p16 l2 bcm22' 'mega p16 l4 bcm01' 'mega_large p16 bcm01' --out a8w4/experiments/20261118_mega_large --name ab1`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_large p16 bcm22#2 delta|, half the base's round spread, 0.3%))

#### 2048 tokens (seed 2055)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm22 | `mega_large p16 bcm22` | 331.5 | +0.0 | 330.6-331.9 | base | 1.8e-03 | 0 |
| mega_large p16 bcm22#2 | `mega_large p16 bcm22` | 331.9 | +0.1 | 331.3-333.2 | noise floor | 1.8e-03 | 3.1e-06 |
| mega p16 l2 bcm22 | `mega p16 l2 bcm22` | 367.7 | +10.9 | 367.2-368.5 | slower | 1.6e-04 | 1.8e-03 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 374.7 | +13.0 | 374.3-375.3 | slower | 1.6e-04 | 1.8e-03 |
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 338.2 | +2.0 | 337.4-340.1 | slower | 1.8e-03 | 4.4e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 4096 tokens (seed 4103)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm22 | `mega_large p16 bcm22` | 547.8 | +0.0 | 538.6-548.3 | base | 1.8e-03 | 0 |
| mega_large p16 bcm22#2 | `mega_large p16 bcm22` | 540.6 | -1.3 | 539.6-543.1 | noise floor | 1.8e-03 | 4.4e-06 |
| mega p16 l2 bcm22 | `mega p16 l2 bcm22` | 551.6 | +0.7 | 550.6-554.0 | ~ | 1.6e-04 | 1.8e-03 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 542.1 | -1.0 | 539.4-543.9 | ~ | 1.6e-04 | 1.8e-03 |
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 520.1 | -5.1 | 518.5-522.1 | faster | 1.8e-03 | 4.9e-06 |
| *noise band +-1.3%* |  |  |  |  |  |  |  |

#### 8192 tokens (seed 8199)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm22 | `mega_large p16 bcm22` | 885.2 | +0.0 | 881.2-904.5 | base | 1.8e-03 | 0 |
| mega_large p16 bcm22#2 | `mega_large p16 bcm22` | 892.0 | +0.8 | 890.9-898.8 | noise floor | 1.8e-03 | 2.0e-06 |
| mega p16 l2 bcm22 | `mega p16 l2 bcm22` | 940.5 | +6.3 | 939.1-943.0 | slower | 1.5e-04 | 1.8e-03 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 905.5 | +2.3 | 903.5-906.3 | slower | 1.5e-04 | 1.8e-03 |
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 825.1 | -6.8 | 819.5-826.6 | faster | 1.8e-03 | 3.4e-06 |
| *noise band +-1.3%* |  |  |  |  |  |  |  |

#### 32768 tokens (seed 32775)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm22 | `mega_large p16 bcm22` | 3262.8 | +0.0 | 3058.2-3418.0 | base | 1.8e-03 | 0 |
| mega_large p16 bcm22#2 | `mega_large p16 bcm22` | 3253.3 | -0.3 | 3001.2-3366.2 | noise floor | 1.8e-03 | 3.2e-06 |
| mega p16 l2 bcm22 | `mega p16 l2 bcm22` | 3385.9 | +3.8 | 3159.1-3437.0 | ~ | 1.5e-04 | 1.8e-03 |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | 3185.9 | -2.4 | 3133.6-3322.9 | ~ | 1.5e-04 | 1.8e-03 |
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 2975.5 | -8.8 | 2864.2-3024.1 | faster | 1.8e-03 | 3.7e-06 |
| *noise band +-5.5%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 2048 | 4096 | 8192 | 32768 |
|---|---|---:|---:|---:|---:|
| mega_large p16 bcm22 | `mega_large p16 bcm22` | 331.5 us | 547.8 us | 885.2 us | 3262.8 us |
| mega_large p16 bcm22#2 | `mega_large p16 bcm22` | +0.1%~ | -1.3%~ | +0.8%~ | -0.3%~ |
| mega p16 l2 bcm22 | `mega p16 l2 bcm22` | +10.9% | +0.7%~ | +6.3% | +3.8%~ |
| mega p16 l4 bcm01 | `mega p16 l4 bcm01` | +13.0% | -1.0%~ | +2.3% | -2.4%~ |
| mega_large p16 bcm01 | `mega_large p16 bcm01` | +2.0% | -5.1% | -6.8% | -8.8% |

