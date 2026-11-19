| tokens | variant | config | cos ref | cos base | repeat | bitwise | graph | result |
|---:|---|---|---:|---:|---:|---|---|---|
| 2048 | @table | `mega_large cadd xrow bcm22` | 1.8e-03 | 0 | 2.9e-06 | - | 3.0e-06 | OK |
| 2048 | mega_large cadd xrow bcm22 | `mega_large cadd xrow bcm22` | 1.8e-03 | 2.5e-06 | 2.9e-06 | - | 3.2e-06 | OK |
| 2048 | mega_large cfp8 x128 bcm01 | `mega_large cfp8 x128 bcm01` | 4.8e-04 | 2.2e-03 | 0 | - | 0 | OK |
| 32768 | @table | `mega_large cfp8 xrow bcm01` | 2.2e-03 | 0 | 0 | - | 0 | OK |
| 32768 | mega_large cadd xrow bcm22 | `mega_large cadd xrow bcm22` | 1.8e-03 | 3.3e-04 | 5.7e-06 | - | 4.7e-06 | OK |
| 32768 | mega_large cfp8 x128 bcm01 | `mega_large cfp8 x128 bcm01` | 4.7e-04 | 2.5e-03 | 0 | - | 0 | OK |
ALL OK
| tokens | variant | config | cos ref | cos base | repeat | bitwise | graph | result |
|---:|---|---|---:|---:|---:|---|---|---|
| 2048 | @table | `mega_large cadd x128 bcm22` | 1.6e-04 | 0 | 3.2e-06 | - | 5.2e-06 | OK |
| 2048 | mega_large cadd xrow bcm22 | `mega_large cadd xrow bcm22` | - | - | 2.7e-06 | - | 3.1e-06 | FAIL ref,base |
| 2048 | mega_large cfp8 x128 bcm01 | `mega_large cfp8 x128 bcm01` | 4.8e-04 | 3.3e-04 | 0 | - | 0 | OK |
| 32768 | @table | `mega_large cfp8 x128 bcm01` | 4.7e-04 | 0 | 0 | - | 0 | OK |
| 32768 | mega_large cadd xrow bcm22 | `mega_large cadd xrow bcm22` | - | - | 6.3e-06 | - | 5.1e-06 | FAIL ref,base |
| 32768 | mega_large cfp8 x128 bcm01 | `mega_large cfp8 x128 bcm01` | 4.7e-04 | 0 | 0 | - | 0 | OK |
2 FAILED
