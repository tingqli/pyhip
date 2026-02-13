# memory access pattern speed
different access will get various speed:
| pattern | detail | performance | comment |
| --- | --- | --- | --- |
| mfma16x16x16| 0-15 threads read [0-15, 0](each element is DWORDX4), then 16-31 threads read [16-31, 1]... | 1.1T/s |  |
| sequential read | 0-63 threads read[0, 0-63]... | 4.0T/s |  |
| len(K)==256,read once | 0-15 threads read[0, 0-15](each element is DWORDX4), 16-31 threads read[1, 0-15]... | 3.8T/s |  |
| len(K)==256,read twice | 0-8 threads read[0, 0-7](each element is DWORDX4), 7-15 threads read[1, 0-7]... | 3.5T/s |  |
| len(K)==256,read four times | 0-3 threads read[0, 0-3](each element is DWORDX4), 4-7 threads read[1, 0-3]... | 1.8T/s |  |
