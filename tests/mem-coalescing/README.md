

- `DWORDx4` is much faster than `DWORDx2`
- coalescing happens as long as data-addresses from 64-lanes fall in same area of cache-lines, they can be permuted w/o damage performance.
- coalescing is best if data-addresses from 64-lanes fall in continous cache-lines (check `sum_1d_16x64`,`sum_1d_8x128`,`sum_1d_4x256`, `sum_1d_2x512`)