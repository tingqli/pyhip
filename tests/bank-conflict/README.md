
according to [lds-bank-conflict](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html), `ds_read_b128` has non-trivial lane-group, but there is no documentation mentioned that. [test.py](./test.py) was designed to :
 - detect LDS bank numbers;
 - detect such lane-groups;

example test results on MI300 GPU:

```bash
======================== LDS bank size
stride_dword:    0   dt:   358.8 us
stride_dword:    1   dt:   358.3 us
stride_dword:    2   dt:   474.4 us
stride_dword:    3   dt:   362.3 us
stride_dword:    4   dt:   907.2 us
stride_dword:    5   dt:   356.8 us
stride_dword:    8   dt:  1805.6 us
stride_dword:    9   dt:   357.4 us
stride_dword:   16   dt:  3600.6 us
stride_dword:   17   dt:   357.8 us
stride_dword:   32   dt:  7196.0 us <=========== bank size 32 DWORDs (128 bytes)
stride_dword:   33   dt:   357.3 us
stride_dword:   64   dt:  7196.2 us
stride_dword:   65   dt:   358.0 us
stride_dword:  128   dt:  7199.9 us
stride_dword:  129   dt:   357.7 us
stride_dword:  256   dt:  7193.8 us
stride_dword:  512   dt:  3714.7 us
stride_dword: 1024   dt:  1919.8 us
======================== ds_read_b64 
lane_group:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
lane_group:  [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
lane_group:  [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
lane_group:  [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
======================== ds_read_b128 
lane_group:  [0, 1, 2, 3, 20, 21, 22, 23]
lane_group:  [4, 5, 6, 7, 16, 17, 18, 19]
lane_group:  [8, 9, 10, 11, 28, 29, 30, 31]
lane_group:  [12, 13, 14, 15, 24, 25, 26, 27]
lane_group:  [32, 33, 34, 35, 52, 53, 54, 55]
lane_group:  [36, 37, 38, 39, 48, 49, 50, 51]
lane_group:  [40, 41, 42, 43, 60, 61, 62, 63]
lane_group:  [44, 45, 46, 47, 56, 57, 58, 59]
```

above tests confirmed lane group arrangement of `ds_read_b128` mentioned in above article, also it shows for `ds_read_b64`, the grouping was much more straight-forward.