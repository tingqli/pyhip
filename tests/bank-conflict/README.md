
according to [lds-bank-conflict](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html), `ds_read_b128` has non-trivial lane-group, but there is no documentation mentioned that. [test.py](./test.py) was designed to :
 - detect LDS bank numbers;
 - detect such lane-groups;

