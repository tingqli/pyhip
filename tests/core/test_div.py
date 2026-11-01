import pyhip
import torch
import math
torch.set_printoptions(linewidth=300)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_div_constant_divisor():
    @pyhip.jit()
    def kernel(J, q, n:"int", output:"int*"):
        out = J.gpr("si32")
        out[0] = n // q
        J.s_store_dword(out, output, mod="glc")

    OUT = torch.ones(64, dtype=torch.int)
    for d in [3, 641, 65537, 1024*1024*1023, -3, -641, -65537, -1024*1024*1023]:
        for n in [2**31-1, d-1, d, d+1]:
            ref = int(n / d)
            kernel([1],[64], d, n, OUT.data_ptr()) 
            assert OUT[0] == ref, f"{OUT[0].item()=} != {ref}  {n} / {d}"

            ref = int(-n / d)
            kernel([1],[64], d, -n, OUT.data_ptr())
            assert OUT[0] == ref, f"{OUT[0].item()=} != {ref} {-n} / {d}"

def test_div_constant_divisor_u32():
    @pyhip.jit()
    def kernel(J, q, n:"uint", output:"int*"):
        out = J.gpr("su32")
        out[0] = n // q
        J.s_store_dword(out, output, mod="glc")

    OUT = torch.ones(64, dtype=torch.int)
    # 7, 14, 21, 84, 112 need a 33-bit magic number
    for d in [3, 7, 14, 21, 84, 112, 641, 65537, 1024*1024*1023]:
        for n in [0, d-1, d, d+1, 123456789, 2**31-1, 2**31, 2**32-1]:
            kernel([1],[64], d, n, OUT.data_ptr())
            got = OUT[0].item() & 0xFFFFFFFF
            assert got == n // d, f"{got=} != {n // d}  {n} / {d}"

def test_div_sgpr_divisor():
    @pyhip.jit()
    def kernel(J, n:"int", q:"int", output:"int*"):
        out = J.gpr("si32")
        out[0] = n // q
        J.s_store_dword(out, output, mod="glc")

    OUT = torch.ones(64, dtype=torch.int)
    for d in [3, 641, 65537, 1024*1024*1023, -3, -641, -65537, -1024*1024*1023]:
        for n in [2**31-1, d-1, d, d+1]:
            ref = int(n / d)
            kernel([1],[64], n, d, OUT.data_ptr()) 
            assert OUT[0] == ref, f"{OUT[0].item()=} != {ref}  {n} / {d}"

            ref = int(-n / d)
            kernel([1],[64], -n, d, OUT.data_ptr())
            assert OUT[0] == ref, f"{OUT[0].item()=} != {ref} {-n} / {d}"

if __name__ == "__main__":
    #test_div_constant_divisor()
    test_div_sgpr_divisor()