import pyhip
import torch
import pytest

torch.set_printoptions(linewidth=300)
torch.set_default_device('cuda')
torch.manual_seed(0)

@pytest.mark.parametrize("cnt", [10,30,50,64,112,128,280,1002])
def test_wg_load_lds(cnt):
    @pyhip.jit()
    def test_lds(J, cnt, pdata:"void*", pout:"void*"):
        lds = J.alloc_lds(cnt * J.sizeof_DW)

        J.wg_load_lds(lds, pdata, cnt * J.sizeof_DW)

        vcnt = (cnt + J.warp_size - 1)//J.warp_size
        data = J.gpr(vcnt, "vu32")
        for i in range(vcnt):
            J.ds_read_b32(data[i], J.lane_id[0] * J.sizeof_DW, mod=f"offset:{lds + i*J.warp_size*J.sizeof_DW}")

        J.s_waitcnt(mod="lgkmcnt(0)")

        pout[:] += J.warp_id[0] * (cnt * J.sizeof_DW)

        vaddr = J.gpr(J.lane_id[0] * J.sizeof_DW)
        for i in range(vcnt):
            with J.ExecMask(vaddr[0] < cnt * J.sizeof_DW):
                J.global_store_dword(vaddr, data[i], pout)
            vaddr[0] += J.warp_size * J.sizeof_DW

    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

    data = torch.arange(0, cnt, dtype = torch.int)
    out = torch.zeros((4, cnt), dtype = torch.int)
    test_lds([1],[256],cnt, data.data_ptr(), out.data_ptr())

    print(out)
    assert torch.allclose(out[0], data)
    assert torch.allclose(out[1], data)
    assert torch.allclose(out[2], data)
    assert torch.allclose(out[3], data)

if __name__ == "__main__":
    test_wg_load_lds(10)
    test_wg_load_lds(1002)