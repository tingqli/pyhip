import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.jit
def test(i:fx.Int32):
    row_base = fx.make_int_tuple(0)
    row_layout = fx.make_layout((128, 64), (1,0))
    coord_tensor = fx.Tensor(fx.make_view(row_base, row_layout))
    ct0 = coord_tensor[6, None]
    ct1 = coord_tensor[i, None]
    print("[compile-time] static slice :", ct0[0])
    print("[compile-time] dynamic slice :", ct1[0])
    fx.printf("[runtime] static slice {}", ct0[0])
    fx.printf("[runtime] dynamic slice {}", ct1[0])

test(6)