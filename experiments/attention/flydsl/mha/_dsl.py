"""Small, local FlyDSL 0.2/0.3 compatibility helpers (no global patching)."""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm


def static_view(value):
    """Materialize type-known layout metadata instead of a deep layout DAG."""
    if isinstance(value, fx.Tensor) and value.type.layout.is_static:
        return fx.make_view(fx.get_iter(value), fx.static(value.type.layout))
    return value


def select(value, indices):
    """Permute tensor modes without touching data or register ordering."""
    if isinstance(value, fx.Tensor):
        if isinstance(value.layout, fx.ComposedLayout):
            raise NotImplementedError("select on a swizzled tensor requires an explicit composed layout")
        if value.shape.is_static and value.stride.is_static:
            shape, stride = value.shape.to_py_value(), value.stride.to_py_value()
            layout = fx.make_layout(tuple(shape[i] for i in indices), tuple(stride[i] for i in indices))
        else:
            layout = fx.make_layout(fx.select(value.shape, indices), fx.select(value.stride, indices))
        return fx.make_view(fx.get_iter(value), layout)
    return fx.select(value, indices)


def group(value, begin, end):
    if isinstance(value, fx.Tensor):
        if value.shape.is_static and value.stride.is_static:
            shape, stride = value.shape.to_py_value(), value.stride.to_py_value()
            grouped = lambda x: tuple(x[:begin]) + (tuple(x[begin:end]),) + tuple(x[end:])
            return fx.make_view(fx.get_iter(value), fx.make_layout(grouped(shape), grouped(stride)))
        return fx.make_view(fx.get_iter(value), fx.make_layout(
            fx.group(value.shape, begin, end), fx.group(value.stride, begin, end)))
    return fx.group(value, begin, end)


def composition(value, tiler):
    if isinstance(value, fx.Tensor):
        return static_view(fx.make_view(fx.get_iter(value), fx.composition(value.layout, tiler)))
    return fx.composition(value, tiler)


def flat_divide(value, divisor):
    if isinstance(value, fx.Tensor):
        return static_view(fx.make_view(fx.get_iter(value), fx.flat_divide(value.layout, divisor)))
    return fx.flat_divide(value, divisor)


def rmem(shape, dtype):
    """Use an explicit compact layout for multidimensional register tensors."""
    def compact(value, stride=1):
        if isinstance(value, (tuple, list)):
            strides = []
            for dimension in value:
                child, stride = compact(dimension, stride)
                strides.append(child)
            return tuple(strides), stride
        return stride, stride * value

    if isinstance(shape, (tuple, list)):
        shape = fx.make_layout(shape, compact(shape)[0])
    return fx.make_rmem_tensor(shape, dtype)


def resource(tensor, size_bytes):
    """A bounded raw ptr<8> buffer descriptor, using stable ROCDL builders."""
    address = fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), address.ir_value())
    return rocdl.make_buffer_rsrc(
        ir.Type.parse("!llvm.ptr<8>"), pointer, fx.Int16(0).ir_value(),
        fx.Int64(size_bytes).ir_value(), fx.Int32(0x27000).ir_value(),
    )


def wait(*, vmcnt=63, expcnt=7, lgkmcnt=63):
    """gfx94x/gfx950 s_waitcnt encoding; omitted counters are not waited on."""
    rocdl.s_waitcnt((vmcnt & 15) | (expcnt << 4) | (lgkmcnt << 8) | ((vmcnt >> 4) << 14))