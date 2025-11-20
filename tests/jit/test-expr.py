

class VarExpr:
    var_id = 0
    @classmethod
    def var(cls, count, dtype:str, reg):
        self = cls("var")
        self.count = count
        self.dtype = dtype
        self.reg = reg
        self.index = cls.var_id
        cls.var_id += count
        return self

    def __init__(self, op:str, vsrc0=None, vsrc1=None, extra=None):
        self.op=op
        self.vsrc0 = vsrc0
        self.vsrc1 = vsrc1
        self.extra = extra

    def __repr__(self):
        if self.op == "var":
            return f"({self.reg}.{self.dtype}[{self.index}:{self.index + self.count}])"
        elif self.vsrc1 is not None:
            return f"({self.vsrc0}{self.op}{self.vsrc1})"
        elif self.op == "getitem":
            return f"({self.vsrc0}[{self.extra}])"
        else:
            return f"({self.op}{self.vsrc0})"
    def __setitem__(self, key, value):
        print(f"设置键值对: {key} ({type(key)}) = {value}")
        assert 0
    def __delitem__(self, key):
        assert 0
    def __getitem__(self, key):
        assert self.op=="var"
        return VarExpr("getitem", self, extra=key)
    def __add__(self, other):
        return VarExpr("+", self, other)
    def __sub__(self, other):
        return VarExpr("-", self, other)
    def __mul__(self, other):
        return VarExpr("*", self, other)
    def __lshift__(self, other):
        return VarExpr("<<", self, other)
    def __rshift__(self, other):
        return VarExpr(">>", self, other)

    def __truediv__(self, other):
        assert 0
    def __floordiv__(self, other):
        return VarExpr("//", self, other)
    def __mod__(self, other):
        return VarExpr("%", self, other)
    def __pow__(self, other, modulo=None):
        assert 0

p = VarExpr.var(16, "I32", "v")
v = VarExpr.var(16, "I32", "v")
print(p[0:3])
print(p[0]+v[2]%8)
print(p[0]+v[2]<<3>>2)


from contextlib import contextmanager


class SimpleContext:
    def __init__(self, name):
        self.name = name
        pass

    """最简单的上下文管理器类"""
    def __enter__(self):
        print("进入上下文")
        return SimpleContext("else")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("退出上下文")
        return False  # 不抑制异常

class SimpleContextElse:
    """最简单的上下文管理器类"""
    def __enter__(self):
        print("进入上下文")
        return "资源对象"
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("退出上下文")
        return False  # 不抑制异常

with simple_context() as resource:
    print(f"在上下文中使用: {resource}")
    print("执行一些操作")
