class Expr:
    def __init__(self, op:str, src0=None, src1=None, extra=None):
        self.op=op
        self.src0 = src0
        self.src1 = src1
        self.extra = extra
        self.depth = 1
        if isinstance(src0, Expr):
            self.depth = max(self.depth, src0.depth + 1)
        if isinstance(src1, Expr):
            self.depth = max(self.depth, src1.depth + 1)

    def __add__(self, other):
        return Expr("+", self, other)
    def __sub__(self, other):
        return Expr("-", self, other)
    def __mul__(self, other):
        return Expr("*", self, other)
    def __lshift__(self, other):
        return Expr("<<", self, other)
    def __rshift__(self, other):
        return Expr(">>", self, other)

    def match(self, other: 'Expr'):
        if self.op != other.op:
            return False
        if (self.src0 is not other.src0) and (not self.src0.match(other.src0)):
            return False
        if (self.src1 is not other.src1) and (not self.src1.match(other.src1)):
            return False
        return True


a=Expr("", "s1")
b=Expr("", "s2")
c=Expr("", "s3")
d=Expr("", "v3")

class PatternNode:
    def match(self, other):
        self.other = other
        return other.startswith("s")

k0 = Expr("", PatternNode())
k1 = Expr("", PatternNode())
k2 = Expr("", PatternNode())

t1 = a<<b + c
t2 = a<<b - c
t3 = a<<b + d
p1 = (k0<<k1 + k2)

print(t1.depth)
print(p1.match(t1))
print(p1.match(t2))
print(p1.match(t3))

p1.match(t1)

print(k0.src0.other, k1.src0.other, k2.src0.other)

Expr.a = 123
Expr.b = 2354

print(Expr.a, Expr.b)
# 
# (a<<b + c)
