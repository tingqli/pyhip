import inspect
from typing import Union

# https://arxiv.org/pdf/2603.02298

# 2.1 Tuples and HTuples
# HTuple describes the hierarchical structure of a tuple, which is used to represent the profile of a layout.
class HTuple:
    # this creation process will recursively convert nested tuples/lists/ints into HTuple, 
    # and also compute the size attribute for each node in the process
    @classmethod
    def get_dtype(cls, h):
        if isinstance(h, HTuple):
            return h.dtype
        if isinstance(h, (tuple,list)):
            return cls.get_dtype(h[0])
        else:
            return type(h)
    @classmethod
    def create(cls, h, dtype=int):
        if dtype is None:
            dtype = cls.get_dtype(h)
        if isinstance(h, dtype):
            return cls(h, dtype=dtype)
        elif isinstance(h, (tuple,list)):
            return cls(*h, dtype=dtype)
        elif isinstance(h, HTuple):
            return h
        else:
            raise ValueError("Invalid input type for HTuple.create")

    def __init__(self, *args, dtype=int): # dtype is element type
        self.dtype = dtype
        if len(args) == 1 and isinstance(args[0], dtype):
            self.is_leaf = True
            self.children = [args[0]]
        else:
            self.is_leaf = False
            self.children = [HTuple.create(arg, dtype=dtype) for arg in args]

        self.depth = self._depth()
        self.rank = len(self.children)

        # add size attribute recursively to each Htuple node (for Shape)
        def assign_size(h):
            if h.is_leaf:
                h.size = h.children[0]
            else:
                size = 1
                for child in h.children:
                    size *= assign_size(child)
                h.size = size
            return h.size

        self.size = assign_size(self)
    
    def get_leaf(self):
        assert self.is_leaf
        return self.children[0]

    def _depth(self):
        if self.is_leaf:
            return 0
        else:
            # every pair of parentheses adds 1 to the depth, so we take the max depth of children and add 1
            return 1 + max(child._depth() for child in self.children)

    def __getitem__(self, key):
        return self.children[key]
    
    def __str__(self):
        return self.__repr__() + f" [depth={self.depth}, rank={self.rank}]"

    def __repr__(self):
        if self.is_leaf:
            ret = f"{self.children[0]}"
        else:
            ret = "(" + ", ".join(repr(c) for c in self.children) + ")"
        return ret

    def __eq__(self, other):
        if self.is_leaf and other.is_leaf:
            return self.children[0] == other.children[0]
        elif not self.is_leaf and not other.is_leaf:
            return self.rank == other.rank and all(p == s for p, s in zip(self.children, other.children))
        else:
            return False

    def Coarsens(self, S):
        return Coarsens(self, S) # defined later
    def Refines(self, P):
        return Refines(self, P) # defined later

# Shape & Stride are just aliases of HTuple with different semantics.
#  Shape describes the hierarchical structure of a layout
#  while Stride describes the memory layout of a tensor.
# They are related by the fact that the stride of a tensor must be compatible with its shape.
Shape = HTuple
Stride = HTuple

# the tree structure of P and S must be the same
def Congruence(P, S):
    if P.is_leaf and S.is_leaf:
        return True
    elif not P.is_leaf and not S.is_leaf:
        if P.rank != S.rank:
            return False
        return all(Congruence(p, s) for p, s in zip(P.children, S.children))
    else:
        return False

# weak congruence tests that the profile of S is at least as refined as P.
# or equivalently, the tree structure of S must be the same as P or more refined.
def WeakCongruence(P, S):
    if P.is_leaf:
        return True
    elif not P.is_leaf and not S.is_leaf:
        if P.rank != S.rank:
            return False
        return all(WeakCongruence(p, s) for p, s in zip(P.children, S.children))
    else:
        return False

def flatten(htuple):
    if htuple.is_leaf:
        return [htuple.get_leaf()]
    else:
        result = []
        for child in htuple.children:
            result.extend(flatten(child))
        return result

assert flatten(HTuple(1, ((2,3),4), 5)) == [1, 2, 3, 4, 5]

# 2.2 Shape
# 2.2.1 Coordinate Sets and Compatibility
# CUTE makes the observation that a 2D shape (M,N) can also be interpreted to describe 1D MN elements indexed
# by an integral coordinate i with 0 ≤ i < MN provided a bijection
# Coarsens is combination of tree structure WeakCongruence with size-compatible
def Coarsens(P, S):
    if P.is_leaf and P.size == S.size:
        return True
    elif not P.is_leaf and not S.is_leaf:
        if P.rank != S.rank:
            return False
        return all(Coarsens(p, s) for p, s in zip(P.children, S.children))
    else:
        return False
def Refines(S, P):
    return Coarsens(P, S)

def Compatible(P, S):
    return Refines(P, S) or Coarsens(P, S)

def test_htuple():
    h = HTuple(1)
    assert str(h) == "1 [depth=0, rank=1]"
    h = HTuple(1, 2)
    assert str(h) == "(1, 2) [depth=1, rank=2]"
    h = HTuple(1, 2, (2,3))
    assert str(h) == "(1, 2, (2, 3)) [depth=2, rank=3]"
    h = HTuple(1, ((2,3),))
    assert str(h) == "(1, ((2, 3))) [depth=3, rank=2]"

    assert Congruence(HTuple(1), HTuple(2)) == True
    assert Congruence(HTuple(1, 2), HTuple(3, 4)) == True
    assert Congruence(HTuple(1, 2, (2,3)), HTuple(4, 5, (6,7))) == True
    assert Congruence(HTuple(1, ((2,3))), HTuple(4, ((6,7)))) == True
    assert Congruence(HTuple(1), HTuple((2,))) == False
    assert Congruence(HTuple(1), HTuple(2,4)) == False

    assert WeakCongruence(HTuple(30), HTuple(2,4)) == True
    assert WeakCongruence(HTuple(2,4), HTuple(2,(4,5))) == True
    assert WeakCongruence(HTuple(30), HTuple(2,4,5)) == True
    assert WeakCongruence(HTuple(2,4,5), HTuple((0,0),0,0)) == True


# A shape S defines a set of compatible coordinate sets, Z(S)
# if shape P coarsens shape S,  then Z(P) ⊆ Z(S).
# S with more refined structure accept all coordinates of P, while having more styles of coordinates

def test_shape():
    assert(Shape(2).size == 2)
    assert(Shape(2,3).size == 6)
    assert(Shape(2,(3,4)).size == 24)

    assert Coarsens(Shape(2), Shape(2)) == True
    assert Coarsens(Shape(2), Shape(3)) == False
    assert Coarsens(Shape(2,3), Shape(2,3)) == True
    assert Coarsens(Shape(2,3), Shape(2,(3,4))) == False
    assert Coarsens(Shape(2,(3,4)), Shape(2,(3,4))) == True
    assert Coarsens(Shape(2,(3,4)), Shape(2,3)) == False

    assert Coarsens(Shape(30), Shape(2,15)) == True
    assert Coarsens(Shape(2,15), Shape(2,(3,5))) == True
    assert Coarsens(Shape(30), Shape(6,5)) == True
    assert Coarsens(Shape(6,5), Shape((2,3),5)) == True

    assert Compatible(Shape(2,(3,5)), Shape((2,3),5)) == False

# 2.2.2 Coordinates
# Definition 2.9. An in-bounds coordinate, or simply coordinate, into a shape S is an element of one of its coordinate
# sets, c ∈ ZS′ ∈ Z(S). Note that a coordinate is always an HTuple(N). When intention is clear, we will simply write
# c ∈ Z(S).
# Definition 2.10. An integral coordinate into a shape S is a coordinate c ∈ Z|S| ∈ Z(S). Note that an integral
# coordinate is always an integer, c ∈ N.
# Definition 2.11. A natural coordinate into a shape S is a coordinate c ∈ ZS ∈ Z(S). Note that a natural coordinate
# is always an HTuple(N) that is congruent to the shape, c ∼ S.

# The colexicographical enumeration defines a bijection on coordinate lists.

# idx2crd is defined for flattened shapes only in the paper. 
# here we recursively apply them to support hierarchical shapes as well, as long as the coordinate is congruent to the shape.
def idx2crd(idx: int, shape: HTuple):
    if shape.is_leaf:
        return idx
    else:
        crd = []
        for child in shape.children[0:-1]:
            # flatten shape only
            #crd.append(idx % child.size) 
            # recursively apply idx2crd to support hierarchical shapes
            crd.append(idx2crd(idx % child.size, child)) 
            idx //= child.size
        crd.append(idx2crd(idx, shape.children[-1]))
        return tuple(crd)

# crd2idx is defined for flattened coordinates only in the paper.
# here we recursively apply them to support hierarchical shapes as well, as long as the coordinate is congruent to the shape.
def crd2idx(crd:Union[HTuple, tuple, list], shape:HTuple):
    if not isinstance(crd, HTuple):
        crd = HTuple(*crd)

    if shape.is_leaf or crd.is_leaf:
        return crd.get_leaf()
    else:
        # assert Congruence(crd, shape)
        idx = 0
        size = 1
        for i in range(shape.rank):
            if crd[i].is_leaf:
                idx += crd[i].get_leaf() * size
            else:
                idx += crd2idx(crd[i], shape[i]) * size
            size *= shape[i].size
        return idx

def is_natural_coordinate(crd:HTuple, shape:HTuple):
    return Congruence(crd, shape)
def is_admissible_coordinate(crd:HTuple, shape:HTuple):
    return WeakCongruence(crd, shape)


def test_coordinate():
    assert idx2crd(5, Shape(2,3)) == (1,2)
    assert idx2crd(5, Shape(6)) == (5)
    assert crd2idx(HTuple(1,2), Shape(2,3)) == 5
    assert crd2idx(HTuple(5), Shape(6)) == 5

    print(idx2crd(7, Shape(2,12)))
    print(idx2crd(7, Shape(2,(3,4))))

    # Out-of-bounds Coordinates
    print(idx2crd(2*12, Shape(2,12)))
    print(idx2crd(2*12, Shape(2,(3,4))))

    print(crd2idx((1,(0,0)), Shape(2,(3,4))))
    print(crd2idx((0,(1,0)), Shape(2,(3,4))))
    print(crd2idx((0,(0,1)), Shape(2,(3,4))))
    print(crd2idx((0,11), Shape(2,(3,4))))

    shape = Shape(2,(3,4))
    for i in range(shape.size):
        crd = idx2crd(i, shape)
        idx = crd2idx(crd, shape)
        print(f"idx={i}, crd={crd}, idx2={idx}")
        assert idx == i

    shape1 = Shape(12)
    shape2 = Shape(6, 2)
    shape3 = Shape((2,3), 2)
    for i in range(12):
        crd1 = idx2crd(i, shape1)
        crd2 = idx2crd(i, shape2)
        crd3 = idx2crd(i, shape3)
        print(f"idx={i}, crd1={crd1}, crd2={crd2}, crd3={crd3}")
        expect2 = (i % 6, i // 6)
        expect3 = (((i % 6) % 2, (i % 6) // 2), i // 6)
        assert crd1 == i
        assert crd2 == expect2, f"{crd2} != {expect2}"
        assert crd3 == expect3, f"{crd3} != {expect3}"

# 2.3 Stride
# Definition2.15. A stride D for a shape S is an HTuple(D) that is congruent with the shape, S∼D. This stride
# defines a mapping from a natural coordinate c ∈ ZS to the codomain D, given by
"""
2.3.1 Integer-Semimodules
This section explores the most basic need for the stride concept and the applicability boundaries of the entire layout concept.

Stride is essentially a "generator" of an integer semimodule, and the index → address computation is a linear combination in that semimodule

- The elements m, n, p ∈ M in a stride need not be integers; they can be vectors or other stuff.
- They only need to satisfy the semimodule linear combination relations with scalar coordinates from the shape.
- The addition and multiplication used in the linear combination need not be the conventional ones, as long as they satisfy:
    1. Multiplicative identity : 1·m = m
    2. Additive associativity : m+(n+p) = (m+n)+p
    3. Multiplicative associativity : a · (b · m) = (ab) · m

 M is the codomain of the Layout function. The linear-combination of natural-coordinates with strides yields
 the output y ∈ M of the Layout function.

Section 2.4.4, "Semi-Linearity," further expresses the stride form in matrix form as a linear form.
"""

class IntegerSemimodule_ZXORm:
    def __init__(self, v): self.v = v
    def __add__(self, other):
        v = other if isinstance(other, int) else other.v
        return IntegerSemimodule_ZXORm(self.v ^ v)
    def __sub__(self, other):
        v = -other if isinstance(other, int) else -other.v
        return IntegerSemimodule_ZXORm(self.v ^ v)        
    def __mul__(self, scalar): return IntegerSemimodule_ZXORm(self.v * scalar)
    def __rmul__(self, scalar): return IntegerSemimodule_ZXORm(self.v * scalar)
    def __abs__(self): return IntegerSemimodule_ZXORm(abs(self.v))
    def __repr__(self): return f"{self.v}_zxor."

class IntegerSemimodule_Vector:
    def __init__(self, *values):
        self.v = values
        self.ndims = len(values)

    def size(self):
        sz = 1
        for a in self.v:
            sz *= abs(a)
        return sz

    def __add__(self, other):
        assert isinstance(other, (int, IntegerSemimodule_Vector)), other
        if isinstance(other, int):
            v = []
            for a in self.v:
                v.append(a + other)
            return IntegerSemimodule_Vector(*v)

        assert self.ndims == other.ndims, "Vector must have the same number of dimensions to be added"
        v = []
        for a, b in zip(self.v, other.v):
            v.append(a + b)
        return IntegerSemimodule_Vector(*v)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, scalar):
        v = []
        for a in self.v:
            v.append(a * scalar)
        return IntegerSemimodule_Vector(*v)

    def __rmul__(self, scalar): return self.__mul__(scalar)

    def __abs__(self):
        v = []
        for a in self.v:
            v.append(abs(a))
        return IntegerSemimodule_Vector(*v)

    def __repr__(self):
        return "<" + ",".join([f"{a}" for a in self.v]) + ">"

ZXORm = IntegerSemimodule_ZXORm
Vector = IntegerSemimodule_Vector

def inner_product(crd:HTuple, stride:HTuple):
    if crd.is_leaf and stride.is_leaf:
        return crd.get_leaf() * stride.get_leaf()
    elif not crd.is_leaf and not stride.is_leaf:
        assert crd.rank == stride.rank
        offset = None
        for c, s in zip(crd.children, stride.children):
            off = inner_product(c, s)
            if offset is None:
                offset = off
            else:
                offset = offset + off
        return offset
    else:
        raise ValueError("crd and stride must have the same structure")

def coshape(layout):
    if layout.shape.is_leaf:
        c = layout.shape.get_leaf() - 1
        return c * abs(layout.stride.get_leaf()) + 1
    else:
        ret = layout(0)
        for i in range(layout.rank):
            ret += coshape(layout[i]) - 1
        return ret + 1

def cosize(codomain_shape):
    if isinstance(codomain_shape, (int, IntegerSemimodule_ZXORm)):
        return codomain_shape
    if isinstance(codomain_shape, IntegerSemimodule_Vector):
        return codomain_shape.size()
    raise ValueError("codomain_shape must be an integer, IntegerSemimodule_ZXORm, or IntegerSemimodule_Vector")

class Layout:
    def __init__(self, shape:Shape, stride:Stride):
        shape = Shape.create(shape)
        stride = Stride.create(stride, dtype=None)
        assert Congruence(shape, stride)
        self.shape = shape
        self.stride = stride
        self.rank = shape.rank
        self.depth = shape.depth
        self.size = shape.size
        # how to get codomain size?
        # it's the minimal size of surrounding box in codomain, which
        # can enclose all valid projections of layout() function
        # for example :
        #   for Layout (2,2):(1,-4), codomain is {0, 1, -4, -3}, so coshape == cosize == 4
        #   for Layout (2,2):(e0,-4*e1)
        #        codomain is {0*e0 + 0*e1,
        #                     1*e0 + 0*e1,
        #                     0*e0 - 4*e1,
        #                     1*e0 - 4*e1}, so coshape is (2*e0, 5*e1), and cosize is 10
        # 
        self.cosize = self(shape.size - 1) + 1
        self.coshape = coshape(self)
        self.cosize = cosize(self.coshape)

    def __call__(self, *crd):
        idx = crd2idx(crd, self.shape)
        natural_crd = idx2crd(idx, self.shape)                      # S : Z ↔ZS, ∀Z ∈Z(S) 
        if isinstance(natural_crd, int):
            addr = inner_product(HTuple(natural_crd), self.stride)     # D : ZS →D
        else:
            addr = inner_product(HTuple(*natural_crd), self.stride)     # D : ZS →D
        return addr

    def Coarsens(self, other):
        # self ⪯ other
        return Coarsens(self.shape, other.shape)

    def Refines(self, other):
        # other ⪯ self
        return Refines(self.shape, other.shape)

    def __getitem__(self, i):
        assert i < self.rank
        sub_shape = self.shape[i]
        sub_stride = self.stride[i]
        return Layout(sub_shape, sub_stride)

    def __str__(self):
        return f"Layout({repr(self.shape)}:{repr(self.stride)})"

    def __eq__(self, value):
        return self.shape == value.shape and self.stride == value.stride
    
    def show(self, stride_names = None, verbose=1):
        if stride_names is not None:
            # stride_names = "T4 V8"
            # reinterpret offset 6 according to shape (4, 8):
            #   T=6%4=2, V=6//4=1
            # so the name is T2V1
            names = []
            sizes = []
            for name_size in stride_names.split():
                names.append(name_size[0])
                sizes.append(int(name_size[1:]))
            last_coord = []
            def offset_name(offset, n):
                nonlocal last_coord
                if n == 0:
                    last_coord = [-1 for _ in range(len(sizes))]
                name_str = ""
                for i, (name, size) in enumerate(zip(names, sizes)):
                    cur_coord = offset % size
                    if last_coord[i] == cur_coord:
                        name_str += ".."
                    else:
                        name_str += f"{name}{offset%size}"
                    offset //= size
                    last_coord[i] = cur_coord
                return f"{name_str:>8s}"
        else:
            def offset_name(offset, n):
                return f"{offset:6d}"

        if verbose:
            caller_frame = inspect.stack()[1]
            if caller_frame.code_context:
                src_line = f"\033[32m{caller_frame.filename}:{caller_frame.lineno}\n" \
                + "  " + caller_frame.code_context[0].strip() + "\033[0m"
            else:
                src_line = "?"

            print(src_line)
        print(self, f"shape(size)->coshape(cosize) {repr(self.shape)}({self.size})->{repr(self.coshape)}({self.cosize}) rank:{self.rank} depth:{self.depth}")
        if self.rank != 2:
            for i in range(self.size):
                print(f"{offset_name(self(i),i)}", end=",")
                if i >= 16:
                    print(end=" ... ")
            print()
        if self.rank == 2:
            for m in range(self.shape[0].size):
                m_coord = idx2crd(m, self.shape[0])
                print(f"{m:>4d}", end="")
                if self.shape[0].depth > 0:
                    print(f"={m_coord}", end="")
                print(f": ", end="")
                for n in range(self.shape[1].size):
                    cur_name = offset_name(self(m, n), n)
                    print(f"{cur_name}", end=",")
                print()
            return self
        return self

if 0:
    layout = Layout((2,2), (1,4))
    print(layout, layout.cosize, layout.coshape, layout(3))
    layout = Layout((2,2), (1,-4))
    print(layout, layout.cosize, layout.coshape, layout(3))

    a = Layout((2,2), (ZXORm(1), ZXORm(5)))
    print(a, a(0,0), a(1,0), a(0,1), a(1,1))

    e0 = Vector(1, 0)
    e1 = Vector(0, 1)
    layout = Layout((2,2), (e0,-4*e1))
    print(layout, layout.cosize, layout.coshape, layout(3))

def concatenation(*layouts):
    new_shape = []
    new_stride = []
    for l in layouts:
        if isinstance(l, list) or isinstance(l, tuple):
            l = concatenation(*l)
        if l.shape.is_leaf:
            new_shape.append(l.shape.get_leaf())
            new_stride.append(l.stride.get_leaf())
        else:
            new_shape.append(l.shape.children)
            new_stride.append(l.stride.children)
    return Layout(Shape(*new_shape), Stride(*new_stride))

assert concatenation(Layout(4,3), Layout((2,2),(1,2))) == Layout((4,(2,2)), (3,(1,2)))
assert concatenation([Layout(1,2), Layout(3,4)], Layout((2,2),(1,2))) == Layout(((1, 3), (2, 2)),((2, 4), (1, 2)))

def test_layout():
    L = Layout(((2,2),(4,2)), ((1,8),(2,16)))
    assert L(22) == L(2,5) == L((0,1),(1,1)) == 26

    layout = Layout((2,3), (3,1))
    assert layout.cosize == 6
    for i in range(6):
        assert layout(i) == (i%2)*3 + (i//2)
    
    assert layout(0,0) == 0
    assert layout(0,1) == 1
    assert layout(0,2) == 2
    assert layout(1,0) == 3
    assert layout(1,1) == 4
    assert layout(1,2) == 5
    layout = Layout(((2,2),(4,2)), ((1,8),(2,16)))
    for n in range(8):
        n_off = (n%4)*2 + (n//4)*16
        assert layout(0, n) == 0 + n_off
        assert layout(1, n) == 1 + n_off
        assert layout(2, n) == 8 + n_off
        assert layout(3, n) == 9 + n_off

    # reproduce Figure4 (c) Binary Swizzle
    
    layout = Layout((4,(4,3)), (ZXORm(1),(ZXORm(5),ZXORm(16))))

    answer = ((0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, ),
              (1, 4, 11, 14, 17, 20, 27, 30, 33, 36, 43, 46, ),
              (2, 7, 8, 13, 18, 23, 24, 29, 34, 39, 40, 45, ),
              (3, 6, 9, 12, 19, 22, 25, 28, 35, 38, 41, 44, ))
    for m in range(4):
        print("(", end="")
        for n in range(12):
            offset = layout(m, n).v
            print(f"{offset}, ", end="")
            assert offset == answer[m][n], f"layout({m}, {n})={offset} != {answer[m][n]}"
        print("),")

# 2.4.4 Semi-Linearity
#  Layout is linear function/mapping when input is limited to natural coordinates
# 3.2 Coalesce
# https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#coalesce
def coalesce(layout, by_mode=False):
    # remove dimensions of size 1 no matter what the stride is
    if layout.shape.is_leaf: # already with minimal rank
        return layout
    
    # recursively coalesce sub-layouts
    sub_layouts = []
    for i in range(layout.rank):
        subl = layout[i]
        subl = coalesce(subl)
        if by_mode:
            sub_layouts.append(subl)
            continue
        # print(f"cur_subl={cur_subl} size={cur_subl.size}")
        if subl.size > 1:
            for k in range(subl.rank):
                cur_subl = subl[k]
                is_coalesced = False
                if len(sub_layouts) > 0:
                    # check if it can be coalesced with the previous sub-layout
                    prev_subl = sub_layouts[-1]
                    if prev_subl.shape.is_leaf and cur_subl.shape.is_leaf:
                        cur_stride = cur_subl.stride.get_leaf()
                        cur_shape = cur_subl.shape.get_leaf()
                        prev_stride = prev_subl.stride.get_leaf()
                        prev_shape = prev_subl.shape.get_leaf()
                        if cur_stride == prev_shape * prev_stride:
                            new_shape = Shape(prev_shape * cur_shape)
                            new_stride = Stride(prev_stride)
                            #print(f"Coalescing {prev_subl} and {cur_subl} into Layout({new_shape}:{new_stride})")
                            sub_layouts[-1] = Layout(new_shape, new_stride)
                            is_coalesced = True
                if not is_coalesced:
                    #print(f"add cur_subl={cur_subl} size={cur_subl.size}")
                    sub_layouts.append(cur_subl)
    return concatenation(*sub_layouts)

def test_coalesce():
    assert coalesce(Layout((2,3), (1,2))) == Layout(6, 1), f"Expected Layout(6, 1) but got {layout}"
    assert coalesce(Layout((2,3), (3,1))) == Layout((2,3), (3,1))
    assert coalesce(Layout(((4,3),5), ((15,1),3))) == Layout((4,15), (15,1)), f"Expected Layout((4,15), (15,1)) but got {layout}"
    assert coalesce(Layout(((4,3),5), ((15,1),3)), by_mode=True) == Layout(((4,3),5), ((15,1),3)), f"Expected Layout(((4,3),5), ((15,1),3)) but got {layout}"
    assert coalesce(Layout((4,(3,5)),(15,(1,3)))) == Layout((4,15),(15,1)), f"Expected Layout((4,15),(15,1)) but got {layout}"
    assert coalesce(Layout((4,(3,5)),(15,(1,3))), by_mode=True) == Layout((4,15),(15,1)), f"Expected Layout((4,15),(15,1)) but got {layout}"
    assert coalesce(Layout((2,(1,6)), (1,(6,2)))) == Layout(12, 1), f"Expected Layout(12, 1) but got {layout}"
    assert coalesce(Layout((2,(1,6)), (1,(6,2))), by_mode=True) == Layout((2,6), (1,2)), f"Expected Layout((2,6), (1,2)) but got {layout}"
    assert coalesce(Layout((3,(4,2)), (1,(3,12)))) == Layout(24,1)

"""
3.3 Composition `R = A o B`
 - `R(c) = A(B(c))`
 - B ⪯ R, 
   B Coarsens R, thus R accepts all coordinates of B:
    - for example, if shape of B is (4, (2,3)), shape of R maybe ((2,2),(2,3)) but not ((2,2),2,3)
   B determines the shape and coordinate sets of the resulting layout by defining the domain of R
 - A determines the codomain of R

Composition Properties:
 - Identity Layouts: ∀c ∈ ZS, IS(c) = c.
   - ID ◦ B = B 
   - A ◦ IS = A
   IS can be used to change the profile of A w/o changing the mapping
   for example: A=(64,64,16):(1024,16,1) IS=(256,256):(1,256) then (A o IS)
   has same mapping as A but only have 2-modes now

 - Associative Property: given image(C) ⊆ Z(B) and image(B) ⊆ Z(A),
   - A ◦ (B ◦ C) = (A ◦ B) ◦ C

methods in paper is complex, here we use intuition form shown here:
 - https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#computing-composition
 - 3.3.3 Intuition and Divisibility

consider inner_product(i, D) one item by one item, we have
   (i*d % S) * D === (i % S') * D'   for any i = 0,1,...,s-1
above equation can only hold when stride divisibility condition is satisfied:
   - d|S :  so S=d*Z (Z is positive integer)  (i*d % d*Z) * D === (i % Z)*(d*D)    so S' = Z = S//d, D' = d*D
            then according to recursive form of idx2crd, i = i*d // S = i//S', go to next mode, we can see
            (i//S' % S2)*D2 becomes exactly inner_product form w/o d in there. so for all rest modes, we 
            have S'== S, D'== D

   - S|d :  so d=Z*S (Z is positive integer)  (i*Z*S % S) * D === 0    no contribution to the offset
            then according to recursive form of idx2crd, i = i*d // S = i*Z, go to next mode, we can see
              (i*Z % S2)*D2 === （i % S2')*D2', for any i =0,1,...,s-1 
            so we have same problem with d becomes Z=d//S

  - last item of idx2crd has no ( % SR) to allow extension, so there is no stride divisibility requirement
        (i*d) * D === (i) * D'   for any i = 0,1,...,s-1
     we just set D'=d*D
"""
def verify_composition(R, A, B):
    assert B.shape.Coarsens(R.shape)
    for i in range(R.size):
        off = R(i)
        ref = A(B(i))
        assert off == ref, f" {R} = {A} o {B} failed R({i})={off}  A(B({i}))={ref}"

def composition_1d(S, D, s, d0):
    # there is no nested structure in S or D, all flattend
    _Sr = 1
    div_up = lambda x, y: (x + y - 1) // y
    S2 = []
    D2 = []
    d = d0
    for r, (Sr,Dr) in enumerate(zip(S[:-1], D[:-1])):
        if Sr >= d:
            assert Sr % d == 0, f"stride divisibility condition voilated S{r}={Sr} and d={d}"
            if Sr > d:
                # skip Sr//d=1 case to avoid extra dimension of size 1
                S2.append(Sr//d)
                D2.append(Dr*d)
            d = 1 # rest modes will be copied
        else:
            assert d % Sr == 0, f"stride divisibility condition voilated S{r}={Sr} and d={d}"
            # current mode has no contribution to the offset， but it makes d smaller
            d = d // Sr
    if d == 0:
        S2.append(S[-1])
        D2.append(0)
    else:
        S2.append(S[-1]//d)
        D2.append(D[-1]*d)

    # now make shape compatible with s
    sz = 1
    for i in range(len(S2)-1):
        if sz >= s:
            S2[i] = 1
        elif sz * S2[i] > s:
            assert s % sz == 0, f"shape divisibility condition voilated s={s} and sz={sz}"
            S2[i] = s // sz
        sz *= S2[i]

    # last item is special to allow extension
    if sz >= s:
        S2[-1] = 1
    else:
        assert s % sz == 0, f"shape divisibility condition voilated s={s} and sz={sz}"
        S2[-1] = s // sz

    # remove extra dimensions of size 1
    max_stride = max(D2)
    S2 = [s for s in S2 if s > 1]
    D2 = D2[:len(S2)]
    if len(S2) == 0:
        # all dimensions are coalesced, we need to keep
        # one dimension to hold the stride information
        S2 = [1]
        D2 = [max_stride]
    return S2, D2

"""
B is a Tiler type which can be:
 - int : as stride-1 layout
 - layout
 - tuple of Tilers : apply composition mode by mode
"""
def composition(A, B, verify=False):
    if isinstance(B, int):
        B = Layout(B, 1)

    if isinstance(B, Layout):
        S = flatten(A.shape)
        D = flatten(A.stride)

        # following recursion keeps the profile of B in the result.
        # thus composition keeps the domain profile of input B
        def _composition(sub_B):
            if sub_B.shape.is_leaf:
                s = sub_B.shape.get_leaf()
                d = sub_B.stride.get_leaf()
                return composition_1d(S, D, s, d)
            else:
                sub_RS = []
                sub_RD = []
                for i in range(sub_B.rank):
                    S2, D2 = _composition(sub_B[i])
                    if len(S2) == 1:
                        sub_RS.extend(S2)
                        sub_RD.extend(D2)
                    else:
                        sub_RS.append(S2)
                        sub_RD.append(D2)
                return sub_RS, sub_RD
        RS, RD = _composition(B)
        R = Layout(RS, RD)
        # print("\t", A, " o " B, " = ", R)
        if verify : verify_composition(R, A, B)
        return R
    else:
        assert isinstance(B, tuple) or isinstance(B, list)
        # B is not a leaf(Tiler), do composition mode by mode
        rank = min(A.rank, len(B))
        RS=[]
        RD=[]
        for i in range(rank):
            rsub = composition(A[i], B[i])
            RS.append(rsub.shape.children)
            RD.append(rsub.stride.children)
        return Layout(RS, RD)

def test_composition():
    assert composition(Layout((6,2),(8,2)), Layout((4,3),(3,1))) == Layout(((2, 2), 3),((24, 2), 8))
    assert composition(Layout(20,2), Layout((5,4),(4,1))) == Layout((5,4),(8,2))
    assert composition(Layout((10,2),(16,4)), Layout((5,4),(1,5))) == Layout((5, (2, 2)),(16, (80, 4)))

    assert composition(Layout((5, 3), (1, 7)),Layout(2, 5)) == Layout(2,7)
    assert composition(Layout((5, 3), (1, 7)),Layout(4, 1)) == Layout(4,1)
    assert composition(Layout(4,1),Layout(2, 5)) == Layout(2,5)

    assert composition(Layout((5, 3), (8, 7)),(Layout(2, 5), Layout(4, 1))) == Layout((2, 4),(40, 7))

"""
3.5 Complement : fill the dimensional gaps (by adding the missing replications one-by-one)

complement `A∗`
R = complement(A, shapeM)

 - size(R) and cosize(R) ≤ size(M).
 - R has positive, increasing strides → unique.
 - A and R operate on disjoint mode sets. R completes the missing modes.

R fills the dimensional gaps not covered by A, following the coalescing assumption.

for example, 30=2x3x5, Possible layouts:
 - (2,3,5):(1,2,6)
 - (3,2,5):(1,3,5)
 - (2,5,3):(1,2,10)
 - ......

complement = `any split of modes into two complementary groups.`

the algorithm is based on the fact that when complement modes are concatenated with the original one,
and sorted strides in ascending order, all modes can coalesce into a single mode.
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#complement-examples
"""
def complement(layout, cosize):
    shape = flatten(layout.shape)
    stride = flatten(layout.stride)
    pairs = sorted(zip(shape, stride), key=lambda x: x[1])
    new_shape = []
    new_strides = []
    next_stride = 1
    for i, (s, d) in enumerate(pairs):
        if d == 0:
            continue
        if d == next_stride:
            next_stride *= s
            continue
        d2 = next_stride
        s2 = d // next_stride
        new_strides.append(d2)
        new_shape.append(s2)
        next_stride = d*s

    if next_stride < cosize:
        new_strides.append(next_stride)
        new_shape.append(cosize // next_stride)

    if len(new_shape) == 0:
        new_shape = [1]
        new_strides = [0]

    return Layout(new_shape, new_strides)

def test_complement():
    assert complement(Layout(4,1), 24) == Layout(6, 4)
    assert complement(Layout(6,4), 24) == Layout(4, 1)
    assert complement(Layout((4,6),(1,4)), 24) == Layout(1, 0)
    assert complement(Layout(4,2), 24) == Layout((2,3), (1,8))
    assert complement(Layout((2,4),(1,6)), 24) == Layout(3, 2)
    assert complement(Layout((2,2),(1,6)), 24) == Layout((3,2), (2,12))
    assert complement(Layout((2,2),(1,2)), 24) == Layout(6,4)


"""
composition & complement are two fundamental layout algebra, internally they are actually profile-agnostic.
"""


"""
division `A⊘B = A o (B, B∗|A|) = A o B⋆`

Given layouts A and B, their division produces a layout R such that A is split into modes:
 - first mode : elements inside each tile pointed to by B  : `A o B`
 - second mode : B-tiles pointed to by B*                  : `A o B*`

After division, size of A is unchanged, cosize is also unchanged(in most cases), 
the only changed is the way we traversal/indexing the elements of A.

tiler_mode controls how top-level tiler results are rearranged with the rest of the modes (the L,... part) in the output layout R.

 Layout Shape : (M, N, L, ...)
 Tiler Shape  : <TileM, TileN>
 logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
 zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
 tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
 flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
"""
def rearrange_layouts(by_mode_layouts, passthrough_layouts, method=None):
    if method is None or method == "logical":
        return concatenation(*by_mode_layouts, *passthrough_layouts)
    if method == "zipped":
        tile_layouts = []
        rest_layouts = []
        for l in by_mode_layouts:
            tile_layouts.append(l[0])
            rest_layouts.append(l[1])
        rest_layouts.extend(passthrough_layouts)
        return concatenation(tile_layouts, rest_layouts)
    if method == "tiled":
        tile_layouts = []
        rest_layouts = []
        for l in by_mode_layouts:
            tile_layouts.append(l[0])
            rest_layouts.append(l[1])
        return concatenation(tile_layouts, *rest_layouts, *passthrough_layouts)
    if method == "flat":
        tile_layouts = []
        rest_layouts = []
        for l in by_mode_layouts:
            tile_layouts.append(l[0])
            rest_layouts.append(l[1])
        return concatenation(*tile_layouts, *rest_layouts, *passthrough_layouts)
    raise ValueError(f"Unsupported rearrange method {method}")

def logical_divide(A, B, tiler_mode=None):
    if isinstance(B, int):
        B = Layout(B, 1)
    if isinstance(B, Layout):
        return composition(A, concatenation(B, complement(B, A.size)), verify=False)
    elif B is None:
        return A
    else:
        assert isinstance(B, tuple) or isinstance(B, list), f"Tiler must be either int, Layout, or tuple/list of Tilers, got {type(B)}"
        div_layouts = []
        keep_layouts = []
        for i in range(A.rank):
            if i < len(B):
                # (Tile, Rest)
                div_layouts.append(logical_divide(A[i], B[i]))
            else:
                # no tiler provided for this mode, keep it as is (the L,... part)
                keep_layouts.append(A[i])
        return rearrange_layouts(div_layouts, keep_layouts, method=tiler_mode)

def zipped_divide(A, B): return logical_divide(A, B, "zipped")
def tiled_divide(A, B): return logical_divide(A, B, "tiled")
def flat_divide(A, B): return logical_divide(A, B, "flat")

def test_div():
    assert logical_divide(Layout((4,2,3),(2,1,8)), Layout(4,2)) == Layout(((2, 2), (2, 3)),((4, 1), (2, 8)))
    assert logical_divide(Layout((9,(4,8)),(59,(13,1))), [Layout(3,3), Layout((2,4),(1,8))]) == Layout(((3, 3), ((2, 4), (2, 2))), ((177, 59), ((13, 2), (26, 1))))

"""
product `R = A ⊗ B = (A, A∗ o B)`

The logical product of two layouts A and B is a layout R where “each element of layout B has been replaced with a
uniquely shifted version of the layout A.”
 - first mode is the layout A
 - the second mode is the layout B composed with A*

 concat 2-modes together, each element in second mode referencing “unique replication” of layout A.


Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_product : ((M,TileM), (N,TileN), L, ...)
zipped_product  : ((M,N), (TileM,TileN,L,...))
tiled_product   : ((M,N), TileM, TileN, L, ...)
flat_product    : (M, N, TileM, TileN, L, ...)

"""

def logical_product(A, B, tiler_mode=None):
    if isinstance(B, int):
        B = Layout(B, 1)
    if isinstance(B, Layout):
        return concatenation(A, composition(complement(A, A.size * B.cosize), B))
    else:
        assert isinstance(B, tuple) or isinstance(B, list), "Tiler must be either int, Layout, or tuple/list of Tilers"
        prod_layouts = []
        keep_layouts = []
        for i in range(A.rank):
            if i < len(B):
                # (Tile, Rest)
                prod_layouts.append(logical_product(A[i], B[i]))
            else:
                # no tiler provided for this mode, keep it as is (the L,... part)
                keep_layouts.append(A[i])
        return rearrange_layouts(prod_layouts, keep_layouts, method=tiler_mode)

def test_product():
    assert logical_product(Layout((2,2),(4,1)),  Layout(6,1)) == Layout(((2,2),(2,3)),((4,1),(2,8)))
    assert logical_product(Layout((2,2),(4,1)),  Layout((4,2),(2,1))) == Layout(((2,2),(4,2)),((4,1),(8,2)))
    assert logical_product(Layout((2,5),(5,1)), [Layout(3,5),Layout(4,6)]) == Layout(((2, 3), (5, 4)), ((5, 10), (1, 30)))

"""
# in result of logical_product(A, B)
# - A is always compatible with mode-0 of the result 
# - B is always compatible with mode-1 of the result
# The blocked_product(LayoutA, LayoutB) and raked_product(LayoutA, LayoutB) are rank-sensitive transformations 
# on top of 1-D logical_product that let us express the more intuitive Layout products that we most often want
# to express.

Layout A Shape : (M, N)
Layout B Shape : (m, n) Note here B is a layout, not a Tiler

                         shape          size of each top-level modes
logical_product(A,B) : ((M, N), (m, n))   M*N, m*n
blocked_product(A,B) : ((M, m), (N, m))   M*m, N*n
raked_product(A,B)   : ((m, M), (n, N))   m*M, n*N
"""
def blocked_product(A, B):
    assert A.rank == B.rank
    result = logical_product(A, B)
    mode0 = result[0]
    mode1 = result[1]
    RS = []
    RD = []
    for i in range(A.rank):
        RS.append((mode0.shape[i], mode1.shape[i]))
        RD.append((mode0.stride[i], mode1.stride[i]))
    return coalesce(Layout(RS, RD), by_mode=True)

def raked_product(A, B):
    assert A.rank == B.rank
    result = logical_product(A, B)
    mode0 = result[0]
    mode1 = result[1]
    RS = []
    RD = []
    for i in range(A.rank):
        RS.append((mode1.shape[i], mode0.shape[i]))
        RD.append((mode1.stride[i], mode0.stride[i]))
    return coalesce(Layout(RS, RD), by_mode=True)

def test_product2():
    assert logical_product(Layout((2,5),(5,1)), Layout((3,4),(1,3))) == Layout(((2, 5), (3, 4)),((5, 1), (10, 30)))
    assert blocked_product(Layout((2,5),(5,1)), Layout((3,4),(1,3))) == Layout((6, (5, 4)),(5, (1, 30)))
    assert raked_product(Layout((2,5),(5,1)), Layout((3,4),(1,3))) == Layout(((3, 2), (4, 5)),((10, 5), (30, 1)))

    thr_layout = Layout((4,64),(64,1))
    val_layout = Layout((16,16),(16,1))
    R = raked_product(thr_layout, val_layout)
    # product(thr_layout, val_layout):
    #    - thr_layout is unchanged
    #    - val_layout repeats thr_layout 16x16 times
    # R((val_m, thr_m),(val_n, thr_n)) is the offset of element assigned to value (val_m, val_n) in thread (thr_m, thr_n)
    # R' = right_inverse(R)
    # R'(offset) is the 
    assert R == Layout(((16, 4), (16, 64)), ((4096, 64), (256, 1)))
    assert blocked_product(val_layout, thr_layout) == Layout(((16, 4), (16, 64)), ((16, 16384), (1, 256)))

"""
R is right inverse of L if:  ∀k ∈ Z|R|, L(R(k)) = k,

  L o R = Identity
  
  L o R = L o concat(R[0], R[1], ...)
        = concat(L o R[0], L o R[1], ...)
        = Identity

  suppose L is biject, R[0] extract stride-1 element, R[1] extract the next big stride one, and so on.
"""

def right_inverse(L, verify = True):
    S = flatten(L.shape)
    D = flatten(L.stride)
    RSD = []
    stride = 1
    for s,d in zip(S, D):
        RSD.append((s,d,stride))
        stride *= s

    RS = []
    RD = []
    for s,d,stride in sorted(RSD, key=lambda x: x[1]):
        RS.append(s)
        RD.append(stride)
    R = coalesce(Layout(RS, RD))

    if verify:
        C = coalesce(composition(L, R, verify=False))
        assert C.rank == 1 and C.depth == 0, f" L o R is unexpected: {C}"
        assert C.shape[0] == R.size and C.stride[0] == 1, f" L o R is unexpected: {C}"

    return R

def test_rinv():

    tv_layout = Layout(((2,4),(2,2)),((8,1),(4,16)))
    #tv_layout.show()
    assert right_inverse(tv_layout) == Layout((8, 2, 2),(2, 1, 16))

    tv_layout = Layout(((64,4),(16,16)),((1024,16),(64,1)))
    assert right_inverse(tv_layout) == Layout((16,64,64),(4096,64,1))


def test_tv_layout():
    # inverse TV-layout is more natural to understand & construct, but TV-layout is what we actually want to use in the code

    # (T8,V4) -> (M4,N8)
    tv_layout = Layout(((2,4),(2,2)),((8,1),(4,16)))
    tv_layout.show("m4 ,8") # tv_layout is not good for understanding

    # right-inverse can not keep the profile, that's why we always see a composition following it
    # to get a real inverse with correct profile, we need to do composition with a Identity layout
    # to group modes correctly:
    rinv = composition(right_inverse(tv_layout), Layout((4,8),(1,4)))
    rinv.show("T8 .4") # inverse of tv_layout is good for understanding if we know target shape

    # most natural way to get tv_layout is to construct Inverse TV Layout, then call make_layout_tv()
    # Inverse TV Layout: (M4,N8) -> (T8,V4)
    """
    Layout A Shape : (M, N)
    Layout B Shape : (m, n) Note here B is a layout, not a Tiler

                            shape          size of each top-level modes
    logical_product(A,B) : ((M, N), (m, n))   M*N, m*n
    blocked_product(A,B) : ((M, m), (N, m))   M*m, N*n
    raked_product(A,B)   : ((m, M), (n, N))   m*M, n*N

    """
    thr_layout = Layout((4,2),(2,1)).show("T8")
    val_layout = Layout((2,4),(4,1)).show("V8")

    # each thread holds a 2x4 rect of values now
    mn_layout_tv = raked_product(thr_layout, val_layout).show("T8 v8")

    # layout_mn: (M64, N256) -> (T256, V64)
    """
    mn_layout_tv: Layout(((16, 4), (4, 64)):((1024, 64), (256, 1)))

    offset will be interpret as 1D index of shape (T256, V64)
    so:
    tid = (offset % 64)
    wid = (offset // 64) % 4
    vid = (offset // 256)

    or:
        offset = tid + wid*64 + vid*256

        so if we add offset by 1, we are increasing TID, the stride is T
        if we add offset by 64, we are increasing WID, the stride is W
        if we add offset by 256, we are increasing VID, the stride is V

    let's rewrite ((16, 4), (4, 64)):((1024, 64), (256, 1)):

        ((16, 4), (4, 64)):
        ((4V, 64T), (V, T))

        codomain = ("T", 256, "V", 256)
        
    thr_layout = Layout((4,64),(64,1)) = (4, 64):(64T, 1T) maps (M4, N64) -> T256
    val_layout = Layout((16,4),(4,1))  = (16, 4):( 4V, 1V) maps (M16, N4) -> V64
    """
    thr_size = thr_layout.size
    val_size = val_layout.size
    tmp = Layout((thr_size,val_size),(1,thr_size))
    tv_layout = composition(right_inverse(mn_layout_tv), tmp).show("M8 n8")

    thr_layout = Layout((4,2),(2,1)).show("T8")
    val_layout = Layout((1,2),(2,1)).show("V8")

    # each thread holds a 2x4 rect of values now
    mn_layout_tv = raked_product(thr_layout, val_layout).show("T8 v8")
    
    tv_layout = Layout(((2,4),(2,2)),((8,1),(4,16))).show("M4 n8")
    mn = Layout((4, 8),(1, 4))
    composition(right_inverse(tv_layout), mn).show("T8 v4")

# test_tv_layout()

if 0:
    N = 256
    K = 128
    preshuffled_B = Layout(((16, N // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512)))
    print(preshuffled_B)

    M32 = 1
    N64 = 32
    tv_layout = Layout(((8, 8, 4), 8), ((8*N64, 1, 8), N64)).show("M32 .64")
    mn = Layout((32, 64),(1, 32))
    composition(right_inverse(tv_layout), mn).show("T256 .8")

    Layout((3,2),(2,12)).show()
                                                #   fx.make_tile(8 * 4, TILE_K)

    print(flat_divide(Layout((512,8192),(8192,1)), [256]))

    at_block = Layout((128,64,128),(8192,1,64)) # (m128, k64, num_TileK) -> offsets
    tv_layout = Layout(((8, 8, 4), 8), ((256, 1, 8), 32)) # (T256, v8) -> (M32, n64)
    at_subtile = tiled_divide(at_block, [32, 64]) # Layout(((32, 64), 4, 1, 128):((8192, 1), 262144, 0, 64))
    subtile_tv = composition(at_subtile[0], tv_layout)
    at_subtile_tv = concatenation(subtile_tv, *[at_subtile[k] for k in range(1, at_subtile.rank)])

    print(tv_layout)
    print(at_subtile)
    print(at_subtile_tv)

# mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))

"""
Tensor<Value(%191 = "fly.tiled_copy.partition_src"(%189, %111, %190) : 
(!fly.tiled_copy<!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>,
 !fly.layout<((8,8,4),8):((256,1,8),32)>,
 !fly.tile<[32|64]>>, 
 !fly.memref<bf16, #fly_rocdl.buffer_desc, (128,64,128):(8192,1,64)>, !fly.int_tuple<?>) -> 
 !fly.memref<bf16, #fly_rocdl.buffer_desc, ((8,1),4,1,128):((1,0),262144,0,64)>)>
                                  
                                  Layout(((32, 64), 4, 1, 128):((8192, 1), 262144, 0, 64))
                                  Layout(((8, 8, 4), 8):((8, 8192, 65536), 1))

"""