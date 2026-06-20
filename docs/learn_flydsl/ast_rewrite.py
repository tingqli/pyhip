import os
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

"""
jit or kernel function body will be rewritten by AST-rewriter,

but pure python function will not be touched, and it can be called inside jit/kernel function.

this is the so-called meta-programming, we can always do pure-python stuff at compile-time,
and also inject the target code for jit/kernel function.

"""

def pure_python_func():
    for i in range(3):
        print("\t", i, type(i))
        fx.printf("runtime-print by pure-python func: i={}", i)

@flyc.jit
def test():
    print("Following print will be rewrite by AST:")
    print("-loop variable will be ArithValue")
    print("-loop body will only execute once at compile-time")

    for i in range(3):
        print("\t", i, type(i))
        fx.printf("runtime-print by jit/kernel: i={}", i)

    print("Following is a pure python function, it will not be changed by AST-rewrite")
    pure_python_func()

test()