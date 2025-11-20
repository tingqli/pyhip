import ast
'''
  <class 'ast.Module'> {'body': [<ast.FunctionDef object at 0x7f5f4ffe4ca0>, <ast.Expr object at 0x7f5f4fdc1960>], 'type_ignores': []}
     <class 'ast.FunctionDef'> {'args': <ast.arguments object at 0x7f5f4fdc15a0>, 'body': [<ast.Expr object at 0x7f5f4fdc1660>, <ast.Expr object at 0x7f5f4fe61690>, <ast.Expr object at 0x7f5f4fe61480>, <ast.Expr object at 0x7f5f4fe61390>, <ast.Assign object at 0x7f5f4fe61270>, <ast.Expr object at 0x7f5f4fe61210>, <ast.Expr object at 0x7f5f4fe61060>], 'decorator_list': [], 'name': 'greet', 'returns': None, 'type_comment': None}
         <class 'ast.arguments'> {'args': [<ast.arg object at 0x7f5f4fe05c30>], 'defaults': [], 'kw_defaults': [], 'kwarg': None, 'kwonlyargs': [], 'posonlyargs': [], 'vararg': None}
             <class 'ast.arg'> {'annotation': None, 'arg': 'name', 'type_comment': None}
         <class 'ast.Expr'> {'value': <ast.Call object at 0x7f5f4fe06770>}
             <class 'ast.Call'> {'args': [<ast.Subscript object at 0x7f5f4fe06110>, <ast.Subscript object at 0x7f5f4fe065f0>, <ast.Constant object at 0x7f5f4fe2bee0>, <ast.Name object at 0x7f5f4fe2bfd0>], 'func': <ast.Name object at 0x7f5f4fe06680>, 'keywords': []}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's_load_dwordx2'}
                     <class 'ast.Load'> {}
                 <class 'ast.Subscript'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'slice': <ast.Slice object at 0x7f5f4fe05420>, 'value': <ast.Name object at 0x7f5f4fe06650>}
                     <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's'}
                         <class 'ast.Load'> {}
                     <class 'ast.Slice'> {'lower': <ast.Constant object at 0x7f5f4fe2b5b0>, 'step': None, 'upper': <ast.Constant object at 0x7f5f4fe2b6d0>}
                         <class 'ast.Constant'> {'kind': None, 'n': 2, 's': 2, 'value': 2}
                         <class 'ast.Constant'> {'kind': None, 'n': 3, 's': 3, 'value': 3}
                     <class 'ast.Load'> {}
                 <class 'ast.Subscript'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'slice': <ast.Slice object at 0x7f5f4fe2b730>, 'value': <ast.Name object at 0x7f5f4fe2b640>}
                     <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's'}
                         <class 'ast.Load'> {}
                     <class 'ast.Slice'> {'lower': <ast.Constant object at 0x7f5f4fe2b820>, 'step': None, 'upper': <ast.Constant object at 0x7f5f4fe2ba00>}
                         <class 'ast.Constant'> {'kind': None, 'n': 0, 's': 0, 'value': 0}
                         <class 'ast.Constant'> {'kind': None, 'n': 1, 's': 1, 'value': 1}
                     <class 'ast.Load'> {}
                 <class 'ast.Constant'> {'kind': None, 'n': 0, 's': 0, 'value': 0}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 'glc'}
                     <class 'ast.Load'> {}
         <class 'ast.Expr'> {'value': <ast.Call object at 0x7f5f4fe61660>}
             <class 'ast.Call'> {'args': [<ast.Name object at 0x7f5f4fe61600>, <ast.Subscript object at 0x7f5f4fe615d0>, <ast.Constant object at 0x7f5f4fe614e0>], 'func': <ast.Name object at 0x7f5f4fe61630>, 'keywords': []}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's_load_dword'}
                     <class 'ast.Load'> {}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's4'}
                     <class 'ast.Load'> {}
                 <class 'ast.Subscript'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'slice': <ast.Slice object at 0x7f5f4fe61570>, 'value': <ast.Name object at 0x7f5f4fe615a0>}
                     <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's'}
                         <class 'ast.Load'> {}
                     <class 'ast.Slice'> {'lower': <ast.Constant object at 0x7f5f4fe61540>, 'step': None, 'upper': <ast.Constant object at 0x7f5f4fe61510>}
                         <class 'ast.Constant'> {'kind': None, 'n': 0, 's': 0, 'value': 0}
                         <class 'ast.Constant'> {'kind': None, 'n': 1, 's': 1, 'value': 1}
                     <class 'ast.Load'> {}
                 <class 'ast.Constant'> {'kind': None, 'n': 8, 's': 8, 'value': 8}
         <class 'ast.Expr'> {'value': <ast.Call object at 0x7f5f4fe61450>}
             <class 'ast.Call'> {'args': [<ast.Constant object at 0x7f5f4fe613f0>], 'func': <ast.Name object at 0x7f5f4fe61420>, 'keywords': []}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's_waitcnt_lgkmcnt'}
                     <class 'ast.Load'> {}
                 <class 'ast.Constant'> {'kind': None, 'n': 0, 's': 0, 'value': 0}
         <class 'ast.Expr'> {'value': <ast.Call object at 0x7f5f4fe61360>}
             <class 'ast.Call'> {'args': [<ast.Name object at 0x7f5f4fe61300>, <ast.Name object at 0x7f5f4fe612d0>], 'func': <ast.Name object at 0x7f5f4fe61330>, 'keywords': []}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 'v_mov_b32'}
                     <class 'ast.Load'> {}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 'v2'}
                     <class 'ast.Load'> {}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's2'}
                     <class 'ast.Load'> {}
         <class 'ast.Assign'> {'targets': [<ast.Name object at 0x7f5f4fe61240>], 'type_comment': None, 'value': <ast.Name object at 0x7f5f4fe611e0>}
             <class 'ast.Name'> {'ctx': <ast.Store object at 0x7f5f4fe285b0>, 'id': 'v3'}
                 <class 'ast.Store'> {}
             <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's3'}
                 <class 'ast.Load'> {}
         <class 'ast.Expr'> {'value': <ast.Call object at 0x7f5f4fe611b0>}
             <class 'ast.Call'> {'args': [<ast.Name object at 0x7f5f4fe61150>, <ast.Name object at 0x7f5f4fe61120>, <ast.Constant object at 0x7f5f4fe610f0>, <ast.Name object at 0x7f5f4fe610c0>], 'func': <ast.Name object at 0x7f5f4fe61180>, 'keywords': []}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 'v_lshl_add_u32'}
                     <class 'ast.Load'> {}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 'v2'}
                     <class 'ast.Load'> {}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 'v0'}
                     <class 'ast.Load'> {}
                 <class 'ast.Constant'> {'kind': None, 'n': 2, 's': 2, 'value': 2}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 'v2'}
                     <class 'ast.Load'> {}
         <class 'ast.Expr'> {'value': <ast.Call object at 0x7f5f4fe61030>}
             <class 'ast.Call'> {'args': [<ast.Name object at 0x7f5f4fe60fd0>, <ast.Subscript object at 0x7f5f4fe60fa0>, <ast.Constant object at 0x7f5f4fe60eb0>, <ast.Name object at 0x7f5f4fe60e80>], 'func': <ast.Name object at 0x7f5f4fe61000>, 'keywords': []}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's_store_dword'}
                     <class 'ast.Load'> {}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's4'}
                     <class 'ast.Load'> {}
                 <class 'ast.Subscript'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'slice': <ast.Slice object at 0x7f5f4fe60f40>, 'value': <ast.Name object at 0x7f5f4fe60f70>}
                     <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 's'}
                         <class 'ast.Load'> {}
                     <class 'ast.Slice'> {'lower': <ast.Constant object at 0x7f5f4fe60f10>, 'step': None, 'upper': <ast.Constant object at 0x7f5f4fe60ee0>}
                         <class 'ast.Constant'> {'kind': None, 'n': 2, 's': 2, 'value': 2}
                         <class 'ast.Constant'> {'kind': None, 'n': 3, 's': 3, 'value': 3}
                     <class 'ast.Load'> {}
                 <class 'ast.Constant'> {'kind': None, 'n': 0, 's': 0, 'value': 0}
                 <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 'glc'}
                     <class 'ast.Load'> {}
     <class 'ast.Expr'> {'value': <ast.Call object at 0x7f5f4fe60e20>}
         <class 'ast.Call'> {'args': [<ast.Constant object at 0x7f5f4fe60dc0>], 'func': <ast.Name object at 0x7f5f4fe60df0>, 'keywords': []}
             <class 'ast.Name'> {'ctx': <ast.Load object at 0x7f5f4fe28610>, 'id': 'greet'}
                 <class 'ast.Load'> {}
             <class 'ast.Constant'> {'kind': None, 'n': 'John', 's': 'John', 'value': 'John'}


'''


def kernel(T, s=[], v=[]):
    T.s_load_dwordx2(s[2:3], s[0:1], 0, T.glc)

    acc = T.compile_time_alloc_agpr(16*16) # 编译期分配

    for i in range(16*16): # 一定是编译期展开
        T.v_accvgpr_write_b32(acc[i], 0)

    #T.s_load_dword(s[4], s[0:1], 8)
    #T.s_waitcnt_lgkmcnt(0)
    #T.v_mov_b32(v[2], s[2])
    # v_mov_b32(v3, s3)
    v[3] = s[3]

    with T.label() as bb0:
        T.v_lshl_add_u32(v[2], v[0], 2, v[2])
        T.goto(bb0)

    # flat_store_dword(v[2:3], v0)
    T.s_store_dword(s[4], s[2:3], 0, T.glc)

import inspect
code = inspect.getsource(kernel)

tree = ast.parse(code)

'''
ast捕获语法之后, 在jit的设计思想上(也就是不引入太多复杂变换), 可以不用DAG建模
直接完成一些简单的替换就生成汇编代码, 这样看来不适合ast,而比较适合直接emit
也就是每个函数调用直接emit了汇编代码

'''

class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.indent = 0

    def show(self, node):
        info = {}
        exclude_attrs = ["col_offset","end_col_offset","end_lineno","lineno"]
        for attr in dir(node):
            if not attr.startswith("_") and attr not in exclude_attrs:
                info[attr] = getattr(node, attr)
        print("    " * self.indent, type(node), info)

    def visit_Expr(self, node):
        self.show(node)
        self.indent += 1
        # 继续 visit children
        self.generic_visit(node)
        self.indent -= 1

    def visit_Call(self, node):
        self.show(node)

    def visit_With(self, node):
        self.show(node)
        self.indent += 1
        # 继续 visit children
        self.generic_visit(node)
        self.indent -= 1

    def visit_FunctionDef(self, node):
        self.show(node)
        #if isinstance(node, ast.FunctionDef):
        #    args_info = node.args
        #    for a in args_info.args:
        #        # a 
        if 0:
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                print(node.func.id)
                args = [arg for arg in node.args if isinstance(arg, ast.Str)]
                if args:
                    print("Detected print statements with string literals:")
                    for arg in args:
                        print(arg.s)  # Print the string literal directly
        self.indent += 1
        # 继续 visit children
        self.generic_visit(node)
        self.indent -= 1


visitor = FunctionCallVisitor()
visitor.visit(tree)
