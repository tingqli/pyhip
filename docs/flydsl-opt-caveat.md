# FlyDSL optimization caveat

## element-wise processing w/o tiled_copy

for 1D element-wise processing, tiled_copy is more complex & limited.
we can simply partion 1D data tensor using divide and slicing it with fx.thread_idx.x.
this pattern is very often see in cuda/hip kernel source code.

this method can be extended to multi-dimension cases by grouping all modes into 1 `fx.group(A, 0, -1)`, 

```python
    A = fx.rocdl.make_buffer_tensor(A, max_size=False)
    B = fx.rocdl.make_buffer_tensor(B, max_size=False)

    vec_width = copy_bits // A.dtype.width
    div_tensorA = fx.logical_divide(A, fx.make_layout(vec_width, 1))
    div_tensorB = fx.logical_divide(B, fx.make_layout(vec_width, 1))

    # make sure neles is integer multiple of vec_width
    num_atoms = fx.Int32(neles // vec_width)

    for i in range(fx.thread_idx.x, num_atoms, num_threads):

        src = div_tensorA[None, i]
        dst = div_tensorB[None, i]
        frag1 = fx.make_fragment_like(src)
        frag2 = fx.make_fragment_like(dst)

        fx.copy(copy_atomA, src, frag1)
        # element-wise process data in frag1 and store to frag2
        fx.copy(copy_atomB, frag2, dst)
```

## avoid atomic instructions

per-tensor-quant in aiter is extreamly slow due to huge work-groups and each work-group uses global atomic to update per-tensor abs-max value, this atomic instruction is very expensive and shold avoid as much as possible. in `flydsl_absmax`, we start just enough number of work-groups (number CU/SM * 8-occupancy), which incur just one global atomic update per work-group.

## gather/scatter use coordinate tensor

the partition_S/D methods of tiled-copy gives thread-view of a global vmem tensor, but it doesn't give the coordinate of the thread-view, somethime we want to gather/scatter data from/to another big tensor using row/col indicies, we need those coordinates.

In this case we can build a broadcast-view of the coordinate tensor, which has exatly same shape as data-tensor, and then partition them along with data tensor, and iterate these thread-views in unit of copy-atom, and directly retrieve the coordinate of each copy-atom by loading from coordinate thread-view.

```python
    row_tensor = fx.make_view(
        fx.get_iter(arg_p_sorted_ids),
        fx.make_layout((BLOCK_M, BLOCK_N), (1, 0)),
    )
    col_tensor = fx.make_view(
        fx.make_int_tuple(0),
        fx.make_layout((BLOCK_M, BLOCK_N), (0, 1))
    )

    thrv_src_data = thr_copy.partition_S(src_data)
    thrv_dst_row = thr_copy.partition_D(row_tensor)
    thrv_dst_col = thr_copy.partition_D(col_tensor)

    for src, row, col in fxh.all_elements(thrv_src_data, thrv_dst_row, thrv_dst_col):
        # src sub-tensor is located at row[0] col[0]
        atom_dst = fxh.atom_tensor(arg_p_output, (row[0], col[0]), 128)
        fx.copy(cp_atom, src, atom_dst)
```


## coelascing within smaller memory-region

MOE down kernel writes output block belong to multiple sorted tokens, and they may be located far away in global memory, which may not be friendly to cache/DRAM locality.
