

import numpy as np

# about LDS & swizzle
# https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/pipeline-descriptions.html#local-data-share-lds
# https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html

lane_group_w=[]
for lane in range(0,64,8):
    lane_group_w.append([lane + i for i in range(8)])

def lg(lane0, lane1):
    return [lane0, lane0+1, lane0+2, lane0+3] + [lane1, lane1+1, lane1+2, lane1+3]

# ds_read_b128 : lane-groups
lane_group_r=[
    lg(0,20),
    lg(4,16),
    lg(8,28),
    lg(12,24),
    lg(32+0,32+20),
    lg(32+4,32+16),
    lg(32+8,32+28),
    lg(32+12,32+24),
]

print(f"{lane_group_w=}")
print(f"{lane_group_r=}")

NUM_BANKS = 32
element_bytes = 16 # DWORDx4/HALFx8/b128

ROWS,COLS = 32, 4

w_shape = (2, 1, 16, 4) # batch_rows, batch_cols, rows, cols
assert w_shape[-1] * w_shape[-2] == 64

# MFMA V32x32x8 case:
r_shape = (1, 2, 32, 2) # batch_rows, batch_cols, rows, cols
assert r_shape[-1] * r_shape[-2] == 64

# lane_w maps to (row,col) natually
# lane_r maps to (row,col) natually
def coord_w(batch_r, lane, batch_c = 0):
    Br, Bc, R, C = w_shape
    assert batch_r < Br
    row = batch_r*R + (lane // C)
    col = batch_c*C + (lane % C)
    return row, col

# MFMA order
def coord_r(batch_c, lane, batch_r=0):
    Br, Bc, R, C = r_shape
    row = batch_r*R + (lane % R)
    col = batch_c*C + (lane//R)
    return row, col

# swizzle mapps logical coord to physical coord
def swizzle(logical_row, logical_col):
    assert logical_row <= ROWS
    assert logical_col <= COLS
    return logical_row, (logical_col^(logical_row//2))&(COLS-1)

reverse_swizzle = {}
for logical_row in range(ROWS):
    for logical_col in range(COLS):
        r,c = swizzle(logical_row, logical_col)
        reverse_swizzle[(r,c)] = (logical_row, logical_col)

def bank_of(physical_row, physical_col):
    assert physical_row <= ROWS
    assert physical_col <= COLS
    assert element_bytes % 4 == 0
    # LDS is 32 banks of 4 bytes,
    banks_per_element = element_bytes//4
    bank = ((physical_row * COLS + physical_col)*banks_per_element) % NUM_BANKS
    return bank

# check bank conflict
def check_bank_conflict(lane_groups, coord, baches=2):
    conflicts = 0
    for batch in range(baches):
        for group in lane_groups:
            banks = []
            for lane in group:
                logical_row, logical_col = coord(batch, lane)
                physical_row, physical_col = swizzle(logical_row, logical_col)
                b = bank_of(physical_row, physical_col)
                conf = b in banks
                conflicts += 1 if conf else 0
                print(f"{'conflict' if conf else '        '} ({logical_row},{logical_col}) => ({physical_row},{physical_col}) {lane} bank{b}")
                banks.append(b)
    return conflicts

total_conflicts = 0
total_conflicts += check_bank_conflict(lane_group_w, coord_w)
total_conflicts += check_bank_conflict(lane_group_r, coord_r)
print(f"{total_conflicts=}")
assert total_conflicts == 0

for physical_row in range(ROWS):
    print(f"{physical_row}:\t", end="")
    for physical_col in range(COLS):
        r,c = reverse_swizzle[(physical_row, physical_col)]
        if r == physical_row:
            print(f"(__,{c:2})", end=",")
        else:
            print(f"({r:2},{c:2})", end=",")
    print("     | banks: ", end="")
    for physical_col in range(COLS):
        bank = bank_of(physical_row, physical_col)
        print(f"{bank:2}", end=",")

    print()

