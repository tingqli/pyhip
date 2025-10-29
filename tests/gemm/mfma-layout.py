


#__builtin_amdgcn_mfma_f32_32x32x8f16
#__builtin_amdgcn_mfma_f32_16x16x16f16

M,N,K,B = 32,32,8,1
M,N,K,B = 16,16,16,1

print(f"MNKB={M}x{N}x{K}_{B}")



print("======== input layout ===============")
K_L = K // (64 // (M * B))
print(f"{K_L=}  each lane contains {K_L} f16s")

def input_item_lane(i, k, b=0):
    return k % K_L, i + M * (b + B * (k // K_L))

A_reg = [ [''] * K_L ]
for i in range(M):
    print(f"{i:2} : ", end="")
    for k in range(0, K, K_L):
        _, lane = input_item_lane(i, k)
        print(f"lane[{lane}] ", end="")
    print()

print("======== output layout ===============")
# https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
def ceildiv(a, b):
    return -(a // -b)
H=4
B_I = ceildiv(64, (N * M // H))
M_I = (64 // B_I) // N
G = M // (H * M_I)
def output_item_lane(i, j, b=0):
    return (i % H) + H * (i//(H * M_I) + G * (b // B_I)), j + N * ((i // H) % M_I + M_I * (b % B_I))


print(f"Matrix C/D: each lane contain {H} rows")
for i in range(0,M,H):
    item0, lane = output_item_lane(i, 0)
    print(f"row {i:2}~{i+H:2} item[{item0:2}:{item0+H:2}] : ", end="")
    
    for j in range(N):
        item, lane = output_item_lane(i, j)
        assert item == item0
        print(f"[{lane:2}]", end="")
    print()