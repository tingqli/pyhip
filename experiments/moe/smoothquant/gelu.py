import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import struct
import pyhip
import torch

pyhip.set_device()

@pyhip.jit()
def cdna_gelu(J, pA:"float*", pB:"float*", N:"int"):
    """
    v = int2float(0x3f4c4229) * (int2float(0x3d372713) * x*x*x + x)
    sign = np.sign(v)
    exp = np.exp(-2*sign*v)
    tanh = (sign - sign*exp)/(1 + exp)
    """
    buff_a = J.Buffer(pA, N*J.sizeof_DW)
    buff_b = J.Buffer(pB, N*J.sizeof_DW)

    idx = J.gpr(J.blockIdx.x * 64 + J.threadIdx.x)
    voffset = J.gpr(idx * J.sizeof_DW)

    v2 = J.gpr("vf32")
    buff_a.load_dword(v2, voffset, 0)
    J.s_waitcnt(mod="vmcnt(0)")
    vgelu = J.gelu(v2)
    buff_b.store_dword(vgelu, voffset, 0)

def int2float(value):
    return struct.unpack('f', struct.pack('I', value))[0]

def gelu1(v3):
    """
    可以在 |v3|<0.625 以内拟合 tanh
	v_mul_f32_e32 v4, v3, v3
	v_mov_b32_e32 v5, 0x3ca908c9                     ;	v5 = 0x3ca908c9;  0.020634072
	v_fmac_f32_e32 v5, 0xbbbac73d, v4				 ;                   -0.0057000206
	v_fmaak_f32 v5, v4, v5, 0xbd5c1c4e               ;                   -0.05373793
	v_fmaak_f32 v5, v4, v5, 0x3e088382               ;                    0.13331416
	v_fmaak_f32 v5, v4, v5, 0xbeaaaa99               ;                    -0.3333328
	v_mul_f32_e64 v5, |v3|, v5
	v_fma_f32 v4, v4, v5, |v3|
    """
    v4 = v3*v3
    v5 = int2float(0x3ca908c9)
    v5 += int2float(0xbbbac73d)*v4
    v5 = v4 * v5 + int2float(0xbd5c1c4e)
    v5 = v4 * v5 + int2float(0x3e088382)
    v5 = v4 * v5 + int2float(0xbeaaaa99)
    v5 *= abs(v3)
    v4 = v4 * v5 + abs(v3)
    # s_brev_b32 s0, -2
    # v_bfi_b32 v3, s0, v4, v3
    v4 *= np.sign(v3) 
    return v4*np.sign(v3) #np.maximum(0, x)


def gelu_opt(x):
    if 0:
        # https://discuss.ai.google.dev/t/fast-gelu-approximation/109588
        p=0.544790
        return 0.5*x*(1.0+x/np.sqrt(p+x*x))
    if 0:
        # https://www.johndcook.com/blog/2025/03/06/gelu/
        return 0.5*x*(1.0 + np.tanh(0.8*x))
    
    if 1:
        tx = torch.tensor(x)
        ty = torch.empty_like(tx)
        cdna_gelu([(tx.numel()+63//64)], [64], tx.data_ptr(), ty.data_ptr(), tx.numel())
        return ty.cpu().numpy()

    v = int2float(0x3f4c4229) * (int2float(0x3d372713) * x*x*x + x)
    sign = np.sign(v)
    exp = np.exp(-2*sign*v)
    tanh = (sign - sign*exp)/(1 + exp)
    return 0.5*x*(1.0 + tanh)

def gelu(x):
    """GELU激活函数 (使用精确公式)"""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    #return np.tanh(x)

# 生成x轴数据（-5到5之间的1000个点）
x = np.linspace(-3, 3, 1000, dtype=np.float32)

# 计算对应的y值
y_relu = gelu_opt(x)
y_gelu = gelu(x)

# 创建图形
plt.figure(figsize=(10, 6), dpi=100)

# 绘制ReLU曲线
plt.plot(x, y_relu, label='GELU2', color='blue', linewidth=2)

# 绘制GELU曲线
plt.plot(x, y_gelu, label='GELU', color='red', linewidth=2)

# 添加参考线
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
plt.axvline(x=0, color='black', linewidth=0.5, linestyle='-')

# 设置图表属性
plt.xlabel('Input (x)', fontsize=12)
plt.ylabel('Output f(x)', fontsize=12)
plt.title('gelu_opt vs GELU Activation Functions', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(-3, 3)
plt.ylim(-1, 1)

# 添加网格和美化
plt.axhline(y=0, color='black', linewidth=0.5)
plt.axvline(x=0, color='black', linewidth=0.5)

# 保存图像
plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
print("图像已保存为 'activation_functions.png'")

# 显示图像
plt.show()

# 打印一些示例值
print("\n示例输出值:")
test_points = np.array([-100, -50, -5, -2, 0, 2, 5, 50, 100], dtype=np.float32)
print("x\tgelu_opt(x)\tGELU(x)")
print("-" * 35)
result1_points = gelu(test_points)
result2_points = gelu_opt(test_points)
for x,y1,y2 in zip(test_points, result1_points, result2_points):
    print(f"{x}\t{y2:.4f}\t{y1:.4f}")
