from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# from https://github.com/ROCm/aiter/blob/9522048dc10de20ba9dcda1c0a3f640867e7a586/aiter/ops/triton/_triton_kernels/attention/pod_attention.py#L15-L62
@gluon.jit
def read_cycle(wait=False):
    if wait:
        asm: gl.constexpr = 's_memtime $0\ns_waitcnt lgkmcnt(0)'
    else:
        asm: gl.constexpr = 's_memtime $0'
    tmp = gl.inline_asm_elementwise(
        asm=asm,
        constraints=("=s"),
        args=[],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )
    return tmp

@gluon.jit
def read_realtime(wait=False):
    if wait:
        asm: gl.constexpr = 's_memrealtime $0\ns_waitcnt lgkmcnt(0)'
    else:
        asm: gl.constexpr = 's_memrealtime $0'
    tmp = gl.inline_asm_elementwise(
        asm=asm,
        # asm="""s_waitcnt vmcnt(0)
        # s_memrealtime $0
        # s_waitcnt lgkmcnt(0)""",
        constraints=("=s"),
        args=[],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )
    return tmp

@gluon.jit
def get_cu_id():
    # HW_ID Register bit structure for GCN and CDNA
    #   WAVE_ID     3:0     Wave buffer slot number. 0-9.
    #   SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
    #   PIPE_ID     7:6     Pipeline from which the wave was dispatched.
    #   CU_ID       11:8    Compute Unit the wave is assigned to.
    #   SH_ID       12      Shader Array (within an SE) the wave is assigned to.
    #   SE_ID       15:13   Shader Engine the wave is assigned to for gfx908, gfx90a
    #               14:13   Shader Engine the wave is assigned to for 942
    #   TG_ID       19:16   Thread-group ID
    #   VM_ID       23:20   Virtual Memory ID
    #   QUEUE_ID    26:24   Queue from which this wave was dispatched.
    #   STATE_ID    29:27   State ID (graphics only, not compute).
    #   ME_ID       31:30   Micro-engine ID.

    # XCC_ID Register bit structure for 942/950
    #   XCC_ID      3:0     XCC the wave is assigned to.

    cu_id, se_id, xcc_id, slot_id = gl.inline_asm_elementwise(
        asm="""
        s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 8, 4)
        s_getreg_b32 $1, hwreg(HW_REG_HW_ID, 13, 2)
        s_getreg_b32 $2, hwreg(HW_REG_XCC_ID, 0, 4)
        s_getreg_b32 $3, hwreg(HW_REG_HW_ID, 0, 4)
        s_waitcnt lgkmcnt(0)
        """,
        constraints=("=s,=s,=s,=s"),  # Three scalar output
        args=[],  # No inputs
        dtype=(gl.int32, gl.int32, gl.int32, gl.int32),  # Output type is int32
        is_pure=False,
        pack=1,
    )
    return (cu_id, se_id, xcc_id, slot_id)