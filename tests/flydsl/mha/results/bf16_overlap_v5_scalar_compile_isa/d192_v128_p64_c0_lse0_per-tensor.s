	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	attn_kernel_0
	.p2align	8
	.type	attn_kernel_0,@function
attn_kernel_0:
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dword s3, s[0:1], 0x38
	s_load_dwordx8 s[68:75], s[0:1], 0x70
	s_mov_b32 s80, 0
	s_waitcnt lgkmcnt(0)
	s_load_dwordx2 s[4:5], s[34:35], 0x0
	s_add_i32 s3, s3, -1
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s4, s5, s4
	s_add_i32 s6, s4, 0xff
	s_ashr_i32 s4, s6, 31
	s_lshr_b32 s4, s4, 24
	s_add_i32 s4, s6, s4
	s_ashr_i32 s8, s4, 8
	s_and_b32 s4, s4, 0xffffff00
	s_cmp_lg_u32 s6, s4
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s6, 0
	s_cselect_b64 s[6:7], -1, 0
	s_and_b64 s[4:5], s[6:7], s[4:5]
	s_subb_u32 s6, s8, 0
	s_cmp_lt_i32 s3, 1
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s2, s6
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[4:5], s[4:5], s[8:9]
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_8
	s_mov_b32 s84, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s80, s8
.LBB0_3:
	s_sub_i32 s33, s33, s6
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s84, 0, s7
	s_cmp_ge_i32 s80, s3
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s33, s98
	s_cselect_b64 s[6:7], -1, 0
	s_or_b64 s[4:5], s[4:5], s[6:7]
	s_and_b64 vcc, exec, s[4:5]
	s_mov_b32 s6, s98
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s7, s84, 1
	s_cmp_gt_i32 s7, 15
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s7, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s8, s80, 1
	s_cmp_ge_i32 s8, s3
	s_mov_b32 s98, s6
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s81, s80, 31
	s_lshl_b64 s[10:11], s[80:81], 2
	s_add_u32 s10, s34, s10
	s_addc_u32 s11, s35, s11
	s_load_dwordx2 s[12:13], s[10:11], 0x4
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s9, s13, s12
	s_addk_i32 s9, 0xff
	s_ashr_i32 s10, s9, 31
	s_lshr_b32 s10, s10, 24
	s_add_i32 s10, s9, s10
	s_ashr_i32 s14, s10, 8
	s_and_b32 s10, s10, 0xffffff00
	s_cmp_lg_u32 s9, s10
	s_cselect_b64 s[10:11], -1, 0
	s_cmp_lt_i32 s9, 0
	s_cselect_b64 s[12:13], -1, 0
	s_and_b64 s[10:11], s[12:13], s[10:11]
	s_subb_u32 s98, s14, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s98, s6
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s98, s6
	s_mov_b32 s84, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s89, s[70:71], 0x0
	s_load_dword s99, s[72:73], 0x0
	s_cmp_ge_i32 s80, s3
	s_cbranch_scc1 .LBB0_126
	v_lshrrev_b32_e32 v2, 6, v0
	v_lshrrev_b32_e32 v3, 2, v0
	v_mul_u32_u24_e32 v2, 0x18000, v2
	s_load_dwordx2 s[46:47], s[0:1], 0x0
	s_load_dwordx2 s[48:49], s[0:1], 0x10
	s_load_dwordx2 s[82:83], s[0:1], 0x20
	s_load_dwordx2 s[50:51], s[0:1], 0x50
	s_load_dwordx2 s[52:53], s[0:1], 0x60
	s_load_dwordx2 s[54:55], s[0:1], 0x98
	s_load_dwordx2 s[90:91], s[0:1], 0xb8
	v_and_b32_e32 v1, 31, v0
	v_and_or_b32 v2, v3, 8, v2
	s_movk_i32 s0, 0xc00
	v_mad_u32_u24 v1, v1, s0, v2
	v_mul_u32_u24_e32 v2, 0xaab, v0
	v_mov_b32_e32 v3, 24
	v_mul_u32_u24_e32 v4, 0x2ab, v0
	v_mov_b32_e32 v5, 2
	v_mov_b32_e32 v6, 5
	v_mul_lo_u16_sdwa v3, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshlrev_b16_sdwa v2, v5, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_u32_u24_e32 v5, 0x156, v0
	v_lshlrev_b16_sdwa v4, v6, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mov_b32_e32 v6, 4
	v_sub_u16_sdwa v3, v0, v3 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_sdwa v5, v6, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v4, 32, v4
	v_or_b32_e32 v3, v3, v5
	v_and_b32_e32 v2, 12, v2
	v_add_u16_e32 v3, v3, v4
	v_or_b32_e32 v85, v3, v2
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v105, v2, v3
	v_and_b32_e32 v2, 63, v0
	s_movk_i32 s0, 0x80
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_u32_sdwa v104, v85, s0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v106, 0x80, v2
	v_mov_b32_e32 v107, 0
	s_mov_b32 s79, 0x27000
	s_movk_i32 s85, 0x180
	s_movk_i32 s72, 0x1000
	v_mov_b32_e32 v108, 0x40e00000
	v_mov_b32_e32 v109, 1.0
	s_mov_b32 s73, 0x7060302
	s_movk_i32 s56, 0xf80
	s_mov_b32 s57, s2
	v_mov_b32_e32 v110, 0xff800000
	s_mov_b32 s36, 0
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s98, s6
.LBB0_12:
	s_mov_b32 s2, s4
	s_cmp_ge_i32 s80, s3
	s_cbranch_scc1 .LBB0_126
.LBB0_13:
	s_ashr_i32 s81, s80, 31
	s_lshl_b32 s6, s33, 8
	s_lshl_b64 s[0:1], s[80:81], 2
	s_add_u32 s4, s34, s0
	s_addc_u32 s5, s35, s1
	global_load_dwordx2 v[4:5], v107, s[4:5]
	global_load_dword v2, v107, s[68:69]
	s_mul_i32 s4, s84, 0xc0
	v_add_lshl_u32 v3, s4, v1, 1
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s70, v4
	s_add_i32 s70, s70, s6
	v_readfirstlane_b32 s5, v5
	s_add_i32 s6, s70, 0x100
	s_min_i32 s5, s6, s5
	s_sub_i32 s81, s5, s70
	s_waitcnt lgkmcnt(0)
	s_add_u32 s6, s50, s0
	s_addc_u32 s7, s51, s1
	s_mul_i32 s4, s70, 0xc00
	s_add_u32 s0, s74, s0
	s_addc_u32 s1, s75, s1
	s_ashr_i32 s5, s4, 31
	global_load_dwordx2 v[4:5], v107, s[6:7]
	global_load_dword v6, v107, s[0:1]
	s_lshl_b64 s[0:1], s[4:5], 1
	s_add_u32 s76, s46, s0
	s_addc_u32 s0, s47, s1
	s_mul_i32 s78, s81, 0x1800
	s_and_b32 s77, s0, 0xffff
	buffer_load_dwordx4 v[130:133], v3, s[76:79], 0 offen
	buffer_load_dwordx4 v[134:137], v3, s[76:79], 0 offen offset:32
	buffer_load_dwordx4 v[138:141], v3, s[76:79], 0 offen offset:64
	buffer_load_dwordx4 v[142:145], v3, s[76:79], 0 offen offset:96
	buffer_load_dwordx4 v[146:149], v3, s[76:79], 0 offen offset:128
	buffer_load_dwordx4 v[150:153], v3, s[76:79], 0 offen offset:160
	buffer_load_dwordx4 v[154:157], v3, s[76:79], 0 offen offset:192
	buffer_load_dwordx4 v[158:161], v3, s[76:79], 0 offen offset:224
	buffer_load_dwordx4 v[162:165], v3, s[76:79], 0 offen offset:256
	buffer_load_dwordx4 v[166:169], v3, s[76:79], 0 offen offset:288
	buffer_load_dwordx4 v[170:173], v3, s[76:79], 0 offen offset:320
	buffer_load_dwordx4 v[174:177], v3, s[76:79], 0 offen offset:352
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_waitcnt vmcnt(13)
	v_readfirstlane_b32 s0, v4
	v_and_b32_e32 v3, 0x100, v3
	v_readfirstlane_b32 s1, v5
	s_waitcnt vmcnt(12)
	v_readfirstlane_b32 s6, v6
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_15
	s_barrier
.LBB0_15:
	s_or_b64 exec, exec, s[4:5]
	s_sub_i32 s92, s1, s0
	s_ashr_i32 s1, s84, 31
	s_lshr_b32 s1, s1, 28
	s_add_i32 s1, s84, s1
	s_ashr_i32 s7, s1, 4
	s_and_b32 s1, s1, -16
	s_cmp_lg_u32 s84, s1
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s84, 0
	s_cselect_b64 s[8:9], -1, 0
	s_and_b64 s[4:5], s[8:9], s[4:5]
	s_subb_u32 s4, s7, 0
	s_mul_i32 s8, s4, 0x3000
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[8:9], s[8:9], 1
	s_add_u32 s94, s48, s8
	s_addc_u32 s95, s49, s9
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s76, s52, s0
	s_addc_u32 s0, s53, s1
	s_lshl_b32 s78, s92, 2
	s_and_b32 s77, s0, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[76:79], 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s11, v4
	v_readfirstlane_b32 s88, v5
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	buffer_load_dword v3, off, s[76:79], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s71, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_mul_i32 s5, s11, 0x1800
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s85, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_17
	v_add_u32_sdwa v4, s5, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[94:95]
	global_load_dwordx4 v[114:117], v[4:5], off
	global_load_dwordx4 v[118:121], v[4:5], off offset:256
.LBB0_17:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	buffer_load_dword v3, off, s[76:79], 0 offset:12
	;;#ASMSTART
	v_mov_b32 v4, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s87, v3
	v_and_b32_e32 v4, 0x180, v4
	v_cmp_ne_u32_e32 vcc, s85, v4
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_19
	v_add_u32_e32 v4, s5, v104
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[94:95]
	global_load_dwordx4 v[122:125], v[4:5], off
	global_load_dwordx4 v[126:129], v[4:5], off offset:256
.LBB0_19:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s85, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_21
	ds_write_b128 v105, v[114:117] offset:12288
	ds_write_b128 v105, v[118:121] offset:18432
.LBB0_21:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s85, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_23
	s_mul_i32 s5, s88, 0x1800
	v_add_u32_sdwa v4, s5, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[94:95]
	global_load_dwordx4 v[114:117], v[4:5], off
	global_load_dwordx4 v[118:121], v[4:5], off offset:256
.LBB0_23:
	s_or_b64 exec, exec, s[0:1]
	v_mul_f32_e32 v2, s89, v2
	s_add_i32 s7, s92, -1
	v_mul_f32_e32 v82, 0x3dd53b95, v2
	s_lshl_b32 s86, s4, 13
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 31, v2
	v_mul_u32_u24_e32 v3, 0xc0, v3
	v_lshrrev_b32_e32 v2, 2, v2
	v_and_or_b32 v2, v2, 8, v3
	v_lshrrev_b32_e32 v3, 3, v3
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v3, v3, v2
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[222:225], v3 offset:12288
	v_add_u32_e32 v3, 0x1810, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[178:181], v3
	v_add_u32_e32 v3, 0x1820, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[182:185], v3
	v_add_u32_e32 v3, 0x1830, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[186:189], v3
	v_add_u32_e32 v3, 0x1840, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[190:193], v3
	v_add_u32_e32 v3, 0x1850, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[194:197], v3
	v_add_u32_e32 v3, 0x1860, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[198:201], v3
	v_add_u32_e32 v3, 0x1870, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[202:205], v3
	v_add_u32_e32 v3, 0x1880, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[206:209], v3
	v_add_u32_e32 v3, 0x1890, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[210:213], v3
	v_add_u32_e32 v3, 0x18a0, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_u32_e32 v2, 0x18b0, v2
	ds_read_b128 v[218:221], v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[214:217], v2
	s_add_i32 s0, s92, -2
	s_bitcmp0_b32 s92, 0
	s_cselect_b32 s96, s0, s7
	s_ashr_i32 s97, s96, 31
	s_cmp_lt_i32 s96, 1
	s_cbranch_scc1 .LBB0_59
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v86, v82
	v_mov_b32_e32 v87, v82
	s_mov_b64 s[0:1], 0
	s_mov_b32 s8, 20
	v_mov_b32_e32 v84, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v111
	v_mov_b32_e32 v4, v111
	v_mov_b32_e32 v5, v111
	v_mov_b32_e32 v6, v111
	v_mov_b32_e32 v7, v111
	v_mov_b32_e32 v8, v111
	v_mov_b32_e32 v9, v111
	v_mov_b32_e32 v10, v111
	v_mov_b32_e32 v11, v111
	v_mov_b32_e32 v12, v111
	v_mov_b32_e32 v13, v111
	v_mov_b32_e32 v14, v111
	v_mov_b32_e32 v15, v111
	v_mov_b32_e32 v16, v111
	v_mov_b32_e32 v17, v111
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v111
	v_mov_b32_e32 v20, v111
	v_mov_b32_e32 v21, v111
	v_mov_b32_e32 v22, v111
	v_mov_b32_e32 v23, v111
	v_mov_b32_e32 v24, v111
	v_mov_b32_e32 v25, v111
	v_mov_b32_e32 v26, v111
	v_mov_b32_e32 v27, v111
	v_mov_b32_e32 v28, v111
	v_mov_b32_e32 v29, v111
	v_mov_b32_e32 v30, v111
	v_mov_b32_e32 v31, v111
	v_mov_b32_e32 v32, v111
	v_mov_b32_e32 v33, v111
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v111
	v_mov_b32_e32 v36, v111
	v_mov_b32_e32 v37, v111
	v_mov_b32_e32 v38, v111
	v_mov_b32_e32 v39, v111
	v_mov_b32_e32 v40, v111
	v_mov_b32_e32 v41, v111
	v_mov_b32_e32 v42, v111
	v_mov_b32_e32 v43, v111
	v_mov_b32_e32 v44, v111
	v_mov_b32_e32 v45, v111
	v_mov_b32_e32 v46, v111
	v_mov_b32_e32 v47, v111
	v_mov_b32_e32 v48, v111
	v_mov_b32_e32 v49, v111
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v111
	v_mov_b32_e32 v52, v111
	v_mov_b32_e32 v53, v111
	v_mov_b32_e32 v54, v111
	v_mov_b32_e32 v55, v111
	v_mov_b32_e32 v56, v111
	v_mov_b32_e32 v57, v111
	v_mov_b32_e32 v58, v111
	v_mov_b32_e32 v59, v111
	v_mov_b32_e32 v60, v111
	v_mov_b32_e32 v61, v111
	v_mov_b32_e32 v62, v111
	v_mov_b32_e32 v63, v111
	v_mov_b32_e32 v64, v111
	v_mov_b32_e32 v65, v111
	s_branch .LBB0_26
.LBB0_25:
	s_or_b64 exec, exec, s[4:5]
	v_add_u32_e32 v73, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s73
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[202:203], v[66:67], v[2:17]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[206:207], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[66:67], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[186:187], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[178:179], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[188:189], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[180:181], v[72:73], v[50:65]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v67, 31, v66
	v_mul_u32_u24_e32 v67, 0xc0, v67
	v_lshrrev_b32_e32 v66, 2, v66
	v_and_or_b32 v66, v66, 8, v67
	v_lshrrev_b32_e32 v67, 3, v67
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v67, v67, v66
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[178:181], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[182:185], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[218:221], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	s_add_u32 s0, s0, 2
	s_addc_u32 s1, s1, 0
	v_mov_b64_e32 v[66:67], s[96:97]
	v_cmp_lt_i64_e32 vcc, s[0:1], v[66:67]
	s_add_i32 s8, s8, 8
	s_mov_b32 s88, s10
	s_mov_b32 s11, s9
	s_cbranch_vccz .LBB0_58
.LBB0_26:
	s_add_i32 s4, s8, -4
	v_mov_b32_e32 v66, s4
	buffer_load_dword v66, v66, s[76:79], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_mov_b32 s9, s71
	v_and_b32_e32 v67, 0x180, v67
	s_mov_b32 s10, s87
	v_cmp_ne_u32_e32 vcc, s85, v67
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s71, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_28
	ds_write_b128 v105, v[122:125]
	ds_write_b128 v105, v[126:129] offset:6144
.LBB0_28:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_30
	s_mul_i32 s12, s88, 0x1800
	v_add_u32_e32 v66, s12, v104
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[122:125], v[66:67], off
	global_load_dwordx4 v[126:129], v[66:67], off offset:256
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[130:131], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_lshl_b32 s12, s11, 13
	v_lshlrev_b32_e32 v88, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	s_add_i32 s12, s12, s86
	v_and_b32_e32 v88, 0xf8, v88
	v_and_b32_e32 v83, 0x400, v83
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[132:133], v[66:81]
	v_or3_b32 v88, v88, v83, s12
	v_ashrrev_i32_e32 v89, 31, v88
	v_lshl_add_u64 v[88:89], v[88:89], 1, s[82:83]
	global_load_dwordx4 v[226:229], v[88:89], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[178:179], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[180:181], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[138:139], v[66:81]
	global_load_dwordx4 v[230:233], v[88:89], off offset:512
	global_load_dwordx4 v[222:225], v[88:89], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[144:145], v[66:81]
	global_load_dwordx4 v[182:185], v[88:89], off offset:1536
	v_add_co_u32_e32 v88, vcc, s72, v88
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	global_load_dwordx4 v[98:101], v[88:89], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	global_load_dwordx4 v[178:181], v[88:89], off offset:512
	global_load_dwordx4 v[94:97], v[88:89], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[156:157], v[66:81]
	global_load_dwordx4 v[90:93], v[88:89], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[168:169], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[172:173], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[174:175], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[176:177], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v83, v82
	s_nop 6
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_pk_mul_f32 v[76:77], v[82:83], v[76:77]
	v_pk_mul_f32 v[74:75], v[82:83], v[74:75]
	v_pk_mul_f32 v[72:73], v[82:83], v[72:73]
	v_pk_mul_f32 v[70:71], v[82:83], v[70:71]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v83, v66, v67
	v_max3_f32 v83, v83, v68, v69
	v_max3_f32 v83, v83, v70, v71
	v_max3_f32 v83, v83, v72, v73
	v_max3_f32 v83, v83, v74, v75
	v_max3_f32 v83, v83, v76, v77
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v88, v106, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v88, v88, v88
	v_max_f32_e32 v88, v83, v88
	;;#ASMSTART
	v_add_f32 v83, v84, v108
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v88, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v88, v88, v109
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v84, v88
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v84, v88
.LBB0_32:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, 0, v66
	v_add_f32_e32 v88, v88, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v68
	v_add_f32_e32 v88, v88, v69
	v_add_f32_e32 v88, v88, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v71
	v_add_f32_e32 v88, v88, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v73
	v_add_f32_e32 v88, v88, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v75
	v_add_f32_e32 v88, v88, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v77
	v_add_f32_e32 v88, v88, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v88, v88, v79
	v_add_f32_e32 v88, v88, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v88, v88, v81
	;;#ASMSTART
	v_fma_f32 v111, v111, v83, v88
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_34
	;;#ASMSTART
	v_mul_f32 v2, v2, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v83
	;;#ASMEND
.LBB0_34:
	s_or_b64 exec, exec, s[4:5]
	v_add_u32_e32 v73, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s73
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[226:227], v[66:67], v[2:17]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[222:223], v[66:67], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[182:183], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[228:229], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[232:233], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[224:225], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[184:185], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[98:99], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[178:179], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[90:91], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[180:181], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[96:97], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[92:93], v[72:73], v[50:65]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v67, 31, v66
	v_mul_u32_u24_e32 v67, 0xc0, v67
	v_lshrrev_b32_e32 v66, 2, v66
	v_and_or_b32 v66, v66, 8, v67
	v_lshrrev_b32_e32 v67, 3, v67
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[182:185], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[88:91], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[92:95], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[100:103], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[96:99], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[178:181], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[226:229], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[230:233], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[222:225], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_36
	ds_write_b128 v105, v[114:117] offset:12288
	ds_write_b128 v105, v[118:121] offset:18432
.LBB0_36:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s11, s9, 0x1800
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_38
	v_add_u32_sdwa v66, s11, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[114:117], v[66:67], off
	global_load_dwordx4 v[118:121], v[66:67], off offset:256
.LBB0_38:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[130:131], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_addk_i32 s12, 0x1000
	v_lshlrev_b32_e32 v112, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	v_and_b32_e32 v112, 0xf8, v112
	v_and_b32_e32 v83, 0x400, v83
	v_or3_b32 v112, v112, v83, s12
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[132:133], v[66:81]
	v_ashrrev_i32_e32 v113, 31, v112
	v_lshl_add_u64 v[112:113], v[112:113], 1, s[82:83]
	global_load_dwordx4 v[202:205], v[112:113], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[88:89], v[134:135], v[66:81]
	v_add_co_u32_e32 v88, vcc, s72, v112
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v113, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[90:91], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[92:93], v[138:139], v[66:81]
	global_load_dwordx4 v[206:209], v[112:113], off offset:512
	global_load_dwordx4 v[194:197], v[112:113], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[94:95], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[144:145], v[66:81]
	global_load_dwordx4 v[198:201], v[112:113], off offset:1536
	global_load_dwordx4 v[186:189], v[88:89], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[178:179], v[150:151], v[66:81]
	global_load_dwordx4 v[190:193], v[88:89], off offset:512
	global_load_dwordx4 v[182:185], v[88:89], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[180:181], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[156:157], v[66:81]
	global_load_dwordx4 v[178:181], v[88:89], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[168:169], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[172:173], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[174:175], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[176:177], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v83, v82
	s_nop 6
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_pk_mul_f32 v[76:77], v[82:83], v[76:77]
	v_pk_mul_f32 v[74:75], v[82:83], v[74:75]
	v_pk_mul_f32 v[72:73], v[82:83], v[72:73]
	v_pk_mul_f32 v[70:71], v[82:83], v[70:71]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v83, v66, v67
	v_max3_f32 v83, v83, v68, v69
	v_max3_f32 v83, v83, v70, v71
	v_max3_f32 v83, v83, v72, v73
	v_max3_f32 v83, v83, v74, v75
	v_max3_f32 v83, v83, v76, v77
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v88, v106, v83
	v_mov_b32_e32 v89, v84
	v_mov_b32_e32 v90, v84
	v_mov_b32_e32 v91, v84
	v_mov_b32_e32 v92, v84
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v88, v88, v88
	v_max_f32_e32 v112, v83, v88
	;;#ASMSTART
	v_add_f32 v83, v84, v108
	;;#ASMEND
	v_mov_b32_e32 v88, v84
	v_cmp_gt_f32_e32 vcc, v112, v83
	v_mov_b32_e32 v83, 1.0
	v_mov_b32_e32 v93, v84
	v_mov_b32_e32 v94, v84
	v_mov_b32_e32 v95, v84
	v_mov_b32_e32 v96, v84
	v_mov_b32_e32 v97, v84
	v_mov_b32_e32 v98, v84
	v_mov_b32_e32 v99, v84
	v_mov_b32_e32 v100, v84
	v_mov_b32_e32 v101, v84
	v_mov_b32_e32 v102, v84
	v_mov_b32_e32 v103, v84
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_40
	;;#ASMSTART
	v_add_f32 v88, v112, v109
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v84, v88
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v84, v88
	v_mov_b32_e32 v89, v88
	v_mov_b32_e32 v90, v88
	v_mov_b32_e32 v91, v88
	v_mov_b32_e32 v92, v88
	v_mov_b32_e32 v93, v88
	v_mov_b32_e32 v94, v88
	v_mov_b32_e32 v95, v88
	v_mov_b32_e32 v96, v88
	v_mov_b32_e32 v97, v88
	v_mov_b32_e32 v98, v88
	v_mov_b32_e32 v99, v88
	v_mov_b32_e32 v100, v88
	v_mov_b32_e32 v101, v88
	v_mov_b32_e32 v102, v88
	v_mov_b32_e32 v103, v88
.LBB0_40:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, 0, v66
	v_add_f32_e32 v112, v112, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v68
	v_add_f32_e32 v112, v112, v69
	v_add_f32_e32 v112, v112, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v71
	v_add_f32_e32 v112, v112, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v73
	v_add_f32_e32 v112, v112, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v75
	v_add_f32_e32 v112, v112, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v77
	v_add_f32_e32 v112, v112, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v112, v112, v79
	v_add_f32_e32 v112, v112, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v112, v112, v81
	;;#ASMSTART
	v_fma_f32 v111, v111, v83, v112
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_42
	;;#ASMSTART
	v_mul_f32 v2, v2, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v83
	;;#ASMEND
.LBB0_42:
	s_or_b64 exec, exec, s[4:5]
	v_add_u32_e32 v73, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s73
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[202:203], v[66:67], v[2:17]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[206:207], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[66:67], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[186:187], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[178:179], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[188:189], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[180:181], v[72:73], v[50:65]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v67, 31, v66
	v_mul_u32_u24_e32 v67, 0xc0, v67
	v_lshrrev_b32_e32 v66, 2, v66
	v_and_or_b32 v66, v66, 8, v67
	v_lshrrev_b32_e32 v67, 3, v67
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v67, v67, v66
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[178:181], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[182:185], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[230:233], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[234:237], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[226:229], v66
	v_mov_b32_e32 v66, s8
	buffer_load_dword v66, v66, s[76:79], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s87, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s85, v67
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_44
	ds_write_b128 v105, v[122:125]
	ds_write_b128 v105, v[126:129] offset:6144
.LBB0_44:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_46
	v_add_u32_e32 v66, s11, v104
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[122:125], v[66:67], off
	global_load_dwordx4 v[126:129], v[66:67], off offset:256
.LBB0_46:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[130:131], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_lshl_b32 s11, s88, 13
	v_lshlrev_b32_e32 v112, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	s_add_i32 s11, s11, s86
	v_and_b32_e32 v112, 0xf8, v112
	v_and_b32_e32 v83, 0x400, v83
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[132:133], v[66:81]
	v_or3_b32 v112, v112, v83, s11
	v_ashrrev_i32_e32 v113, 31, v112
	v_lshl_add_u64 v[112:113], v[112:113], 1, s[82:83]
	global_load_dwordx4 v[202:205], v[112:113], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[178:179], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[180:181], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[138:139], v[66:81]
	global_load_dwordx4 v[206:209], v[112:113], off offset:512
	global_load_dwordx4 v[194:197], v[112:113], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[144:145], v[66:81]
	global_load_dwordx4 v[198:201], v[112:113], off offset:1536
	v_add_co_u32_e32 v112, vcc, s72, v112
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	global_load_dwordx4 v[186:189], v[112:113], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[150:151], v[66:81]
	global_load_dwordx4 v[190:193], v[112:113], off offset:512
	global_load_dwordx4 v[182:185], v[112:113], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	global_load_dwordx4 v[178:181], v[112:113], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[168:169], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[172:173], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[174:175], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[176:177], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v83, v82
	s_nop 6
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_pk_mul_f32 v[76:77], v[82:83], v[76:77]
	v_pk_mul_f32 v[74:75], v[82:83], v[74:75]
	v_pk_mul_f32 v[72:73], v[82:83], v[72:73]
	v_pk_mul_f32 v[70:71], v[82:83], v[70:71]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v83, v66, v67
	v_max3_f32 v83, v83, v68, v69
	v_max3_f32 v83, v83, v70, v71
	v_max3_f32 v83, v83, v72, v73
	v_max3_f32 v83, v83, v74, v75
	v_max3_f32 v83, v83, v76, v77
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v112, v106, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v112, v112, v112
	v_max_f32_e32 v112, v83, v112
	;;#ASMSTART
	v_add_f32 v83, v84, v108
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v112, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_48
	;;#ASMSTART
	v_add_f32 v88, v112, v109
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v84, v88
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v84, v88
	v_mov_b32_e32 v89, v88
	v_mov_b32_e32 v90, v88
	v_mov_b32_e32 v91, v88
	v_mov_b32_e32 v92, v88
	v_mov_b32_e32 v93, v88
	v_mov_b32_e32 v94, v88
	v_mov_b32_e32 v95, v88
	v_mov_b32_e32 v96, v88
	v_mov_b32_e32 v97, v88
	v_mov_b32_e32 v98, v88
	v_mov_b32_e32 v99, v88
	v_mov_b32_e32 v100, v88
	v_mov_b32_e32 v101, v88
	v_mov_b32_e32 v102, v88
	v_mov_b32_e32 v103, v88
.LBB0_48:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, 0, v66
	v_add_f32_e32 v112, v112, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v68
	v_add_f32_e32 v112, v112, v69
	v_add_f32_e32 v112, v112, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v71
	v_add_f32_e32 v112, v112, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v73
	v_add_f32_e32 v112, v112, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v75
	v_add_f32_e32 v112, v112, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v77
	v_add_f32_e32 v112, v112, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v112, v112, v79
	v_add_f32_e32 v112, v112, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v112, v112, v81
	;;#ASMSTART
	v_fma_f32 v111, v111, v83, v112
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_50
	;;#ASMSTART
	v_mul_f32 v2, v2, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v83
	;;#ASMEND
.LBB0_50:
	s_or_b64 exec, exec, s[4:5]
	v_add_u32_e32 v73, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s73
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[202:203], v[66:67], v[2:17]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[206:207], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[66:67], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[186:187], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[178:179], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[188:189], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[180:181], v[72:73], v[50:65]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v67, 31, v66
	v_mul_u32_u24_e32 v67, 0xc0, v67
	v_lshrrev_b32_e32 v66, 2, v66
	v_and_or_b32 v66, v66, 8, v67
	v_lshrrev_b32_e32 v67, 3, v67
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[194:197], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[178:181], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[182:185], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[230:233], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[234:237], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[226:229], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_52
	ds_write_b128 v105, v[114:117] offset:12288
	ds_write_b128 v105, v[118:121] offset:18432
.LBB0_52:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_54
	s_mul_i32 s12, s10, 0x1800
	v_add_u32_sdwa v66, s12, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[114:117], v[66:67], off
	global_load_dwordx4 v[118:121], v[66:67], off offset:256
.LBB0_54:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[130:131], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_addk_i32 s11, 0x1000
	v_lshlrev_b32_e32 v112, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	v_and_b32_e32 v112, 0xf8, v112
	v_and_b32_e32 v83, 0x400, v83
	v_or3_b32 v112, v112, v83, s11
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[132:133], v[66:81]
	v_ashrrev_i32_e32 v113, 31, v112
	v_lshl_add_u64 v[112:113], v[112:113], 1, s[82:83]
	global_load_dwordx4 v[202:205], v[112:113], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[178:179], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[180:181], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[138:139], v[66:81]
	global_load_dwordx4 v[206:209], v[112:113], off offset:512
	global_load_dwordx4 v[194:197], v[112:113], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[144:145], v[66:81]
	global_load_dwordx4 v[198:201], v[112:113], off offset:1536
	v_add_co_u32_e32 v112, vcc, s72, v112
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	global_load_dwordx4 v[186:189], v[112:113], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[150:151], v[66:81]
	global_load_dwordx4 v[190:193], v[112:113], off offset:512
	global_load_dwordx4 v[182:185], v[112:113], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	global_load_dwordx4 v[178:181], v[112:113], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[168:169], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[172:173], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[174:175], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[176:177], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v83, v82
	s_nop 6
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_pk_mul_f32 v[76:77], v[82:83], v[76:77]
	v_pk_mul_f32 v[74:75], v[82:83], v[74:75]
	v_pk_mul_f32 v[72:73], v[82:83], v[72:73]
	v_pk_mul_f32 v[70:71], v[82:83], v[70:71]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v83, v66, v67
	v_max3_f32 v83, v83, v68, v69
	v_max3_f32 v83, v83, v70, v71
	v_max3_f32 v83, v83, v72, v73
	v_max3_f32 v83, v83, v74, v75
	v_max3_f32 v83, v83, v76, v77
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v112, v106, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v112, v112, v112
	v_max_f32_e32 v112, v83, v112
	;;#ASMSTART
	v_add_f32 v83, v84, v108
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v112, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_56
	;;#ASMSTART
	v_add_f32 v88, v112, v109
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v84, v88
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v84, v88
	v_mov_b32_e32 v89, v88
	v_mov_b32_e32 v90, v88
	v_mov_b32_e32 v91, v88
	v_mov_b32_e32 v92, v88
	v_mov_b32_e32 v93, v88
	v_mov_b32_e32 v94, v88
	v_mov_b32_e32 v95, v88
	v_mov_b32_e32 v96, v88
	v_mov_b32_e32 v97, v88
	v_mov_b32_e32 v98, v88
	v_mov_b32_e32 v99, v88
	v_mov_b32_e32 v100, v88
	v_mov_b32_e32 v101, v88
	v_mov_b32_e32 v102, v88
	v_mov_b32_e32 v103, v88
.LBB0_56:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, 0, v66
	v_add_f32_e32 v88, v88, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v68
	v_add_f32_e32 v88, v88, v69
	v_add_f32_e32 v88, v88, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v71
	v_add_f32_e32 v88, v88, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v73
	v_add_f32_e32 v88, v88, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v75
	v_add_f32_e32 v88, v88, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v77
	v_add_f32_e32 v88, v88, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v88, v88, v79
	v_add_f32_e32 v88, v88, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v88, v88, v81
	;;#ASMSTART
	v_fma_f32 v111, v111, v83, v88
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_25
	;;#ASMSTART
	v_mul_f32 v2, v2, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v83
	;;#ASMEND
	s_branch .LBB0_25
.LBB0_58:
	s_mov_b32 s88, s10
	s_mov_b32 s11, s9
	s_cmp_ge_i32 s96, s92
	s_cbranch_scc0 .LBB0_60
	s_branch .LBB0_97
.LBB0_59:
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_mov_b64 s[0:1], s[46:47]
	s_mov_b32 s46, s36
	s_mov_b32 s47, s36
	s_mov_b64 s[4:5], s[48:49]
	s_mov_b32 s48, s36
	s_mov_b32 s49, s36
	s_mov_b64 s[8:9], s[50:51]
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	s_mov_b64 s[12:13], s[52:53]
	s_mov_b32 s52, s36
	s_mov_b32 s53, s36
	s_mov_b64 s[14:15], s[54:55]
	s_mov_b32 s54, s36
	s_mov_b32 s55, s36
	s_mov_b32 s56, s36
	s_mov_b32 s10, s57
	s_mov_b32 s57, s36
	s_mov_b32 s58, s36
	s_mov_b32 s59, s36
	s_mov_b32 s60, s36
	s_mov_b32 s61, s36
	s_mov_b32 s62, s36
	s_mov_b32 s63, s36
	s_mov_b32 s64, s36
	s_mov_b32 s65, s36
	s_mov_b32 s66, s36
	s_mov_b32 s67, s36
	v_mov_b64_e32 v[2:3], s[36:37]
	v_mov_b64_e32 v[4:5], s[38:39]
	v_mov_b64_e32 v[6:7], s[40:41]
	v_mov_b64_e32 v[8:9], s[42:43]
	v_mov_b64_e32 v[10:11], s[44:45]
	v_mov_b64_e32 v[12:13], s[46:47]
	v_mov_b64_e32 v[14:15], s[48:49]
	v_mov_b64_e32 v[16:17], s[50:51]
	v_mov_b64_e32 v[18:19], s[52:53]
	v_mov_b64_e32 v[20:21], s[54:55]
	v_mov_b64_e32 v[22:23], s[56:57]
	v_mov_b64_e32 v[24:25], s[58:59]
	v_mov_b64_e32 v[26:27], s[60:61]
	v_mov_b64_e32 v[28:29], s[62:63]
	v_mov_b64_e32 v[30:31], s[64:65]
	v_mov_b64_e32 v[32:33], s[66:67]
	v_mov_b32_e32 v84, 0xff800000
	v_mov_b32_e32 v111, 0
	s_mov_b32 s57, s10
	s_movk_i32 s56, 0xf80
	s_mov_b64 s[54:55], s[14:15]
	s_mov_b64 s[52:53], s[12:13]
	s_mov_b64 s[50:51], s[8:9]
	s_mov_b64 s[48:49], s[4:5]
	s_mov_b64 s[46:47], s[0:1]
	v_mov_b64_e32 v[64:65], v[32:33]
	v_mov_b64_e32 v[62:63], v[30:31]
	v_mov_b64_e32 v[60:61], v[28:29]
	v_mov_b64_e32 v[58:59], v[26:27]
	v_mov_b64_e32 v[56:57], v[24:25]
	v_mov_b64_e32 v[54:55], v[22:23]
	v_mov_b64_e32 v[52:53], v[20:21]
	v_mov_b64_e32 v[50:51], v[18:19]
	v_mov_b64_e32 v[48:49], v[16:17]
	v_mov_b64_e32 v[46:47], v[14:15]
	v_mov_b64_e32 v[44:45], v[12:13]
	v_mov_b64_e32 v[42:43], v[10:11]
	v_mov_b64_e32 v[40:41], v[8:9]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	s_cmp_ge_i32 s96, s92
	s_cbranch_scc1 .LBB0_97
.LBB0_60:
	s_lshl_b32 s37, s7, 6
	s_lshl_b32 s0, s96, 6
	s_ashr_i32 s93, s92, 31
	s_add_i32 s37, s37, s6
	v_mov_b32_e32 v86, v82
	v_mov_b32_e32 v87, v82
	s_add_i32 s40, s0, 0x77
	s_add_i32 s41, s96, 1
	s_lshl2_add_u32 s42, s96, 20
	s_branch .LBB0_63
.LBB0_61:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v73, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s73
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[202:203], v[66:67], v[2:17]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[206:207], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[66:67], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[186:187], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[178:179], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[188:189], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[180:181], v[72:73], v[50:65]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v67, 31, v66
	v_mul_u32_u24_e32 v67, 0xc0, v67
	v_lshrrev_b32_e32 v66, 2, v66
	v_and_or_b32 v66, v66, 8, v67
	v_lshrrev_b32_e32 v67, 3, v67
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v67, v67, v66
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[178:181], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[182:185], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[218:221], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
.LBB0_62:
	s_and_b64 s[0:1], s[38:39], exec
	s_cselect_b32 s11, s71, s88
	s_cselect_b32 s88, s87, s71
	s_cselect_b32 s71, s43, s87
	s_add_u32 s96, s96, 2
	s_addc_u32 s97, s97, 0
	v_mov_b64_e32 v[66:67], s[92:93]
	v_cmp_lt_i64_e32 vcc, s[96:97], v[66:67]
	s_addk_i32 s40, 0x80
	s_add_i32 s41, s41, 2
	s_add_i32 s42, s42, 8
	s_mov_b32 s87, s44
	s_cbranch_vccz .LBB0_97
.LBB0_63:
	s_add_i32 s0, s42, -4
	v_mov_b32_e32 v66, s0
	buffer_load_dword v66, v66, s[76:79], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s85, v67
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_65
	ds_write_b128 v105, v[122:125]
	ds_write_b128 v105, v[126:129] offset:6144
.LBB0_65:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_67
	s_mul_i32 s4, s88, 0x1800
	v_add_u32_e32 v66, s4, v104
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[122:125], v[66:67], off
	global_load_dwordx4 v[126:129], v[66:67], off offset:256
.LBB0_67:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[130:131], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_lshl_b32 s38, s11, 13
	v_lshlrev_b32_e32 v88, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	s_add_i32 s38, s38, s86
	v_and_b32_e32 v88, 0xf8, v88
	v_and_b32_e32 v83, 0x400, v83
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[132:133], v[66:81]
	v_or3_b32 v88, v88, v83, s38
	v_ashrrev_i32_e32 v89, 31, v88
	v_lshl_add_u64 v[88:89], v[88:89], 1, s[82:83]
	global_load_dwordx4 v[226:229], v[88:89], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[178:179], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[180:181], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[138:139], v[66:81]
	global_load_dwordx4 v[230:233], v[88:89], off offset:512
	global_load_dwordx4 v[222:225], v[88:89], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[144:145], v[66:81]
	global_load_dwordx4 v[182:185], v[88:89], off offset:1536
	v_add_co_u32_e32 v88, vcc, s72, v88
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	global_load_dwordx4 v[98:101], v[88:89], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	global_load_dwordx4 v[178:181], v[88:89], off offset:512
	global_load_dwordx4 v[94:97], v[88:89], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[156:157], v[66:81]
	global_load_dwordx4 v[90:93], v[88:89], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[168:169], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[172:173], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[174:175], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[176:177], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v83, 2, v83
	v_and_b32_e32 v83, 8, v83
	v_add_u32_e32 v83, s40, v83
	v_add_u32_e32 v88, 0xffffff89, v83
	v_cmp_gt_i32_e32 vcc, s37, v88
	v_add_u32_e32 v88, 0xffffff8a, v83
	v_cmp_gt_i32_e64 s[0:1], s37, v88
	v_add_u32_e32 v88, 0xffffff8b, v83
	v_cmp_gt_i32_e64 s[4:5], s37, v88
	v_add_u32_e32 v88, 0xffffff8c, v83
	v_cmp_gt_i32_e64 s[6:7], s37, v88
	v_add_u32_e32 v88, 0xffffff8d, v83
	v_cmp_gt_i32_e64 s[8:9], s37, v88
	v_add_u32_e32 v88, 0xffffff8e, v83
	v_cmp_gt_i32_e64 s[10:11], s37, v88
	v_add_u32_e32 v88, 0xffffff8f, v83
	v_cmp_gt_i32_e64 s[12:13], s37, v88
	v_add_u32_e32 v88, 0xffffff90, v83
	v_cmp_gt_i32_e64 s[14:15], s37, v88
	v_add_u32_e32 v88, 0xffffff99, v83
	v_cmp_gt_i32_e64 s[16:17], s37, v88
	v_add_u32_e32 v88, 0xffffff9a, v83
	v_cmp_gt_i32_e64 s[18:19], s37, v88
	v_add_u32_e32 v88, 0xffffff9b, v83
	v_cmp_gt_i32_e64 s[20:21], s37, v88
	v_add_u32_e32 v88, 0xffffff9c, v83
	v_cmp_gt_i32_e64 s[22:23], s37, v88
	v_add_u32_e32 v88, 0xffffff9d, v83
	v_cmp_gt_i32_e64 s[24:25], s37, v88
	v_add_u32_e32 v88, 0xffffff9e, v83
	v_cmp_gt_i32_e64 s[26:27], s37, v88
	v_add_u32_e32 v88, 0xffffff9f, v83
	v_add_u32_e32 v83, 0xffffffa0, v83
	v_cmp_gt_i32_e64 s[28:29], s37, v88
	v_cmp_gt_i32_e64 s[30:31], s37, v83
	s_or_b64 s[28:29], s[30:31], s[28:29]
	s_or_b64 s[26:27], s[28:29], s[26:27]
	s_or_b64 s[24:25], s[26:27], s[24:25]
	s_or_b64 s[22:23], s[24:25], s[22:23]
	s_or_b64 s[20:21], s[22:23], s[20:21]
	s_or_b64 s[18:19], s[20:21], s[18:19]
	s_or_b64 s[16:17], s[18:19], s[16:17]
	s_or_b64 s[14:15], s[16:17], s[14:15]
	s_or_b64 s[12:13], s[14:15], s[12:13]
	s_or_b64 s[10:11], s[12:13], s[10:11]
	s_or_b64 s[8:9], s[10:11], s[8:9]
	s_or_b64 s[6:7], s[8:9], s[6:7]
	s_or_b64 s[4:5], s[6:7], s[4:5]
	s_or_b64 s[0:1], s[4:5], s[0:1]
	s_or_b64 vcc, s[0:1], vcc
	v_cndmask_b32_e64 v69, v110, v69, s[6:7]
	v_cndmask_b32_e64 v68, v110, v68, s[4:5]
	v_cndmask_b32_e64 v89, v110, v67, s[0:1]
	v_cndmask_b32_e32 v88, v110, v66, vcc
	v_mov_b32_e32 v83, v82
	v_cndmask_b32_e64 v81, v110, v81, s[30:31]
	v_cndmask_b32_e64 v80, v110, v80, s[28:29]
	v_cndmask_b32_e64 v79, v110, v79, s[26:27]
	v_cndmask_b32_e64 v78, v110, v78, s[24:25]
	v_cndmask_b32_e64 v77, v110, v77, s[22:23]
	v_cndmask_b32_e64 v76, v110, v76, s[20:21]
	v_cndmask_b32_e64 v75, v110, v75, s[18:19]
	v_cndmask_b32_e64 v74, v110, v74, s[16:17]
	v_cndmask_b32_e64 v73, v110, v73, s[14:15]
	v_cndmask_b32_e64 v72, v110, v72, s[12:13]
	v_cndmask_b32_e64 v71, v110, v71, s[10:11]
	v_cndmask_b32_e64 v70, v110, v70, s[8:9]
	v_pk_mul_f32 v[66:67], v[82:83], v[68:69]
	v_pk_mul_f32 v[68:69], v[86:87], v[88:89]
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_pk_mul_f32 v[76:77], v[82:83], v[76:77]
	v_pk_mul_f32 v[74:75], v[82:83], v[74:75]
	v_pk_mul_f32 v[72:73], v[82:83], v[72:73]
	v_pk_mul_f32 v[70:71], v[82:83], v[70:71]
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
	v_max_f32_e32 v83, v68, v69
	v_max3_f32 v83, v83, v66, v67
	v_max3_f32 v83, v83, v70, v71
	v_max3_f32 v83, v83, v72, v73
	v_max3_f32 v83, v83, v74, v75
	v_max3_f32 v83, v83, v76, v77
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v88, v106, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v88, v88, v88
	v_max_f32_e32 v88, v83, v88
	;;#ASMSTART
	v_add_f32 v83, v84, v108
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v88, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_69
	;;#ASMSTART
	v_add_f32 v88, v88, v109
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v84, v88
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v84, v88
.LBB0_69:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[88:89], v[66:67], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67], v[68:69], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v68, v88
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v89
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, 0, v66
	v_add_f32_e32 v88, v88, v67
	v_add_f32_e32 v88, v88, v68
	v_add_f32_e32 v88, v88, v69
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v70
	v_add_f32_e32 v88, v88, v71
	v_add_f32_e32 v88, v88, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v73
	v_add_f32_e32 v88, v88, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v75
	v_add_f32_e32 v88, v88, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v77
	v_add_f32_e32 v88, v88, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v88, v88, v79
	v_add_f32_e32 v88, v88, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v88, v88, v81
	;;#ASMSTART
	v_fma_f32 v111, v111, v83, v88
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_71
	;;#ASMSTART
	v_mul_f32 v2, v2, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v83
	;;#ASMEND
.LBB0_71:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v73, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s73
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[226:227], v[66:67], v[2:17]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[222:223], v[66:67], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[182:183], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[228:229], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[232:233], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[224:225], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[184:185], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[98:99], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[178:179], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[90:91], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[180:181], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[96:97], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[92:93], v[72:73], v[50:65]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v67, 31, v66
	v_mul_u32_u24_e32 v67, 0xc0, v67
	v_lshrrev_b32_e32 v66, 2, v66
	v_and_or_b32 v66, v66, 8, v67
	v_lshrrev_b32_e32 v67, 3, v67
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[182:185], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[88:91], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[92:95], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[100:103], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[96:99], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[178:181], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[226:229], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[230:233], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[222:225], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_73
	ds_write_b128 v105, v[114:117] offset:12288
	ds_write_b128 v105, v[118:121] offset:18432
.LBB0_73:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s45, s71, 0x1800
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_75
	v_add_u32_sdwa v66, s45, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[114:117], v[66:67], off
	global_load_dwordx4 v[118:121], v[66:67], off offset:256
.LBB0_75:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[130:131], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_addk_i32 s38, 0x1000
	v_lshlrev_b32_e32 v112, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	v_and_b32_e32 v112, 0xf8, v112
	v_and_b32_e32 v83, 0x400, v83
	v_or3_b32 v112, v112, v83, s38
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[132:133], v[66:81]
	v_ashrrev_i32_e32 v113, 31, v112
	v_lshl_add_u64 v[112:113], v[112:113], 1, s[82:83]
	global_load_dwordx4 v[202:205], v[112:113], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[88:89], v[134:135], v[66:81]
	v_add_co_u32_e32 v88, vcc, s72, v112
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v113, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[90:91], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[92:93], v[138:139], v[66:81]
	global_load_dwordx4 v[206:209], v[112:113], off offset:512
	global_load_dwordx4 v[194:197], v[112:113], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[94:95], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[144:145], v[66:81]
	global_load_dwordx4 v[198:201], v[112:113], off offset:1536
	global_load_dwordx4 v[186:189], v[88:89], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[178:179], v[150:151], v[66:81]
	global_load_dwordx4 v[190:193], v[88:89], off offset:512
	global_load_dwordx4 v[182:185], v[88:89], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[180:181], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[156:157], v[66:81]
	global_load_dwordx4 v[178:181], v[88:89], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[168:169], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[172:173], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[174:175], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[176:177], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	v_mov_b32_e32 v89, v84
	v_lshrrev_b32_e32 v83, 2, v83
	v_and_b32_e32 v83, 8, v83
	v_add_u32_e32 v83, s40, v83
	v_add_u32_e32 v88, 0xffffffa9, v83
	v_cmp_gt_i32_e32 vcc, s37, v88
	v_add_u32_e32 v88, 0xffffffaa, v83
	v_cmp_gt_i32_e64 s[0:1], s37, v88
	v_add_u32_e32 v88, 0xffffffab, v83
	v_cmp_gt_i32_e64 s[4:5], s37, v88
	v_add_u32_e32 v88, 0xffffffac, v83
	v_cmp_gt_i32_e64 s[6:7], s37, v88
	v_add_u32_e32 v88, 0xffffffad, v83
	v_cmp_gt_i32_e64 s[8:9], s37, v88
	v_add_u32_e32 v88, 0xffffffae, v83
	v_cmp_gt_i32_e64 s[10:11], s37, v88
	v_add_u32_e32 v88, 0xffffffaf, v83
	v_cmp_gt_i32_e64 s[12:13], s37, v88
	v_add_u32_e32 v88, 0xffffffb0, v83
	v_cmp_gt_i32_e64 s[14:15], s37, v88
	v_add_u32_e32 v88, 0xffffffb9, v83
	v_cmp_gt_i32_e64 s[16:17], s37, v88
	v_add_u32_e32 v88, 0xffffffba, v83
	v_cmp_gt_i32_e64 s[18:19], s37, v88
	v_add_u32_e32 v88, 0xffffffbb, v83
	v_cmp_gt_i32_e64 s[20:21], s37, v88
	v_add_u32_e32 v88, 0xffffffbc, v83
	v_cmp_gt_i32_e64 s[22:23], s37, v88
	v_add_u32_e32 v88, 0xffffffbd, v83
	v_cmp_gt_i32_e64 s[24:25], s37, v88
	v_add_u32_e32 v88, 0xffffffbe, v83
	v_cmp_gt_i32_e64 s[26:27], s37, v88
	v_add_u32_e32 v88, 0xffffffbf, v83
	v_subrev_u32_e32 v83, 64, v83
	v_cmp_gt_i32_e64 s[28:29], s37, v88
	v_cmp_gt_i32_e64 s[30:31], s37, v83
	s_or_b64 s[28:29], s[30:31], s[28:29]
	s_or_b64 s[26:27], s[28:29], s[26:27]
	s_or_b64 s[24:25], s[26:27], s[24:25]
	s_or_b64 s[22:23], s[24:25], s[22:23]
	s_or_b64 s[20:21], s[22:23], s[20:21]
	s_or_b64 s[18:19], s[20:21], s[18:19]
	s_or_b64 s[16:17], s[18:19], s[16:17]
	s_or_b64 s[14:15], s[16:17], s[14:15]
	s_or_b64 s[12:13], s[14:15], s[12:13]
	s_or_b64 s[10:11], s[12:13], s[10:11]
	s_or_b64 s[8:9], s[10:11], s[8:9]
	s_or_b64 s[6:7], s[8:9], s[6:7]
	s_or_b64 s[4:5], s[6:7], s[4:5]
	s_or_b64 s[0:1], s[4:5], s[0:1]
	s_or_b64 vcc, s[0:1], vcc
	v_cndmask_b32_e64 v67, v110, v67, s[0:1]
	v_cndmask_b32_e32 v66, v110, v66, vcc
	v_cndmask_b32_e64 v81, v110, v81, s[30:31]
	v_cndmask_b32_e64 v80, v110, v80, s[28:29]
	v_cndmask_b32_e64 v79, v110, v79, s[26:27]
	v_cndmask_b32_e64 v78, v110, v78, s[24:25]
	v_cndmask_b32_e64 v77, v110, v77, s[22:23]
	v_cndmask_b32_e64 v76, v110, v76, s[20:21]
	v_cndmask_b32_e64 v75, v110, v75, s[18:19]
	v_cndmask_b32_e64 v74, v110, v74, s[16:17]
	v_cndmask_b32_e64 v73, v110, v73, s[14:15]
	v_cndmask_b32_e64 v72, v110, v72, s[12:13]
	v_cndmask_b32_e64 v71, v110, v71, s[10:11]
	v_cndmask_b32_e64 v70, v110, v70, s[8:9]
	v_cndmask_b32_e64 v69, v110, v69, s[6:7]
	v_cndmask_b32_e64 v68, v110, v68, s[4:5]
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_pk_mul_f32 v[76:77], v[82:83], v[76:77]
	v_pk_mul_f32 v[74:75], v[82:83], v[74:75]
	v_pk_mul_f32 v[72:73], v[82:83], v[72:73]
	v_pk_mul_f32 v[70:71], v[82:83], v[70:71]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
	v_max_f32_e32 v83, v66, v67
	v_max3_f32 v83, v83, v68, v69
	v_max3_f32 v83, v83, v70, v71
	v_max3_f32 v83, v83, v72, v73
	v_max3_f32 v83, v83, v74, v75
	v_max3_f32 v83, v83, v76, v77
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v88, v106, v83
	v_mov_b32_e32 v90, v84
	v_mov_b32_e32 v91, v84
	v_mov_b32_e32 v92, v84
	v_mov_b32_e32 v93, v84
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v88, v88, v88
	v_max_f32_e32 v112, v83, v88
	;;#ASMSTART
	v_add_f32 v83, v84, v108
	;;#ASMEND
	v_mov_b32_e32 v88, v84
	v_cmp_gt_f32_e32 vcc, v112, v83
	v_mov_b32_e32 v83, 1.0
	v_mov_b32_e32 v94, v84
	v_mov_b32_e32 v95, v84
	v_mov_b32_e32 v96, v84
	v_mov_b32_e32 v97, v84
	v_mov_b32_e32 v98, v84
	v_mov_b32_e32 v99, v84
	v_mov_b32_e32 v100, v84
	v_mov_b32_e32 v101, v84
	v_mov_b32_e32 v102, v84
	v_mov_b32_e32 v103, v84
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_77
	;;#ASMSTART
	v_add_f32 v88, v112, v109
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v84, v88
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v84, v88
	v_mov_b32_e32 v89, v88
	v_mov_b32_e32 v90, v88
	v_mov_b32_e32 v91, v88
	v_mov_b32_e32 v92, v88
	v_mov_b32_e32 v93, v88
	v_mov_b32_e32 v94, v88
	v_mov_b32_e32 v95, v88
	v_mov_b32_e32 v96, v88
	v_mov_b32_e32 v97, v88
	v_mov_b32_e32 v98, v88
	v_mov_b32_e32 v99, v88
	v_mov_b32_e32 v100, v88
	v_mov_b32_e32 v101, v88
	v_mov_b32_e32 v102, v88
	v_mov_b32_e32 v103, v88
.LBB0_77:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, 0, v66
	v_add_f32_e32 v112, v112, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v68
	v_add_f32_e32 v112, v112, v69
	v_add_f32_e32 v112, v112, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v71
	v_add_f32_e32 v112, v112, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v73
	v_add_f32_e32 v112, v112, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v75
	v_add_f32_e32 v112, v112, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v77
	v_add_f32_e32 v112, v112, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v112, v112, v79
	v_add_f32_e32 v112, v112, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v112, v112, v81
	;;#ASMSTART
	v_fma_f32 v111, v111, v83, v112
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_79
	;;#ASMSTART
	v_mul_f32 v2, v2, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v83
	;;#ASMEND
.LBB0_79:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v73, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s73
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[202:203], v[66:67], v[2:17]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[206:207], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[66:67], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[186:187], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[178:179], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[188:189], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[180:181], v[72:73], v[50:65]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v67, 31, v66
	v_mul_u32_u24_e32 v67, 0xc0, v67
	v_lshrrev_b32_e32 v66, 2, v66
	v_and_or_b32 v66, v66, 8, v67
	v_lshrrev_b32_e32 v67, 3, v67
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v67, v67, v66
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[178:181], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[182:185], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[218:221], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	s_cmp_gt_i32 s92, s41
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_le_i32 s92, s41
	s_cbranch_scc1 .LBB0_96
	v_mov_b32_e32 v66, s42
	buffer_load_dword v66, v66, s[76:79], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s85, v67
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_82
	ds_write_b128 v105, v[122:125]
	ds_write_b128 v105, v[126:129] offset:6144
.LBB0_82:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_84
	v_add_u32_e32 v66, s45, v104
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[122:125], v[66:67], off
	global_load_dwordx4 v[126:129], v[66:67], off offset:256
.LBB0_84:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[130:131], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_lshl_b32 s45, s88, 13
	v_lshlrev_b32_e32 v112, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	s_add_i32 s45, s45, s86
	v_and_b32_e32 v112, 0xf8, v112
	v_and_b32_e32 v83, 0x400, v83
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[132:133], v[66:81]
	v_or3_b32 v112, v112, v83, s45
	v_ashrrev_i32_e32 v113, 31, v112
	v_lshl_add_u64 v[112:113], v[112:113], 1, s[82:83]
	global_load_dwordx4 v[230:233], v[112:113], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[178:179], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[180:181], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[138:139], v[66:81]
	global_load_dwordx4 v[234:237], v[112:113], off offset:512
	global_load_dwordx4 v[222:225], v[112:113], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[144:145], v[66:81]
	global_load_dwordx4 v[226:229], v[112:113], off offset:1536
	v_add_co_u32_e32 v112, vcc, s72, v112
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	global_load_dwordx4 v[186:189], v[112:113], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	global_load_dwordx4 v[190:193], v[112:113], off offset:512
	global_load_dwordx4 v[182:185], v[112:113], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[156:157], v[66:81]
	global_load_dwordx4 v[178:181], v[112:113], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[168:169], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[172:173], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[174:175], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[176:177], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v83, 2, v83
	v_and_b32_e32 v83, 8, v83
	v_add_u32_e32 v83, s40, v83
	v_subrev_u32_e32 v112, 55, v83
	v_cmp_gt_i32_e32 vcc, s37, v112
	v_subrev_u32_e32 v112, 54, v83
	v_cmp_gt_i32_e64 s[0:1], s37, v112
	v_subrev_u32_e32 v112, 53, v83
	v_cmp_gt_i32_e64 s[4:5], s37, v112
	v_subrev_u32_e32 v112, 52, v83
	v_cmp_gt_i32_e64 s[6:7], s37, v112
	v_subrev_u32_e32 v112, 51, v83
	v_cmp_gt_i32_e64 s[8:9], s37, v112
	v_subrev_u32_e32 v112, 50, v83
	v_cmp_gt_i32_e64 s[10:11], s37, v112
	v_subrev_u32_e32 v112, 49, v83
	v_cmp_gt_i32_e64 s[12:13], s37, v112
	v_subrev_u32_e32 v112, 48, v83
	v_cmp_gt_i32_e64 s[14:15], s37, v112
	v_subrev_u32_e32 v112, 39, v83
	v_cmp_gt_i32_e64 s[16:17], s37, v112
	v_subrev_u32_e32 v112, 38, v83
	v_cmp_gt_i32_e64 s[18:19], s37, v112
	v_subrev_u32_e32 v112, 37, v83
	v_cmp_gt_i32_e64 s[20:21], s37, v112
	v_subrev_u32_e32 v112, 36, v83
	v_cmp_gt_i32_e64 s[22:23], s37, v112
	v_subrev_u32_e32 v112, 35, v83
	v_cmp_gt_i32_e64 s[24:25], s37, v112
	v_subrev_u32_e32 v112, 34, v83
	v_cmp_gt_i32_e64 s[26:27], s37, v112
	v_subrev_u32_e32 v112, 33, v83
	v_subrev_u32_e32 v83, 32, v83
	v_cmp_gt_i32_e64 s[28:29], s37, v112
	v_cmp_gt_i32_e64 s[30:31], s37, v83
	s_or_b64 s[28:29], s[30:31], s[28:29]
	s_or_b64 s[26:27], s[28:29], s[26:27]
	s_or_b64 s[24:25], s[26:27], s[24:25]
	s_or_b64 s[22:23], s[24:25], s[22:23]
	s_or_b64 s[20:21], s[22:23], s[20:21]
	s_or_b64 s[18:19], s[20:21], s[18:19]
	s_or_b64 s[16:17], s[18:19], s[16:17]
	s_or_b64 s[14:15], s[16:17], s[14:15]
	s_or_b64 s[12:13], s[14:15], s[12:13]
	s_or_b64 s[10:11], s[12:13], s[10:11]
	s_or_b64 s[8:9], s[10:11], s[8:9]
	s_or_b64 s[6:7], s[8:9], s[6:7]
	s_or_b64 s[4:5], s[6:7], s[4:5]
	s_or_b64 s[0:1], s[4:5], s[0:1]
	s_or_b64 vcc, s[0:1], vcc
	v_cndmask_b32_e64 v67, v110, v67, s[0:1]
	v_cndmask_b32_e32 v66, v110, v66, vcc
	v_cndmask_b32_e64 v81, v110, v81, s[30:31]
	v_cndmask_b32_e64 v80, v110, v80, s[28:29]
	v_cndmask_b32_e64 v79, v110, v79, s[26:27]
	v_cndmask_b32_e64 v78, v110, v78, s[24:25]
	v_cndmask_b32_e64 v77, v110, v77, s[22:23]
	v_cndmask_b32_e64 v76, v110, v76, s[20:21]
	v_cndmask_b32_e64 v75, v110, v75, s[18:19]
	v_cndmask_b32_e64 v74, v110, v74, s[16:17]
	v_cndmask_b32_e64 v73, v110, v73, s[14:15]
	v_cndmask_b32_e64 v72, v110, v72, s[12:13]
	v_cndmask_b32_e64 v71, v110, v71, s[10:11]
	v_cndmask_b32_e64 v70, v110, v70, s[8:9]
	v_cndmask_b32_e64 v69, v110, v69, s[6:7]
	v_cndmask_b32_e64 v68, v110, v68, s[4:5]
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_pk_mul_f32 v[76:77], v[82:83], v[76:77]
	v_pk_mul_f32 v[74:75], v[82:83], v[74:75]
	v_pk_mul_f32 v[72:73], v[82:83], v[72:73]
	v_pk_mul_f32 v[70:71], v[82:83], v[70:71]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
	v_max_f32_e32 v83, v66, v67
	v_max3_f32 v83, v83, v68, v69
	v_max3_f32 v83, v83, v70, v71
	v_max3_f32 v83, v83, v72, v73
	v_max3_f32 v83, v83, v74, v75
	v_max3_f32 v83, v83, v76, v77
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v112, v106, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v112, v112, v112
	v_max_f32_e32 v112, v83, v112
	;;#ASMSTART
	v_add_f32 v83, v84, v108
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v112, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_86
	;;#ASMSTART
	v_add_f32 v88, v112, v109
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v84, v88
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v84, v88
	v_mov_b32_e32 v89, v88
	v_mov_b32_e32 v90, v88
	v_mov_b32_e32 v91, v88
	v_mov_b32_e32 v92, v88
	v_mov_b32_e32 v93, v88
	v_mov_b32_e32 v94, v88
	v_mov_b32_e32 v95, v88
	v_mov_b32_e32 v96, v88
	v_mov_b32_e32 v97, v88
	v_mov_b32_e32 v98, v88
	v_mov_b32_e32 v99, v88
	v_mov_b32_e32 v100, v88
	v_mov_b32_e32 v101, v88
	v_mov_b32_e32 v102, v88
	v_mov_b32_e32 v103, v88
.LBB0_86:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, 0, v66
	v_add_f32_e32 v112, v112, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v68
	v_add_f32_e32 v112, v112, v69
	v_add_f32_e32 v112, v112, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v71
	v_add_f32_e32 v112, v112, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v73
	v_add_f32_e32 v112, v112, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v75
	v_add_f32_e32 v112, v112, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v77
	v_add_f32_e32 v112, v112, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v112, v112, v79
	v_add_f32_e32 v112, v112, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v112, v112, v81
	;;#ASMSTART
	v_fma_f32 v111, v111, v83, v112
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_88
	;;#ASMSTART
	v_mul_f32 v2, v2, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v83
	;;#ASMEND
.LBB0_88:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v73, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s73
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[230:231], v[66:67], v[2:17]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[234:235], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[222:223], v[66:67], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[226:227], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[232:233], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[236:237], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[224:225], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[228:229], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[186:187], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[178:179], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[188:189], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[180:181], v[72:73], v[50:65]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v67, 31, v66
	v_mul_u32_u24_e32 v67, 0xc0, v67
	v_lshrrev_b32_e32 v66, 2, v66
	v_and_or_b32 v66, v66, 8, v67
	v_lshrrev_b32_e32 v67, 3, v67
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[194:197], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[178:181], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[182:185], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[230:233], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[234:237], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[226:229], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_90
	ds_write_b128 v105, v[114:117] offset:12288
	ds_write_b128 v105, v[118:121] offset:18432
.LBB0_90:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s85, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_92
	s_mul_i32 s4, s87, 0x1800
	v_add_u32_sdwa v66, s4, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[114:117], v[66:67], off
	global_load_dwordx4 v[118:121], v[66:67], off offset:256
.LBB0_92:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[130:131], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_addk_i32 s45, 0x1000
	v_lshlrev_b32_e32 v112, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	v_and_b32_e32 v112, 0xf8, v112
	v_and_b32_e32 v83, 0x400, v83
	v_or3_b32 v112, v112, v83, s45
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[132:133], v[66:81]
	v_ashrrev_i32_e32 v113, 31, v112
	v_lshl_add_u64 v[112:113], v[112:113], 1, s[82:83]
	global_load_dwordx4 v[202:205], v[112:113], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[178:179], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[180:181], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[138:139], v[66:81]
	global_load_dwordx4 v[206:209], v[112:113], off offset:512
	global_load_dwordx4 v[194:197], v[112:113], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[144:145], v[66:81]
	global_load_dwordx4 v[198:201], v[112:113], off offset:1536
	v_add_co_u32_e32 v112, vcc, s72, v112
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	global_load_dwordx4 v[186:189], v[112:113], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[150:151], v[66:81]
	global_load_dwordx4 v[190:193], v[112:113], off offset:512
	global_load_dwordx4 v[182:185], v[112:113], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	global_load_dwordx4 v[178:181], v[112:113], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[168:169], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[172:173], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[174:175], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[176:177], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v83, 2, v83
	v_and_b32_e32 v83, 8, v83
	v_add_u32_e32 v83, s40, v83
	v_subrev_u32_e32 v112, 23, v83
	v_cmp_gt_i32_e32 vcc, s37, v112
	v_subrev_u32_e32 v112, 22, v83
	v_cmp_gt_i32_e64 s[0:1], s37, v112
	v_subrev_u32_e32 v112, 21, v83
	v_cmp_gt_i32_e64 s[4:5], s37, v112
	v_subrev_u32_e32 v112, 20, v83
	v_cmp_gt_i32_e64 s[6:7], s37, v112
	v_subrev_u32_e32 v112, 19, v83
	v_cmp_gt_i32_e64 s[8:9], s37, v112
	v_subrev_u32_e32 v112, 18, v83
	v_cmp_gt_i32_e64 s[10:11], s37, v112
	v_subrev_u32_e32 v112, 17, v83
	v_cmp_gt_i32_e64 s[12:13], s37, v112
	v_add_u32_e32 v112, -16, v83
	v_cmp_gt_i32_e64 s[14:15], s37, v112
	v_add_u32_e32 v112, -7, v83
	v_cmp_gt_i32_e64 s[16:17], s37, v112
	v_add_u32_e32 v112, -6, v83
	v_cmp_gt_i32_e64 s[18:19], s37, v112
	v_add_u32_e32 v112, -5, v83
	v_cmp_gt_i32_e64 s[20:21], s37, v112
	v_add_u32_e32 v112, -4, v83
	v_cmp_gt_i32_e64 s[22:23], s37, v112
	v_add_u32_e32 v112, -3, v83
	v_cmp_gt_i32_e64 s[24:25], s37, v112
	v_add_u32_e32 v112, -2, v83
	v_cmp_gt_i32_e64 s[26:27], s37, v112
	v_add_u32_e32 v112, -1, v83
	v_cmp_gt_i32_e64 s[28:29], s37, v112
	v_cmp_gt_i32_e64 s[30:31], s37, v83
	s_or_b64 s[28:29], s[30:31], s[28:29]
	s_or_b64 s[26:27], s[28:29], s[26:27]
	s_or_b64 s[24:25], s[26:27], s[24:25]
	s_or_b64 s[22:23], s[24:25], s[22:23]
	s_or_b64 s[20:21], s[22:23], s[20:21]
	s_or_b64 s[18:19], s[20:21], s[18:19]
	s_or_b64 s[16:17], s[18:19], s[16:17]
	s_or_b64 s[14:15], s[16:17], s[14:15]
	s_or_b64 s[12:13], s[14:15], s[12:13]
	s_or_b64 s[10:11], s[12:13], s[10:11]
	s_or_b64 s[8:9], s[10:11], s[8:9]
	s_or_b64 s[6:7], s[8:9], s[6:7]
	s_or_b64 s[4:5], s[6:7], s[4:5]
	s_or_b64 s[0:1], s[4:5], s[0:1]
	s_or_b64 vcc, s[0:1], vcc
	v_cndmask_b32_e64 v67, v110, v67, s[0:1]
	v_cndmask_b32_e32 v66, v110, v66, vcc
	v_cndmask_b32_e64 v81, v110, v81, s[30:31]
	v_cndmask_b32_e64 v80, v110, v80, s[28:29]
	v_cndmask_b32_e64 v79, v110, v79, s[26:27]
	v_cndmask_b32_e64 v78, v110, v78, s[24:25]
	v_cndmask_b32_e64 v77, v110, v77, s[22:23]
	v_cndmask_b32_e64 v76, v110, v76, s[20:21]
	v_cndmask_b32_e64 v75, v110, v75, s[18:19]
	v_cndmask_b32_e64 v74, v110, v74, s[16:17]
	v_cndmask_b32_e64 v73, v110, v73, s[14:15]
	v_cndmask_b32_e64 v72, v110, v72, s[12:13]
	v_cndmask_b32_e64 v71, v110, v71, s[10:11]
	v_cndmask_b32_e64 v70, v110, v70, s[8:9]
	v_cndmask_b32_e64 v69, v110, v69, s[6:7]
	v_cndmask_b32_e64 v68, v110, v68, s[4:5]
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_pk_mul_f32 v[76:77], v[82:83], v[76:77]
	v_pk_mul_f32 v[74:75], v[82:83], v[74:75]
	v_pk_mul_f32 v[72:73], v[82:83], v[72:73]
	v_pk_mul_f32 v[70:71], v[82:83], v[70:71]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
	v_max_f32_e32 v83, v66, v67
	v_max3_f32 v83, v83, v68, v69
	v_max3_f32 v83, v83, v70, v71
	v_max3_f32 v83, v83, v72, v73
	v_max3_f32 v83, v83, v74, v75
	v_max3_f32 v83, v83, v76, v77
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v112, v106, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v112, v112, v112
	v_max_f32_e32 v112, v83, v112
	;;#ASMSTART
	v_add_f32 v83, v84, v108
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v112, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_94
	;;#ASMSTART
	v_add_f32 v88, v112, v109
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v84, v88
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v84, v88
	v_mov_b32_e32 v89, v88
	v_mov_b32_e32 v90, v88
	v_mov_b32_e32 v91, v88
	v_mov_b32_e32 v92, v88
	v_mov_b32_e32 v93, v88
	v_mov_b32_e32 v94, v88
	v_mov_b32_e32 v95, v88
	v_mov_b32_e32 v96, v88
	v_mov_b32_e32 v97, v88
	v_mov_b32_e32 v98, v88
	v_mov_b32_e32 v99, v88
	v_mov_b32_e32 v100, v88
	v_mov_b32_e32 v101, v88
	v_mov_b32_e32 v102, v88
	v_mov_b32_e32 v103, v88
.LBB0_94:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, 0, v66
	v_add_f32_e32 v88, v88, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v68
	v_add_f32_e32 v88, v88, v69
	v_add_f32_e32 v88, v88, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v71
	v_add_f32_e32 v88, v88, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v73
	v_add_f32_e32 v88, v88, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v75
	v_add_f32_e32 v88, v88, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v77
	v_add_f32_e32 v88, v88, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v88, v88, v79
	v_add_f32_e32 v88, v88, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v88, v88, v81
	;;#ASMSTART
	v_fma_f32 v111, v111, v83, v88
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_61
	;;#ASMSTART
	v_mul_f32 v2, v2, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v83
	;;#ASMEND
	s_branch .LBB0_61
.LBB0_96:
	s_mov_b32 s44, s43
	s_branch .LBB0_62
.LBB0_97:
	ds_bpermute_b32 v66, v106, v111
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v66, v111, v66
	;;#ASMEND
	s_nop 0
	v_div_scale_f32 v67, s[0:1], v66, v66, s99
	v_rcp_f32_e32 v68, v67
	v_div_scale_f32 v69, vcc, s99, v66, s99
	v_fma_f32 v70, -v67, v68, 1.0
	v_fmac_f32_e32 v68, v70, v68
	v_mul_f32_e32 v70, v69, v68
	v_fma_f32 v71, -v67, v70, v69
	v_fmac_f32_e32 v70, v71, v68
	v_fma_f32 v67, -v67, v70, v69
	v_div_fmas_f32 v67, v67, v68, v70
	v_div_fixup_f32 v66, v67, v66, s99
	v_pk_mul_f32 v[2:3], v[2:3], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[66:67] op_sel_hi:[1,0]
	v_add_u32_e32 v34, 0x8000, v34
	v_add_u32_e32 v23, 0x8000, v23
	v_add_u32_e32 v22, 0x8000, v22
	v_add_u32_e32 v21, 0x8000, v21
	v_add_u32_e32 v20, 0x8000, v20
	v_add_u32_e32 v19, 0x8000, v19
	v_add_u32_e32 v18, 0x8000, v18
	v_add_u32_e32 v17, 0x8000, v17
	v_add_u32_e32 v16, 0x8000, v16
	v_add_u32_e32 v15, 0x8000, v15
	v_add_u32_e32 v14, 0x8000, v14
	v_add_u32_e32 v13, 0x8000, v13
	v_add_u32_e32 v12, 0x8000, v12
	v_add_u32_e32 v11, 0x8000, v11
	v_add_u32_e32 v10, 0x8000, v10
	v_add_u32_e32 v9, 0x8000, v9
	v_add_u32_e32 v8, 0x8000, v8
	v_add_u32_e32 v7, 0x8000, v7
	v_add_u32_e32 v6, 0x8000, v6
	v_add_u32_e32 v5, 0x8000, v5
	v_add_u32_e32 v4, 0x8000, v4
	v_add_u32_e32 v3, 0x8000, v3
	v_add_u32_e32 v2, 0x8000, v2
	v_add_u32_e32 v65, 0x8000, v65
	v_add_u32_e32 v64, 0x8000, v64
	v_add_u32_e32 v63, 0x8000, v63
	v_add_u32_e32 v62, 0x8000, v62
	v_add_u32_e32 v61, 0x8000, v61
	v_add_u32_e32 v60, 0x8000, v60
	v_add_u32_e32 v59, 0x8000, v59
	v_add_u32_e32 v58, 0x8000, v58
	v_add_u32_e32 v57, 0x8000, v57
	v_add_u32_e32 v56, 0x8000, v56
	v_add_u32_e32 v55, 0x8000, v55
	v_add_u32_e32 v54, 0x8000, v54
	v_add_u32_e32 v53, 0x8000, v53
	v_add_u32_e32 v52, 0x8000, v52
	v_add_u32_e32 v51, 0x8000, v51
	v_add_u32_e32 v50, 0x8000, v50
	v_add_u32_e32 v49, 0x8000, v49
	v_add_u32_e32 v48, 0x8000, v48
	v_add_u32_e32 v47, 0x8000, v47
	v_add_u32_e32 v46, 0x8000, v46
	v_add_u32_e32 v45, 0x8000, v45
	v_add_u32_e32 v44, 0x8000, v44
	v_add_u32_e32 v43, 0x8000, v43
	v_add_u32_e32 v42, 0x8000, v42
	v_add_u32_e32 v41, 0x8000, v41
	v_add_u32_e32 v40, 0x8000, v40
	v_add_u32_e32 v39, 0x8000, v39
	v_add_u32_e32 v38, 0x8000, v38
	v_add_u32_e32 v37, 0x8000, v37
	v_add_u32_e32 v36, 0x8000, v36
	v_add_u32_e32 v35, 0x8000, v35
	v_add_u32_e32 v66, 0x8000, v33
	v_add_u32_e32 v67, 0x8000, v32
	v_add_u32_e32 v68, 0x8000, v31
	v_add_u32_e32 v69, 0x8000, v30
	v_add_u32_e32 v70, 0x8000, v29
	v_add_u32_e32 v71, 0x8000, v28
	v_add_u32_e32 v72, 0x8000, v27
	v_add_u32_e32 v73, 0x8000, v26
	v_add_u32_e32 v74, 0x8000, v25
	v_add_u32_e32 v75, 0x8000, v24
	;;#ASMSTART
	v_perm_b32 v28, v3, v2, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v5, v4, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v7, v6, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v9, v8, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v11, v10, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v13, v12, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v15, v14, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v17, v16, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v19, v18, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v21, v20, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v23, v22, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v74, v75, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v72, v73, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v70, v71, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v68, v69, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v66, v67, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v35, v34, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v37, v36, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v39, v38, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v41, v40, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v43, v42, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v45, v44, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v47, v46, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v49, v48, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v51, v50, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v53, v52, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v55, v54, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v57, v56, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v59, v58, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v61, v60, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v63, v62, s73
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v65, v64, s73
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v34, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v34, 0x100, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_99
	s_barrier
.LBB0_99:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v37, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v34, 3, v37
	v_bfe_u32 v36, v37, 6, 3
	v_lshlrev_b32_e32 v35, 7, v37
	v_and_b32_e32 v34, 4, v34
	v_and_or_b32 v34, v35, s56, v34
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_101
	v_lshrrev_b32_e32 v38, 3, v35
	v_and_b32_e32 v38, 48, v38
	v_lshlrev_b32_e32 v39, 1, v34
	v_lshl_add_u32 v40, v38, 1, v39
	ds_write2_b64 v40, v[28:29], v[32:33] offset1:2
	v_or_b32_e32 v40, 16, v34
	v_xor_b32_e32 v40, v40, v38
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[30:31]
	v_or_b32_e32 v40, 24, v34
	v_xor_b32_e32 v40, v40, v38
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[26:27]
	v_or_b32_e32 v40, 32, v34
	v_xor_b32_e32 v40, v40, v38
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[24:25]
	v_or_b32_e32 v40, 40, v34
	v_xor_b32_e32 v40, v40, v38
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[22:23]
	v_or_b32_e32 v40, 48, v34
	v_xor_b32_e32 v40, v40, v38
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[20:21]
	v_or_b32_e32 v40, 56, v34
	v_xor_b32_e32 v38, v40, v38
	v_lshlrev_b32_e32 v38, 1, v38
	ds_write_b64 v38, v[18:19]
	v_or_b32_e32 v38, 64, v34
	v_lshrrev_b32_e32 v38, 2, v38
	v_and_b32_e32 v38, 0x70, v38
	v_add_u32_e32 v38, v39, v38
	ds_write_b64 v38, v[16:17] offset:128
	v_or_b32_e32 v38, 0x48, v34
	v_lshrrev_b32_e32 v39, 3, v38
	v_and_b32_e32 v39, 56, v39
	v_xor_b32_e32 v38, v39, v38
	v_lshlrev_b32_e32 v38, 1, v38
	ds_write_b64 v38, v[14:15]
	v_or_b32_e32 v38, 0x50, v34
	v_lshrrev_b32_e32 v39, 3, v38
	v_and_b32_e32 v39, 56, v39
	v_xor_b32_e32 v38, v39, v38
	v_lshlrev_b32_e32 v38, 1, v38
	ds_write_b64 v38, v[12:13]
	v_or_b32_e32 v38, 0x58, v34
	v_lshrrev_b32_e32 v39, 3, v38
	v_and_b32_e32 v39, 56, v39
	v_xor_b32_e32 v38, v39, v38
	v_lshlrev_b32_e32 v38, 1, v38
	ds_write_b64 v38, v[10:11]
	v_or_b32_e32 v38, 0x60, v34
	v_lshrrev_b32_e32 v39, 3, v38
	v_and_b32_e32 v39, 56, v39
	v_xor_b32_e32 v38, v39, v38
	v_lshlrev_b32_e32 v38, 1, v38
	ds_write_b64 v38, v[8:9]
	v_or_b32_e32 v38, 0x68, v34
	v_lshrrev_b32_e32 v39, 3, v38
	v_and_b32_e32 v39, 56, v39
	v_xor_b32_e32 v38, v39, v38
	v_lshlrev_b32_e32 v38, 1, v38
	ds_write_b64 v38, v[6:7]
	v_or_b32_e32 v38, 0x70, v34
	v_lshrrev_b32_e32 v39, 3, v38
	v_and_b32_e32 v39, 56, v39
	v_xor_b32_e32 v38, v39, v38
	v_lshlrev_b32_e32 v38, 1, v38
	ds_write_b64 v38, v[4:5]
	v_or_b32_e32 v38, 0x78, v34
	v_lshrrev_b32_e32 v39, 3, v38
	v_and_b32_e32 v39, 56, v39
	v_xor_b32_e32 v38, v39, v38
	v_lshlrev_b32_e32 v38, 1, v38
	ds_write_b64 v38, v[2:3]
.LBB0_101:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v39, 0x1ff, v37
	s_lshl_b32 s0, s70, 11
	v_lshlrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v37, 56, v37
	s_ashr_i32 s1, s0, 31
	v_xor_b32_e32 v37, v40, v37
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v37, 1, v37
	s_add_u32 s76, s54, s0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v39, 7, v39
	ds_read_b128 v[42:45], v37
	s_addc_u32 s0, s55, s1
	s_lshl_b32 s4, s84, 7
	v_and_b32_e32 v39, 0xf800, v39
	v_and_b32_e32 v38, 0x78, v40
	v_add_u32_e32 v40, s4, v39
	v_or_b32_e32 v40, v40, v38
	s_lshl_b32 s78, s81, 12
	s_and_b32 s77, s0, 0xffff
	v_lshlrev_b32_e32 v40, 1, v40
	v_cmp_eq_u32_e32 vcc, 1, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[42:45], v40, s[76:79], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_103
	v_lshrrev_b32_e32 v40, 3, v35
	v_and_b32_e32 v40, 48, v40
	v_lshlrev_b32_e32 v41, 1, v34
	v_lshl_add_u32 v42, v40, 1, v41
	ds_write2_b64 v42, v[28:29], v[32:33] offset1:2
	v_or_b32_e32 v42, 16, v34
	v_xor_b32_e32 v42, v42, v40
	v_lshlrev_b32_e32 v42, 1, v42
	ds_write_b64 v42, v[30:31]
	v_or_b32_e32 v42, 24, v34
	v_xor_b32_e32 v42, v42, v40
	v_lshlrev_b32_e32 v42, 1, v42
	ds_write_b64 v42, v[26:27]
	v_or_b32_e32 v42, 32, v34
	v_xor_b32_e32 v42, v42, v40
	v_lshlrev_b32_e32 v42, 1, v42
	ds_write_b64 v42, v[24:25]
	v_or_b32_e32 v42, 40, v34
	v_xor_b32_e32 v42, v42, v40
	v_lshlrev_b32_e32 v42, 1, v42
	ds_write_b64 v42, v[22:23]
	v_or_b32_e32 v42, 48, v34
	v_xor_b32_e32 v42, v42, v40
	v_lshlrev_b32_e32 v42, 1, v42
	ds_write_b64 v42, v[20:21]
	v_or_b32_e32 v42, 56, v34
	v_xor_b32_e32 v40, v42, v40
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[18:19]
	v_or_b32_e32 v40, 64, v34
	v_lshrrev_b32_e32 v40, 2, v40
	v_and_b32_e32 v40, 0x70, v40
	v_add_u32_e32 v40, v41, v40
	ds_write_b64 v40, v[16:17] offset:128
	v_or_b32_e32 v40, 0x48, v34
	v_lshrrev_b32_e32 v41, 3, v40
	v_and_b32_e32 v41, 56, v41
	v_xor_b32_e32 v40, v41, v40
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[14:15]
	v_or_b32_e32 v40, 0x50, v34
	v_lshrrev_b32_e32 v41, 3, v40
	v_and_b32_e32 v41, 56, v41
	v_xor_b32_e32 v40, v41, v40
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[12:13]
	v_or_b32_e32 v40, 0x58, v34
	v_lshrrev_b32_e32 v41, 3, v40
	v_and_b32_e32 v41, 56, v41
	v_xor_b32_e32 v40, v41, v40
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[10:11]
	v_or_b32_e32 v40, 0x60, v34
	v_lshrrev_b32_e32 v41, 3, v40
	v_and_b32_e32 v41, 56, v41
	v_xor_b32_e32 v40, v41, v40
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[8:9]
	v_or_b32_e32 v40, 0x68, v34
	v_lshrrev_b32_e32 v41, 3, v40
	v_and_b32_e32 v41, 56, v41
	v_xor_b32_e32 v40, v41, v40
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[6:7]
	v_or_b32_e32 v40, 0x70, v34
	v_lshrrev_b32_e32 v41, 3, v40
	v_and_b32_e32 v41, 56, v41
	v_xor_b32_e32 v40, v41, v40
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[4:5]
	v_or_b32_e32 v40, 0x78, v34
	v_lshrrev_b32_e32 v41, 3, v40
	v_and_b32_e32 v41, 56, v41
	v_xor_b32_e32 v40, v41, v40
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[2:3]
.LBB0_103:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s4, s4, 0x10000
	v_or_b32_e32 v38, v38, v39
	v_add_lshl_u32 v39, s4, v38, 1
	v_cmp_eq_u32_e32 vcc, 2, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[76:79], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_105
	v_lshrrev_b32_e32 v39, 3, v35
	v_and_b32_e32 v39, 48, v39
	v_lshlrev_b32_e32 v40, 1, v34
	v_lshl_add_u32 v41, v39, 1, v40
	ds_write2_b64 v41, v[28:29], v[32:33] offset1:2
	v_or_b32_e32 v41, 16, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[30:31]
	v_or_b32_e32 v41, 24, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[26:27]
	v_or_b32_e32 v41, 32, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[24:25]
	v_or_b32_e32 v41, 40, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[22:23]
	v_or_b32_e32 v41, 48, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[20:21]
	v_or_b32_e32 v41, 56, v34
	v_xor_b32_e32 v39, v41, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[18:19]
	v_or_b32_e32 v39, 64, v34
	v_lshrrev_b32_e32 v39, 2, v39
	v_and_b32_e32 v39, 0x70, v39
	v_add_u32_e32 v39, v40, v39
	ds_write_b64 v39, v[16:17] offset:128
	v_or_b32_e32 v39, 0x48, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[14:15]
	v_or_b32_e32 v39, 0x50, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[12:13]
	v_or_b32_e32 v39, 0x58, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[10:11]
	v_or_b32_e32 v39, 0x60, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[8:9]
	v_or_b32_e32 v39, 0x68, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[6:7]
	v_or_b32_e32 v39, 0x70, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[4:5]
	v_or_b32_e32 v39, 0x78, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[2:3]
.LBB0_105:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x10000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 3, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[76:79], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_107
	v_lshrrev_b32_e32 v39, 3, v35
	v_and_b32_e32 v39, 48, v39
	v_lshlrev_b32_e32 v40, 1, v34
	v_lshl_add_u32 v41, v39, 1, v40
	ds_write2_b64 v41, v[28:29], v[32:33] offset1:2
	v_or_b32_e32 v41, 16, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[30:31]
	v_or_b32_e32 v41, 24, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[26:27]
	v_or_b32_e32 v41, 32, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[24:25]
	v_or_b32_e32 v41, 40, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[22:23]
	v_or_b32_e32 v41, 48, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[20:21]
	v_or_b32_e32 v41, 56, v34
	v_xor_b32_e32 v39, v41, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[18:19]
	v_or_b32_e32 v39, 64, v34
	v_lshrrev_b32_e32 v39, 2, v39
	v_and_b32_e32 v39, 0x70, v39
	v_add_u32_e32 v39, v40, v39
	ds_write_b64 v39, v[16:17] offset:128
	v_or_b32_e32 v39, 0x48, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[14:15]
	v_or_b32_e32 v39, 0x50, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[12:13]
	v_or_b32_e32 v39, 0x58, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[10:11]
	v_or_b32_e32 v39, 0x60, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[8:9]
	v_or_b32_e32 v39, 0x68, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[6:7]
	v_or_b32_e32 v39, 0x70, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[4:5]
	v_or_b32_e32 v39, 0x78, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[2:3]
.LBB0_107:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x20000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 4, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[76:79], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_109
	v_lshrrev_b32_e32 v39, 3, v35
	v_and_b32_e32 v39, 48, v39
	v_lshlrev_b32_e32 v40, 1, v34
	v_lshl_add_u32 v41, v39, 1, v40
	ds_write2_b64 v41, v[28:29], v[32:33] offset1:2
	v_or_b32_e32 v41, 16, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[30:31]
	v_or_b32_e32 v41, 24, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[26:27]
	v_or_b32_e32 v41, 32, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[24:25]
	v_or_b32_e32 v41, 40, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[22:23]
	v_or_b32_e32 v41, 48, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[20:21]
	v_or_b32_e32 v41, 56, v34
	v_xor_b32_e32 v39, v41, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[18:19]
	v_or_b32_e32 v39, 64, v34
	v_lshrrev_b32_e32 v39, 2, v39
	v_and_b32_e32 v39, 0x70, v39
	v_add_u32_e32 v39, v40, v39
	ds_write_b64 v39, v[16:17] offset:128
	v_or_b32_e32 v39, 0x48, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[14:15]
	v_or_b32_e32 v39, 0x50, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[12:13]
	v_or_b32_e32 v39, 0x58, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[10:11]
	v_or_b32_e32 v39, 0x60, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[8:9]
	v_or_b32_e32 v39, 0x68, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[6:7]
	v_or_b32_e32 v39, 0x70, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[4:5]
	v_or_b32_e32 v39, 0x78, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[2:3]
.LBB0_109:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x30000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 5, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[76:79], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_111
	v_lshrrev_b32_e32 v39, 3, v35
	v_and_b32_e32 v39, 48, v39
	v_lshlrev_b32_e32 v40, 1, v34
	v_lshl_add_u32 v41, v39, 1, v40
	ds_write2_b64 v41, v[28:29], v[32:33] offset1:2
	v_or_b32_e32 v41, 16, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[30:31]
	v_or_b32_e32 v41, 24, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[26:27]
	v_or_b32_e32 v41, 32, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[24:25]
	v_or_b32_e32 v41, 40, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[22:23]
	v_or_b32_e32 v41, 48, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[20:21]
	v_or_b32_e32 v41, 56, v34
	v_xor_b32_e32 v39, v41, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[18:19]
	v_or_b32_e32 v39, 64, v34
	v_lshrrev_b32_e32 v39, 2, v39
	v_and_b32_e32 v39, 0x70, v39
	v_add_u32_e32 v39, v40, v39
	ds_write_b64 v39, v[16:17] offset:128
	v_or_b32_e32 v39, 0x48, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[14:15]
	v_or_b32_e32 v39, 0x50, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[12:13]
	v_or_b32_e32 v39, 0x58, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[10:11]
	v_or_b32_e32 v39, 0x60, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[8:9]
	v_or_b32_e32 v39, 0x68, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[6:7]
	v_or_b32_e32 v39, 0x70, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[4:5]
	v_or_b32_e32 v39, 0x78, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[2:3]
.LBB0_111:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x40000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 6, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[76:79], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_113
	v_lshrrev_b32_e32 v39, 3, v35
	v_and_b32_e32 v39, 48, v39
	v_lshlrev_b32_e32 v40, 1, v34
	v_lshl_add_u32 v41, v39, 1, v40
	ds_write2_b64 v41, v[28:29], v[32:33] offset1:2
	v_or_b32_e32 v41, 16, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[30:31]
	v_or_b32_e32 v41, 24, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[26:27]
	v_or_b32_e32 v41, 32, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[24:25]
	v_or_b32_e32 v41, 40, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[22:23]
	v_or_b32_e32 v41, 48, v34
	v_xor_b32_e32 v41, v41, v39
	v_lshlrev_b32_e32 v41, 1, v41
	ds_write_b64 v41, v[20:21]
	v_or_b32_e32 v41, 56, v34
	v_xor_b32_e32 v39, v41, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[18:19]
	v_or_b32_e32 v39, 64, v34
	v_lshrrev_b32_e32 v39, 2, v39
	v_and_b32_e32 v39, 0x70, v39
	v_add_u32_e32 v39, v40, v39
	ds_write_b64 v39, v[16:17] offset:128
	v_or_b32_e32 v39, 0x48, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[14:15]
	v_or_b32_e32 v39, 0x50, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[12:13]
	v_or_b32_e32 v39, 0x58, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[10:11]
	v_or_b32_e32 v39, 0x60, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[8:9]
	v_or_b32_e32 v39, 0x68, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[6:7]
	v_or_b32_e32 v39, 0x70, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[4:5]
	v_or_b32_e32 v39, 0x78, v34
	v_lshrrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v40, 56, v40
	v_xor_b32_e32 v39, v40, v39
	v_lshlrev_b32_e32 v39, 1, v39
	ds_write_b64 v39, v[2:3]
.LBB0_113:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x50000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 7, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[76:79], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_115
	v_lshrrev_b32_e32 v35, 3, v35
	v_and_b32_e32 v35, 48, v35
	v_lshlrev_b32_e32 v36, 1, v34
	v_lshl_add_u32 v39, v35, 1, v36
	ds_write2_b64 v39, v[28:29], v[32:33] offset1:2
	v_or_b32_e32 v28, 16, v34
	v_xor_b32_e32 v28, v28, v35
	v_lshlrev_b32_e32 v28, 1, v28
	ds_write_b64 v28, v[30:31]
	v_or_b32_e32 v28, 24, v34
	v_xor_b32_e32 v28, v28, v35
	v_lshlrev_b32_e32 v28, 1, v28
	ds_write_b64 v28, v[26:27]
	v_or_b32_e32 v26, 32, v34
	v_xor_b32_e32 v26, v26, v35
	v_lshlrev_b32_e32 v26, 1, v26
	ds_write_b64 v26, v[24:25]
	v_or_b32_e32 v24, 40, v34
	v_xor_b32_e32 v24, v24, v35
	v_lshlrev_b32_e32 v24, 1, v24
	ds_write_b64 v24, v[22:23]
	v_or_b32_e32 v22, 48, v34
	v_xor_b32_e32 v22, v22, v35
	v_lshlrev_b32_e32 v22, 1, v22
	ds_write_b64 v22, v[20:21]
	v_or_b32_e32 v20, 56, v34
	v_xor_b32_e32 v20, v20, v35
	v_lshlrev_b32_e32 v20, 1, v20
	ds_write_b64 v20, v[18:19]
	v_or_b32_e32 v18, 64, v34
	v_lshrrev_b32_e32 v18, 2, v18
	v_and_b32_e32 v18, 0x70, v18
	v_add_u32_e32 v18, v36, v18
	ds_write_b64 v18, v[16:17] offset:128
	v_or_b32_e32 v16, 0x48, v34
	v_lshrrev_b32_e32 v17, 3, v16
	v_and_b32_e32 v17, 56, v17
	v_xor_b32_e32 v16, v17, v16
	v_lshlrev_b32_e32 v16, 1, v16
	ds_write_b64 v16, v[14:15]
	v_or_b32_e32 v14, 0x50, v34
	v_lshrrev_b32_e32 v15, 3, v14
	v_and_b32_e32 v15, 56, v15
	v_xor_b32_e32 v14, v15, v14
	v_lshlrev_b32_e32 v14, 1, v14
	ds_write_b64 v14, v[12:13]
	v_or_b32_e32 v12, 0x58, v34
	v_lshrrev_b32_e32 v13, 3, v12
	v_and_b32_e32 v13, 56, v13
	v_xor_b32_e32 v12, v13, v12
	v_lshlrev_b32_e32 v12, 1, v12
	ds_write_b64 v12, v[10:11]
	v_or_b32_e32 v10, 0x60, v34
	v_lshrrev_b32_e32 v11, 3, v10
	v_and_b32_e32 v11, 56, v11
	v_xor_b32_e32 v10, v11, v10
	v_lshlrev_b32_e32 v10, 1, v10
	ds_write_b64 v10, v[8:9]
	v_or_b32_e32 v8, 0x68, v34
	v_lshrrev_b32_e32 v9, 3, v8
	v_and_b32_e32 v9, 56, v9
	v_xor_b32_e32 v8, v9, v8
	v_lshlrev_b32_e32 v8, 1, v8
	ds_write_b64 v8, v[6:7]
	v_or_b32_e32 v6, 0x70, v34
	v_lshrrev_b32_e32 v7, 3, v6
	v_and_b32_e32 v7, 56, v7
	v_xor_b32_e32 v6, v7, v6
	v_lshlrev_b32_e32 v6, 1, v6
	ds_write_b64 v6, v[4:5]
	v_or_b32_e32 v4, 0x78, v34
	v_lshrrev_b32_e32 v5, 3, v4
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v4, v5, v4
	v_lshlrev_b32_e32 v4, 1, v4
	ds_write_b64 v4, v[2:3]
.LBB0_115:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[4:7], v37
	s_add_i32 s4, s4, 0x60000
	v_add_lshl_u32 v2, s4, v38, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v2, s[76:79], 0 offen
	s_barrier
	;;#ASMSTART
	s_mov_b32 s0, s57
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_119
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_118
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v3, s8
	global_atomic_add v3, v107, v3, s[90:91] sc0
.LBB0_118:
	s_or_b64 exec, exec, s[6:7]
	s_lshl_b64 s[6:7], s[0:1], 2
	s_add_u32 s6, s90, s6
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v3
	s_addc_u32 s7, s91, s7
	s_nop 0
	v_add_u32_e32 v2, s8, v2
	global_store_dword v107, v2, s[6:7]
	s_waitcnt vmcnt(0)
.LBB0_119:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s90, s0
	s_addc_u32 s1, s91, s1
	s_barrier
	global_load_dword v2, v107, s[0:1]
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s4, v2
	s_add_i32 s0, s4, s33
	s_sub_i32 s33, s0, s2
	s_cmp_ge_i32 s80, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s98
	s_cselect_b64 s[6:7], -1, 0
	s_or_b64 s[0:1], s[0:1], s[6:7]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_122
	s_branch .LBB0_12
.LBB0_120:
	s_mov_b32 s80, s5
.LBB0_121:
	s_sub_i32 s33, s33, s98
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s84, 0, s2
	s_cmp_ge_i32 s80, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s98, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_122:
	s_add_i32 s2, s84, 1
	s_cmp_gt_i32 s2, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s2, 16
	s_cbranch_scc1 .LBB0_125
	s_add_i32 s5, s80, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s98
	s_cbranch_scc1 .LBB0_120
	s_ashr_i32 s81, s80, 31
	s_lshl_b64 s[6:7], s[80:81], 2
	s_add_u32 s6, s34, s6
	s_addc_u32 s7, s35, s7
	global_load_dwordx2 v[2:3], v107, s[6:7] offset:4
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s6, v3
	v_readfirstlane_b32 s7, v2
	s_sub_i32 s6, s6, s7
	s_add_i32 s8, s6, 0xff
	s_ashr_i32 s6, s8, 31
	s_lshr_b32 s6, s6, 24
	s_add_i32 s6, s8, s6
	s_ashr_i32 s10, s6, 8
	s_and_b32 s6, s6, 0xffffff00
	s_cmp_lg_u32 s8, s6
	s_cselect_b64 s[6:7], -1, 0
	s_cmp_lt_i32 s8, 0
	s_cselect_b64 s[8:9], -1, 0
	s_and_b64 s[6:7], s[8:9], s[6:7]
	s_subb_u32 s6, s10, 0
	s_branch .LBB0_120
.LBB0_125:
	s_mov_b32 s6, s98
	s_branch .LBB0_121
.LBB0_126:
	s_endpgm
.Lfunc_end0:
	.size	attn_kernel_0, .Lfunc_end0-attn_kernel_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_kernel_0
		.amdhsa_group_segment_fixed_size 24576
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 196
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 238
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 240
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text

	.set .Lattn_kernel_0.num_vgpr, 238
	.set .Lattn_kernel_0.num_agpr, 0
	.set .Lattn_kernel_0.numbered_sgpr, 100
	.set .Lattn_kernel_0.num_named_barrier, 0
	.set .Lattn_kernel_0.private_seg_size, 0
	.set .Lattn_kernel_0.uses_vcc, 1
	.set .Lattn_kernel_0.uses_flat_scratch, 0
	.set .Lattn_kernel_0.has_dyn_sized_stack, 0
	.set .Lattn_kernel_0.has_recursion, 0
	.set .Lattn_kernel_0.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     global_buffer
      - .offset:         144
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     global_buffer
      - .offset:         160
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         168
        .size:           8
        .value_kind:     global_buffer
      - .offset:         176
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         184
        .size:           8
        .value_kind:     global_buffer
      - .offset:         192
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 24576
    .kernarg_segment_align: 8
    .kernarg_segment_size: 196
    .max_flat_workgroup_size: 512
    .name:           attn_kernel_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 512
      - 1
      - 1
    .sgpr_count:     106
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     238
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
