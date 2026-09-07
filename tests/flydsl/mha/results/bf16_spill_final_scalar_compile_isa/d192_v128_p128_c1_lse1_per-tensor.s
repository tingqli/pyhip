	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	attn_kernel_0
	.p2align	8
	.type	attn_kernel_0,@function
attn_kernel_0:
	s_load_dwordx2 s[16:17], s[0:1], 0x30
	s_load_dword s3, s[0:1], 0x38
	s_load_dwordx8 s[4:11], s[0:1], 0x70
	s_mov_b32 s18, 0
	s_waitcnt lgkmcnt(0)
	s_load_dwordx2 s[12:13], s[16:17], 0x0
	s_add_i32 s3, s3, -1
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s12, s13, s12
	s_add_i32 s14, s12, 0xff
	s_ashr_i32 s12, s14, 31
	s_lshr_b32 s12, s12, 24
	s_add_i32 s12, s14, s12
	s_ashr_i32 s19, s12, 8
	s_and_b32 s12, s12, 0xffffff00
	s_cmp_lg_u32 s14, s12
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s14, 0
	s_cselect_b64 s[14:15], -1, 0
	s_and_b64 s[12:13], s[14:15], s[12:13]
	s_subb_u32 s14, s19, 0
	s_cmp_lt_i32 s3, 1
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s2, s14
	s_cselect_b64 s[20:21], -1, 0
	s_or_b64 s[12:13], s[12:13], s[20:21]
	s_and_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_8
	s_mov_b32 s24, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s18, s20
.LBB0_3:
	s_sub_i32 s33, s33, s14
	s_and_b64 s[12:13], s[12:13], exec
	s_cselect_b32 s24, 0, s15
	s_cmp_ge_i32 s18, s3
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s33, s72
	s_cselect_b64 s[14:15], -1, 0
	s_or_b64 s[12:13], s[12:13], s[14:15]
	s_and_b64 vcc, exec, s[12:13]
	s_mov_b32 s14, s72
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s15, s24, 1
	s_cmp_gt_i32 s15, 15
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s15, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s20, s18, 1
	s_cmp_ge_i32 s20, s3
	s_mov_b32 s72, s14
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s19, s18, 31
	s_lshl_b64 s[18:19], s[18:19], 2
	s_add_u32 s18, s16, s18
	s_addc_u32 s19, s17, s19
	s_load_dwordx2 s[22:23], s[18:19], 0x4
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s18, s23, s22
	s_add_i32 s21, s18, 0xff
	s_ashr_i32 s18, s21, 31
	s_lshr_b32 s18, s18, 24
	s_add_i32 s18, s21, s18
	s_ashr_i32 s24, s18, 8
	s_and_b32 s18, s18, 0xffffff00
	s_cmp_lg_u32 s21, s18
	s_cselect_b64 s[18:19], -1, 0
	s_cmp_lt_i32 s21, 0
	s_cselect_b64 s[22:23], -1, 0
	s_and_b64 s[18:19], s[22:23], s[18:19]
	s_subb_u32 s72, s24, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s72, s14
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s72, s14
	s_mov_b32 s24, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s73, s[6:7], 0x0
	s_load_dword s74, s[8:9], 0x0
	s_cmp_ge_i32 s18, s3
	s_cbranch_scc1 .LBB0_161
	v_lshrrev_b32_e32 v2, 6, v0
	v_lshrrev_b32_e32 v3, 2, v0
	v_mul_u32_u24_e32 v2, 0x18000, v2
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_load_dwordx2 s[20:21], s[0:1], 0x20
	s_load_dwordx2 s[22:23], s[0:1], 0x50
	s_load_dwordx2 s[26:27], s[0:1], 0x60
	s_load_dwordx2 s[28:29], s[0:1], 0x98
	s_load_dwordx2 s[30:31], s[0:1], 0xa8
	s_load_dwordx2 s[34:35], s[0:1], 0xb8
	v_and_b32_e32 v1, 31, v0
	v_and_or_b32 v2, v3, 8, v2
	s_movk_i32 s0, 0xc00
	v_mad_u32_u24 v85, v1, s0, v2
	v_mul_u32_u24_e32 v2, 0xaab, v0
	v_mov_b32_e32 v3, 24
	v_mul_lo_u16_sdwa v3, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_mul_u32_u24_e32 v4, 0x2ab, v0
	v_mov_b32_e32 v5, 2
	v_mov_b32_e32 v6, 5
	v_sub_u16_e32 v3, v0, v3
	v_lshlrev_b16_sdwa v2, v5, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_u32_u24_e32 v5, 0x156, v0
	v_lshlrev_b16_sdwa v4, v6, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mov_b32_e32 v6, 4
	v_lshlrev_b16_e32 v3, 9, v3
	v_lshlrev_b16_sdwa v5, v6, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v4, 32, v4
	v_or_b32_e32 v3, v3, v5
	v_and_b32_e32 v2, 12, v2
	v_add_u16_e32 v3, v3, v4
	v_or_b32_e32 v104, v3, v2
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	s_movk_i32 s0, 0x80
	v_xor_b32_e32 v106, v2, v3
	v_and_b32_e32 v2, 63, v0
	v_add_u32_sdwa v105, v104, s0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_movk_i32 s0, 0x100
	s_movk_i32 s75, 0x180
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_u32_sdwa v107, v104, s0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u32_sdwa v108, v104, s75 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v109, 0x80, v2
	v_mov_b32_e32 v110, 0
	s_mov_b32 s15, 0x27000
	v_mov_b32_e32 v111, 0x40e00000
	v_mov_b32_e32 v112, 1.0
	s_mov_b32 s76, 0x7060302
	s_movk_i32 s77, 0x1000
	s_movk_i32 s78, 0xe0
	s_mov_b32 s79, 0x800000
	s_movk_i32 s80, 0xf80
	s_mov_b32 s81, s2
	v_mov_b32_e32 v113, 0xff800000
	v_mov_b32_e32 v114, 0x42000000
	s_mov_b32 s36, 0
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s72, s14
.LBB0_12:
	s_cmp_ge_i32 s18, s3
	s_cbranch_scc1 .LBB0_161
.LBB0_13:
	s_ashr_i32 s19, s18, 31
	s_lshl_b32 s86, s33, 8
	s_lshl_b64 s[0:1], s[18:19], 2
	s_add_u32 s12, s16, s0
	s_addc_u32 s13, s17, s1
	global_load_dwordx2 v[4:5], v110, s[12:13]
	global_load_dword v2, v110, s[4:5]
	s_mul_i32 s12, s24, 0xc0
	v_add_lshl_u32 v3, s12, v85, 1
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s40, v4
	s_add_i32 s68, s40, s86
	v_readfirstlane_b32 s41, v5
	s_add_i32 s13, s68, 0x100
	s_min_i32 s13, s13, s41
	s_sub_i32 s19, s13, s68
	s_waitcnt lgkmcnt(0)
	s_add_u32 s38, s22, s0
	s_addc_u32 s39, s23, s1
	s_mul_i32 s12, s68, 0xc00
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_ashr_i32 s13, s12, 31
	global_load_dwordx2 v[4:5], v110, s[38:39]
	global_load_dword v6, v110, s[0:1]
	s_lshl_b64 s[0:1], s[12:13], 1
	s_add_u32 s12, s6, s0
	s_addc_u32 s0, s7, s1
	s_mul_i32 s14, s19, 0x1800
	s_and_b32 s13, s0, 0xffff
	buffer_load_dwordx4 v[118:121], v3, s[12:15], 0 offen
	buffer_load_dwordx4 v[122:125], v3, s[12:15], 0 offen offset:32
	buffer_load_dwordx4 v[126:129], v3, s[12:15], 0 offen offset:64
	buffer_load_dwordx4 v[130:133], v3, s[12:15], 0 offen offset:96
	buffer_load_dwordx4 v[134:137], v3, s[12:15], 0 offen offset:128
	buffer_load_dwordx4 v[138:141], v3, s[12:15], 0 offen offset:160
	buffer_load_dwordx4 v[142:145], v3, s[12:15], 0 offen offset:192
	buffer_load_dwordx4 v[146:149], v3, s[12:15], 0 offen offset:224
	buffer_load_dwordx4 v[150:153], v3, s[12:15], 0 offen offset:256
	buffer_load_dwordx4 v[154:157], v3, s[12:15], 0 offen offset:288
	buffer_load_dwordx4 v[158:161], v3, s[12:15], 0 offen offset:320
	buffer_load_dwordx4 v[162:165], v3, s[12:15], 0 offen offset:352
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_waitcnt vmcnt(13)
	v_readfirstlane_b32 s12, v4
	v_and_b32_e32 v3, 0x100, v3
	v_readfirstlane_b32 s13, v5
	s_waitcnt vmcnt(12)
	v_readfirstlane_b32 s37, v6
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_15
	s_barrier
.LBB0_15:
	s_or_b64 exec, exec, s[0:1]
	s_ashr_i32 s25, s24, 31
	s_lshr_b32 s0, s25, 28
	s_add_i32 s0, s24, s0
	s_sub_i32 s87, s13, s12
	s_ashr_i32 s13, s0, 4
	s_and_b32 s0, s0, -16
	s_cmp_lg_u32 s24, s0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s24, 0
	s_cselect_b64 s[38:39], -1, 0
	s_and_b64 s[0:1], s[38:39], s[0:1]
	s_subb_u32 s42, s13, 0
	s_mul_i32 s0, s42, 0x6000
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s8, s0
	s_addc_u32 s1, s9, s1
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s12, s26, s12
	s_addc_u32 s13, s27, s13
	s_lshl_b32 s14, s87, 2
	s_and_b32 s13, s13, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[12:15], 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s90, v4
	v_readfirstlane_b32 s84, v5
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
	buffer_load_dword v3, off, s[12:15], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s69, v3
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
	buffer_load_dword v3, off, s[12:15], 0 offset:12
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s85, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_mul_i32 s43, s90, 0x3000
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s75, v3
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_17
	v_add_u32_sdwa v4, s43, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[0:1]
	global_load_dwordx4 v[166:169], v[4:5], off
	global_load_dwordx4 v[170:173], v[4:5], off offset:256
.LBB0_17:
	s_or_b64 exec, exec, s[38:39]
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
	v_cmp_ne_u32_e32 vcc, s75, v3
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_19
	v_add_u32_e32 v4, s43, v105
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[0:1]
	global_load_dwordx4 v[174:177], v[4:5], off
	global_load_dwordx4 v[178:181], v[4:5], off offset:256
.LBB0_19:
	s_or_b64 exec, exec, s[38:39]
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
	v_cmp_ne_u32_e32 vcc, s75, v3
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_21
	s_waitcnt vmcnt(1)
	ds_write_b128 v106, v[166:169] offset:12288
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[170:173] offset:18432
.LBB0_21:
	s_or_b64 exec, exec, s[38:39]
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s75, v3
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_23
	v_add_u32_e32 v4, s43, v107
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[0:1]
	global_load_dwordx4 v[166:169], v[4:5], off
	global_load_dwordx4 v[170:173], v[4:5], off offset:256
.LBB0_23:
	s_or_b64 exec, exec, s[38:39]
	v_mul_f32_e32 v2, s73, v2
	v_mul_f32_e32 v82, 0x3dd53b95, v2
	s_lshl_b32 s82, s42, 14
	s_sub_i32 s38, s40, s41
	s_lshl_b32 s39, s87, 7
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
	ds_read_b128 v[182:185], v3 offset:12288
	v_add_u32_e32 v3, 0x1810, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[186:189], v3
	v_add_u32_e32 v3, 0x1820, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[190:193], v3
	v_add_u32_e32 v3, 0x1830, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[194:197], v3
	v_add_u32_e32 v3, 0x1840, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[198:201], v3
	v_add_u32_e32 v3, 0x1850, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[202:205], v3
	v_add_u32_e32 v3, 0x1860, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[206:209], v3
	v_add_u32_e32 v3, 0x1870, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[210:213], v3
	v_add_u32_e32 v3, 0x1880, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[214:217], v3
	v_add_u32_e32 v3, 0x1890, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[218:221], v3
	v_add_u32_e32 v3, 0x18a0, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_u32_e32 v2, 0x18b0, v2
	ds_read_b128 v[222:225], v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[226:229], v2
	s_add_i32 s38, s38, s39
	s_add_i32 s83, s38, s37
	s_addk_i32 s83, 0xff80
	s_add_i32 s89, s83, s86
	s_add_i32 s37, s89, 1
	s_ashr_i32 s38, s37, 31
	s_lshr_b32 s38, s38, 25
	s_add_i32 s38, s37, s38
	s_ashr_i32 s42, s38, 7
	s_and_b32 s38, s38, 0xffffff80
	s_cmp_lg_u32 s37, s38
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_lt_i32 s37, 0
	s_cselect_b64 s[40:41], -1, 0
	s_and_b64 s[38:39], s[40:41], s[38:39]
	s_subb_u32 s37, s42, 0
	s_lshr_b32 s38, s37, 31
	s_add_i32 s38, s37, s38
	s_ashr_i32 s42, s38, 1
	s_and_b32 s38, s38, -2
	s_cmp_lg_u32 s37, s38
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_lt_i32 s37, 0
	s_cselect_b64 s[40:41], -1, 0
	s_and_b64 s[38:39], s[40:41], s[38:39]
	s_subb_u32 s88, s42, 0
	s_lshl_b32 s70, s88, 1
	s_ashr_i32 s71, s70, 31
	s_cmp_lt_i32 s88, 1
	s_cbranch_scc1 .LBB0_75
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v86, v82
	v_mov_b32_e32 v87, v82
	s_mov_b64 s[38:39], 0
	s_mov_b32 s37, 20
	v_mov_b32_e32 v84, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v116
	v_mov_b32_e32 v4, v116
	v_mov_b32_e32 v5, v116
	v_mov_b32_e32 v6, v116
	v_mov_b32_e32 v7, v116
	v_mov_b32_e32 v8, v116
	v_mov_b32_e32 v9, v116
	v_mov_b32_e32 v10, v116
	v_mov_b32_e32 v11, v116
	v_mov_b32_e32 v12, v116
	v_mov_b32_e32 v13, v116
	v_mov_b32_e32 v14, v116
	v_mov_b32_e32 v15, v116
	v_mov_b32_e32 v16, v116
	v_mov_b32_e32 v17, v116
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v116
	v_mov_b32_e32 v20, v116
	v_mov_b32_e32 v21, v116
	v_mov_b32_e32 v22, v116
	v_mov_b32_e32 v23, v116
	v_mov_b32_e32 v24, v116
	v_mov_b32_e32 v25, v116
	v_mov_b32_e32 v26, v116
	v_mov_b32_e32 v27, v116
	v_mov_b32_e32 v28, v116
	v_mov_b32_e32 v29, v116
	v_mov_b32_e32 v30, v116
	v_mov_b32_e32 v31, v116
	v_mov_b32_e32 v32, v116
	v_mov_b32_e32 v33, v116
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v116
	v_mov_b32_e32 v36, v116
	v_mov_b32_e32 v37, v116
	v_mov_b32_e32 v38, v116
	v_mov_b32_e32 v39, v116
	v_mov_b32_e32 v40, v116
	v_mov_b32_e32 v41, v116
	v_mov_b32_e32 v42, v116
	v_mov_b32_e32 v43, v116
	v_mov_b32_e32 v44, v116
	v_mov_b32_e32 v45, v116
	v_mov_b32_e32 v46, v116
	v_mov_b32_e32 v47, v116
	v_mov_b32_e32 v48, v116
	v_mov_b32_e32 v49, v116
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v116
	v_mov_b32_e32 v52, v116
	v_mov_b32_e32 v53, v116
	v_mov_b32_e32 v54, v116
	v_mov_b32_e32 v55, v116
	v_mov_b32_e32 v56, v116
	v_mov_b32_e32 v57, v116
	v_mov_b32_e32 v58, v116
	v_mov_b32_e32 v59, v116
	v_mov_b32_e32 v60, v116
	v_mov_b32_e32 v61, v116
	v_mov_b32_e32 v62, v116
	v_mov_b32_e32 v63, v116
	v_mov_b32_e32 v64, v116
	v_mov_b32_e32 v65, v116
	s_branch .LBB0_26
.LBB0_25:
	s_or_b64 exec, exec, s[40:41]
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
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v88, v88, v79
	v_add_f32_e32 v88, v88, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v88, v88, v81
	;;#ASMSTART
	v_fma_f32 v116, v115, v83, v88
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s45, 0x2000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s45
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[88:91], v[74:75], off offset:512
	global_load_dwordx4 v[92:95], v[74:75], off offset:1024
	global_load_dwordx4 v[96:99], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[88:89], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[92:93], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[96:97], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[90:91], v[68:69], v[18:33]
	global_load_dwordx4 v[88:91], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[98:99], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[88:89], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[90:91], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[182:185], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[222:225], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[226:229], v66
	s_add_u32 s38, s38, 2
	s_addc_u32 s39, s39, 0
	v_mov_b64_e32 v[66:67], s[70:71]
	v_cmp_lt_i64_e32 vcc, s[38:39], v[66:67]
	s_add_i32 s37, s37, 8
	s_mov_b32 s84, s43
	s_mov_b32 s90, s42
	s_cbranch_vccz .LBB0_74
.LBB0_26:
	s_add_i32 s40, s37, -4
	v_mov_b32_e32 v66, s40
	buffer_load_dword v66, v66, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_mov_b32 s42, s69
	v_and_b32_e32 v67, 0x180, v67
	s_mov_b32 s43, s85
	v_cmp_ne_u32_e32 vcc, s75, v67
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s69, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_28
	ds_write_b128 v106, v[174:177]
	ds_write_b128 v106, v[178:181] offset:6144
.LBB0_28:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_30
	s_mul_i32 s44, s90, 0x3000
	v_add_u32_e32 v66, s44, v108
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[174:177], v[66:67], off
	global_load_dwordx4 v[178:181], v[66:67], off offset:256
.LBB0_30:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
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
	ds_bpermute_b32 v88, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v88, v88, v88
	v_max_f32_e32 v88, v83, v88
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v88, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v88, v88, v112
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v84, v88
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v84, v88
.LBB0_32:
	s_or_b64 exec, exec, s[40:41]
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
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v88, v88, v79
	v_add_f32_e32 v88, v88, v80
	v_add_f32_e32 v88, v88, v81
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	;;#ASMSTART
	v_fma_f32 v115, v116, v83, v88
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_lshl_b32 s45, s90, 14
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	s_add_i32 s45, s45, s82
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s45
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[88:91], v[74:75], off offset:512
	global_load_dwordx4 v[92:95], v[74:75], off offset:1024
	global_load_dwordx4 v[96:99], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[88:89], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[92:93], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[96:97], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[90:91], v[68:69], v[18:33]
	global_load_dwordx4 v[88:91], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[98:99], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[88:89], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[90:91], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[210:213], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[206:209], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[202:205], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[182:185], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[100:103], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[96:99], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[92:95], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[88:91], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_34
	ds_write_b128 v106, v[166:169] offset:12288
	ds_write_b128 v106, v[170:173] offset:18432
.LBB0_34:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s44, s84, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_36
	v_add_u32_sdwa v66, s44, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[166:169], v[66:67], off
	global_load_dwordx4 v[170:173], v[66:67], off offset:256
.LBB0_36:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[92:93], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[94:95], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[88:89], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[90:91], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
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
	ds_bpermute_b32 v88, v109, v83
	v_mov_b32_e32 v89, v84
	v_mov_b32_e32 v90, v84
	v_mov_b32_e32 v91, v84
	v_mov_b32_e32 v92, v84
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v88, v88, v88
	v_max_f32_e32 v116, v83, v88
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	v_mov_b32_e32 v88, v84
	v_cmp_gt_f32_e32 vcc, v116, v83
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
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_38
	;;#ASMSTART
	v_add_f32 v88, v116, v112
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
.LBB0_38:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, 0, v66
	v_add_f32_e32 v116, v116, v67
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
	v_add_f32_e32 v116, v116, v68
	v_add_f32_e32 v116, v116, v69
	v_add_f32_e32 v116, v116, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v71
	v_add_f32_e32 v116, v116, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v73
	v_add_f32_e32 v116, v116, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v75
	v_add_f32_e32 v116, v116, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v77
	v_add_f32_e32 v116, v116, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v116, v116, v79
	v_add_f32_e32 v116, v116, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v116, v116, v81
	;;#ASMSTART
	v_fma_f32 v115, v115, v83, v116
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s45, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s45
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_40
	ds_write_b128 v106, v[174:177]
	ds_write_b128 v106, v[178:181] offset:6144
.LBB0_40:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_42
	v_add_u32_e32 v66, s44, v105
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[174:177], v[66:67], off
	global_load_dwordx4 v[178:181], v[66:67], off offset:256
.LBB0_42:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
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
	ds_bpermute_b32 v116, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v116, v116, v116
	v_max_f32_e32 v116, v83, v116
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v116, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_44
	;;#ASMSTART
	v_add_f32 v88, v116, v112
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
.LBB0_44:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, 0, v66
	v_add_f32_e32 v116, v116, v67
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
	v_add_f32_e32 v116, v116, v68
	v_add_f32_e32 v116, v116, v69
	v_add_f32_e32 v116, v116, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v71
	v_add_f32_e32 v116, v116, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v73
	v_add_f32_e32 v116, v116, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v75
	v_add_f32_e32 v116, v116, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v77
	v_add_f32_e32 v116, v116, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v116, v116, v79
	v_add_f32_e32 v116, v116, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v116, v116, v81
	;;#ASMSTART
	v_fma_f32 v115, v115, v83, v116
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_add_i32 s40, s45, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s40
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[222:225], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[218:221], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_46
	ds_write_b128 v106, v[166:169] offset:12288
	ds_write_b128 v106, v[170:173] offset:18432
.LBB0_46:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_48
	v_add_u32_e32 v66, s44, v107
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[166:169], v[66:67], off
	global_load_dwordx4 v[170:173], v[66:67], off offset:256
.LBB0_48:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
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
	ds_bpermute_b32 v116, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v116, v116, v116
	v_max_f32_e32 v116, v83, v116
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v116, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_50
	;;#ASMSTART
	v_add_f32 v88, v116, v112
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
.LBB0_50:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, 0, v66
	v_add_f32_e32 v116, v116, v67
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
	v_add_f32_e32 v116, v116, v68
	v_add_f32_e32 v116, v116, v69
	v_add_f32_e32 v116, v116, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v71
	v_add_f32_e32 v116, v116, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v73
	v_add_f32_e32 v116, v116, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v75
	v_add_f32_e32 v116, v116, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v77
	v_add_f32_e32 v116, v116, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v116, v116, v79
	v_add_f32_e32 v116, v116, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v116, v116, v81
	;;#ASMSTART
	v_fma_f32 v115, v115, v83, v116
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s45, 0x2000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s45
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	v_mov_b32_e32 v66, s37
	buffer_load_dword v66, v66, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s85, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s75, v67
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_52
	ds_write_b128 v106, v[174:177]
	ds_write_b128 v106, v[178:181] offset:6144
.LBB0_52:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_54
	v_add_u32_e32 v66, s44, v108
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[174:177], v[66:67], off
	global_load_dwordx4 v[178:181], v[66:67], off offset:256
.LBB0_54:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
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
	ds_bpermute_b32 v116, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v116, v116, v116
	v_max_f32_e32 v116, v83, v116
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v116, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_56
	;;#ASMSTART
	v_add_f32 v88, v116, v112
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
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, 0, v66
	v_add_f32_e32 v116, v116, v67
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
	v_add_f32_e32 v116, v116, v68
	v_add_f32_e32 v116, v116, v69
	v_add_f32_e32 v116, v116, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v71
	v_add_f32_e32 v116, v116, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v73
	v_add_f32_e32 v116, v116, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v75
	v_add_f32_e32 v116, v116, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v77
	v_add_f32_e32 v116, v116, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v116, v116, v79
	v_add_f32_e32 v116, v116, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v116, v116, v81
	;;#ASMSTART
	v_fma_f32 v115, v115, v83, v116
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_lshl_b32 s45, s84, 14
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	s_add_i32 s45, s45, s82
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s45
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[222:225], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[218:221], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_58
	ds_write_b128 v106, v[166:169] offset:12288
	ds_write_b128 v106, v[170:173] offset:18432
.LBB0_58:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s44, s42, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_60
	v_add_u32_sdwa v66, s44, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[166:169], v[66:67], off
	global_load_dwordx4 v[170:173], v[66:67], off offset:256
.LBB0_60:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
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
	ds_bpermute_b32 v116, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v116, v116, v116
	v_max_f32_e32 v116, v83, v116
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v116, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_62
	;;#ASMSTART
	v_add_f32 v88, v116, v112
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
.LBB0_62:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, 0, v66
	v_add_f32_e32 v116, v116, v67
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
	v_add_f32_e32 v116, v116, v68
	v_add_f32_e32 v116, v116, v69
	v_add_f32_e32 v116, v116, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v71
	v_add_f32_e32 v116, v116, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v73
	v_add_f32_e32 v116, v116, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v75
	v_add_f32_e32 v116, v116, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v77
	v_add_f32_e32 v116, v116, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v116, v116, v79
	v_add_f32_e32 v116, v116, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v116, v116, v81
	;;#ASMSTART
	v_fma_f32 v115, v115, v83, v116
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s45, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s45
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_64
	ds_write_b128 v106, v[174:177]
	ds_write_b128 v106, v[178:181] offset:6144
.LBB0_64:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_66
	v_add_u32_e32 v66, s44, v105
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[174:177], v[66:67], off
	global_load_dwordx4 v[178:181], v[66:67], off offset:256
.LBB0_66:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
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
	ds_bpermute_b32 v116, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v116, v116, v116
	v_max_f32_e32 v116, v83, v116
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v116, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_68
	;;#ASMSTART
	v_add_f32 v88, v116, v112
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
.LBB0_68:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, 0, v66
	v_add_f32_e32 v116, v116, v67
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
	v_add_f32_e32 v116, v116, v68
	v_add_f32_e32 v116, v116, v69
	v_add_f32_e32 v116, v116, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v71
	v_add_f32_e32 v116, v116, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v73
	v_add_f32_e32 v116, v116, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v75
	v_add_f32_e32 v116, v116, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v77
	v_add_f32_e32 v116, v116, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v116, v116, v79
	v_add_f32_e32 v116, v116, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v116, v116, v81
	;;#ASMSTART
	v_fma_f32 v115, v115, v83, v116
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_add_i32 s40, s45, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s40
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[222:225], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[218:221], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_70
	ds_write_b128 v106, v[166:169] offset:12288
	ds_write_b128 v106, v[170:173] offset:18432
.LBB0_70:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_72
	v_add_u32_e32 v66, s44, v107
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[166:169], v[66:67], off
	global_load_dwordx4 v[170:173], v[66:67], off offset:256
.LBB0_72:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
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
	ds_bpermute_b32 v116, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v116, v116, v116
	v_max_f32_e32 v116, v83, v116
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v116, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_25
	;;#ASMSTART
	v_add_f32 v88, v116, v112
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
	s_branch .LBB0_25
.LBB0_74:
	s_mov_b32 s84, s43
	s_mov_b32 s90, s42
	s_branch .LBB0_76
.LBB0_75:
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_mov_b32 s46, s36
	s_mov_b32 s47, s36
	s_mov_b32 s48, s36
	s_mov_b32 s49, s36
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	s_mov_b32 s52, s36
	s_mov_b32 s53, s36
	s_mov_b32 s54, s36
	s_mov_b32 s55, s36
	s_mov_b32 s56, s36
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
	v_mov_b32_e32 v116, 0
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
.LBB0_76:
	s_add_i32 s37, s19, s89
	s_addk_i32 s37, 0x7f
	s_ashr_i32 s38, s37, 31
	s_lshr_b32 s38, s38, 25
	s_add_i32 s38, s37, s38
	s_ashr_i32 s42, s38, 7
	s_and_b32 s38, s38, 0xffffff80
	s_cmp_lg_u32 s37, s38
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_lt_i32 s37, 0
	s_cselect_b64 s[40:41], -1, 0
	s_and_b64 s[38:39], s[40:41], s[38:39]
	s_subb_u32 s37, s42, 0
	s_min_i32 s38, s37, s87
	s_cmp_ge_i32 s70, s38
	s_cbranch_scc1 .LBB0_130
	s_lshl_b32 s37, s88, 8
	s_ashr_i32 s39, s38, 31
	v_mov_b32_e32 v86, v82
	v_mov_b32_e32 v87, v82
	v_or_b32_e32 v115, s86, v1
	s_addk_i32 s37, 0xf7
	s_lshl3_add_u32 s44, s88, 20
	s_branch .LBB0_80
.LBB0_78:
	s_or_b64 exec, exec, s[42:43]
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
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v88, v88, v79
	v_add_f32_e32 v88, v88, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v88, v88, v81
	;;#ASMSTART
	v_fma_f32 v116, v116, v83, v88
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s48, 0x2000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s48
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[88:91], v[74:75], off offset:512
	global_load_dwordx4 v[92:95], v[74:75], off offset:1024
	global_load_dwordx4 v[96:99], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[88:89], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[92:93], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[96:97], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[90:91], v[68:69], v[18:33]
	global_load_dwordx4 v[88:91], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[98:99], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[88:89], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[90:91], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[182:185], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[222:225], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[226:229], v66
.LBB0_79:
	s_and_b64 s[40:41], s[40:41], exec
	s_cselect_b32 s90, s69, s84
	s_cselect_b32 s84, s85, s69
	s_cselect_b32 s69, s45, s85
	s_add_u32 s70, s70, 2
	s_addc_u32 s71, s71, 0
	v_mov_b64_e32 v[66:67], s[38:39]
	v_cmp_lt_i64_e32 vcc, s[70:71], v[66:67]
	s_addk_i32 s37, 0x100
	s_add_i32 s44, s44, 8
	s_mov_b32 s85, s46
	s_cbranch_vccz .LBB0_130
.LBB0_80:
	s_add_i32 s40, s44, -4
	v_mov_b32_e32 v66, s40
	buffer_load_dword v66, v66, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s75, v67
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_82
	ds_write_b128 v106, v[174:177]
	ds_write_b128 v106, v[178:181] offset:6144
.LBB0_82:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_84
	s_mul_i32 s42, s90, 0x3000
	v_add_u32_e32 v66, s42, v108
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[174:177], v[66:67], off
	global_load_dwordx4 v[178:181], v[66:67], off offset:256
.LBB0_84:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v88, 2, v83
	v_and_b32_e32 v88, 8, v88
	v_lshrrev_b32_e32 v83, 1, v83
	v_and_or_b32 v83, v83, s78, v115
	v_add_u32_e32 v90, s37, v88
	v_add_u32_e32 v83, s83, v83
	v_add_u32_e32 v88, 0xffffff09, v90
	v_cmp_lt_i32_e32 vcc, v88, v83
	s_nop 1
	v_cndmask_b32_e32 v89, v113, v67, vcc
	v_cmp_le_i32_e32 vcc, v88, v83
	v_add_u32_e32 v67, 0xffffff0c, v90
	s_nop 0
	v_cndmask_b32_e32 v88, v113, v66, vcc
	v_add_u32_e32 v66, 0xffffff0b, v90
	v_cmp_le_i32_e32 vcc, v66, v83
	s_nop 1
	v_cndmask_b32_e32 v66, v113, v68, vcc
	v_cmp_le_i32_e32 vcc, v67, v83
	v_add_u32_e32 v68, 0xffffff0d, v90
	s_nop 0
	v_cndmask_b32_e32 v67, v113, v69, vcc
	v_cmp_le_i32_e32 vcc, v68, v83
	v_add_u32_e32 v69, 0xffffff0e, v90
	s_nop 0
	v_cndmask_b32_e32 v68, v113, v70, vcc
	v_cmp_le_i32_e32 vcc, v69, v83
	v_add_u32_e32 v70, 0xffffff0f, v90
	s_nop 0
	v_cndmask_b32_e32 v69, v113, v71, vcc
	v_cmp_le_i32_e32 vcc, v70, v83
	v_add_u32_e32 v71, 0xffffff10, v90
	s_nop 0
	v_cndmask_b32_e32 v70, v113, v72, vcc
	v_cmp_le_i32_e32 vcc, v71, v83
	v_add_u32_e32 v72, 0xffffff19, v90
	s_nop 0
	v_cndmask_b32_e32 v71, v113, v73, vcc
	v_cmp_le_i32_e32 vcc, v72, v83
	v_add_u32_e32 v73, 0xffffff1a, v90
	s_nop 0
	v_cndmask_b32_e32 v72, v113, v74, vcc
	v_cmp_le_i32_e32 vcc, v73, v83
	v_add_u32_e32 v74, 0xffffff1b, v90
	s_nop 0
	v_cndmask_b32_e32 v73, v113, v75, vcc
	v_cmp_le_i32_e32 vcc, v74, v83
	v_add_u32_e32 v75, 0xffffff1c, v90
	s_nop 0
	v_cndmask_b32_e32 v74, v113, v76, vcc
	v_cmp_le_i32_e32 vcc, v75, v83
	v_add_u32_e32 v76, 0xffffff1d, v90
	s_nop 0
	v_cndmask_b32_e32 v75, v113, v77, vcc
	v_cmp_le_i32_e32 vcc, v76, v83
	v_add_u32_e32 v77, 0xffffff1e, v90
	s_nop 0
	v_cndmask_b32_e32 v76, v113, v78, vcc
	v_cmp_le_i32_e32 vcc, v77, v83
	v_add_u32_e32 v78, 0xffffff1f, v90
	s_nop 0
	v_cndmask_b32_e32 v77, v113, v79, vcc
	v_cmp_le_i32_e32 vcc, v78, v83
	v_add_u32_e32 v79, 0xffffff20, v90
	s_nop 0
	v_cndmask_b32_e32 v78, v113, v80, vcc
	v_cmp_le_i32_e32 vcc, v79, v83
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e32 v79, v113, v81, vcc
	v_pk_mul_f32 v[80:81], v[82:83], v[78:79]
	v_pk_mul_f32 v[78:79], v[82:83], v[76:77]
	v_pk_mul_f32 v[76:77], v[82:83], v[74:75]
	v_pk_mul_f32 v[74:75], v[82:83], v[72:73]
	v_pk_mul_f32 v[72:73], v[82:83], v[70:71]
	v_pk_mul_f32 v[70:71], v[82:83], v[68:69]
	v_pk_mul_f32 v[68:69], v[86:87], v[88:89]
	s_nop 0
	v_max_f32_e32 v83, v68, v69
	v_max3_f32 v83, v83, v66, v67
	v_max3_f32 v83, v83, v70, v71
	v_max3_f32 v83, v83, v72, v73
	v_max3_f32 v83, v83, v74, v75
	v_max3_f32 v83, v83, v76, v77
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v88, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v88, v88, v88
	v_max_f32_e32 v88, v83, v88
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v88, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_86
	;;#ASMSTART
	v_add_f32 v88, v88, v112
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v84, v88
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v84, v88
.LBB0_86:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[68:69], v[68:69], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67], v[66:67], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, 0, v68
	v_add_f32_e32 v88, v88, v69
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v88, v88, v66
	v_add_f32_e32 v88, v88, v67
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
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v83
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v83
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v88, v88, v79
	v_add_f32_e32 v88, v88, v80
	v_add_f32_e32 v88, v88, v81
	;;#ASMSTART
	v_fma_f32 v116, v116, v83, v88
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
	v_add_u32_e32 v73, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v83, 0x8000, v66
	v_add_u32_e32 v66, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v66, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v67, v83, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_lshl_b32 s42, s90, 14
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	s_add_i32 s42, s42, s82
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s42
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[88:91], v[74:75], off offset:512
	global_load_dwordx4 v[92:95], v[74:75], off offset:1024
	global_load_dwordx4 v[96:99], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[88:89], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[92:93], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[96:97], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[90:91], v[68:69], v[18:33]
	global_load_dwordx4 v[88:91], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[98:99], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[88:89], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[90:91], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[210:213], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[206:209], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[202:205], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[182:185], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[100:103], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[96:99], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[92:95], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[88:91], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_88
	ds_write_b128 v106, v[166:169] offset:12288
	ds_write_b128 v106, v[170:173] offset:18432
.LBB0_88:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s47, s84, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_90
	v_add_u32_sdwa v66, s47, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[166:169], v[66:67], off
	global_load_dwordx4 v[170:173], v[66:67], off offset:256
.LBB0_90:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[92:93], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[94:95], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[88:89], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[90:91], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	v_mov_b32_e32 v90, v84
	v_lshrrev_b32_e32 v88, 2, v83
	v_and_b32_e32 v88, 8, v88
	v_lshrrev_b32_e32 v83, 1, v83
	v_and_or_b32 v83, v83, s78, v115
	v_add_u32_e32 v88, s37, v88
	v_add_u32_e32 v83, s83, v83
	v_add_u32_e32 v89, 0xffffff29, v88
	v_cmp_lt_i32_e32 vcc, v89, v83
	v_mov_b32_e32 v91, v84
	v_mov_b32_e32 v92, v84
	v_cndmask_b32_e32 v67, v113, v67, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff2b, v88
	v_mov_b32_e32 v93, v84
	v_cndmask_b32_e32 v66, v113, v66, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff2c, v88
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_cndmask_b32_e32 v68, v113, v68, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff2d, v88
	v_mov_b32_e32 v94, v84
	v_cndmask_b32_e32 v69, v113, v69, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff2e, v88
	v_mov_b32_e32 v95, v84
	v_cndmask_b32_e32 v70, v113, v70, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff2f, v88
	v_mov_b32_e32 v96, v84
	v_cndmask_b32_e32 v71, v113, v71, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff30, v88
	v_mov_b32_e32 v97, v84
	v_cndmask_b32_e32 v72, v113, v72, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff39, v88
	v_mov_b32_e32 v98, v84
	v_cndmask_b32_e32 v73, v113, v73, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff3a, v88
	v_mov_b32_e32 v99, v84
	v_cndmask_b32_e32 v74, v113, v74, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff3b, v88
	v_mov_b32_e32 v100, v84
	v_cndmask_b32_e32 v75, v113, v75, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff3c, v88
	v_mov_b32_e32 v101, v84
	v_cndmask_b32_e32 v76, v113, v76, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff3d, v88
	v_mov_b32_e32 v102, v84
	v_cndmask_b32_e32 v77, v113, v77, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff3e, v88
	v_mov_b32_e32 v103, v84
	v_cndmask_b32_e32 v78, v113, v78, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_add_u32_e32 v89, 0xffffff3f, v88
	v_add_u32_e32 v88, 0xffffff40, v88
	v_cndmask_b32_e32 v79, v113, v79, vcc
	v_cmp_le_i32_e32 vcc, v89, v83
	v_mov_b32_e32 v89, v84
	s_nop 0
	v_cndmask_b32_e32 v80, v113, v80, vcc
	v_cmp_le_i32_e32 vcc, v88, v83
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_cndmask_b32_e32 v81, v113, v81, vcc
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
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
	ds_bpermute_b32 v88, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v88, v88, v88
	v_max_f32_e32 v117, v83, v88
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	v_mov_b32_e32 v88, v84
	v_cmp_gt_f32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_92
	;;#ASMSTART
	v_add_f32 v88, v117, v112
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
.LBB0_92:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, 0, v66
	v_add_f32_e32 v117, v117, v67
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
	v_add_f32_e32 v117, v117, v68
	v_add_f32_e32 v117, v117, v69
	v_add_f32_e32 v117, v117, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v71
	v_add_f32_e32 v117, v117, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v73
	v_add_f32_e32 v117, v117, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v75
	v_add_f32_e32 v117, v117, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v77
	v_add_f32_e32 v117, v117, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v117, v117, v79
	v_add_f32_e32 v117, v117, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v117, v117, v81
	;;#ASMSTART
	v_fma_f32 v116, v116, v83, v117
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s42, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s42
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_94
	ds_write_b128 v106, v[174:177]
	ds_write_b128 v106, v[178:181] offset:6144
.LBB0_94:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_96
	v_add_u32_e32 v66, s47, v105
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[174:177], v[66:67], off
	global_load_dwordx4 v[178:181], v[66:67], off offset:256
.LBB0_96:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v83
	v_and_b32_e32 v117, 8, v117
	v_lshrrev_b32_e32 v83, 1, v83
	v_and_or_b32 v83, v83, s78, v115
	v_add_u32_e32 v117, s37, v117
	v_add_u32_e32 v83, s83, v83
	v_add_u32_e32 v182, 0xffffff49, v117
	v_cmp_lt_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v67, v113, v67, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff4b, v117
	s_nop 0
	v_cndmask_b32_e32 v66, v113, v66, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff4c, v117
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_cndmask_b32_e32 v68, v113, v68, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff4d, v117
	s_nop 0
	v_cndmask_b32_e32 v69, v113, v69, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff4e, v117
	s_nop 0
	v_cndmask_b32_e32 v70, v113, v70, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff4f, v117
	s_nop 0
	v_cndmask_b32_e32 v71, v113, v71, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff50, v117
	s_nop 0
	v_cndmask_b32_e32 v72, v113, v72, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff59, v117
	s_nop 0
	v_cndmask_b32_e32 v73, v113, v73, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff5a, v117
	s_nop 0
	v_cndmask_b32_e32 v74, v113, v74, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff5b, v117
	s_nop 0
	v_cndmask_b32_e32 v75, v113, v75, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff5c, v117
	s_nop 0
	v_cndmask_b32_e32 v76, v113, v76, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff5d, v117
	s_nop 0
	v_cndmask_b32_e32 v77, v113, v77, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff5e, v117
	s_nop 0
	v_cndmask_b32_e32 v78, v113, v78, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff5f, v117
	v_add_u32_e32 v117, 0xffffff60, v117
	v_cndmask_b32_e32 v79, v113, v79, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v80, v113, v80, vcc
	v_cmp_le_i32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_cndmask_b32_e32 v81, v113, v81, vcc
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
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
	ds_bpermute_b32 v117, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v117, v117, v117
	v_max_f32_e32 v117, v83, v117
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_98
	;;#ASMSTART
	v_add_f32 v88, v117, v112
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
.LBB0_98:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, 0, v66
	v_add_f32_e32 v117, v117, v67
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
	v_add_f32_e32 v117, v117, v68
	v_add_f32_e32 v117, v117, v69
	v_add_f32_e32 v117, v117, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v71
	v_add_f32_e32 v117, v117, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v73
	v_add_f32_e32 v117, v117, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v75
	v_add_f32_e32 v117, v117, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v77
	v_add_f32_e32 v117, v117, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v117, v117, v79
	v_add_f32_e32 v117, v117, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v117, v117, v81
	;;#ASMSTART
	v_fma_f32 v116, v116, v83, v117
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_add_i32 s40, s42, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s40
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[222:225], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[218:221], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_100
	ds_write_b128 v106, v[166:169] offset:12288
	ds_write_b128 v106, v[170:173] offset:18432
.LBB0_100:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_102
	v_add_u32_e32 v66, s47, v107
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[166:169], v[66:67], off
	global_load_dwordx4 v[170:173], v[66:67], off offset:256
.LBB0_102:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v83
	v_and_b32_e32 v117, 8, v117
	v_lshrrev_b32_e32 v83, 1, v83
	v_and_or_b32 v83, v83, s78, v115
	v_add_u32_e32 v117, s37, v117
	v_add_u32_e32 v83, s83, v83
	v_add_u32_e32 v182, 0xffffff69, v117
	v_cmp_lt_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v67, v113, v67, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff6b, v117
	s_nop 0
	v_cndmask_b32_e32 v66, v113, v66, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff6c, v117
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_cndmask_b32_e32 v68, v113, v68, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff6d, v117
	s_nop 0
	v_cndmask_b32_e32 v69, v113, v69, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff6e, v117
	s_nop 0
	v_cndmask_b32_e32 v70, v113, v70, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff6f, v117
	s_nop 0
	v_cndmask_b32_e32 v71, v113, v71, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff70, v117
	s_nop 0
	v_cndmask_b32_e32 v72, v113, v72, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff79, v117
	s_nop 0
	v_cndmask_b32_e32 v73, v113, v73, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff7a, v117
	s_nop 0
	v_cndmask_b32_e32 v74, v113, v74, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff7b, v117
	s_nop 0
	v_cndmask_b32_e32 v75, v113, v75, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff7c, v117
	s_nop 0
	v_cndmask_b32_e32 v76, v113, v76, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff7d, v117
	s_nop 0
	v_cndmask_b32_e32 v77, v113, v77, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff7e, v117
	s_nop 0
	v_cndmask_b32_e32 v78, v113, v78, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff7f, v117
	v_add_u32_e32 v117, 0xffffff80, v117
	v_cndmask_b32_e32 v79, v113, v79, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v80, v113, v80, vcc
	v_cmp_le_i32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_cndmask_b32_e32 v81, v113, v81, vcc
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
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
	ds_bpermute_b32 v117, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v117, v117, v117
	v_max_f32_e32 v117, v83, v117
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_104
	;;#ASMSTART
	v_add_f32 v88, v117, v112
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
.LBB0_104:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, 0, v66
	v_add_f32_e32 v117, v117, v67
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
	v_add_f32_e32 v117, v117, v68
	v_add_f32_e32 v117, v117, v69
	v_add_f32_e32 v117, v117, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v71
	v_add_f32_e32 v117, v117, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v73
	v_add_f32_e32 v117, v117, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v75
	v_add_f32_e32 v117, v117, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v77
	v_add_f32_e32 v117, v117, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v117, v117, v79
	v_add_f32_e32 v117, v117, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v117, v117, v81
	;;#ASMSTART
	v_fma_f32 v116, v116, v83, v117
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s42, 0x2000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s42
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[182:185], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[186:189], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[222:225], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[226:229], v66
	s_add_i32 s42, s70, 1
	s_cmp_gt_i32 s38, s42
	s_cselect_b64 s[40:41], -1, 0
	s_cmp_le_i32 s38, s42
	s_cbranch_scc1 .LBB0_129
	v_mov_b32_e32 v66, s44
	buffer_load_dword v66, v66, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s46, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s75, v67
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_107
	ds_write_b128 v106, v[174:177]
	ds_write_b128 v106, v[178:181] offset:6144
.LBB0_107:
	s_or_b64 exec, exec, s[42:43]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_109
	v_add_u32_e32 v66, s47, v108
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[174:177], v[66:67], off
	global_load_dwordx4 v[178:181], v[66:67], off offset:256
.LBB0_109:
	s_or_b64 exec, exec, s[42:43]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v83
	v_and_b32_e32 v117, 8, v117
	v_lshrrev_b32_e32 v83, 1, v83
	v_and_or_b32 v83, v83, s78, v115
	v_add_u32_e32 v117, s37, v117
	v_add_u32_e32 v83, s83, v83
	v_add_u32_e32 v182, 0xffffff89, v117
	v_cmp_lt_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v67, v113, v67, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff8b, v117
	s_nop 0
	v_cndmask_b32_e32 v66, v113, v66, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff8c, v117
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_cndmask_b32_e32 v68, v113, v68, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff8d, v117
	s_nop 0
	v_cndmask_b32_e32 v69, v113, v69, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff8e, v117
	s_nop 0
	v_cndmask_b32_e32 v70, v113, v70, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff8f, v117
	s_nop 0
	v_cndmask_b32_e32 v71, v113, v71, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff90, v117
	s_nop 0
	v_cndmask_b32_e32 v72, v113, v72, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff99, v117
	s_nop 0
	v_cndmask_b32_e32 v73, v113, v73, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff9a, v117
	s_nop 0
	v_cndmask_b32_e32 v74, v113, v74, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff9b, v117
	s_nop 0
	v_cndmask_b32_e32 v75, v113, v75, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff9c, v117
	s_nop 0
	v_cndmask_b32_e32 v76, v113, v76, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff9d, v117
	s_nop 0
	v_cndmask_b32_e32 v77, v113, v77, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff9e, v117
	s_nop 0
	v_cndmask_b32_e32 v78, v113, v78, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffff9f, v117
	v_add_u32_e32 v117, 0xffffffa0, v117
	v_cndmask_b32_e32 v79, v113, v79, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v80, v113, v80, vcc
	v_cmp_le_i32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_cndmask_b32_e32 v81, v113, v81, vcc
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
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
	ds_bpermute_b32 v117, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v117, v117, v117
	v_max_f32_e32 v117, v83, v117
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_111
	;;#ASMSTART
	v_add_f32 v88, v117, v112
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
.LBB0_111:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, 0, v66
	v_add_f32_e32 v117, v117, v67
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
	v_add_f32_e32 v117, v117, v68
	v_add_f32_e32 v117, v117, v69
	v_add_f32_e32 v117, v117, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v71
	v_add_f32_e32 v117, v117, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v73
	v_add_f32_e32 v117, v117, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v75
	v_add_f32_e32 v117, v117, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v77
	v_add_f32_e32 v117, v117, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v117, v117, v79
	v_add_f32_e32 v117, v117, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v117, v117, v81
	;;#ASMSTART
	v_fma_f32 v116, v116, v83, v117
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_lshl_b32 s48, s84, 14
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	s_add_i32 s48, s48, s82
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s48
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[222:225], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[218:221], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_113
	ds_write_b128 v106, v[166:169] offset:12288
	ds_write_b128 v106, v[170:173] offset:18432
.LBB0_113:
	s_or_b64 exec, exec, s[42:43]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s47, s69, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_115
	v_add_u32_sdwa v66, s47, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[166:169], v[66:67], off
	global_load_dwordx4 v[170:173], v[66:67], off offset:256
.LBB0_115:
	s_or_b64 exec, exec, s[42:43]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v83
	v_and_b32_e32 v117, 8, v117
	v_lshrrev_b32_e32 v83, 1, v83
	v_and_or_b32 v83, v83, s78, v115
	v_add_u32_e32 v117, s37, v117
	v_add_u32_e32 v83, s83, v83
	v_add_u32_e32 v182, 0xffffffa9, v117
	v_cmp_lt_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v67, v113, v67, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffab, v117
	s_nop 0
	v_cndmask_b32_e32 v66, v113, v66, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffac, v117
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_cndmask_b32_e32 v68, v113, v68, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffad, v117
	s_nop 0
	v_cndmask_b32_e32 v69, v113, v69, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffae, v117
	s_nop 0
	v_cndmask_b32_e32 v70, v113, v70, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffaf, v117
	s_nop 0
	v_cndmask_b32_e32 v71, v113, v71, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffb0, v117
	s_nop 0
	v_cndmask_b32_e32 v72, v113, v72, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffb9, v117
	s_nop 0
	v_cndmask_b32_e32 v73, v113, v73, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffba, v117
	s_nop 0
	v_cndmask_b32_e32 v74, v113, v74, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffbb, v117
	s_nop 0
	v_cndmask_b32_e32 v75, v113, v75, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffbc, v117
	s_nop 0
	v_cndmask_b32_e32 v76, v113, v76, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffbd, v117
	s_nop 0
	v_cndmask_b32_e32 v77, v113, v77, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffbe, v117
	s_nop 0
	v_cndmask_b32_e32 v78, v113, v78, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, 0xffffffbf, v117
	v_subrev_u32_e32 v117, 64, v117
	v_cndmask_b32_e32 v79, v113, v79, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v80, v113, v80, vcc
	v_cmp_le_i32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_cndmask_b32_e32 v81, v113, v81, vcc
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
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
	ds_bpermute_b32 v117, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v117, v117, v117
	v_max_f32_e32 v117, v83, v117
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_117
	;;#ASMSTART
	v_add_f32 v88, v117, v112
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
.LBB0_117:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, 0, v66
	v_add_f32_e32 v117, v117, v67
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
	v_add_f32_e32 v117, v117, v68
	v_add_f32_e32 v117, v117, v69
	v_add_f32_e32 v117, v117, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v71
	v_add_f32_e32 v117, v117, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v73
	v_add_f32_e32 v117, v117, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v75
	v_add_f32_e32 v117, v117, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v77
	v_add_f32_e32 v117, v117, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v117, v117, v79
	v_add_f32_e32 v117, v117, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v117, v117, v81
	;;#ASMSTART
	v_fma_f32 v116, v116, v83, v117
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s48, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s48
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[222:225], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[218:221], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_119
	ds_write_b128 v106, v[174:177]
	ds_write_b128 v106, v[178:181] offset:6144
.LBB0_119:
	s_or_b64 exec, exec, s[42:43]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_121
	v_add_u32_e32 v66, s47, v105
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[174:177], v[66:67], off
	global_load_dwordx4 v[178:181], v[66:67], off offset:256
.LBB0_121:
	s_or_b64 exec, exec, s[42:43]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v83
	v_and_b32_e32 v117, 8, v117
	v_lshrrev_b32_e32 v83, 1, v83
	v_and_or_b32 v83, v83, s78, v115
	v_add_u32_e32 v117, s37, v117
	v_add_u32_e32 v83, s83, v83
	v_subrev_u32_e32 v182, 55, v117
	v_cmp_lt_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v67, v113, v67, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 53, v117
	s_nop 0
	v_cndmask_b32_e32 v66, v113, v66, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 52, v117
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_cndmask_b32_e32 v68, v113, v68, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 51, v117
	s_nop 0
	v_cndmask_b32_e32 v69, v113, v69, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 50, v117
	s_nop 0
	v_cndmask_b32_e32 v70, v113, v70, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 49, v117
	s_nop 0
	v_cndmask_b32_e32 v71, v113, v71, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 48, v117
	s_nop 0
	v_cndmask_b32_e32 v72, v113, v72, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 39, v117
	s_nop 0
	v_cndmask_b32_e32 v73, v113, v73, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 38, v117
	s_nop 0
	v_cndmask_b32_e32 v74, v113, v74, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 37, v117
	s_nop 0
	v_cndmask_b32_e32 v75, v113, v75, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 36, v117
	s_nop 0
	v_cndmask_b32_e32 v76, v113, v76, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 35, v117
	s_nop 0
	v_cndmask_b32_e32 v77, v113, v77, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 34, v117
	s_nop 0
	v_cndmask_b32_e32 v78, v113, v78, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 33, v117
	v_subrev_u32_e32 v117, 32, v117
	v_cndmask_b32_e32 v79, v113, v79, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v80, v113, v80, vcc
	v_cmp_le_i32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_cndmask_b32_e32 v81, v113, v81, vcc
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
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
	ds_bpermute_b32 v117, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v117, v117, v117
	v_max_f32_e32 v117, v83, v117
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_123
	;;#ASMSTART
	v_add_f32 v88, v117, v112
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
.LBB0_123:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[66:67], v[66:67], v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[90:91] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[92:93] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, 0, v66
	v_add_f32_e32 v117, v117, v67
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
	v_add_f32_e32 v117, v117, v68
	v_add_f32_e32 v117, v117, v69
	v_add_f32_e32 v117, v117, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v71
	v_add_f32_e32 v117, v117, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v73
	v_add_f32_e32 v117, v117, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v75
	v_add_f32_e32 v117, v117, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v77
	v_add_f32_e32 v117, v117, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v117, v117, v79
	v_add_f32_e32 v117, v117, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v117, v117, v81
	;;#ASMSTART
	v_fma_f32 v116, v116, v83, v117
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s76
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_add_i32 s42, s48, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s42
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[20:21]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[182:185], v[74:75], off offset:512
	global_load_dwordx4 v[186:189], v[74:75], off offset:1024
	global_load_dwordx4 v[190:193], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[190:191], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s77, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[68:69], v[18:33]
	global_load_dwordx4 v[182:185], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[188:189], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[68:69], v[72:73], v[50:65]
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
	ds_read_b128 v[226:229], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[222:225], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[218:221], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[214:217], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[210:213], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[206:209], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[202:205], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[198:201], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[194:197], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[190:193], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[186:189], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[182:185], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_125
	ds_write_b128 v106, v[166:169] offset:12288
	ds_write_b128 v106, v[170:173] offset:18432
.LBB0_125:
	s_or_b64 exec, exec, s[42:43]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s75, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_127
	v_add_u32_e32 v66, s47, v107
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[166:169], v[66:67], off
	global_load_dwordx4 v[170:173], v[66:67], off offset:256
.LBB0_127:
	s_or_b64 exec, exec, s[42:43]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[118:119], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[122:123], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[124:125], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[128:129], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[132:133], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[156:157], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[162:163], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[164:165], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v83
	v_and_b32_e32 v117, 8, v117
	v_lshrrev_b32_e32 v83, 1, v83
	v_and_or_b32 v83, v83, s78, v115
	v_add_u32_e32 v117, s37, v117
	v_add_u32_e32 v83, s83, v83
	v_subrev_u32_e32 v182, 23, v117
	v_cmp_lt_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v67, v113, v67, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 21, v117
	s_nop 0
	v_cndmask_b32_e32 v66, v113, v66, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 20, v117
	v_pk_mul_f32 v[66:67], v[86:87], v[66:67]
	v_cndmask_b32_e32 v68, v113, v68, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 19, v117
	s_nop 0
	v_cndmask_b32_e32 v69, v113, v69, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 18, v117
	s_nop 0
	v_cndmask_b32_e32 v70, v113, v70, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_subrev_u32_e32 v182, 17, v117
	s_nop 0
	v_cndmask_b32_e32 v71, v113, v71, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, -16, v117
	s_nop 0
	v_cndmask_b32_e32 v72, v113, v72, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, -7, v117
	s_nop 0
	v_cndmask_b32_e32 v73, v113, v73, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, -6, v117
	s_nop 0
	v_cndmask_b32_e32 v74, v113, v74, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, -5, v117
	s_nop 0
	v_cndmask_b32_e32 v75, v113, v75, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, -4, v117
	s_nop 0
	v_cndmask_b32_e32 v76, v113, v76, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, -3, v117
	s_nop 0
	v_cndmask_b32_e32 v77, v113, v77, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, -2, v117
	s_nop 0
	v_cndmask_b32_e32 v78, v113, v78, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	v_add_u32_e32 v182, -1, v117
	s_nop 0
	v_cndmask_b32_e32 v79, v113, v79, vcc
	v_cmp_le_i32_e32 vcc, v182, v83
	s_nop 1
	v_cndmask_b32_e32 v80, v113, v80, vcc
	v_cmp_le_i32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, v82
	v_pk_mul_f32 v[78:79], v[82:83], v[78:79]
	v_cndmask_b32_e32 v81, v113, v81, vcc
	v_pk_mul_f32 v[80:81], v[82:83], v[80:81]
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
	ds_bpermute_b32 v117, v109, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v117, v117, v117
	v_max_f32_e32 v117, v83, v117
	;;#ASMSTART
	v_add_f32 v83, v84, v111
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v117, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_78
	;;#ASMSTART
	v_add_f32 v88, v117, v112
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
	s_branch .LBB0_78
.LBB0_129:
	s_mov_b32 s46, s45
	s_branch .LBB0_79
.LBB0_130:
	;;#ASMSTART
	v_mov_b32 v68, v0
	;;#ASMEND
	ds_bpermute_b32 v66, v109, v116
	v_lshrrev_b32_e32 v67, 1, v68
	v_and_b32_e32 v69, 31, v68
	v_and_or_b32 v67, v67, s78, v69
	v_and_b32_e32 v68, 32, v68
	v_cmp_eq_u32_e32 vcc, 0, v68
	v_cmp_gt_i32_e64 s[0:1], s19, v67
	s_and_b64 s[12:13], vcc, s[0:1]
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v66, v116, v66
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[12:13]
	s_cbranch_execz .LBB0_132
	v_cmp_gt_f32_e32 vcc, s79, v66
	s_ashr_i32 s69, s68, 31
	s_lshl_b64 s[12:13], s[68:69], 6
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v66, v69
	v_log_f32_e32 v69, v69
	v_cndmask_b32_e32 v68, 0, v114, vcc
	s_add_u32 s14, s30, s12
	s_addc_u32 s37, s31, s13
	v_sub_f32_e32 v68, v69, v68
	v_add_f32_e32 v68, v84, v68
	s_lshl_b64 s[12:13], s[24:25], 2
	v_mul_f32_e32 v68, 0x3f317218, v68
	v_cmp_lt_f32_e32 vcc, 0, v66
	s_add_u32 s12, s14, s12
	s_addc_u32 s13, s37, s13
	v_cndmask_b32_e32 v68, v113, v68, vcc
	v_lshlrev_b32_e32 v67, 6, v67
	global_store_dword v67, v68, s[12:13]
.LBB0_132:
	s_or_b64 exec, exec, s[0:1]
	v_div_scale_f32 v67, s[0:1], v66, v66, s74
	v_rcp_f32_e32 v68, v67
	v_div_scale_f32 v69, vcc, s74, v66, s74
	v_fma_f32 v70, -v67, v68, 1.0
	v_fmac_f32_e32 v68, v70, v68
	v_mul_f32_e32 v70, v69, v68
	v_fma_f32 v71, -v67, v70, v69
	v_fmac_f32_e32 v70, v71, v68
	v_fma_f32 v67, -v67, v70, v69
	v_div_fmas_f32 v67, v67, v68, v70
	v_div_fixup_f32 v66, v67, v66, s74
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
	v_perm_b32 v28, v3, v2, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v5, v4, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v7, v6, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v9, v8, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v11, v10, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v13, v12, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v15, v14, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v17, v16, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v19, v18, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v21, v20, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v23, v22, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v74, v75, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v72, v73, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v70, v71, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v68, v69, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v66, v67, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v35, v34, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v37, v36, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v39, v38, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v41, v40, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v43, v42, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v45, v44, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v47, v46, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v49, v48, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v51, v50, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v53, v52, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v55, v54, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v57, v56, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v59, v58, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v61, v60, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v63, v62, s76
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v65, v64, s76
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v34, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v34, 0x100, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_134
	s_barrier
.LBB0_134:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v37, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v34, 3, v37
	v_bfe_u32 v36, v37, 6, 3
	v_lshlrev_b32_e32 v35, 7, v37
	v_and_b32_e32 v34, 4, v34
	v_and_or_b32 v34, v35, s80, v34
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_136
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
.LBB0_136:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v39, 0x1ff, v37
	s_lshl_b32 s0, s68, 11
	v_lshlrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v37, 56, v37
	s_ashr_i32 s1, s0, 31
	v_xor_b32_e32 v37, v40, v37
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v37, 1, v37
	s_add_u32 s12, s28, s0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v39, 7, v39
	ds_read_b128 v[42:45], v37
	s_addc_u32 s0, s29, s1
	s_lshl_b32 s14, s19, 12
	s_lshl_b32 s19, s24, 7
	v_and_b32_e32 v39, 0xf800, v39
	v_and_b32_e32 v38, 0x78, v40
	v_add_u32_e32 v40, s19, v39
	v_or_b32_e32 v40, v40, v38
	s_and_b32 s13, s0, 0xffff
	v_lshlrev_b32_e32 v40, 1, v40
	v_cmp_eq_u32_e32 vcc, 1, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[42:45], v40, s[12:15], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_138
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
.LBB0_138:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s19, s19, 0x10000
	v_or_b32_e32 v38, v38, v39
	v_add_lshl_u32 v39, s19, v38, 1
	v_cmp_eq_u32_e32 vcc, 2, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[12:15], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_140
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
.LBB0_140:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s19, 0x10000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 3, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[12:15], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_142
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
.LBB0_142:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s19, 0x20000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 4, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[12:15], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_144
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
.LBB0_144:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s19, 0x30000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 5, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[12:15], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_146
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
.LBB0_146:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s19, 0x40000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 6, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[12:15], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_148
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
.LBB0_148:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s19, 0x50000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 7, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[12:15], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_150
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
.LBB0_150:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[4:7], v37
	s_add_i32 s19, s19, 0x60000
	v_add_lshl_u32 v2, s19, v38, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v2, s[12:15], 0 offen
	s_barrier
	;;#ASMSTART
	s_mov_b32 s0, s81
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_154
	s_mov_b64 s[40:41], exec
	v_mbcnt_lo_u32_b32 v2, s40, 0
	v_mbcnt_hi_u32_b32 v2, s41, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_153
	s_bcnt1_i32_b64 s14, s[40:41]
	v_mov_b32_e32 v3, s14
	global_atomic_add v3, v110, v3, s[34:35] sc0
.LBB0_153:
	s_or_b64 exec, exec, s[38:39]
	s_lshl_b64 s[38:39], s[0:1], 2
	s_add_u32 s38, s34, s38
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s14, v3
	s_addc_u32 s39, s35, s39
	s_nop 0
	v_add_u32_e32 v2, s14, v2
	global_store_dword v110, v2, s[38:39]
	s_waitcnt vmcnt(0)
.LBB0_154:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s34, s0
	s_addc_u32 s1, s35, s1
	s_barrier
	global_load_dword v2, v110, s[0:1]
	s_sub_i32 s0, s33, s2
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s2, v2
	s_add_i32 s33, s0, s2
	s_cmp_ge_i32 s18, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s72
	s_cselect_b64 s[12:13], -1, 0
	s_or_b64 s[0:1], s[0:1], s[12:13]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_157
	s_branch .LBB0_12
.LBB0_155:
	s_mov_b32 s18, s13
.LBB0_156:
	s_sub_i32 s33, s33, s72
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s24, 0, s12
	s_cmp_ge_i32 s18, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s14
	s_cselect_b64 s[12:13], -1, 0
	s_or_b64 s[0:1], s[0:1], s[12:13]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s72, s14
	s_cbranch_vccnz .LBB0_11
.LBB0_157:
	s_add_i32 s12, s24, 1
	s_cmp_gt_i32 s12, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s12, 16
	s_cbranch_scc1 .LBB0_160
	s_add_i32 s13, s18, 1
	s_cmp_ge_i32 s13, s3
	s_mov_b32 s14, s72
	s_cbranch_scc1 .LBB0_155
	s_ashr_i32 s19, s18, 31
	s_lshl_b64 s[18:19], s[18:19], 2
	s_add_u32 s18, s16, s18
	s_addc_u32 s19, s17, s19
	global_load_dwordx2 v[2:3], v110, s[18:19] offset:4
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s14, v3
	v_readfirstlane_b32 s18, v2
	s_sub_i32 s14, s14, s18
	s_addk_i32 s14, 0xff
	s_ashr_i32 s18, s14, 31
	s_lshr_b32 s18, s18, 24
	s_add_i32 s18, s14, s18
	s_ashr_i32 s37, s18, 8
	s_and_b32 s18, s18, 0xffffff00
	s_cmp_lg_u32 s14, s18
	s_cselect_b64 s[18:19], -1, 0
	s_cmp_lt_i32 s14, 0
	s_cselect_b64 s[24:25], -1, 0
	s_and_b64 s[18:19], s[24:25], s[18:19]
	s_subb_u32 s14, s37, 0
	s_branch .LBB0_155
.LBB0_160:
	s_mov_b32 s14, s72
	s_branch .LBB0_156
.LBB0_161:
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
		.amdhsa_next_free_vgpr 230
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 232
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

	.set .Lattn_kernel_0.num_vgpr, 230
	.set .Lattn_kernel_0.num_agpr, 0
	.set .Lattn_kernel_0.numbered_sgpr, 91
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
    .sgpr_count:     97
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     230
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
