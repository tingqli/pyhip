	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	attn_kernel_0
	.p2align	8
	.type	attn_kernel_0,@function
attn_kernel_0:
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dword s3, s[0:1], 0x38
	s_load_dwordx2 s[72:73], s[0:1], 0x90
	s_load_dwordx4 s[4:7], s[0:1], 0x80
	s_mov_b32 s74, 0
	s_waitcnt lgkmcnt(0)
	s_load_dwordx2 s[8:9], s[34:35], 0x0
	s_add_i32 s3, s3, -1
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s8, s9, s8
	s_add_i32 s10, s8, 0xff
	s_ashr_i32 s8, s10, 31
	s_lshr_b32 s8, s8, 24
	s_add_i32 s8, s10, s8
	s_ashr_i32 s12, s8, 8
	s_and_b32 s8, s8, 0xffffff00
	s_cmp_lg_u32 s10, s8
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s10, 0
	s_cselect_b64 s[10:11], -1, 0
	s_and_b64 s[8:9], s[10:11], s[8:9]
	s_subb_u32 s10, s12, 0
	s_cmp_lt_i32 s3, 1
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s2, s10
	s_cselect_b64 s[12:13], -1, 0
	s_or_b64 s[8:9], s[8:9], s[12:13]
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_8
	s_mov_b32 s76, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s74, s12
.LBB0_3:
	s_sub_i32 s33, s33, s10
	s_and_b64 s[8:9], s[8:9], exec
	s_cselect_b32 s76, 0, s11
	s_cmp_ge_i32 s74, s3
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s33, s98
	s_cselect_b64 s[10:11], -1, 0
	s_or_b64 s[8:9], s[8:9], s[10:11]
	s_and_b64 vcc, exec, s[8:9]
	s_mov_b32 s10, s98
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s11, s76, 1
	s_cmp_gt_i32 s11, 15
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s11, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s12, s74, 1
	s_cmp_ge_i32 s12, s3
	s_mov_b32 s98, s10
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s75, s74, 31
	s_lshl_b64 s[14:15], s[74:75], 2
	s_add_u32 s14, s34, s14
	s_addc_u32 s15, s35, s15
	s_load_dwordx2 s[16:17], s[14:15], 0x4
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s13, s17, s16
	s_addk_i32 s13, 0xff
	s_ashr_i32 s14, s13, 31
	s_lshr_b32 s14, s14, 24
	s_add_i32 s14, s13, s14
	s_ashr_i32 s18, s14, 8
	s_and_b32 s14, s14, 0xffffff00
	s_cmp_lg_u32 s13, s14
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_lt_i32 s13, 0
	s_cselect_b64 s[16:17], -1, 0
	s_and_b64 s[14:15], s[16:17], s[14:15]
	s_subb_u32 s98, s18, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s98, s10
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s98, s10
	s_mov_b32 s76, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s87, s[4:5], 0x0
	s_load_dword s99, s[6:7], 0x0
	s_cmp_ge_i32 s74, s3
	s_cbranch_scc1 .LBB0_191
	s_load_dwordx2 s[48:49], s[0:1], 0x0
	s_load_dwordx2 s[50:51], s[0:1], 0x10
	s_load_dwordx2 s[80:81], s[0:1], 0x20
	s_load_dwordx2 s[52:53], s[0:1], 0x50
	s_load_dwordx2 s[54:55], s[0:1], 0x60
	s_load_dwordx2 s[56:57], s[0:1], 0x70
	s_load_dwordx2 s[88:89], s[0:1], 0xa0
	s_load_dwordx2 s[90:91], s[0:1], 0xc0
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v2, 31, v0
	s_movk_i32 s0, 0xe0
	v_and_or_b32 v1, v1, s0, v2
	v_lshlrev_b32_e32 v254, 6, v1
	v_lshrrev_b32_e32 v1, 6, v0
	v_lshrrev_b32_e32 v3, 2, v0
	v_mul_u32_u24_e32 v1, 0x18000, v1
	v_and_or_b32 v1, v3, 8, v1
	s_movk_i32 s0, 0xc00
	v_mad_u32_u24 v255, v2, s0, v1
	s_mov_b32 s0, 0xaaaaaab
	v_mul_hi_u32 v1, v0, s0
	s_mov_b32 s0, 0x2aaaaab
	v_mul_hi_u32 v3, v0, s0
	s_mov_b32 s0, 0x1555556
	v_mul_u32_u24_e32 v2, 24, v1
	v_mul_hi_u32 v4, v0, s0
	v_sub_u32_e32 v2, v0, v2
	v_lshlrev_b32_e32 v3, 5, v3
	v_lshlrev_b32_e32 v4, 4, v4
	v_and_b32_e32 v3, 32, v3
	v_lshl_or_b32 v2, v2, 9, v4
	v_lshlrev_b32_e32 v1, 2, v1
	v_add_u32_e32 v2, v2, v3
	v_lshlrev_b32_e32 v3, 1, v0
	v_and_or_b32 v1, v1, 12, v2
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v152, v2, v3
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_or_b32_e32 v119, 0x80, v1
	v_or_b32_e32 v153, 0x100, v1
	s_movk_i32 s77, 0x180
	v_or_b32_e32 v154, 0x180, v1
	v_xor_b32_e32 v155, 0x80, v2
	v_mov_b32_e32 v156, 0
	s_mov_b32 s71, 0x27000
	s_movk_i32 s78, 0x1000
	v_mov_b32_e32 v157, 0x40e00000
	v_mov_b32_e32 v158, 1.0
	s_mov_b32 s79, 0x7060302
	s_movk_i32 s47, 0xf80
	s_mov_b32 s58, s2
	v_mov_b32_e32 v159, 0xff800000
	s_mov_b32 s36, 0
	scratch_store_dword off, v254, off offset:64
	scratch_store_dword off, v255, off offset:68
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s98, s6
.LBB0_12:
	s_cmp_ge_i32 s74, s3
	s_cbranch_scc1 .LBB0_191
.LBB0_13:
	s_ashr_i32 s75, s74, 31
	s_lshl_b32 s6, s33, 8
	s_lshl_b64 s[0:1], s[74:75], 2
	s_add_u32 s4, s34, s0
	s_addc_u32 s5, s35, s1
	global_load_dwordx2 v[2:3], v156, s[4:5]
	s_mul_i32 s4, s76, 0xc0
	v_add_lshl_u32 v7, s4, v255, 1
	s_mov_b32 s7, s71
	v_lshl_add_u32 v6, s76, 2, v254
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s82, v2
	s_add_i32 s82, s82, s6
	v_readfirstlane_b32 s5, v3
	s_add_i32 s6, s82, 0x100
	s_min_i32 s5, s6, s5
	s_sub_i32 s75, s5, s82
	s_waitcnt lgkmcnt(0)
	s_add_u32 s8, s52, s0
	s_addc_u32 s9, s53, s1
	s_mul_i32 s4, s82, 0xc00
	s_add_u32 s0, s72, s0
	s_addc_u32 s1, s73, s1
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s10, s82, 4
	global_load_dwordx2 v[4:5], v156, s[8:9]
	global_load_dword v3, v156, s[0:1]
	s_lshl_b64 s[0:1], s[4:5], 1
	s_add_u32 s68, s48, s0
	s_addc_u32 s0, s49, s1
	s_mul_i32 s70, s75, 0x1800
	s_and_b32 s69, s0, 0xffff
	buffer_load_dwordx4 v[178:181], v7, s[68:71], 0 offen
	buffer_load_dwordx4 v[182:185], v7, s[68:71], 0 offen offset:32
	buffer_load_dwordx4 v[186:189], v7, s[68:71], 0 offen offset:64
	buffer_load_dwordx4 v[190:193], v7, s[68:71], 0 offen offset:96
	buffer_load_dwordx4 v[194:197], v7, s[68:71], 0 offen offset:128
	buffer_load_dwordx4 v[198:201], v7, s[68:71], 0 offen offset:160
	buffer_load_dwordx4 v[202:205], v7, s[68:71], 0 offen offset:192
	buffer_load_dwordx4 v[206:209], v7, s[68:71], 0 offen offset:224
	buffer_load_dwordx4 v[210:213], v7, s[68:71], 0 offen offset:256
	buffer_load_dwordx4 v[214:217], v7, s[68:71], 0 offen offset:288
	s_ashr_i32 s11, s10, 31
	s_lshl_b64 s[0:1], s[10:11], 2
	s_add_u32 s4, s56, s0
	s_addc_u32 s0, s57, s1
	s_lshl_b32 s6, s75, 6
	s_and_b32 s5, s0, 0xffff
	buffer_load_dword v2, v6, s[4:7], 0 offen
	buffer_load_dwordx4 v[218:221], v7, s[68:71], 0 offen offset:320
	buffer_load_dwordx4 v[222:225], v7, s[68:71], 0 offen offset:352
	s_waitcnt vmcnt(14)
	v_readfirstlane_b32 s0, v4
	s_waitcnt vmcnt(13)
	v_readfirstlane_b32 s6, v3
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	v_readfirstlane_b32 s1, v5
	v_and_b32_e32 v3, 0x100, v3
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_15
	s_barrier
.LBB0_15:
	s_or_b64 exec, exec, s[4:5]
	s_sub_i32 s92, s1, s0
	s_ashr_i32 s1, s76, 31
	s_lshr_b32 s1, s1, 28
	s_add_i32 s1, s76, s1
	s_ashr_i32 s7, s1, 4
	s_and_b32 s1, s1, -16
	s_cmp_lg_u32 s76, s1
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s76, 0
	s_cselect_b64 s[8:9], -1, 0
	s_and_b64 s[4:5], s[8:9], s[4:5]
	s_subb_u32 s4, s7, 0
	s_mul_i32 s8, s4, 0x6000
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[8:9], s[8:9], 1
	s_add_u32 s94, s50, s8
	s_addc_u32 s95, s51, s9
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s68, s54, s0
	s_addc_u32 s0, s55, s1
	s_lshl_b32 s70, s92, 2
	s_and_b32 s69, s0, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[68:71], 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s11, v4
	v_readfirstlane_b32 s85, v5
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
	buffer_load_dword v3, off, s[68:71], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s83, v3
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
	buffer_load_dword v3, off, s[68:71], 0 offset:12
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s86, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_mul_i32 s5, s11, 0x3000
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s77, v3
	scratch_store_dwordx4 off, v[4:7], off
	scratch_store_dwordx4 off, v[4:7], off offset:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_17
	v_add_u32_e32 v4, s5, v1
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[94:95]
	global_load_dwordx4 v[6:9], v[4:5], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[6:9], off
	global_load_dwordx4 v[4:7], v[4:5], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[4:7], off offset:16
.LBB0_17:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	scratch_store_dwordx4 off, v[4:7], off offset:32
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s77, v3
	scratch_store_dwordx4 off, v[4:7], off offset:48
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_19
	v_add_u32_e32 v4, s5, v119
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[94:95]
	global_load_dwordx4 v[6:9], v[4:5], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[6:9], off offset:32
	global_load_dwordx4 v[4:7], v[4:5], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[4:7], off offset:48
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
	v_cmp_ne_u32_e32 vcc, s77, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_21
	scratch_load_dwordx4 v[4:7], off, off
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[4:7] offset:12288
	scratch_load_dwordx4 v[4:7], off, off offset:16
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[4:7] offset:18432
.LBB0_21:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s77, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_23
	v_add_u32_e32 v4, s5, v153
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[94:95]
	global_load_dwordx4 v[6:9], v[4:5], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[6:9], off
	global_load_dwordx4 v[4:7], v[4:5], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[4:7], off offset:16
.LBB0_23:
	s_or_b64 exec, exec, s[0:1]
	v_mul_f32_e32 v2, s87, v2
	s_add_i32 s7, s92, -1
	v_mul_f32_e32 v116, 0x3dd53b95, v2
	s_lshl_b32 s84, s4, 14
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
	ds_read_b128 v[120:123], v3 offset:12288
	v_add_u32_e32 v3, 0x1810, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[162:165], v3
	v_add_u32_e32 v3, 0x1820, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[166:169], v3
	v_add_u32_e32 v3, 0x1830, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[170:173], v3
	v_add_u32_e32 v3, 0x1840, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[174:177], v3
	v_add_u32_e32 v3, 0x1850, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[242:245], v3
	v_add_u32_e32 v3, 0x1860, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[246:249], v3
	v_add_u32_e32 v3, 0x1870, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[250:253], v3
	v_add_u32_e32 v3, 0x1880, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[234:237], v3
	v_add_u32_e32 v3, 0x1890, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[238:241], v3
	v_add_u32_e32 v3, 0x18a0, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_u32_e32 v2, 0x18b0, v2
	ds_read_b128 v[230:233], v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[226:229], v2
	s_add_i32 s0, s92, -2
	s_bitcmp0_b32 s92, 0
	s_cselect_b32 s96, s0, s7
	s_ashr_i32 s97, s96, 31
	s_cmp_lt_i32 s96, 1
	s_cbranch_scc1 .LBB0_91
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v117, v116
	v_mov_b32_e32 v82, v116
	v_mov_b32_e32 v83, v116
	v_mov_b32_e32 v84, v116
	v_mov_b32_e32 v85, v116
	v_mov_b32_e32 v86, v116
	v_mov_b32_e32 v87, v116
	v_mov_b32_e32 v88, v116
	v_mov_b32_e32 v89, v116
	v_mov_b32_e32 v90, v116
	v_mov_b32_e32 v91, v116
	v_mov_b32_e32 v92, v116
	v_mov_b32_e32 v93, v116
	v_mov_b32_e32 v94, v116
	v_mov_b32_e32 v95, v116
	s_mov_b64 s[0:1], 0
	s_mov_b32 s8, 20
	v_mov_b32_e32 v118, 0xff800000
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v160
	v_mov_b32_e32 v36, v160
	v_mov_b32_e32 v37, v160
	v_mov_b32_e32 v38, v160
	v_mov_b32_e32 v39, v160
	v_mov_b32_e32 v40, v160
	v_mov_b32_e32 v41, v160
	v_mov_b32_e32 v42, v160
	v_mov_b32_e32 v43, v160
	v_mov_b32_e32 v44, v160
	v_mov_b32_e32 v45, v160
	v_mov_b32_e32 v46, v160
	v_mov_b32_e32 v47, v160
	v_mov_b32_e32 v48, v160
	v_mov_b32_e32 v49, v160
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v160
	v_mov_b32_e32 v52, v160
	v_mov_b32_e32 v53, v160
	v_mov_b32_e32 v54, v160
	v_mov_b32_e32 v55, v160
	v_mov_b32_e32 v56, v160
	v_mov_b32_e32 v57, v160
	v_mov_b32_e32 v58, v160
	v_mov_b32_e32 v59, v160
	v_mov_b32_e32 v60, v160
	v_mov_b32_e32 v61, v160
	v_mov_b32_e32 v62, v160
	v_mov_b32_e32 v63, v160
	v_mov_b32_e32 v64, v160
	v_mov_b32_e32 v65, v160
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v160
	v_mov_b32_e32 v4, v160
	v_mov_b32_e32 v5, v160
	v_mov_b32_e32 v6, v160
	v_mov_b32_e32 v7, v160
	v_mov_b32_e32 v8, v160
	v_mov_b32_e32 v9, v160
	v_mov_b32_e32 v10, v160
	v_mov_b32_e32 v11, v160
	v_mov_b32_e32 v12, v160
	v_mov_b32_e32 v13, v160
	v_mov_b32_e32 v14, v160
	v_mov_b32_e32 v15, v160
	v_mov_b32_e32 v16, v160
	v_mov_b32_e32 v17, v160
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v160
	v_mov_b32_e32 v20, v160
	v_mov_b32_e32 v21, v160
	v_mov_b32_e32 v22, v160
	v_mov_b32_e32 v23, v160
	v_mov_b32_e32 v24, v160
	v_mov_b32_e32 v25, v160
	v_mov_b32_e32 v26, v160
	v_mov_b32_e32 v27, v160
	v_mov_b32_e32 v28, v160
	v_mov_b32_e32 v29, v160
	v_mov_b32_e32 v30, v160
	v_mov_b32_e32 v31, v160
	v_mov_b32_e32 v32, v160
	v_mov_b32_e32 v33, v160
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[136:137], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[140:141], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[138:139], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[142:143], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[124:125], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[126:127], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[72:73], v[18:33]
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
	ds_read_b128 v[120:123], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[162:165], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[166:169], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[170:173], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[174:177], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[242:245], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[246:249], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[250:253], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[234:237], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[238:241], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[230:233], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[226:229], v66
	s_add_u32 s0, s0, 2
	s_addc_u32 s1, s1, 0
	v_mov_b64_e32 v[66:67], s[96:97]
	v_cmp_lt_i64_e32 vcc, s[0:1], v[66:67]
	s_add_i32 s8, s8, 8
	s_mov_b32 s85, s10
	s_mov_b32 s11, s9
	s_cbranch_vccz .LBB0_90
.LBB0_26:
	s_add_i32 s4, s8, -4
	v_mov_b32_e32 v66, s4
	buffer_load_dword v66, v66, s[68:71], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_mov_b32 s9, s83
	v_and_b32_e32 v67, 0x180, v67
	s_mov_b32 s10, s86
	v_cmp_ne_u32_e32 vcc, s77, v67
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s83, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_28
	scratch_load_dwordx4 v[66:69], off, off offset:32
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69]
	scratch_load_dwordx4 v[66:69], off, off offset:48
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:6144
.LBB0_28:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_30
	s_mul_i32 s12, s11, 0x3000
	v_add_u32_e32 v66, s12, v154
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off offset:32
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:48
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v96, v0
	;;#ASMEND
	s_lshl_b32 s12, s11, 14
	v_lshlrev_b32_e32 v97, 3, v96
	v_lshlrev_b32_e32 v96, 5, v96
	s_add_i32 s12, s12, s84
	v_and_b32_e32 v97, 0xf8, v97
	v_and_b32_e32 v96, 0x400, v96
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[180:181], v[66:81]
	v_or3_b32 v96, v97, v96, s12
	v_ashrrev_i32_e32 v97, 31, v96
	v_lshl_add_u64 v[96:97], v[96:97], 1, s[80:81]
	global_load_dwordx4 v[132:135], v[96:97], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[182:183], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[186:187], v[66:81]
	global_load_dwordx4 v[136:139], v[96:97], off offset:512
	global_load_dwordx4 v[124:127], v[96:97], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[192:193], v[66:81]
	global_load_dwordx4 v[128:131], v[96:97], off offset:1536
	v_add_co_u32_e32 v96, vcc, s78, v96
	s_nop 1
	v_addc_co_u32_e32 v97, vcc, 0, v97, vcc
	global_load_dwordx4 v[106:109], v[96:97], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[176:177], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[198:199], v[66:81]
	global_load_dwordx4 v[120:123], v[96:97], off offset:512
	global_load_dwordx4 v[102:105], v[96:97], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[244:245], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[246:247], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[248:249], v[204:205], v[66:81]
	global_load_dwordx4 v[98:101], v[96:97], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[250:251], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[252:253], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v96, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v96, v96, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v96, v96, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v96, v96, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v96, v96, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v96, v96, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v96, v96, v78, v79
	v_max3_f32 v96, v96, v80, v81
	ds_bpermute_b32 v97, v155, v96
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v97, v97, v97
	v_max_f32_e32 v97, v96, v97
	;;#ASMSTART
	v_add_f32 v96, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v97, v96
	v_mov_b32_e32 v96, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v97, v97, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v96, v118, v97
	v_exp_f32_e32 v96, v96
	v_mov_b32_e32 v118, v97
.LBB0_32:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v97, 0, v66
	v_add_f32_e32 v97, v97, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v97, v97, v68
	v_add_f32_e32 v97, v97, v69
	v_add_f32_e32 v97, v97, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v97, v97, v71
	v_add_f32_e32 v97, v97, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v97, v97, v73
	v_add_f32_e32 v97, v97, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v97, v97, v75
	v_add_f32_e32 v97, v97, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v97, v97, v77
	v_add_f32_e32 v97, v97, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v96
	v_add_f32_e32 v97, v97, v79
	v_add_f32_e32 v97, v97, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v97, v97, v81
	;;#ASMSTART
	v_fma_f32 v112, v160, v96, v97
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_34
	;;#ASMSTART
	v_mul_f32 v34, v34, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v96
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v96
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[132:133], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[136:137], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[124:125], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[128:129], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[134:135], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[126:127], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[130:131], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[106:107], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[120:121], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[98:99], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[108:109], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[122:123], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[104:105], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[100:101], v[72:73], v[18:33]
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
	ds_read_b128 v[124:127], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[96:99], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[100:103], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[108:111], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[104:107], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[120:123], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[160:163], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[168:171], v67
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
	ds_read_b128 v[172:175], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_36
	scratch_load_dwordx4 v[66:69], off, off
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:12288
	scratch_load_dwordx4 v[66:69], off, off offset:16
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:18432
.LBB0_36:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s11, s85, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_38
	v_add_u32_e32 v66, s11, v1
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:16
.LBB0_38:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v113, v0
	;;#ASMEND
	s_addk_i32 s12, 0x1000
	v_lshlrev_b32_e32 v114, 3, v113
	v_lshlrev_b32_e32 v113, 5, v113
	v_and_b32_e32 v114, 0xf8, v114
	v_and_b32_e32 v113, 0x400, v113
	v_or3_b32 v114, v114, v113, s12
	v_mfma_f32_32x32x8_bf16 v[66:81], v[126:127], v[180:181], v[66:81]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, s[80:81]
	global_load_dwordx4 v[144:147], v[114:115], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[182:183], v[66:81]
	v_add_co_u32_e32 v96, vcc, s78, v114
	s_nop 1
	v_addc_co_u32_e32 v97, vcc, 0, v115, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[186:187], v[66:81]
	global_load_dwordx4 v[148:151], v[114:115], off offset:512
	global_load_dwordx4 v[136:139], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[192:193], v[66:81]
	global_load_dwordx4 v[140:143], v[114:115], off offset:1536
	global_load_dwordx4 v[128:131], v[96:97], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[198:199], v[66:81]
	global_load_dwordx4 v[132:135], v[96:97], off offset:512
	global_load_dwordx4 v[124:127], v[96:97], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[204:205], v[66:81]
	global_load_dwordx4 v[120:123], v[96:97], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v96, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v96, v96, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v96, v96, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v96, v96, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v96, v96, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v96, v96, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v96, v96, v78, v79
	v_max3_f32 v96, v96, v80, v81
	ds_bpermute_b32 v97, v155, v96
	v_mov_b32_e32 v113, 1.0
	v_mov_b32_e32 v98, v118
	v_mov_b32_e32 v99, v118
	v_mov_b32_e32 v100, v118
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v97, v97, v97
	v_max_f32_e32 v114, v96, v97
	;;#ASMSTART
	v_add_f32 v96, v118, v157
	;;#ASMEND
	v_mov_b32_e32 v97, v118
	v_cmp_gt_f32_e32 vcc, v114, v96
	v_mov_b32_e32 v96, v118
	v_mov_b32_e32 v101, v118
	v_mov_b32_e32 v102, v118
	v_mov_b32_e32 v103, v118
	v_mov_b32_e32 v104, v118
	v_mov_b32_e32 v105, v118
	v_mov_b32_e32 v106, v118
	v_mov_b32_e32 v107, v118
	v_mov_b32_e32 v108, v118
	v_mov_b32_e32 v109, v118
	v_mov_b32_e32 v110, v118
	v_mov_b32_e32 v111, v118
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_40
	;;#ASMSTART
	v_add_f32 v96, v114, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v118, v96
	v_exp_f32_e32 v113, v97
	v_mov_b32_e32 v118, v96
	v_mov_b32_e32 v97, v96
	v_mov_b32_e32 v98, v96
	v_mov_b32_e32 v99, v96
	v_mov_b32_e32 v100, v96
	v_mov_b32_e32 v101, v96
	v_mov_b32_e32 v102, v96
	v_mov_b32_e32 v103, v96
	v_mov_b32_e32 v104, v96
	v_mov_b32_e32 v105, v96
	v_mov_b32_e32 v106, v96
	v_mov_b32_e32 v107, v96
	v_mov_b32_e32 v108, v96
	v_mov_b32_e32 v109, v96
	v_mov_b32_e32 v110, v96
	v_mov_b32_e32 v111, v96
.LBB0_40:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, 0, v66
	v_add_f32_e32 v114, v114, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v68
	v_add_f32_e32 v114, v114, v69
	v_add_f32_e32 v114, v114, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v71
	v_add_f32_e32 v114, v114, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v73
	v_add_f32_e32 v114, v114, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v75
	v_add_f32_e32 v114, v114, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v77
	v_add_f32_e32 v114, v114, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v113
	v_add_f32_e32 v114, v114, v79
	v_add_f32_e32 v114, v114, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v114, v114, v81
	;;#ASMSTART
	v_fma_f32 v112, v112, v113, v114
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_42
	;;#ASMSTART
	v_mul_f32 v34, v34, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v113
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[136:137], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[140:141], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[138:139], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[142:143], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[124:125], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[126:127], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[72:73], v[18:33]
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
	ds_read_b128 v[136:139], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[120:123], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[124:127], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[128:131], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[132:135], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[160:163], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[168:171], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[172:175], v67
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
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_44
	scratch_load_dwordx4 v[66:69], off, off offset:32
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69]
	scratch_load_dwordx4 v[66:69], off, off offset:48
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:6144
.LBB0_44:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_46
	v_add_u32_e32 v66, s11, v119
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off offset:32
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:48
.LBB0_46:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v113, v0
	;;#ASMEND
	s_add_i32 s4, s12, 0x1000
	v_lshlrev_b32_e32 v114, 3, v113
	v_lshlrev_b32_e32 v113, 5, v113
	v_and_b32_e32 v114, 0xf8, v114
	v_and_b32_e32 v113, 0x400, v113
	v_or3_b32 v114, v114, v113, s4
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, s[80:81]
	global_load_dwordx4 v[144:147], v[114:115], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[182:183], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[186:187], v[66:81]
	global_load_dwordx4 v[148:151], v[114:115], off offset:512
	global_load_dwordx4 v[136:139], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[126:127], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[128:129], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[130:131], v[192:193], v[66:81]
	global_load_dwordx4 v[140:143], v[114:115], off offset:1536
	v_add_co_u32_e32 v114, vcc, s78, v114
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	global_load_dwordx4 v[128:131], v[114:115], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[198:199], v[66:81]
	global_load_dwordx4 v[132:135], v[114:115], off offset:512
	global_load_dwordx4 v[124:127], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[204:205], v[66:81]
	global_load_dwordx4 v[120:123], v[114:115], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v113, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v113, v113, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v113, v113, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v113, v113, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v113, v113, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v113, v113, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v113, v113, v78, v79
	v_max3_f32 v113, v113, v80, v81
	ds_bpermute_b32 v114, v155, v113
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v114, v114, v114
	v_max_f32_e32 v114, v113, v114
	;;#ASMSTART
	v_add_f32 v113, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v114, v113
	v_mov_b32_e32 v113, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_48
	;;#ASMSTART
	v_add_f32 v96, v114, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v118, v96
	v_exp_f32_e32 v113, v97
	v_mov_b32_e32 v118, v96
	v_mov_b32_e32 v97, v96
	v_mov_b32_e32 v98, v96
	v_mov_b32_e32 v99, v96
	v_mov_b32_e32 v100, v96
	v_mov_b32_e32 v101, v96
	v_mov_b32_e32 v102, v96
	v_mov_b32_e32 v103, v96
	v_mov_b32_e32 v104, v96
	v_mov_b32_e32 v105, v96
	v_mov_b32_e32 v106, v96
	v_mov_b32_e32 v107, v96
	v_mov_b32_e32 v108, v96
	v_mov_b32_e32 v109, v96
	v_mov_b32_e32 v110, v96
	v_mov_b32_e32 v111, v96
.LBB0_48:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, 0, v66
	v_add_f32_e32 v114, v114, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v68
	v_add_f32_e32 v114, v114, v69
	v_add_f32_e32 v114, v114, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v71
	v_add_f32_e32 v114, v114, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v73
	v_add_f32_e32 v114, v114, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v75
	v_add_f32_e32 v114, v114, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v77
	v_add_f32_e32 v114, v114, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v113
	v_add_f32_e32 v114, v114, v79
	v_add_f32_e32 v114, v114, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v114, v114, v81
	;;#ASMSTART
	v_fma_f32 v112, v112, v113, v114
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_50
	;;#ASMSTART
	v_mul_f32 v34, v34, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v113
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[136:137], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[140:141], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[138:139], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[142:143], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[124:125], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[126:127], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[72:73], v[18:33]
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
	ds_read_b128 v[136:139], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[120:123], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[124:127], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[128:131], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[132:135], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[160:163], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[168:171], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[172:175], v67
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
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_52
	scratch_load_dwordx4 v[66:69], off, off
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:12288
	scratch_load_dwordx4 v[66:69], off, off offset:16
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:18432
.LBB0_52:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_54
	v_add_u32_e32 v66, s11, v153
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:16
.LBB0_54:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v113, v0
	;;#ASMEND
	s_addk_i32 s12, 0x2000
	v_lshlrev_b32_e32 v114, 3, v113
	v_lshlrev_b32_e32 v113, 5, v113
	v_and_b32_e32 v114, 0xf8, v114
	v_and_b32_e32 v113, 0x400, v113
	v_or3_b32 v114, v114, v113, s12
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, s[80:81]
	global_load_dwordx4 v[144:147], v[114:115], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[182:183], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[186:187], v[66:81]
	global_load_dwordx4 v[148:151], v[114:115], off offset:512
	global_load_dwordx4 v[136:139], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[126:127], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[128:129], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[130:131], v[192:193], v[66:81]
	global_load_dwordx4 v[140:143], v[114:115], off offset:1536
	v_add_co_u32_e32 v114, vcc, s78, v114
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	global_load_dwordx4 v[128:131], v[114:115], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[198:199], v[66:81]
	global_load_dwordx4 v[132:135], v[114:115], off offset:512
	global_load_dwordx4 v[124:127], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[204:205], v[66:81]
	global_load_dwordx4 v[120:123], v[114:115], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v113, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v113, v113, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v113, v113, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v113, v113, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v113, v113, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v113, v113, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v113, v113, v78, v79
	v_max3_f32 v113, v113, v80, v81
	ds_bpermute_b32 v114, v155, v113
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v114, v114, v114
	v_max_f32_e32 v114, v113, v114
	;;#ASMSTART
	v_add_f32 v113, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v114, v113
	v_mov_b32_e32 v113, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_56
	;;#ASMSTART
	v_add_f32 v96, v114, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v118, v96
	v_exp_f32_e32 v113, v97
	v_mov_b32_e32 v118, v96
	v_mov_b32_e32 v97, v96
	v_mov_b32_e32 v98, v96
	v_mov_b32_e32 v99, v96
	v_mov_b32_e32 v100, v96
	v_mov_b32_e32 v101, v96
	v_mov_b32_e32 v102, v96
	v_mov_b32_e32 v103, v96
	v_mov_b32_e32 v104, v96
	v_mov_b32_e32 v105, v96
	v_mov_b32_e32 v106, v96
	v_mov_b32_e32 v107, v96
	v_mov_b32_e32 v108, v96
	v_mov_b32_e32 v109, v96
	v_mov_b32_e32 v110, v96
	v_mov_b32_e32 v111, v96
.LBB0_56:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, 0, v66
	v_add_f32_e32 v114, v114, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v68
	v_add_f32_e32 v114, v114, v69
	v_add_f32_e32 v114, v114, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v71
	v_add_f32_e32 v114, v114, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v73
	v_add_f32_e32 v114, v114, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v75
	v_add_f32_e32 v114, v114, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v77
	v_add_f32_e32 v114, v114, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v113
	v_add_f32_e32 v114, v114, v79
	v_add_f32_e32 v114, v114, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v114, v114, v81
	;;#ASMSTART
	v_fma_f32 v112, v112, v113, v114
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_58
	;;#ASMSTART
	v_mul_f32 v34, v34, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v113
	;;#ASMEND
.LBB0_58:
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[136:137], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[140:141], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[138:139], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[142:143], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[124:125], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[126:127], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[72:73], v[18:33]
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
	ds_read_b128 v[136:139], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[120:123], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[124:127], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[128:131], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[132:135], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[160:163], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[168:171], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[172:175], v67
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
	buffer_load_dword v66, v66, s[68:71], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s86, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s77, v67
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_60
	scratch_load_dwordx4 v[66:69], off, off offset:32
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69]
	scratch_load_dwordx4 v[66:69], off, off offset:48
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:6144
.LBB0_60:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_62
	v_add_u32_e32 v66, s11, v154
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off offset:32
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:48
.LBB0_62:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v113, v0
	;;#ASMEND
	s_lshl_b32 s12, s85, 14
	v_lshlrev_b32_e32 v114, 3, v113
	v_lshlrev_b32_e32 v113, 5, v113
	s_add_i32 s12, s12, s84
	v_and_b32_e32 v114, 0xf8, v114
	v_and_b32_e32 v113, 0x400, v113
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	v_or3_b32 v114, v114, v113, s12
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, s[80:81]
	global_load_dwordx4 v[144:147], v[114:115], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[182:183], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[186:187], v[66:81]
	global_load_dwordx4 v[148:151], v[114:115], off offset:512
	global_load_dwordx4 v[136:139], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[126:127], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[128:129], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[130:131], v[192:193], v[66:81]
	global_load_dwordx4 v[140:143], v[114:115], off offset:1536
	v_add_co_u32_e32 v114, vcc, s78, v114
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	global_load_dwordx4 v[128:131], v[114:115], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[198:199], v[66:81]
	global_load_dwordx4 v[132:135], v[114:115], off offset:512
	global_load_dwordx4 v[124:127], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[204:205], v[66:81]
	global_load_dwordx4 v[120:123], v[114:115], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v113, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v113, v113, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v113, v113, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v113, v113, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v113, v113, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v113, v113, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v113, v113, v78, v79
	v_max3_f32 v113, v113, v80, v81
	ds_bpermute_b32 v114, v155, v113
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v114, v114, v114
	v_max_f32_e32 v114, v113, v114
	;;#ASMSTART
	v_add_f32 v113, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v114, v113
	v_mov_b32_e32 v113, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_64
	;;#ASMSTART
	v_add_f32 v96, v114, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v118, v96
	v_exp_f32_e32 v113, v97
	v_mov_b32_e32 v118, v96
	v_mov_b32_e32 v97, v96
	v_mov_b32_e32 v98, v96
	v_mov_b32_e32 v99, v96
	v_mov_b32_e32 v100, v96
	v_mov_b32_e32 v101, v96
	v_mov_b32_e32 v102, v96
	v_mov_b32_e32 v103, v96
	v_mov_b32_e32 v104, v96
	v_mov_b32_e32 v105, v96
	v_mov_b32_e32 v106, v96
	v_mov_b32_e32 v107, v96
	v_mov_b32_e32 v108, v96
	v_mov_b32_e32 v109, v96
	v_mov_b32_e32 v110, v96
	v_mov_b32_e32 v111, v96
.LBB0_64:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, 0, v66
	v_add_f32_e32 v114, v114, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v68
	v_add_f32_e32 v114, v114, v69
	v_add_f32_e32 v114, v114, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v71
	v_add_f32_e32 v114, v114, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v73
	v_add_f32_e32 v114, v114, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v75
	v_add_f32_e32 v114, v114, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v77
	v_add_f32_e32 v114, v114, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v113
	v_add_f32_e32 v114, v114, v79
	v_add_f32_e32 v114, v114, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v114, v114, v81
	;;#ASMSTART
	v_fma_f32 v112, v112, v113, v114
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_66
	;;#ASMSTART
	v_mul_f32 v34, v34, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v113
	;;#ASMEND
.LBB0_66:
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[136:137], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[140:141], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[138:139], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[142:143], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[124:125], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[126:127], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[72:73], v[18:33]
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
	ds_read_b128 v[136:139], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[120:123], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[124:127], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[128:131], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[132:135], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[160:163], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[168:171], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[172:175], v67
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
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_68
	scratch_load_dwordx4 v[66:69], off, off
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:12288
	scratch_load_dwordx4 v[66:69], off, off offset:16
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:18432
.LBB0_68:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s11, s9, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_70
	v_add_u32_e32 v66, s11, v1
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:16
.LBB0_70:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v113, v0
	;;#ASMEND
	s_addk_i32 s12, 0x1000
	v_lshlrev_b32_e32 v114, 3, v113
	v_lshlrev_b32_e32 v113, 5, v113
	v_and_b32_e32 v114, 0xf8, v114
	v_and_b32_e32 v113, 0x400, v113
	v_or3_b32 v114, v114, v113, s12
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, s[80:81]
	global_load_dwordx4 v[144:147], v[114:115], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[182:183], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[186:187], v[66:81]
	global_load_dwordx4 v[148:151], v[114:115], off offset:512
	global_load_dwordx4 v[136:139], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[126:127], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[128:129], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[130:131], v[192:193], v[66:81]
	global_load_dwordx4 v[140:143], v[114:115], off offset:1536
	v_add_co_u32_e32 v114, vcc, s78, v114
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	global_load_dwordx4 v[128:131], v[114:115], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[198:199], v[66:81]
	global_load_dwordx4 v[132:135], v[114:115], off offset:512
	global_load_dwordx4 v[124:127], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[204:205], v[66:81]
	global_load_dwordx4 v[120:123], v[114:115], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v113, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v113, v113, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v113, v113, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v113, v113, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v113, v113, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v113, v113, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v113, v113, v78, v79
	v_max3_f32 v113, v113, v80, v81
	ds_bpermute_b32 v114, v155, v113
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v114, v114, v114
	v_max_f32_e32 v114, v113, v114
	;;#ASMSTART
	v_add_f32 v113, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v114, v113
	v_mov_b32_e32 v113, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_72
	;;#ASMSTART
	v_add_f32 v96, v114, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v118, v96
	v_exp_f32_e32 v113, v97
	v_mov_b32_e32 v118, v96
	v_mov_b32_e32 v97, v96
	v_mov_b32_e32 v98, v96
	v_mov_b32_e32 v99, v96
	v_mov_b32_e32 v100, v96
	v_mov_b32_e32 v101, v96
	v_mov_b32_e32 v102, v96
	v_mov_b32_e32 v103, v96
	v_mov_b32_e32 v104, v96
	v_mov_b32_e32 v105, v96
	v_mov_b32_e32 v106, v96
	v_mov_b32_e32 v107, v96
	v_mov_b32_e32 v108, v96
	v_mov_b32_e32 v109, v96
	v_mov_b32_e32 v110, v96
	v_mov_b32_e32 v111, v96
.LBB0_72:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, 0, v66
	v_add_f32_e32 v114, v114, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v68
	v_add_f32_e32 v114, v114, v69
	v_add_f32_e32 v114, v114, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v71
	v_add_f32_e32 v114, v114, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v73
	v_add_f32_e32 v114, v114, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v75
	v_add_f32_e32 v114, v114, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v77
	v_add_f32_e32 v114, v114, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v113
	v_add_f32_e32 v114, v114, v79
	v_add_f32_e32 v114, v114, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v114, v114, v81
	;;#ASMSTART
	v_fma_f32 v112, v112, v113, v114
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_74
	;;#ASMSTART
	v_mul_f32 v34, v34, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v113
	;;#ASMEND
.LBB0_74:
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[136:137], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[140:141], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[138:139], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[142:143], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[124:125], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[126:127], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[72:73], v[18:33]
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
	ds_read_b128 v[136:139], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[120:123], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[124:127], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[128:131], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[132:135], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[160:163], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[168:171], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[172:175], v67
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
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_76
	scratch_load_dwordx4 v[66:69], off, off offset:32
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69]
	scratch_load_dwordx4 v[66:69], off, off offset:48
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:6144
.LBB0_76:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_78
	v_add_u32_e32 v66, s11, v119
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off offset:32
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:48
.LBB0_78:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v113, v0
	;;#ASMEND
	s_add_i32 s4, s12, 0x1000
	v_lshlrev_b32_e32 v114, 3, v113
	v_lshlrev_b32_e32 v113, 5, v113
	v_and_b32_e32 v114, 0xf8, v114
	v_and_b32_e32 v113, 0x400, v113
	v_or3_b32 v114, v114, v113, s4
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, s[80:81]
	global_load_dwordx4 v[144:147], v[114:115], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[182:183], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[186:187], v[66:81]
	global_load_dwordx4 v[148:151], v[114:115], off offset:512
	global_load_dwordx4 v[136:139], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[126:127], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[128:129], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[130:131], v[192:193], v[66:81]
	global_load_dwordx4 v[140:143], v[114:115], off offset:1536
	v_add_co_u32_e32 v114, vcc, s78, v114
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	global_load_dwordx4 v[128:131], v[114:115], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[198:199], v[66:81]
	global_load_dwordx4 v[132:135], v[114:115], off offset:512
	global_load_dwordx4 v[124:127], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[204:205], v[66:81]
	global_load_dwordx4 v[120:123], v[114:115], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v113, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v113, v113, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v113, v113, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v113, v113, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v113, v113, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v113, v113, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v113, v113, v78, v79
	v_max3_f32 v113, v113, v80, v81
	ds_bpermute_b32 v114, v155, v113
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v114, v114, v114
	v_max_f32_e32 v114, v113, v114
	;;#ASMSTART
	v_add_f32 v113, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v114, v113
	v_mov_b32_e32 v113, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_80
	;;#ASMSTART
	v_add_f32 v96, v114, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v118, v96
	v_exp_f32_e32 v113, v97
	v_mov_b32_e32 v118, v96
	v_mov_b32_e32 v97, v96
	v_mov_b32_e32 v98, v96
	v_mov_b32_e32 v99, v96
	v_mov_b32_e32 v100, v96
	v_mov_b32_e32 v101, v96
	v_mov_b32_e32 v102, v96
	v_mov_b32_e32 v103, v96
	v_mov_b32_e32 v104, v96
	v_mov_b32_e32 v105, v96
	v_mov_b32_e32 v106, v96
	v_mov_b32_e32 v107, v96
	v_mov_b32_e32 v108, v96
	v_mov_b32_e32 v109, v96
	v_mov_b32_e32 v110, v96
	v_mov_b32_e32 v111, v96
.LBB0_80:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, 0, v66
	v_add_f32_e32 v114, v114, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v68
	v_add_f32_e32 v114, v114, v69
	v_add_f32_e32 v114, v114, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v71
	v_add_f32_e32 v114, v114, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v73
	v_add_f32_e32 v114, v114, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v75
	v_add_f32_e32 v114, v114, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v114, v114, v77
	v_add_f32_e32 v114, v114, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v113
	v_add_f32_e32 v114, v114, v79
	v_add_f32_e32 v114, v114, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v114, v114, v81
	;;#ASMSTART
	v_fma_f32 v112, v112, v113, v114
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_82
	;;#ASMSTART
	v_mul_f32 v34, v34, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v113
	;;#ASMEND
.LBB0_82:
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[136:137], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[140:141], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[138:139], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[142:143], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[124:125], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[126:127], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[72:73], v[18:33]
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
	ds_read_b128 v[136:139], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[120:123], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[124:127], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[128:131], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[132:135], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[160:163], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[168:171], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[172:175], v67
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
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_84
	scratch_load_dwordx4 v[66:69], off, off
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:12288
	scratch_load_dwordx4 v[66:69], off, off offset:16
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:18432
.LBB0_84:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_86
	v_add_u32_e32 v66, s11, v153
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:16
.LBB0_86:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v113, v0
	;;#ASMEND
	s_addk_i32 s12, 0x2000
	v_lshlrev_b32_e32 v114, 3, v113
	v_lshlrev_b32_e32 v113, 5, v113
	v_and_b32_e32 v114, 0xf8, v114
	v_and_b32_e32 v113, 0x400, v113
	v_or3_b32 v114, v114, v113, s12
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, s[80:81]
	global_load_dwordx4 v[144:147], v[114:115], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[182:183], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[186:187], v[66:81]
	global_load_dwordx4 v[148:151], v[114:115], off offset:512
	global_load_dwordx4 v[136:139], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[126:127], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[128:129], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[130:131], v[192:193], v[66:81]
	global_load_dwordx4 v[140:143], v[114:115], off offset:1536
	v_add_co_u32_e32 v114, vcc, s78, v114
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	global_load_dwordx4 v[128:131], v[114:115], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[198:199], v[66:81]
	global_load_dwordx4 v[132:135], v[114:115], off offset:512
	global_load_dwordx4 v[124:127], v[114:115], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[204:205], v[66:81]
	global_load_dwordx4 v[120:123], v[114:115], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v113, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v113, v113, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v113, v113, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v113, v113, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v113, v113, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v113, v113, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v113, v113, v78, v79
	v_max3_f32 v113, v113, v80, v81
	ds_bpermute_b32 v114, v155, v113
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v114, v114, v114
	v_max_f32_e32 v114, v113, v114
	;;#ASMSTART
	v_add_f32 v113, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v114, v113
	v_mov_b32_e32 v113, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_88
	;;#ASMSTART
	v_add_f32 v96, v114, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v118, v96
	v_exp_f32_e32 v113, v97
	v_mov_b32_e32 v118, v96
	v_mov_b32_e32 v97, v96
	v_mov_b32_e32 v98, v96
	v_mov_b32_e32 v99, v96
	v_mov_b32_e32 v100, v96
	v_mov_b32_e32 v101, v96
	v_mov_b32_e32 v102, v96
	v_mov_b32_e32 v103, v96
	v_mov_b32_e32 v104, v96
	v_mov_b32_e32 v105, v96
	v_mov_b32_e32 v106, v96
	v_mov_b32_e32 v107, v96
	v_mov_b32_e32 v108, v96
	v_mov_b32_e32 v109, v96
	v_mov_b32_e32 v110, v96
	v_mov_b32_e32 v111, v96
.LBB0_88:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[98:99] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v96, 0, v66
	v_add_f32_e32 v96, v96, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v96, v96, v68
	v_add_f32_e32 v96, v96, v69
	v_add_f32_e32 v96, v96, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v96, v96, v71
	v_add_f32_e32 v96, v96, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v96, v96, v73
	v_add_f32_e32 v96, v96, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v96, v96, v75
	v_add_f32_e32 v96, v96, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v96, v96, v77
	v_add_f32_e32 v96, v96, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v113
	v_add_f32_e32 v96, v96, v79
	v_add_f32_e32 v96, v96, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v96, v96, v81
	;;#ASMSTART
	v_fma_f32 v160, v112, v113, v96
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_25
	;;#ASMSTART
	v_mul_f32 v34, v34, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v113
	;;#ASMEND
	s_branch .LBB0_25
.LBB0_90:
	s_mov_b32 s85, s10
	s_mov_b32 s11, s9
	s_cmp_ge_i32 s96, s92
	s_cbranch_scc0 .LBB0_92
	s_branch .LBB0_162
.LBB0_91:
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
	s_mov_b64 s[0:1], s[48:49]
	s_mov_b32 s48, s36
	s_mov_b32 s49, s36
	s_mov_b64 s[4:5], s[50:51]
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	s_mov_b64 s[8:9], s[52:53]
	s_mov_b32 s52, s36
	s_mov_b32 s53, s36
	s_mov_b64 s[12:13], s[54:55]
	s_mov_b32 s54, s36
	s_mov_b32 s55, s36
	s_mov_b64 s[14:15], s[56:57]
	s_mov_b32 s56, s36
	s_mov_b32 s57, s36
	s_mov_b32 s10, s58
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
	v_mov_b64_e32 v[34:35], s[36:37]
	v_mov_b64_e32 v[36:37], s[38:39]
	v_mov_b64_e32 v[38:39], s[40:41]
	v_mov_b64_e32 v[40:41], s[42:43]
	v_mov_b64_e32 v[42:43], s[44:45]
	v_mov_b64_e32 v[44:45], s[46:47]
	v_mov_b64_e32 v[46:47], s[48:49]
	v_mov_b64_e32 v[48:49], s[50:51]
	v_mov_b64_e32 v[50:51], s[52:53]
	v_mov_b64_e32 v[52:53], s[54:55]
	v_mov_b64_e32 v[54:55], s[56:57]
	v_mov_b64_e32 v[56:57], s[58:59]
	v_mov_b64_e32 v[58:59], s[60:61]
	v_mov_b64_e32 v[60:61], s[62:63]
	v_mov_b64_e32 v[62:63], s[64:65]
	v_mov_b64_e32 v[64:65], s[66:67]
	v_mov_b32_e32 v118, 0xff800000
	v_mov_b32_e32 v160, 0
	s_mov_b32 s58, s10
	s_movk_i32 s47, 0xf80
	s_mov_b64 s[56:57], s[14:15]
	s_mov_b64 s[54:55], s[12:13]
	s_mov_b64 s[52:53], s[8:9]
	s_mov_b64 s[50:51], s[4:5]
	s_mov_b64 s[48:49], s[0:1]
	v_mov_b64_e32 v[2:3], v[34:35]
	v_mov_b64_e32 v[4:5], v[36:37]
	v_mov_b64_e32 v[6:7], v[38:39]
	v_mov_b64_e32 v[8:9], v[40:41]
	v_mov_b64_e32 v[10:11], v[42:43]
	v_mov_b64_e32 v[12:13], v[44:45]
	v_mov_b64_e32 v[14:15], v[46:47]
	v_mov_b64_e32 v[16:17], v[48:49]
	v_mov_b64_e32 v[18:19], v[50:51]
	v_mov_b64_e32 v[20:21], v[52:53]
	v_mov_b64_e32 v[22:23], v[54:55]
	v_mov_b64_e32 v[24:25], v[56:57]
	v_mov_b64_e32 v[26:27], v[58:59]
	v_mov_b64_e32 v[28:29], v[60:61]
	v_mov_b64_e32 v[30:31], v[62:63]
	v_mov_b64_e32 v[32:33], v[64:65]
	s_cmp_ge_i32 s96, s92
	s_cbranch_scc1 .LBB0_162
.LBB0_92:
	s_lshl_b32 s37, s7, 7
	s_lshl_b32 s0, s96, 7
	s_ashr_i32 s93, s92, 31
	s_add_i32 s37, s37, s6
	v_mov_b32_e32 v117, v116
	v_mov_b32_e32 v254, v116
	v_mov_b32_e32 v255, v116
	v_mov_b32_e32 v150, v116
	v_mov_b32_e32 v151, v116
	v_mov_b32_e32 v124, v116
	v_mov_b32_e32 v125, v116
	v_mov_b32_e32 v126, v116
	v_mov_b32_e32 v127, v116
	v_mov_b32_e32 v128, v116
	v_mov_b32_e32 v129, v116
	v_mov_b32_e32 v130, v116
	v_mov_b32_e32 v131, v116
	v_mov_b32_e32 v132, v116
	v_mov_b32_e32 v133, v116
	s_add_i32 s40, s0, 0xf7
	s_add_i32 s41, s96, 1
	s_lshl2_add_u32 s42, s96, 20
	s_branch .LBB0_95
.LBB0_93:
	s_or_b64 exec, exec, s[0:1]
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
	;;#ASMSTART
	v_perm_b32 v2, v3, v2, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v5, v4, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v7, v6, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v9, v8, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v11, v10, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v13, v12, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v15, v14, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v17, v16, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[52:67], v[46:47], v[2:3], v[52:67]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[68:83], v[120:121], v[2:3], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[38:39], v[2:3], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[42:43], v[2:3], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[48:49], v[4:5], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[122:123], v[4:5], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[40:41], v[4:5], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[44:45], v[4:5], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[30:31], v[6:7], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[34:35], v[6:7], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[26:27], v[6:7], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[22:23], v[6:7], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[32:33], v[8:9], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[36:37], v[8:9], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[28:29], v[8:9], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[24:25], v[8:9], v[100:115]
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
	ds_read_b128 v[120:123], v3 offset:12288
	v_add_u32_e32 v3, 0x1810, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[162:165], v3
	v_add_u32_e32 v3, 0x1820, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[166:169], v3
	v_add_u32_e32 v3, 0x1830, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[170:173], v3
	v_add_u32_e32 v3, 0x1840, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[174:177], v3
	v_add_u32_e32 v3, 0x1850, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[242:245], v3
	v_add_u32_e32 v3, 0x1860, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[246:249], v3
	v_add_u32_e32 v3, 0x1870, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[250:253], v3
	v_add_u32_e32 v3, 0x1880, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[234:237], v3
	v_add_u32_e32 v3, 0x1890, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[238:241], v3
	v_add_u32_e32 v3, 0x18a0, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_u32_e32 v2, 0x18b0, v2
	ds_read_b128 v[230:233], v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[226:229], v2
	v_mov_b64_e32 v[34:35], v[52:53]
	v_mov_b64_e32 v[36:37], v[54:55]
	v_mov_b64_e32 v[38:39], v[56:57]
	v_mov_b64_e32 v[40:41], v[58:59]
	v_mov_b64_e32 v[42:43], v[60:61]
	v_mov_b64_e32 v[44:45], v[62:63]
	v_mov_b64_e32 v[46:47], v[64:65]
	v_mov_b64_e32 v[48:49], v[66:67]
	v_mov_b64_e32 v[50:51], v[68:69]
	v_mov_b64_e32 v[52:53], v[70:71]
	v_mov_b64_e32 v[54:55], v[72:73]
	v_mov_b64_e32 v[56:57], v[74:75]
	v_mov_b64_e32 v[58:59], v[76:77]
	v_mov_b64_e32 v[60:61], v[78:79]
	v_mov_b64_e32 v[62:63], v[80:81]
	v_mov_b64_e32 v[64:65], v[82:83]
	v_mov_b64_e32 v[2:3], v[84:85]
	v_mov_b64_e32 v[4:5], v[86:87]
	v_mov_b64_e32 v[6:7], v[88:89]
	v_mov_b64_e32 v[8:9], v[90:91]
	v_mov_b64_e32 v[10:11], v[92:93]
	v_mov_b64_e32 v[12:13], v[94:95]
	v_mov_b64_e32 v[14:15], v[96:97]
	v_mov_b64_e32 v[16:17], v[98:99]
	v_mov_b64_e32 v[18:19], v[100:101]
	v_mov_b64_e32 v[20:21], v[102:103]
	v_mov_b64_e32 v[22:23], v[104:105]
	v_mov_b64_e32 v[24:25], v[106:107]
	v_mov_b64_e32 v[26:27], v[108:109]
	v_mov_b64_e32 v[28:29], v[110:111]
	v_mov_b64_e32 v[30:31], v[112:113]
	v_mov_b64_e32 v[32:33], v[114:115]
.LBB0_94:
	s_and_b64 s[0:1], s[38:39], exec
	s_cselect_b32 s11, s83, s85
	s_cselect_b32 s85, s86, s83
	s_cselect_b32 s83, s43, s86
	s_add_u32 s96, s96, 2
	s_addc_u32 s97, s97, 0
	v_mov_b64_e32 v[66:67], s[92:93]
	v_cmp_lt_i64_e32 vcc, s[96:97], v[66:67]
	s_addk_i32 s40, 0x100
	s_add_i32 s41, s41, 2
	s_add_i32 s42, s42, 8
	s_mov_b32 s86, s44
	s_cbranch_vccz .LBB0_161
.LBB0_95:
	s_add_i32 s0, s42, -4
	v_mov_b32_e32 v66, s0
	buffer_load_dword v66, v66, s[68:71], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s77, v67
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_97
	scratch_load_dwordx4 v[66:69], off, off offset:32
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69]
	scratch_load_dwordx4 v[66:69], off, off offset:48
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:6144
.LBB0_97:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_99
	s_mul_i32 s4, s11, 0x3000
	v_add_u32_e32 v66, s4, v154
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off offset:32
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:48
.LBB0_99:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v82, v0
	;;#ASMEND
	s_lshl_b32 s38, s11, 14
	v_lshlrev_b32_e32 v83, 3, v82
	v_lshlrev_b32_e32 v82, 5, v82
	s_add_i32 s38, s38, s84
	v_and_b32_e32 v83, 0xf8, v83
	v_and_b32_e32 v82, 0x400, v82
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[180:181], v[66:81]
	v_or3_b32 v82, v83, v82, s38
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, s[80:81]
	global_load_dwordx4 v[108:111], v[82:83], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[182:183], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[186:187], v[66:81]
	global_load_dwordx4 v[112:115], v[82:83], off offset:512
	global_load_dwordx4 v[100:103], v[82:83], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[192:193], v[66:81]
	global_load_dwordx4 v[104:107], v[82:83], off offset:1536
	v_add_co_u32_e32 v82, vcc, s78, v82
	s_nop 1
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	global_load_dwordx4 v[92:95], v[82:83], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[176:177], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[198:199], v[66:81]
	global_load_dwordx4 v[96:99], v[82:83], off offset:512
	global_load_dwordx4 v[88:91], v[82:83], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[244:245], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[246:247], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[248:249], v[204:205], v[66:81]
	global_load_dwordx4 v[84:87], v[82:83], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[250:251], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[252:253], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v82, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v82, 2, v82
	v_and_b32_e32 v82, 8, v82
	v_add_u32_e32 v82, s40, v82
	v_add_u32_e32 v83, 0xffffff09, v82
	v_cmp_gt_i32_e32 vcc, s37, v83
	v_add_u32_e32 v83, 0xffffff0a, v82
	v_cmp_gt_i32_e64 s[0:1], s37, v83
	v_add_u32_e32 v83, 0xffffff0b, v82
	v_cmp_gt_i32_e64 s[4:5], s37, v83
	v_add_u32_e32 v83, 0xffffff0c, v82
	v_cmp_gt_i32_e64 s[6:7], s37, v83
	v_add_u32_e32 v83, 0xffffff0d, v82
	v_cmp_gt_i32_e64 s[8:9], s37, v83
	v_add_u32_e32 v83, 0xffffff0e, v82
	v_cmp_gt_i32_e64 s[10:11], s37, v83
	v_add_u32_e32 v83, 0xffffff0f, v82
	v_cmp_gt_i32_e64 s[12:13], s37, v83
	v_add_u32_e32 v83, 0xffffff10, v82
	v_cmp_gt_i32_e64 s[14:15], s37, v83
	v_add_u32_e32 v83, 0xffffff19, v82
	v_cmp_gt_i32_e64 s[16:17], s37, v83
	v_add_u32_e32 v83, 0xffffff1a, v82
	v_cmp_gt_i32_e64 s[18:19], s37, v83
	v_add_u32_e32 v83, 0xffffff1b, v82
	v_cmp_gt_i32_e64 s[20:21], s37, v83
	v_add_u32_e32 v83, 0xffffff1c, v82
	v_cmp_gt_i32_e64 s[22:23], s37, v83
	v_add_u32_e32 v83, 0xffffff1d, v82
	v_cmp_gt_i32_e64 s[24:25], s37, v83
	v_add_u32_e32 v83, 0xffffff1e, v82
	v_cmp_gt_i32_e64 s[26:27], s37, v83
	v_add_u32_e32 v83, 0xffffff1f, v82
	v_add_u32_e32 v82, 0xffffff20, v82
	v_cmp_gt_i32_e64 s[28:29], s37, v83
	v_cmp_gt_i32_e64 s[30:31], s37, v82
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
	v_cndmask_b32_e64 v79, v159, v79, s[26:27]
	v_cndmask_b32_e64 v78, v159, v78, s[24:25]
	v_cndmask_b32_e64 v123, v159, v67, s[0:1]
	v_cndmask_b32_e32 v122, v159, v66, vcc
	v_cndmask_b32_e64 v77, v159, v77, s[22:23]
	v_cndmask_b32_e64 v76, v159, v76, s[20:21]
	v_cndmask_b32_e64 v75, v159, v75, s[18:19]
	v_cndmask_b32_e64 v74, v159, v74, s[16:17]
	v_cndmask_b32_e64 v83, v159, v71, s[10:11]
	v_cndmask_b32_e64 v82, v159, v70, s[8:9]
	v_cndmask_b32_e64 v121, v159, v69, s[6:7]
	v_cndmask_b32_e64 v120, v159, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[130:131], v[78:79]
	v_pk_mul_f32 v[78:79], v[116:117], v[122:123]
	v_pk_mul_f32 v[68:69], v[128:129], v[76:77]
	v_pk_mul_f32 v[70:71], v[126:127], v[74:75]
	v_pk_mul_f32 v[74:75], v[150:151], v[82:83]
	v_pk_mul_f32 v[76:77], v[254:255], v[120:121]
	v_max_f32_e32 v82, v78, v79
	v_cndmask_b32_e64 v73, v159, v73, s[14:15]
	v_cndmask_b32_e64 v72, v159, v72, s[12:13]
	v_max3_f32 v82, v82, v76, v77
	v_pk_mul_f32 v[72:73], v[124:125], v[72:73]
	v_max3_f32 v82, v82, v74, v75
	v_max3_f32 v82, v82, v72, v73
	v_max3_f32 v82, v82, v70, v71
	v_cndmask_b32_e64 v81, v159, v81, s[30:31]
	v_cndmask_b32_e64 v80, v159, v80, s[28:29]
	v_max3_f32 v82, v82, v68, v69
	v_pk_mul_f32 v[80:81], v[132:133], v[80:81]
	v_max3_f32 v82, v82, v66, v67
	v_max3_f32 v82, v82, v80, v81
	ds_bpermute_b32 v83, v155, v82
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v83, v83, v83
	v_max_f32_e32 v82, v82, v83
	;;#ASMSTART
	v_add_f32 v83, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v82, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_101
	;;#ASMSTART
	v_add_f32 v82, v82, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v118, v82
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v118, v82
.LBB0_101:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[120:121], v[66:67], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67], v[78:79], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[122:123], v[68:69], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	v_pk_add_f32 v[68:69], v[76:77], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, 0, v66
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	v_pk_add_f32 v[134:135], v[70:71], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v67
	v_add_f32_e32 v82, v82, v68
	v_pk_add_f32 v[70:71], v[74:75], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v69
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v134
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v75, v135
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v82, v82, v70
	v_add_f32_e32 v82, v82, v71
	v_add_f32_e32 v82, v82, v72
	v_add_f32_e32 v82, v82, v73
	v_add_f32_e32 v82, v82, v74
	v_add_f32_e32 v82, v82, v75
	;;#ASMSTART
	v_exp_f32 v76, v122
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v77, v123
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v120
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v76
	v_add_f32_e32 v82, v82, v77
	v_add_f32_e32 v82, v82, v78
	;;#ASMSTART
	v_exp_f32 v79, v121
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v82, v82, v79
	v_add_f32_e32 v82, v82, v80
	v_add_f32_e32 v82, v82, v81
	;;#ASMSTART
	v_fma_f32 v82, v160, v83, v82
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_103
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
.LBB0_103:
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[108:109], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[112:113], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[104:105], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[110:111], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[114:115], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[92:93], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[96:97], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[88:89], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[84:85], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[98:99], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[90:91], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[86:87], v[72:73], v[18:33]
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
	ds_read_b128 v[102:105], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[84:87], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[88:91], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[92:95], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[98:101], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[134:137], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[138:141], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[142:145], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[146:149], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[168:171], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[160:163], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_105
	scratch_load_dwordx4 v[66:69], off, off
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:12288
	scratch_load_dwordx4 v[66:69], off, off offset:16
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:18432
.LBB0_105:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s45, s85, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_107
	v_add_u32_e32 v66, s45, v1
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:16
.LBB0_107:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_addk_i32 s38, 0x1000
	v_lshlrev_b32_e32 v96, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	v_and_b32_e32 v96, 0xf8, v96
	v_and_b32_e32 v83, 0x400, v83
	v_or3_b32 v96, v96, v83, s38
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[180:181], v[66:81]
	v_ashrrev_i32_e32 v97, 31, v96
	v_lshl_add_u64 v[96:97], v[96:97], 1, s[80:81]
	global_load_dwordx4 v[110:113], v[96:97], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[84:85], v[182:183], v[66:81]
	v_add_co_u32_e32 v84, vcc, s78, v96
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v97, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[86:87], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[88:89], v[186:187], v[66:81]
	global_load_dwordx4 v[120:123], v[96:97], off offset:512
	global_load_dwordx4 v[102:105], v[96:97], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[90:91], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[92:93], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[94:95], v[192:193], v[66:81]
	global_load_dwordx4 v[106:109], v[96:97], off offset:1536
	s_nop 0
	global_load_dwordx4 v[94:97], v[84:85], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[198:199], v[66:81]
	global_load_dwordx4 v[98:101], v[84:85], off offset:512
	global_load_dwordx4 v[90:93], v[84:85], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[140:141], v[204:205], v[66:81]
	global_load_dwordx4 v[86:89], v[84:85], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[148:149], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	v_mov_b32_e32 v134, v118
	v_lshrrev_b32_e32 v83, 2, v83
	v_and_b32_e32 v83, 8, v83
	v_add_u32_e32 v83, s40, v83
	v_add_u32_e32 v84, 0xffffff29, v83
	v_cmp_gt_i32_e32 vcc, s37, v84
	v_add_u32_e32 v84, 0xffffff2a, v83
	v_cmp_gt_i32_e64 s[0:1], s37, v84
	v_add_u32_e32 v84, 0xffffff2b, v83
	v_cmp_gt_i32_e64 s[4:5], s37, v84
	v_add_u32_e32 v84, 0xffffff2c, v83
	v_cmp_gt_i32_e64 s[6:7], s37, v84
	v_add_u32_e32 v84, 0xffffff2d, v83
	v_cmp_gt_i32_e64 s[8:9], s37, v84
	v_add_u32_e32 v84, 0xffffff2e, v83
	v_cmp_gt_i32_e64 s[10:11], s37, v84
	v_add_u32_e32 v84, 0xffffff2f, v83
	v_cmp_gt_i32_e64 s[12:13], s37, v84
	v_add_u32_e32 v84, 0xffffff30, v83
	v_cmp_gt_i32_e64 s[14:15], s37, v84
	v_add_u32_e32 v84, 0xffffff39, v83
	v_cmp_gt_i32_e64 s[16:17], s37, v84
	v_add_u32_e32 v84, 0xffffff3a, v83
	v_cmp_gt_i32_e64 s[18:19], s37, v84
	v_add_u32_e32 v84, 0xffffff3b, v83
	v_cmp_gt_i32_e64 s[20:21], s37, v84
	v_add_u32_e32 v84, 0xffffff3c, v83
	v_cmp_gt_i32_e64 s[22:23], s37, v84
	v_add_u32_e32 v84, 0xffffff3d, v83
	v_cmp_gt_i32_e64 s[24:25], s37, v84
	v_add_u32_e32 v84, 0xffffff3e, v83
	v_cmp_gt_i32_e64 s[26:27], s37, v84
	v_add_u32_e32 v84, 0xffffff3f, v83
	v_add_u32_e32 v83, 0xffffff40, v83
	v_cmp_gt_i32_e64 s[28:29], s37, v84
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
	v_cndmask_b32_e64 v67, v159, v67, s[0:1]
	v_cndmask_b32_e32 v66, v159, v66, vcc
	v_cndmask_b32_e64 v69, v159, v69, s[6:7]
	v_cndmask_b32_e64 v68, v159, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_cndmask_b32_e64 v71, v159, v71, s[10:11]
	v_cndmask_b32_e64 v70, v159, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[254:255], v[68:69]
	v_max_f32_e32 v83, v66, v67
	v_cndmask_b32_e64 v73, v159, v73, s[14:15]
	v_cndmask_b32_e64 v72, v159, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[150:151], v[70:71]
	v_max3_f32 v83, v83, v68, v69
	v_cndmask_b32_e64 v75, v159, v75, s[18:19]
	v_cndmask_b32_e64 v74, v159, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[124:125], v[72:73]
	v_max3_f32 v83, v83, v70, v71
	v_cndmask_b32_e64 v77, v159, v77, s[22:23]
	v_cndmask_b32_e64 v76, v159, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[126:127], v[74:75]
	v_max3_f32 v83, v83, v72, v73
	v_cndmask_b32_e64 v79, v159, v79, s[26:27]
	v_cndmask_b32_e64 v78, v159, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[128:129], v[76:77]
	v_max3_f32 v83, v83, v74, v75
	v_cndmask_b32_e64 v81, v159, v81, s[30:31]
	v_cndmask_b32_e64 v80, v159, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[130:131], v[78:79]
	v_max3_f32 v83, v83, v76, v77
	v_pk_mul_f32 v[80:81], v[132:133], v[80:81]
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v84, v155, v83
	v_mov_b32_e32 v135, v118
	v_mov_b32_e32 v136, v118
	v_mov_b32_e32 v137, v118
	v_mov_b32_e32 v138, v118
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v84, v84, v84
	v_max_f32_e32 v84, v83, v84
	;;#ASMSTART
	v_add_f32 v83, v118, v157
	;;#ASMEND
	v_mov_b32_e32 v139, v118
	v_cmp_gt_f32_e32 vcc, v84, v83
	v_mov_b32_e32 v83, 1.0
	v_mov_b32_e32 v140, v118
	v_mov_b32_e32 v141, v118
	v_mov_b32_e32 v142, v118
	v_mov_b32_e32 v143, v118
	v_mov_b32_e32 v144, v118
	v_mov_b32_e32 v145, v118
	v_mov_b32_e32 v146, v118
	v_mov_b32_e32 v147, v118
	v_mov_b32_e32 v148, v118
	v_mov_b32_e32 v149, v118
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_109
	;;#ASMSTART
	v_add_f32 v134, v84, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v118, v134
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v118, v134
	v_mov_b32_e32 v135, v134
	v_mov_b32_e32 v136, v134
	v_mov_b32_e32 v137, v134
	v_mov_b32_e32 v138, v134
	v_mov_b32_e32 v139, v134
	v_mov_b32_e32 v140, v134
	v_mov_b32_e32 v141, v134
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v134
	v_mov_b32_e32 v144, v134
	v_mov_b32_e32 v145, v134
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v134
	v_mov_b32_e32 v148, v134
	v_mov_b32_e32 v149, v134
.LBB0_109:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, 0, v66
	v_add_f32_e32 v84, v84, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v68
	v_add_f32_e32 v84, v84, v69
	v_add_f32_e32 v84, v84, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v71
	v_add_f32_e32 v84, v84, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v73
	v_add_f32_e32 v84, v84, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v75
	v_add_f32_e32 v84, v84, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v77
	v_add_f32_e32 v84, v84, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v84, v84, v79
	v_add_f32_e32 v84, v84, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v84, v84, v81
	;;#ASMSTART
	v_fma_f32 v82, v82, v83, v84
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_111
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
.LBB0_111:
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[110:111], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[120:121], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[112:113], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[122:123], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[104:105], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[98:99], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[90:91], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[86:87], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[96:97], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[100:101], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[92:93], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[88:89], v[72:73], v[18:33]
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
	ds_read_b128 v[102:105], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[84:87], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[88:91], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[92:95], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[98:101], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[160:163], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[168:171], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[172:175], v67
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
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_113
	scratch_load_dwordx4 v[66:69], off, off offset:32
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69]
	scratch_load_dwordx4 v[66:69], off, off offset:48
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:6144
.LBB0_113:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_115
	v_add_u32_e32 v66, s45, v119
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off offset:32
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:48
.LBB0_115:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_add_i32 s0, s38, 0x1000
	v_lshlrev_b32_e32 v96, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	v_and_b32_e32 v96, 0xf8, v96
	v_and_b32_e32 v83, 0x400, v83
	v_or3_b32 v96, v96, v83, s0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[180:181], v[66:81]
	v_ashrrev_i32_e32 v97, 31, v96
	v_lshl_add_u64 v[96:97], v[96:97], 1, s[80:81]
	global_load_dwordx4 v[110:113], v[96:97], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[84:85], v[182:183], v[66:81]
	v_add_co_u32_e32 v84, vcc, s78, v96
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v97, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[86:87], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[88:89], v[186:187], v[66:81]
	global_load_dwordx4 v[120:123], v[96:97], off offset:512
	global_load_dwordx4 v[102:105], v[96:97], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[90:91], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[92:93], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[94:95], v[192:193], v[66:81]
	global_load_dwordx4 v[106:109], v[96:97], off offset:1536
	s_nop 0
	global_load_dwordx4 v[94:97], v[84:85], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[198:199], v[66:81]
	global_load_dwordx4 v[98:101], v[84:85], off offset:512
	global_load_dwordx4 v[90:93], v[84:85], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[204:205], v[66:81]
	global_load_dwordx4 v[86:89], v[84:85], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
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
	v_add_u32_e32 v84, 0xffffff49, v83
	v_cmp_gt_i32_e32 vcc, s37, v84
	v_add_u32_e32 v84, 0xffffff4a, v83
	v_cmp_gt_i32_e64 s[0:1], s37, v84
	v_add_u32_e32 v84, 0xffffff4b, v83
	v_cmp_gt_i32_e64 s[4:5], s37, v84
	v_add_u32_e32 v84, 0xffffff4c, v83
	v_cmp_gt_i32_e64 s[6:7], s37, v84
	v_add_u32_e32 v84, 0xffffff4d, v83
	v_cmp_gt_i32_e64 s[8:9], s37, v84
	v_add_u32_e32 v84, 0xffffff4e, v83
	v_cmp_gt_i32_e64 s[10:11], s37, v84
	v_add_u32_e32 v84, 0xffffff4f, v83
	v_cmp_gt_i32_e64 s[12:13], s37, v84
	v_add_u32_e32 v84, 0xffffff50, v83
	v_cmp_gt_i32_e64 s[14:15], s37, v84
	v_add_u32_e32 v84, 0xffffff59, v83
	v_cmp_gt_i32_e64 s[16:17], s37, v84
	v_add_u32_e32 v84, 0xffffff5a, v83
	v_cmp_gt_i32_e64 s[18:19], s37, v84
	v_add_u32_e32 v84, 0xffffff5b, v83
	v_cmp_gt_i32_e64 s[20:21], s37, v84
	v_add_u32_e32 v84, 0xffffff5c, v83
	v_cmp_gt_i32_e64 s[22:23], s37, v84
	v_add_u32_e32 v84, 0xffffff5d, v83
	v_cmp_gt_i32_e64 s[24:25], s37, v84
	v_add_u32_e32 v84, 0xffffff5e, v83
	v_cmp_gt_i32_e64 s[26:27], s37, v84
	v_add_u32_e32 v84, 0xffffff5f, v83
	v_add_u32_e32 v83, 0xffffff60, v83
	v_cmp_gt_i32_e64 s[28:29], s37, v84
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
	v_cndmask_b32_e64 v67, v159, v67, s[0:1]
	v_cndmask_b32_e32 v66, v159, v66, vcc
	v_cndmask_b32_e64 v69, v159, v69, s[6:7]
	v_cndmask_b32_e64 v68, v159, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_cndmask_b32_e64 v71, v159, v71, s[10:11]
	v_cndmask_b32_e64 v70, v159, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[254:255], v[68:69]
	v_max_f32_e32 v83, v66, v67
	v_cndmask_b32_e64 v73, v159, v73, s[14:15]
	v_cndmask_b32_e64 v72, v159, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[150:151], v[70:71]
	v_max3_f32 v83, v83, v68, v69
	v_cndmask_b32_e64 v75, v159, v75, s[18:19]
	v_cndmask_b32_e64 v74, v159, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[124:125], v[72:73]
	v_max3_f32 v83, v83, v70, v71
	v_cndmask_b32_e64 v77, v159, v77, s[22:23]
	v_cndmask_b32_e64 v76, v159, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[126:127], v[74:75]
	v_max3_f32 v83, v83, v72, v73
	v_cndmask_b32_e64 v79, v159, v79, s[26:27]
	v_cndmask_b32_e64 v78, v159, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[128:129], v[76:77]
	v_max3_f32 v83, v83, v74, v75
	v_cndmask_b32_e64 v81, v159, v81, s[30:31]
	v_cndmask_b32_e64 v80, v159, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[130:131], v[78:79]
	v_max3_f32 v83, v83, v76, v77
	v_pk_mul_f32 v[80:81], v[132:133], v[80:81]
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v84, v155, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v84, v84, v84
	v_max_f32_e32 v84, v83, v84
	;;#ASMSTART
	v_add_f32 v83, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v84, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_117
	;;#ASMSTART
	v_add_f32 v134, v84, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v118, v134
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v118, v134
	v_mov_b32_e32 v135, v134
	v_mov_b32_e32 v136, v134
	v_mov_b32_e32 v137, v134
	v_mov_b32_e32 v138, v134
	v_mov_b32_e32 v139, v134
	v_mov_b32_e32 v140, v134
	v_mov_b32_e32 v141, v134
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v134
	v_mov_b32_e32 v144, v134
	v_mov_b32_e32 v145, v134
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v134
	v_mov_b32_e32 v148, v134
	v_mov_b32_e32 v149, v134
.LBB0_117:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, 0, v66
	v_add_f32_e32 v84, v84, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v68
	v_add_f32_e32 v84, v84, v69
	v_add_f32_e32 v84, v84, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v71
	v_add_f32_e32 v84, v84, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v73
	v_add_f32_e32 v84, v84, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v75
	v_add_f32_e32 v84, v84, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v77
	v_add_f32_e32 v84, v84, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v84, v84, v79
	v_add_f32_e32 v84, v84, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v84, v84, v81
	;;#ASMSTART
	v_fma_f32 v82, v82, v83, v84
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_119
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
.LBB0_119:
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[110:111], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[120:121], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[112:113], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[122:123], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[104:105], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[98:99], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[90:91], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[86:87], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[96:97], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[100:101], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[92:93], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[88:89], v[72:73], v[18:33]
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
	ds_read_b128 v[102:105], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[84:87], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[88:91], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[92:95], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[98:101], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[160:163], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[168:171], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[172:175], v67
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
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_121
	scratch_load_dwordx4 v[66:69], off, off
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:12288
	scratch_load_dwordx4 v[66:69], off, off offset:16
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:18432
.LBB0_121:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_123
	v_add_u32_e32 v66, s45, v153
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:16
.LBB0_123:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_addk_i32 s38, 0x2000
	v_lshlrev_b32_e32 v96, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	v_and_b32_e32 v96, 0xf8, v96
	v_and_b32_e32 v83, 0x400, v83
	v_or3_b32 v96, v96, v83, s38
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[180:181], v[66:81]
	v_ashrrev_i32_e32 v97, 31, v96
	v_lshl_add_u64 v[96:97], v[96:97], 1, s[80:81]
	global_load_dwordx4 v[110:113], v[96:97], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[84:85], v[182:183], v[66:81]
	v_add_co_u32_e32 v84, vcc, s78, v96
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v97, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[86:87], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[88:89], v[186:187], v[66:81]
	global_load_dwordx4 v[120:123], v[96:97], off offset:512
	global_load_dwordx4 v[102:105], v[96:97], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[90:91], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[92:93], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[94:95], v[192:193], v[66:81]
	global_load_dwordx4 v[106:109], v[96:97], off offset:1536
	s_nop 0
	global_load_dwordx4 v[94:97], v[84:85], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[198:199], v[66:81]
	global_load_dwordx4 v[98:101], v[84:85], off offset:512
	global_load_dwordx4 v[90:93], v[84:85], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[204:205], v[66:81]
	global_load_dwordx4 v[86:89], v[84:85], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
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
	v_add_u32_e32 v84, 0xffffff69, v83
	v_cmp_gt_i32_e32 vcc, s37, v84
	v_add_u32_e32 v84, 0xffffff6a, v83
	v_cmp_gt_i32_e64 s[0:1], s37, v84
	v_add_u32_e32 v84, 0xffffff6b, v83
	v_cmp_gt_i32_e64 s[4:5], s37, v84
	v_add_u32_e32 v84, 0xffffff6c, v83
	v_cmp_gt_i32_e64 s[6:7], s37, v84
	v_add_u32_e32 v84, 0xffffff6d, v83
	v_cmp_gt_i32_e64 s[8:9], s37, v84
	v_add_u32_e32 v84, 0xffffff6e, v83
	v_cmp_gt_i32_e64 s[10:11], s37, v84
	v_add_u32_e32 v84, 0xffffff6f, v83
	v_cmp_gt_i32_e64 s[12:13], s37, v84
	v_add_u32_e32 v84, 0xffffff70, v83
	v_cmp_gt_i32_e64 s[14:15], s37, v84
	v_add_u32_e32 v84, 0xffffff79, v83
	v_cmp_gt_i32_e64 s[16:17], s37, v84
	v_add_u32_e32 v84, 0xffffff7a, v83
	v_cmp_gt_i32_e64 s[18:19], s37, v84
	v_add_u32_e32 v84, 0xffffff7b, v83
	v_cmp_gt_i32_e64 s[20:21], s37, v84
	v_add_u32_e32 v84, 0xffffff7c, v83
	v_cmp_gt_i32_e64 s[22:23], s37, v84
	v_add_u32_e32 v84, 0xffffff7d, v83
	v_cmp_gt_i32_e64 s[24:25], s37, v84
	v_add_u32_e32 v84, 0xffffff7e, v83
	v_cmp_gt_i32_e64 s[26:27], s37, v84
	v_add_u32_e32 v84, 0xffffff7f, v83
	v_add_u32_e32 v83, 0xffffff80, v83
	v_cmp_gt_i32_e64 s[28:29], s37, v84
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
	v_cndmask_b32_e64 v67, v159, v67, s[0:1]
	v_cndmask_b32_e32 v66, v159, v66, vcc
	v_cndmask_b32_e64 v69, v159, v69, s[6:7]
	v_cndmask_b32_e64 v68, v159, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_cndmask_b32_e64 v71, v159, v71, s[10:11]
	v_cndmask_b32_e64 v70, v159, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[254:255], v[68:69]
	v_max_f32_e32 v83, v66, v67
	v_cndmask_b32_e64 v73, v159, v73, s[14:15]
	v_cndmask_b32_e64 v72, v159, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[150:151], v[70:71]
	v_max3_f32 v83, v83, v68, v69
	v_cndmask_b32_e64 v75, v159, v75, s[18:19]
	v_cndmask_b32_e64 v74, v159, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[124:125], v[72:73]
	v_max3_f32 v83, v83, v70, v71
	v_cndmask_b32_e64 v77, v159, v77, s[22:23]
	v_cndmask_b32_e64 v76, v159, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[126:127], v[74:75]
	v_max3_f32 v83, v83, v72, v73
	v_cndmask_b32_e64 v79, v159, v79, s[26:27]
	v_cndmask_b32_e64 v78, v159, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[128:129], v[76:77]
	v_max3_f32 v83, v83, v74, v75
	v_cndmask_b32_e64 v81, v159, v81, s[30:31]
	v_cndmask_b32_e64 v80, v159, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[130:131], v[78:79]
	v_max3_f32 v83, v83, v76, v77
	v_pk_mul_f32 v[80:81], v[132:133], v[80:81]
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v84, v155, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v84, v84, v84
	v_max_f32_e32 v84, v83, v84
	;;#ASMSTART
	v_add_f32 v83, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v84, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_125
	;;#ASMSTART
	v_add_f32 v134, v84, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v118, v134
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v118, v134
	v_mov_b32_e32 v135, v134
	v_mov_b32_e32 v136, v134
	v_mov_b32_e32 v137, v134
	v_mov_b32_e32 v138, v134
	v_mov_b32_e32 v139, v134
	v_mov_b32_e32 v140, v134
	v_mov_b32_e32 v141, v134
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v134
	v_mov_b32_e32 v144, v134
	v_mov_b32_e32 v145, v134
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v134
	v_mov_b32_e32 v148, v134
	v_mov_b32_e32 v149, v134
.LBB0_125:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, 0, v66
	v_add_f32_e32 v84, v84, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v68
	v_add_f32_e32 v84, v84, v69
	v_add_f32_e32 v84, v84, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v71
	v_add_f32_e32 v84, v84, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v73
	v_add_f32_e32 v84, v84, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v75
	v_add_f32_e32 v84, v84, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v77
	v_add_f32_e32 v84, v84, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v84, v84, v79
	v_add_f32_e32 v84, v84, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v84, v84, v81
	;;#ASMSTART
	v_fma_f32 v160, v82, v83, v84
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_127
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
.LBB0_127:
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[110:111], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[120:121], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[112:113], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[122:123], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[104:105], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[98:99], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[90:91], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[86:87], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[96:97], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[100:101], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[92:93], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[88:89], v[72:73], v[18:33]
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
	ds_read_b128 v[120:123], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[162:165], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[166:169], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[170:173], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[174:177], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[242:245], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[246:249], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[250:253], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[234:237], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[238:241], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[230:233], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[226:229], v66
	s_cmp_gt_i32 s92, s41
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_le_i32 s92, s41
	s_cbranch_scc1 .LBB0_160
	v_mov_b32_e32 v66, s42
	buffer_load_dword v66, v66, s[68:71], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s77, v67
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_130
	scratch_load_dwordx4 v[66:69], off, off offset:32
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69]
	scratch_load_dwordx4 v[66:69], off, off offset:48
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:6144
.LBB0_130:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_132
	v_add_u32_e32 v66, s45, v154
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off offset:32
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:48
.LBB0_132:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v82, v0
	;;#ASMEND
	s_lshl_b32 s46, s85, 14
	v_lshlrev_b32_e32 v83, 3, v82
	v_lshlrev_b32_e32 v82, 5, v82
	s_add_i32 s46, s46, s84
	v_and_b32_e32 v83, 0xf8, v83
	v_and_b32_e32 v82, 0x400, v82
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[180:181], v[66:81]
	v_or3_b32 v82, v83, v82, s46
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, s[80:81]
	global_load_dwordx4 v[108:111], v[82:83], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[182:183], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[186:187], v[66:81]
	global_load_dwordx4 v[112:115], v[82:83], off offset:512
	global_load_dwordx4 v[100:103], v[82:83], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[192:193], v[66:81]
	global_load_dwordx4 v[104:107], v[82:83], off offset:1536
	v_add_co_u32_e32 v82, vcc, s78, v82
	s_nop 1
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	global_load_dwordx4 v[92:95], v[82:83], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[174:175], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[176:177], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[198:199], v[66:81]
	global_load_dwordx4 v[96:99], v[82:83], off offset:512
	global_load_dwordx4 v[88:91], v[82:83], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[244:245], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[246:247], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[248:249], v[204:205], v[66:81]
	global_load_dwordx4 v[84:87], v[82:83], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[250:251], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[252:253], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[224:225], v[66:81]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v82, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v82, 2, v82
	v_and_b32_e32 v82, 8, v82
	v_add_u32_e32 v82, s40, v82
	v_add_u32_e32 v83, 0xffffff89, v82
	v_cmp_gt_i32_e32 vcc, s37, v83
	v_add_u32_e32 v83, 0xffffff8a, v82
	v_cmp_gt_i32_e64 s[0:1], s37, v83
	v_add_u32_e32 v83, 0xffffff8b, v82
	v_cmp_gt_i32_e64 s[4:5], s37, v83
	v_add_u32_e32 v83, 0xffffff8c, v82
	v_cmp_gt_i32_e64 s[6:7], s37, v83
	v_add_u32_e32 v83, 0xffffff8d, v82
	v_cmp_gt_i32_e64 s[8:9], s37, v83
	v_add_u32_e32 v83, 0xffffff8e, v82
	v_cmp_gt_i32_e64 s[10:11], s37, v83
	v_add_u32_e32 v83, 0xffffff8f, v82
	v_cmp_gt_i32_e64 s[12:13], s37, v83
	v_add_u32_e32 v83, 0xffffff90, v82
	v_cmp_gt_i32_e64 s[14:15], s37, v83
	v_add_u32_e32 v83, 0xffffff99, v82
	v_cmp_gt_i32_e64 s[16:17], s37, v83
	v_add_u32_e32 v83, 0xffffff9a, v82
	v_cmp_gt_i32_e64 s[18:19], s37, v83
	v_add_u32_e32 v83, 0xffffff9b, v82
	v_cmp_gt_i32_e64 s[20:21], s37, v83
	v_add_u32_e32 v83, 0xffffff9c, v82
	v_cmp_gt_i32_e64 s[22:23], s37, v83
	v_add_u32_e32 v83, 0xffffff9d, v82
	v_cmp_gt_i32_e64 s[24:25], s37, v83
	v_add_u32_e32 v83, 0xffffff9e, v82
	v_cmp_gt_i32_e64 s[26:27], s37, v83
	v_add_u32_e32 v83, 0xffffff9f, v82
	v_add_u32_e32 v82, 0xffffffa0, v82
	v_cmp_gt_i32_e64 s[28:29], s37, v83
	v_cmp_gt_i32_e64 s[30:31], s37, v82
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
	v_cndmask_b32_e64 v67, v159, v67, s[0:1]
	v_cndmask_b32_e32 v66, v159, v66, vcc
	v_cndmask_b32_e64 v69, v159, v69, s[6:7]
	v_cndmask_b32_e64 v68, v159, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_cndmask_b32_e64 v71, v159, v71, s[10:11]
	v_cndmask_b32_e64 v70, v159, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[254:255], v[68:69]
	v_max_f32_e32 v82, v66, v67
	v_cndmask_b32_e64 v73, v159, v73, s[14:15]
	v_cndmask_b32_e64 v72, v159, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[150:151], v[70:71]
	v_max3_f32 v82, v82, v68, v69
	v_cndmask_b32_e64 v75, v159, v75, s[18:19]
	v_cndmask_b32_e64 v74, v159, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[124:125], v[72:73]
	v_max3_f32 v82, v82, v70, v71
	v_cndmask_b32_e64 v77, v159, v77, s[22:23]
	v_cndmask_b32_e64 v76, v159, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[126:127], v[74:75]
	v_max3_f32 v82, v82, v72, v73
	v_cndmask_b32_e64 v79, v159, v79, s[26:27]
	v_cndmask_b32_e64 v78, v159, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[128:129], v[76:77]
	v_max3_f32 v82, v82, v74, v75
	v_cndmask_b32_e64 v81, v159, v81, s[30:31]
	v_cndmask_b32_e64 v80, v159, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[130:131], v[78:79]
	v_max3_f32 v82, v82, v76, v77
	v_pk_mul_f32 v[80:81], v[132:133], v[80:81]
	v_max3_f32 v82, v82, v78, v79
	v_max3_f32 v82, v82, v80, v81
	ds_bpermute_b32 v83, v155, v82
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v83, v83, v83
	v_max_f32_e32 v82, v82, v83
	;;#ASMSTART
	v_add_f32 v83, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v82, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_134
	;;#ASMSTART
	v_add_f32 v134, v82, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v82, v118, v134
	v_exp_f32_e32 v83, v82
	v_mov_b32_e32 v118, v134
	v_mov_b32_e32 v135, v134
	v_mov_b32_e32 v136, v134
	v_mov_b32_e32 v137, v134
	v_mov_b32_e32 v138, v134
	v_mov_b32_e32 v139, v134
	v_mov_b32_e32 v140, v134
	v_mov_b32_e32 v141, v134
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v134
	v_mov_b32_e32 v144, v134
	v_mov_b32_e32 v145, v134
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v134
	v_mov_b32_e32 v148, v134
	v_mov_b32_e32 v149, v134
.LBB0_134:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, 0, v66
	v_add_f32_e32 v82, v82, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v68
	v_add_f32_e32 v82, v82, v69
	v_add_f32_e32 v82, v82, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v71
	v_add_f32_e32 v82, v82, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v73
	v_add_f32_e32 v82, v82, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v75
	v_add_f32_e32 v82, v82, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v77
	v_add_f32_e32 v82, v82, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v83
	v_add_f32_e32 v82, v82, v79
	v_add_f32_e32 v82, v82, v80
	v_add_f32_e32 v82, v82, v81
	;;#ASMSTART
	v_fma_f32 v82, v160, v83, v82
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_136
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
.LBB0_136:
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
	v_perm_b32 v66, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[108:109], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[112:113], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[104:105], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[110:111], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[114:115], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[92:93], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[96:97], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[88:89], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[84:85], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[94:95], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[98:99], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[90:91], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[86:87], v[72:73], v[18:33]
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
	ds_read_b128 v[120:123], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[84:87], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[88:91], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[96:99], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[92:95], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[100:103], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[104:107], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[108:111], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[112:115], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[164:167], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[168:171], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[160:163], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_138
	scratch_load_dwordx4 v[66:69], off, off
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:12288
	scratch_load_dwordx4 v[66:69], off, off offset:16
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[66:69] offset:18432
.LBB0_138:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s45, s83, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s77, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_140
	v_add_u32_e32 v66, s45, v1
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[94:95]
	global_load_dwordx4 v[68:71], v[66:67], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[68:71], off
	global_load_dwordx4 v[66:69], v[66:67], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[66:69], off offset:16
.LBB0_140:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[120:121], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_addk_i32 s46, 0x1000
	v_lshlrev_b32_e32 v120, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	v_and_b32_e32 v120, 0xf8, v120
	v_and_b32_e32 v83, 0x400, v83
	v_or3_b32 v120, v120, v83, s46
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[180:181], v[66:81]
	v_ashrrev_i32_e32 v121, 31, v120
	v_lshl_add_u64 v[120:121], v[120:121], 1, s[80:81]
	global_load_dwordx4 v[246:249], v[120:121], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[84:85], v[182:183], v[66:81]
	v_add_co_u32_e32 v84, vcc, s78, v120
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v121, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[86:87], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[88:89], v[186:187], v[66:81]
	global_load_dwordx4 v[250:253], v[120:121], off offset:512
	global_load_dwordx4 v[238:241], v[120:121], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[90:91], v[188:189], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[192:193], v[66:81]
	global_load_dwordx4 v[242:245], v[120:121], off offset:1536
	global_load_dwordx4 v[230:233], v[84:85], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[92:93], v[194:195], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[94:95], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[198:199], v[66:81]
	global_load_dwordx4 v[234:237], v[84:85], off offset:512
	global_load_dwordx4 v[226:229], v[84:85], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[200:201], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[204:205], v[66:81]
	global_load_dwordx4 v[120:123], v[84:85], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[208:209], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[114:115], v[212:213], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[216:217], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[218:219], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[220:221], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[224:225], v[66:81]
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
	v_add_u32_e32 v84, 0xffffffa9, v83
	v_cmp_gt_i32_e32 vcc, s37, v84
	v_add_u32_e32 v84, 0xffffffaa, v83
	v_cmp_gt_i32_e64 s[0:1], s37, v84
	v_add_u32_e32 v84, 0xffffffab, v83
	v_cmp_gt_i32_e64 s[4:5], s37, v84
	v_add_u32_e32 v84, 0xffffffac, v83
	v_cmp_gt_i32_e64 s[6:7], s37, v84
	v_add_u32_e32 v84, 0xffffffad, v83
	v_cmp_gt_i32_e64 s[8:9], s37, v84
	v_add_u32_e32 v84, 0xffffffae, v83
	v_cmp_gt_i32_e64 s[10:11], s37, v84
	v_add_u32_e32 v84, 0xffffffaf, v83
	v_cmp_gt_i32_e64 s[12:13], s37, v84
	v_add_u32_e32 v84, 0xffffffb0, v83
	v_cmp_gt_i32_e64 s[14:15], s37, v84
	v_add_u32_e32 v84, 0xffffffb9, v83
	v_cmp_gt_i32_e64 s[16:17], s37, v84
	v_add_u32_e32 v84, 0xffffffba, v83
	v_cmp_gt_i32_e64 s[18:19], s37, v84
	v_add_u32_e32 v84, 0xffffffbb, v83
	v_cmp_gt_i32_e64 s[20:21], s37, v84
	v_add_u32_e32 v84, 0xffffffbc, v83
	v_cmp_gt_i32_e64 s[22:23], s37, v84
	v_add_u32_e32 v84, 0xffffffbd, v83
	v_cmp_gt_i32_e64 s[24:25], s37, v84
	v_add_u32_e32 v84, 0xffffffbe, v83
	v_cmp_gt_i32_e64 s[26:27], s37, v84
	v_add_u32_e32 v84, 0xffffffbf, v83
	v_subrev_u32_e32 v83, 64, v83
	v_cmp_gt_i32_e64 s[28:29], s37, v84
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
	v_cndmask_b32_e64 v67, v159, v67, s[0:1]
	v_cndmask_b32_e32 v66, v159, v66, vcc
	v_cndmask_b32_e64 v69, v159, v69, s[6:7]
	v_cndmask_b32_e64 v68, v159, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[116:117], v[66:67]
	v_cndmask_b32_e64 v71, v159, v71, s[10:11]
	v_cndmask_b32_e64 v70, v159, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[254:255], v[68:69]
	v_max_f32_e32 v83, v66, v67
	v_cndmask_b32_e64 v73, v159, v73, s[14:15]
	v_cndmask_b32_e64 v72, v159, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[150:151], v[70:71]
	v_max3_f32 v83, v83, v68, v69
	v_cndmask_b32_e64 v75, v159, v75, s[18:19]
	v_cndmask_b32_e64 v74, v159, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[124:125], v[72:73]
	v_max3_f32 v83, v83, v70, v71
	v_cndmask_b32_e64 v77, v159, v77, s[22:23]
	v_cndmask_b32_e64 v76, v159, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[126:127], v[74:75]
	v_max3_f32 v83, v83, v72, v73
	v_cndmask_b32_e64 v79, v159, v79, s[26:27]
	v_cndmask_b32_e64 v78, v159, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[128:129], v[76:77]
	v_max3_f32 v83, v83, v74, v75
	v_cndmask_b32_e64 v81, v159, v81, s[30:31]
	v_cndmask_b32_e64 v80, v159, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[130:131], v[78:79]
	v_max3_f32 v83, v83, v76, v77
	v_pk_mul_f32 v[80:81], v[132:133], v[80:81]
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v84, v155, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v84, v84, v84
	v_max_f32_e32 v83, v83, v84
	;;#ASMSTART
	v_add_f32 v84, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v83, v84
	v_mov_b32_e32 v84, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_142
	;;#ASMSTART
	v_add_f32 v134, v83, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v118, v134
	v_exp_f32_e32 v84, v83
	v_mov_b32_e32 v118, v134
	v_mov_b32_e32 v135, v134
	v_mov_b32_e32 v136, v134
	v_mov_b32_e32 v137, v134
	v_mov_b32_e32 v138, v134
	v_mov_b32_e32 v139, v134
	v_mov_b32_e32 v140, v134
	v_mov_b32_e32 v141, v134
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v134
	v_mov_b32_e32 v144, v134
	v_mov_b32_e32 v145, v134
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v134
	v_mov_b32_e32 v148, v134
	v_mov_b32_e32 v149, v134
.LBB0_142:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v161, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v162, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, 0, v161
	v_add_f32_e32 v66, v66, v162
	;;#ASMSTART
	v_exp_f32 v163, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v164, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v165, v70
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v166, v71
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, v66, v163
	v_add_f32_e32 v66, v66, v164
	v_add_f32_e32 v66, v66, v165
	v_add_f32_e32 v66, v66, v166
	;;#ASMSTART
	v_exp_f32 v167, v72
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v168, v73
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, v66, v167
	v_add_f32_e32 v66, v66, v168
	;;#ASMSTART
	v_exp_f32 v169, v74
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v170, v75
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, v66, v169
	v_add_f32_e32 v66, v66, v170
	;;#ASMSTART
	v_exp_f32 v171, v76
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v172, v77
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, v66, v171
	v_add_f32_e32 v66, v66, v172
	;;#ASMSTART
	v_exp_f32 v173, v78
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v174, v79
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, v66, v173
	v_add_f32_e32 v66, v66, v174
	;;#ASMSTART
	v_exp_f32 v175, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v176, v81
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v84
	v_add_f32_e32 v66, v66, v175
	v_add_f32_e32 v66, v66, v176
	;;#ASMSTART
	v_fma_f32 v160, v82, v84, v66
	;;#ASMEND
	v_mov_b32_e32 v66, v48
	v_mov_b32_e32 v67, v49
	v_mov_b32_e32 v68, v50
	v_mov_b32_e32 v69, v51
	v_mov_b32_e32 v70, v52
	v_mov_b32_e32 v71, v53
	v_mov_b32_e32 v72, v54
	v_mov_b32_e32 v73, v55
	v_mov_b32_e32 v74, v56
	v_mov_b32_e32 v75, v57
	v_mov_b32_e32 v76, v58
	v_mov_b32_e32 v77, v59
	v_mov_b32_e32 v78, v60
	v_mov_b32_e32 v79, v61
	v_mov_b32_e32 v80, v62
	v_mov_b32_e32 v81, v63
	v_mov_b32_e32 v82, v64
	v_mov_b32_e32 v83, v65
	v_mov_b32_e32 v98, v16
	v_mov_b32_e32 v99, v17
	v_mov_b32_e32 v100, v18
	v_mov_b32_e32 v101, v19
	v_mov_b32_e32 v102, v20
	v_mov_b32_e32 v103, v21
	v_mov_b32_e32 v104, v22
	v_mov_b32_e32 v105, v23
	v_mov_b32_e32 v106, v24
	v_mov_b32_e32 v107, v25
	v_mov_b32_e32 v108, v26
	v_mov_b32_e32 v109, v27
	v_mov_b32_e32 v110, v28
	v_mov_b32_e32 v111, v29
	v_mov_b32_e32 v112, v30
	v_mov_b32_e32 v113, v31
	v_mov_b32_e32 v114, v32
	v_mov_b32_e32 v115, v33
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_144
	;;#ASMSTART
	v_mul_f32 v34, v34, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v48, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v49, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v50, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v51, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v52, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v53, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v54, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v55, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v56, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v57, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v58, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v59, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v60, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v61, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v62, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v63, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v64, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v65, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v98, v16, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v99, v17, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v100, v18, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v101, v19, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v102, v20, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v103, v21, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v104, v22, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v105, v23, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v106, v24, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v107, v25, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v108, v26, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v109, v27, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v110, v28, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v111, v29, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v112, v30, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v113, v31, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v114, v32, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v115, v33, v84
	;;#ASMEND
.LBB0_144:
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v97, v15
	v_mov_b32_e32 v96, v14
	v_mov_b32_e32 v95, v13
	v_mov_b32_e32 v94, v12
	v_mov_b32_e32 v93, v11
	v_mov_b32_e32 v92, v10
	v_mov_b32_e32 v91, v9
	v_mov_b32_e32 v90, v8
	v_mov_b32_e32 v89, v7
	v_mov_b32_e32 v88, v6
	v_mov_b32_e32 v87, v5
	v_mov_b32_e32 v86, v4
	v_mov_b32_e32 v85, v3
	v_mov_b32_e32 v84, v2
	v_mov_b32_e32 v65, v47
	v_mov_b32_e32 v64, v46
	v_mov_b32_e32 v63, v45
	v_mov_b32_e32 v62, v44
	v_mov_b32_e32 v61, v43
	v_mov_b32_e32 v60, v42
	v_mov_b32_e32 v59, v41
	v_mov_b32_e32 v58, v40
	v_mov_b32_e32 v57, v39
	v_mov_b32_e32 v56, v38
	v_mov_b32_e32 v55, v37
	v_mov_b32_e32 v54, v36
	v_mov_b32_e32 v53, v35
	v_mov_b32_e32 v52, v34
	v_add_u32_e32 v9, 0x8000, v176
	v_add_u32_e32 v8, 0x8000, v174
	v_add_u32_e32 v7, 0x8000, v172
	v_add_u32_e32 v6, 0x8000, v170
	v_add_u32_e32 v5, 0x8000, v168
	v_add_u32_e32 v4, 0x8000, v166
	v_add_u32_e32 v3, 0x8000, v164
	v_add_u32_e32 v2, 0x8000, v162
	v_add_u32_e32 v10, 0x8000, v175
	v_add_u32_e32 v11, 0x8000, v173
	v_add_u32_e32 v12, 0x8000, v171
	v_add_u32_e32 v13, 0x8000, v169
	v_add_u32_e32 v14, 0x8000, v167
	v_add_u32_e32 v15, 0x8000, v165
	v_add_u32_e32 v16, 0x8000, v163
	v_add_u32_e32 v17, 0x8000, v161
	;;#ASMSTART
	v_perm_b32 v2, v2, v17, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v3, v16, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v4, v15, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v5, v14, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v13, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v7, v12, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v8, v11, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v9, v10, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[52:67], v[246:247], v[2:3], v[52:67]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[68:83], v[250:251], v[2:3], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[238:239], v[2:3], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[242:243], v[2:3], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[248:249], v[4:5], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[252:253], v[4:5], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[240:241], v[4:5], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[244:245], v[4:5], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[230:231], v[6:7], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[234:235], v[6:7], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[226:227], v[6:7], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[120:121], v[6:7], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[232:233], v[8:9], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[236:237], v[8:9], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[228:229], v[8:9], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[122:123], v[8:9], v[100:115]
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
	ds_read_b128 v[36:39], v3 offset:12288
	v_add_u32_e32 v3, 0x1810, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[18:21], v3
	v_add_u32_e32 v3, 0x1820, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[22:25], v3
	v_add_u32_e32 v3, 0x1830, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[26:29], v3
	v_add_u32_e32 v3, 0x1840, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[32:35], v3
	v_add_u32_e32 v3, 0x1850, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[120:123], v3
	v_add_u32_e32 v3, 0x1860, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[162:165], v3
	v_add_u32_e32 v3, 0x1870, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[166:169], v3
	v_add_u32_e32 v3, 0x1880, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[170:173], v3
	v_add_u32_e32 v3, 0x1890, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[226:229], v3
	v_add_u32_e32 v3, 0x18a0, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_u32_e32 v2, 0x18b0, v2
	ds_read_b128 v[230:233], v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[174:177], v2
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v2, 0x180, v2
	v_cmp_ne_u32_e32 vcc, s77, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_146
	scratch_load_dwordx4 v[2:5], off, off offset:32
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[2:5]
	scratch_load_dwordx4 v[2:5], off, off offset:48
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[2:5] offset:6144
.LBB0_146:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v2, 0x180, v2
	v_cmp_ne_u32_e32 vcc, s77, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_148
	v_add_u32_e32 v2, s45, v119
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[94:95]
	global_load_dwordx4 v[4:7], v[2:3], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[4:7], off offset:32
	global_load_dwordx4 v[2:5], v[2:3], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:48
.LBB0_148:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[36:37], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v30, v0
	;;#ASMEND
	s_add_i32 s0, s46, 0x1000
	v_lshlrev_b32_e32 v31, 3, v30
	v_lshlrev_b32_e32 v30, 5, v30
	v_and_b32_e32 v31, 0xf8, v31
	v_and_b32_e32 v30, 0x400, v30
	v_or3_b32 v30, v31, v30, s0
	v_mfma_f32_32x32x8_bf16 v[2:17], v[38:39], v[180:181], v[2:17]
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshl_add_u64 v[30:31], v[30:31], 1, s[80:81]
	global_load_dwordx4 v[44:47], v[30:31], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[18:19], v[182:183], v[2:17]
	v_add_co_u32_e32 v18, vcc, s78, v30
	s_nop 1
	v_addc_co_u32_e32 v19, vcc, 0, v31, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[20:21], v[184:185], v[2:17]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[22:23], v[186:187], v[2:17]
	global_load_dwordx4 v[48:51], v[30:31], off offset:512
	global_load_dwordx4 v[36:39], v[30:31], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[2:17], v[24:25], v[188:189], v[2:17]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[26:27], v[190:191], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[28:29], v[192:193], v[2:17]
	global_load_dwordx4 v[40:43], v[30:31], off offset:1536
	s_nop 0
	global_load_dwordx4 v[28:31], v[18:19], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[32:33], v[194:195], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[34:35], v[196:197], v[2:17]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[120:121], v[198:199], v[2:17]
	global_load_dwordx4 v[32:35], v[18:19], off offset:512
	global_load_dwordx4 v[24:27], v[18:19], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[2:17], v[122:123], v[200:201], v[2:17]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[162:163], v[202:203], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[164:165], v[204:205], v[2:17]
	global_load_dwordx4 v[20:23], v[18:19], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[166:167], v[206:207], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[168:169], v[208:209], v[2:17]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[170:171], v[210:211], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[172:173], v[212:213], v[2:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[226:227], v[214:215], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[228:229], v[216:217], v[2:17]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[230:231], v[218:219], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[232:233], v[220:221], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[174:175], v[222:223], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[176:177], v[224:225], v[2:17]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v18, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v18, 2, v18
	v_and_b32_e32 v18, 8, v18
	v_add_u32_e32 v18, s40, v18
	v_subrev_u32_e32 v19, 55, v18
	v_cmp_gt_i32_e32 vcc, s37, v19
	v_subrev_u32_e32 v19, 54, v18
	v_cmp_gt_i32_e64 s[0:1], s37, v19
	v_subrev_u32_e32 v19, 53, v18
	v_cmp_gt_i32_e64 s[4:5], s37, v19
	v_subrev_u32_e32 v19, 52, v18
	v_cmp_gt_i32_e64 s[6:7], s37, v19
	v_subrev_u32_e32 v19, 51, v18
	v_cmp_gt_i32_e64 s[8:9], s37, v19
	v_subrev_u32_e32 v19, 50, v18
	v_cmp_gt_i32_e64 s[10:11], s37, v19
	v_subrev_u32_e32 v19, 49, v18
	v_cmp_gt_i32_e64 s[12:13], s37, v19
	v_subrev_u32_e32 v19, 48, v18
	v_cmp_gt_i32_e64 s[14:15], s37, v19
	v_subrev_u32_e32 v19, 39, v18
	v_cmp_gt_i32_e64 s[16:17], s37, v19
	v_subrev_u32_e32 v19, 38, v18
	v_cmp_gt_i32_e64 s[18:19], s37, v19
	v_subrev_u32_e32 v19, 37, v18
	v_cmp_gt_i32_e64 s[20:21], s37, v19
	v_subrev_u32_e32 v19, 36, v18
	v_cmp_gt_i32_e64 s[22:23], s37, v19
	v_subrev_u32_e32 v19, 35, v18
	v_cmp_gt_i32_e64 s[24:25], s37, v19
	v_subrev_u32_e32 v19, 34, v18
	v_cmp_gt_i32_e64 s[26:27], s37, v19
	v_subrev_u32_e32 v19, 33, v18
	v_subrev_u32_e32 v18, 32, v18
	v_cmp_gt_i32_e64 s[28:29], s37, v19
	v_cmp_gt_i32_e64 s[30:31], s37, v18
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
	v_cndmask_b32_e64 v3, v159, v3, s[0:1]
	v_cndmask_b32_e32 v2, v159, v2, vcc
	v_cndmask_b32_e64 v5, v159, v5, s[6:7]
	v_cndmask_b32_e64 v4, v159, v4, s[4:5]
	v_pk_mul_f32 v[2:3], v[116:117], v[2:3]
	v_cndmask_b32_e64 v7, v159, v7, s[10:11]
	v_cndmask_b32_e64 v6, v159, v6, s[8:9]
	v_pk_mul_f32 v[4:5], v[254:255], v[4:5]
	v_max_f32_e32 v18, v2, v3
	v_cndmask_b32_e64 v9, v159, v9, s[14:15]
	v_cndmask_b32_e64 v8, v159, v8, s[12:13]
	v_pk_mul_f32 v[6:7], v[150:151], v[6:7]
	v_max3_f32 v18, v18, v4, v5
	v_cndmask_b32_e64 v11, v159, v11, s[18:19]
	v_cndmask_b32_e64 v10, v159, v10, s[16:17]
	v_pk_mul_f32 v[8:9], v[124:125], v[8:9]
	v_max3_f32 v18, v18, v6, v7
	v_cndmask_b32_e64 v13, v159, v13, s[22:23]
	v_cndmask_b32_e64 v12, v159, v12, s[20:21]
	v_pk_mul_f32 v[10:11], v[126:127], v[10:11]
	v_max3_f32 v18, v18, v8, v9
	v_cndmask_b32_e64 v15, v159, v15, s[26:27]
	v_cndmask_b32_e64 v14, v159, v14, s[24:25]
	v_pk_mul_f32 v[12:13], v[128:129], v[12:13]
	v_max3_f32 v18, v18, v10, v11
	v_cndmask_b32_e64 v17, v159, v17, s[30:31]
	v_cndmask_b32_e64 v16, v159, v16, s[28:29]
	v_pk_mul_f32 v[14:15], v[130:131], v[14:15]
	v_max3_f32 v18, v18, v12, v13
	v_pk_mul_f32 v[16:17], v[132:133], v[16:17]
	v_max3_f32 v18, v18, v14, v15
	v_max3_f32 v18, v18, v16, v17
	ds_bpermute_b32 v19, v155, v18
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v19, v19, v19
	v_max_f32_e32 v18, v18, v19
	;;#ASMSTART
	v_add_f32 v19, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v18, v19
	v_mov_b32_e32 v19, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_150
	;;#ASMSTART
	v_add_f32 v134, v18, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v18, v118, v134
	v_exp_f32_e32 v19, v18
	v_mov_b32_e32 v118, v134
	v_mov_b32_e32 v135, v134
	v_mov_b32_e32 v136, v134
	v_mov_b32_e32 v137, v134
	v_mov_b32_e32 v138, v134
	v_mov_b32_e32 v139, v134
	v_mov_b32_e32 v140, v134
	v_mov_b32_e32 v141, v134
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v134
	v_mov_b32_e32 v144, v134
	v_mov_b32_e32 v145, v134
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v134
	v_mov_b32_e32 v148, v134
	v_mov_b32_e32 v149, v134
.LBB0_150:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[2:3], v[2:3], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v2, v2
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v3, v3
	;;#ASMEND
	v_pk_add_f32 v[6:7], v[6:7], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v18, 0, v2
	v_add_f32_e32 v18, v18, v3
	;;#ASMSTART
	v_exp_f32 v4, v4
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v5, v5
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v6, v6
	;;#ASMEND
	v_pk_add_f32 v[8:9], v[8:9], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v18, v18, v4
	v_add_f32_e32 v18, v18, v5
	v_add_f32_e32 v18, v18, v6
	;;#ASMSTART
	v_exp_f32 v7, v7
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v8, v8
	;;#ASMEND
	v_pk_add_f32 v[10:11], v[10:11], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v18, v18, v7
	v_add_f32_e32 v18, v18, v8
	;;#ASMSTART
	v_exp_f32 v9, v9
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v10, v10
	;;#ASMEND
	v_pk_add_f32 v[12:13], v[12:13], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v18, v18, v9
	v_add_f32_e32 v18, v18, v10
	;;#ASMSTART
	v_exp_f32 v11, v11
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v12, v12
	;;#ASMEND
	v_pk_add_f32 v[14:15], v[14:15], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v18, v18, v11
	v_add_f32_e32 v18, v18, v12
	;;#ASMSTART
	v_exp_f32 v13, v13
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v14, v14
	;;#ASMEND
	v_pk_add_f32 v[16:17], v[16:17], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v18, v18, v13
	v_add_f32_e32 v18, v18, v14
	;;#ASMSTART
	v_exp_f32 v15, v15
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v16, v16
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v17, v17
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v19
	v_add_f32_e32 v18, v18, v15
	v_add_f32_e32 v18, v18, v16
	v_add_f32_e32 v18, v18, v17
	;;#ASMSTART
	v_fma_f32 v18, v160, v19, v18
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_152
	;;#ASMSTART
	v_mul_f32 v52, v52, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v98, v98, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v99, v99, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v100, v100, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v101, v101, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v102, v102, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v103, v103, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v104, v104, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v105, v105, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v106, v106, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v107, v107, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v108, v108, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v109, v109, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v110, v110, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v111, v111, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v112, v112, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v113, v113, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v114, v114, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v115, v115, v19
	;;#ASMEND
.LBB0_152:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v9, 0x8000, v9
	v_add_u32_e32 v8, 0x8000, v8
	v_add_u32_e32 v7, 0x8000, v7
	v_add_u32_e32 v6, 0x8000, v6
	v_add_u32_e32 v5, 0x8000, v5
	v_add_u32_e32 v4, 0x8000, v4
	v_add_u32_e32 v3, 0x8000, v3
	v_add_u32_e32 v2, 0x8000, v2
	v_add_u32_e32 v17, 0x8000, v17
	v_add_u32_e32 v16, 0x8000, v16
	v_add_u32_e32 v15, 0x8000, v15
	v_add_u32_e32 v14, 0x8000, v14
	v_add_u32_e32 v13, 0x8000, v13
	v_add_u32_e32 v12, 0x8000, v12
	v_add_u32_e32 v11, 0x8000, v11
	v_add_u32_e32 v10, 0x8000, v10
	;;#ASMSTART
	v_perm_b32 v2, v3, v2, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v5, v4, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v7, v6, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v9, v8, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v11, v10, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v13, v12, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v15, v14, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v17, v16, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[52:67], v[44:45], v[2:3], v[52:67]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[68:83], v[48:49], v[2:3], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[36:37], v[2:3], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[40:41], v[2:3], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[46:47], v[4:5], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[50:51], v[4:5], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[38:39], v[4:5], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[42:43], v[4:5], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[28:29], v[6:7], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[32:33], v[6:7], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[24:25], v[6:7], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[20:21], v[6:7], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[30:31], v[8:9], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[34:35], v[8:9], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[26:27], v[8:9], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[22:23], v[8:9], v[100:115]
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
	v_xor_b32_e32 v4, v3, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[38:41], v4
	v_or_b32_e32 v4, 16, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[20:23], v4
	v_or_b32_e32 v4, 32, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[24:27], v4
	v_or_b32_e32 v4, 48, v2
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[28:31], v3
	v_add_u32_e32 v3, 64, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[34:37], v3
	v_add_u32_e32 v3, 0x50, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[160:163], v3
	v_add_u32_e32 v3, 0x60, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[164:167], v3
	v_add_u32_e32 v3, 0x70, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[168:171], v3
	v_add_u32_e32 v3, 0x80, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[172:175], v3
	v_add_u32_e32 v3, 0x90, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[230:233], v3
	v_add_u32_e32 v3, 0xa0, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_u32_e32 v2, 0xb0, v2
	ds_read_b128 v[234:237], v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[226:229], v2
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v2, 0x180, v2
	v_cmp_ne_u32_e32 vcc, s77, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_154
	scratch_load_dwordx4 v[2:5], off, off
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[2:5] offset:12288
	scratch_load_dwordx4 v[2:5], off, off offset:16
	s_waitcnt vmcnt(0)
	ds_write_b128 v152, v[2:5] offset:18432
.LBB0_154:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v2, 0x180, v2
	v_cmp_ne_u32_e32 vcc, s77, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_156
	v_add_u32_e32 v2, s45, v153
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[94:95]
	global_load_dwordx4 v[4:7], v[2:3], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[4:7], off
	global_load_dwordx4 v[2:5], v[2:3], off offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:16
.LBB0_156:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[38:39], v[178:179], 0
	;;#ASMSTART
	v_mov_b32 v19, v0
	;;#ASMEND
	s_addk_i32 s46, 0x2000
	v_lshlrev_b32_e32 v32, 3, v19
	v_lshlrev_b32_e32 v19, 5, v19
	v_and_b32_e32 v32, 0xf8, v32
	v_and_b32_e32 v19, 0x400, v19
	v_or3_b32 v32, v32, v19, s46
	v_mfma_f32_32x32x8_bf16 v[2:17], v[40:41], v[180:181], v[2:17]
	v_ashrrev_i32_e32 v33, 31, v32
	v_lshl_add_u64 v[32:33], v[32:33], 1, s[80:81]
	global_load_dwordx4 v[46:49], v[32:33], off
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[20:21], v[182:183], v[2:17]
	v_add_co_u32_e32 v20, vcc, s78, v32
	s_nop 1
	v_addc_co_u32_e32 v21, vcc, 0, v33, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[22:23], v[184:185], v[2:17]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[24:25], v[186:187], v[2:17]
	global_load_dwordx4 v[120:123], v[32:33], off offset:512
	global_load_dwordx4 v[38:41], v[32:33], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[2:17], v[26:27], v[188:189], v[2:17]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[28:29], v[190:191], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[30:31], v[192:193], v[2:17]
	global_load_dwordx4 v[42:45], v[32:33], off offset:1536
	s_nop 0
	global_load_dwordx4 v[30:33], v[20:21], off
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[34:35], v[194:195], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[36:37], v[196:197], v[2:17]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[160:161], v[198:199], v[2:17]
	global_load_dwordx4 v[34:37], v[20:21], off offset:512
	global_load_dwordx4 v[26:29], v[20:21], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[2:17], v[162:163], v[200:201], v[2:17]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[164:165], v[202:203], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[166:167], v[204:205], v[2:17]
	global_load_dwordx4 v[22:25], v[20:21], off offset:1536
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[168:169], v[206:207], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[170:171], v[208:209], v[2:17]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[172:173], v[210:211], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[174:175], v[212:213], v[2:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[230:231], v[214:215], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[232:233], v[216:217], v[2:17]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[234:235], v[218:219], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[236:237], v[220:221], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[226:227], v[222:223], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[228:229], v[224:225], v[2:17]
	s_waitcnt vmcnt(10) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v19, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v19, 2, v19
	v_and_b32_e32 v19, 8, v19
	v_add_u32_e32 v19, s40, v19
	v_subrev_u32_e32 v20, 23, v19
	v_cmp_gt_i32_e32 vcc, s37, v20
	v_subrev_u32_e32 v20, 22, v19
	v_cmp_gt_i32_e64 s[0:1], s37, v20
	v_subrev_u32_e32 v20, 21, v19
	v_cmp_gt_i32_e64 s[4:5], s37, v20
	v_subrev_u32_e32 v20, 20, v19
	v_cmp_gt_i32_e64 s[6:7], s37, v20
	v_subrev_u32_e32 v20, 19, v19
	v_cmp_gt_i32_e64 s[8:9], s37, v20
	v_subrev_u32_e32 v20, 18, v19
	v_cmp_gt_i32_e64 s[10:11], s37, v20
	v_subrev_u32_e32 v20, 17, v19
	v_cmp_gt_i32_e64 s[12:13], s37, v20
	v_add_u32_e32 v20, -16, v19
	v_cmp_gt_i32_e64 s[14:15], s37, v20
	v_add_u32_e32 v20, -7, v19
	v_cmp_gt_i32_e64 s[16:17], s37, v20
	v_add_u32_e32 v20, -6, v19
	v_cmp_gt_i32_e64 s[18:19], s37, v20
	v_add_u32_e32 v20, -5, v19
	v_cmp_gt_i32_e64 s[20:21], s37, v20
	v_add_u32_e32 v20, -4, v19
	v_cmp_gt_i32_e64 s[22:23], s37, v20
	v_add_u32_e32 v20, -3, v19
	v_cmp_gt_i32_e64 s[24:25], s37, v20
	v_add_u32_e32 v20, -2, v19
	v_cmp_gt_i32_e64 s[26:27], s37, v20
	v_add_u32_e32 v20, -1, v19
	v_cmp_gt_i32_e64 s[28:29], s37, v20
	v_cmp_gt_i32_e64 s[30:31], s37, v19
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
	v_cndmask_b32_e64 v3, v159, v3, s[0:1]
	v_cndmask_b32_e32 v2, v159, v2, vcc
	v_cndmask_b32_e64 v5, v159, v5, s[6:7]
	v_cndmask_b32_e64 v4, v159, v4, s[4:5]
	v_pk_mul_f32 v[2:3], v[116:117], v[2:3]
	v_cndmask_b32_e64 v7, v159, v7, s[10:11]
	v_cndmask_b32_e64 v6, v159, v6, s[8:9]
	v_pk_mul_f32 v[4:5], v[254:255], v[4:5]
	v_max_f32_e32 v19, v2, v3
	v_cndmask_b32_e64 v9, v159, v9, s[14:15]
	v_cndmask_b32_e64 v8, v159, v8, s[12:13]
	v_pk_mul_f32 v[6:7], v[150:151], v[6:7]
	v_max3_f32 v19, v19, v4, v5
	v_cndmask_b32_e64 v11, v159, v11, s[18:19]
	v_cndmask_b32_e64 v10, v159, v10, s[16:17]
	v_pk_mul_f32 v[8:9], v[124:125], v[8:9]
	v_max3_f32 v19, v19, v6, v7
	v_cndmask_b32_e64 v13, v159, v13, s[22:23]
	v_cndmask_b32_e64 v12, v159, v12, s[20:21]
	v_pk_mul_f32 v[10:11], v[126:127], v[10:11]
	v_max3_f32 v19, v19, v8, v9
	v_cndmask_b32_e64 v15, v159, v15, s[26:27]
	v_cndmask_b32_e64 v14, v159, v14, s[24:25]
	v_pk_mul_f32 v[12:13], v[128:129], v[12:13]
	v_max3_f32 v19, v19, v10, v11
	v_cndmask_b32_e64 v17, v159, v17, s[30:31]
	v_cndmask_b32_e64 v16, v159, v16, s[28:29]
	v_pk_mul_f32 v[14:15], v[130:131], v[14:15]
	v_max3_f32 v19, v19, v12, v13
	v_pk_mul_f32 v[16:17], v[132:133], v[16:17]
	v_max3_f32 v19, v19, v14, v15
	v_max3_f32 v19, v19, v16, v17
	ds_bpermute_b32 v20, v155, v19
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v20, v20, v20
	v_max_f32_e32 v20, v19, v20
	;;#ASMSTART
	v_add_f32 v19, v118, v157
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v20, v19
	v_mov_b32_e32 v19, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_158
	;;#ASMSTART
	v_add_f32 v134, v20, v158
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v19, v118, v134
	v_exp_f32_e32 v19, v19
	v_mov_b32_e32 v118, v134
	v_mov_b32_e32 v135, v134
	v_mov_b32_e32 v136, v134
	v_mov_b32_e32 v137, v134
	v_mov_b32_e32 v138, v134
	v_mov_b32_e32 v139, v134
	v_mov_b32_e32 v140, v134
	v_mov_b32_e32 v141, v134
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v134
	v_mov_b32_e32 v144, v134
	v_mov_b32_e32 v145, v134
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v134
	v_mov_b32_e32 v148, v134
	v_mov_b32_e32 v149, v134
.LBB0_158:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[2:3], v[2:3], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v2, v2
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v3, v3
	;;#ASMEND
	v_pk_add_f32 v[6:7], v[6:7], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, 0, v2
	v_add_f32_e32 v20, v20, v3
	;;#ASMSTART
	v_exp_f32 v4, v4
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v5, v5
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v6, v6
	;;#ASMEND
	v_pk_add_f32 v[8:9], v[8:9], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, v20, v4
	v_add_f32_e32 v20, v20, v5
	v_add_f32_e32 v20, v20, v6
	;;#ASMSTART
	v_exp_f32 v7, v7
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v8, v8
	;;#ASMEND
	v_pk_add_f32 v[10:11], v[10:11], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, v20, v7
	v_add_f32_e32 v20, v20, v8
	;;#ASMSTART
	v_exp_f32 v9, v9
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v10, v10
	;;#ASMEND
	v_pk_add_f32 v[12:13], v[12:13], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, v20, v9
	v_add_f32_e32 v20, v20, v10
	;;#ASMSTART
	v_exp_f32 v11, v11
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v12, v12
	;;#ASMEND
	v_pk_add_f32 v[14:15], v[14:15], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, v20, v11
	v_add_f32_e32 v20, v20, v12
	;;#ASMSTART
	v_exp_f32 v13, v13
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v14, v14
	;;#ASMEND
	v_pk_add_f32 v[16:17], v[16:17], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, v20, v13
	v_add_f32_e32 v20, v20, v14
	;;#ASMSTART
	v_exp_f32 v15, v15
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v16, v16
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v19
	v_add_f32_e32 v20, v20, v15
	v_add_f32_e32 v20, v20, v16
	;;#ASMSTART
	v_exp_f32 v17, v17
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v20, v20, v17
	;;#ASMSTART
	v_fma_f32 v160, v18, v19, v20
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_93
	;;#ASMSTART
	v_mul_f32 v52, v52, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v98, v98, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v99, v99, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v100, v100, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v101, v101, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v102, v102, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v103, v103, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v104, v104, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v105, v105, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v106, v106, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v107, v107, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v108, v108, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v109, v109, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v110, v110, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v111, v111, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v112, v112, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v113, v113, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v114, v114, v19
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v115, v115, v19
	;;#ASMEND
	s_branch .LBB0_93
.LBB0_160:
	s_mov_b32 s44, s43
	s_branch .LBB0_94
.LBB0_161:
	scratch_load_dword v254, off, off offset:64
	scratch_load_dword v255, off, off offset:68
.LBB0_162:
	ds_bpermute_b32 v66, v155, v160
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v66, v160, v66
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
	v_pk_mul_f32 v[34:35], v[34:35], v[66:67] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[24:25], v[24:25], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[66:67] op_sel_hi:[1,0]
	v_add_u32_e32 v68, 0x8000, v31
	v_add_u32_e32 v66, 0x8000, v33
	v_add_u32_e32 v67, 0x8000, v32
	v_add_u32_e32 v69, 0x8000, v30
	v_add_u32_e32 v70, 0x8000, v29
	v_add_u32_e32 v71, 0x8000, v28
	v_add_u32_e32 v72, 0x8000, v27
	v_add_u32_e32 v73, 0x8000, v26
	v_add_u32_e32 v74, 0x8000, v25
	v_add_u32_e32 v75, 0x8000, v24
	v_add_u32_e32 v76, 0x8000, v23
	v_add_u32_e32 v77, 0x8000, v22
	v_add_u32_e32 v78, 0x8000, v21
	v_add_u32_e32 v79, 0x8000, v20
	v_add_u32_e32 v80, 0x8000, v19
	v_add_u32_e32 v81, 0x8000, v18
	v_add_u32_e32 v82, 0x8000, v17
	v_add_u32_e32 v83, 0x8000, v16
	v_add_u32_e32 v84, 0x8000, v15
	v_add_u32_e32 v85, 0x8000, v14
	v_add_u32_e32 v13, 0x8000, v13
	v_add_u32_e32 v86, 0x8000, v12
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
	v_add_u32_e32 v12, 0x8000, v65
	v_add_u32_e32 v14, 0x8000, v64
	v_add_u32_e32 v15, 0x8000, v63
	v_add_u32_e32 v16, 0x8000, v62
	v_add_u32_e32 v17, 0x8000, v61
	v_add_u32_e32 v18, 0x8000, v60
	v_add_u32_e32 v19, 0x8000, v59
	v_add_u32_e32 v20, 0x8000, v58
	v_add_u32_e32 v21, 0x8000, v57
	v_add_u32_e32 v23, 0x8000, v56
	v_add_u32_e32 v22, 0x8000, v55
	v_add_u32_e32 v25, 0x8000, v53
	v_add_u32_e32 v24, 0x8000, v51
	v_add_u32_e32 v27, 0x8000, v49
	v_add_u32_e32 v26, 0x8000, v47
	v_add_u32_e32 v31, 0x8000, v45
	v_add_u32_e32 v30, 0x8000, v43
	v_add_u32_e32 v33, 0x8000, v41
	v_add_u32_e32 v32, 0x8000, v39
	v_add_u32_e32 v29, 0x8000, v37
	v_add_u32_e32 v28, 0x8000, v35
	v_add_u32_e32 v34, 0x8000, v34
	v_add_u32_e32 v54, 0x8000, v54
	v_add_u32_e32 v52, 0x8000, v52
	v_add_u32_e32 v50, 0x8000, v50
	v_add_u32_e32 v48, 0x8000, v48
	v_add_u32_e32 v46, 0x8000, v46
	v_add_u32_e32 v44, 0x8000, v44
	v_add_u32_e32 v42, 0x8000, v42
	v_add_u32_e32 v40, 0x8000, v40
	v_add_u32_e32 v38, 0x8000, v38
	v_add_u32_e32 v36, 0x8000, v36
	;;#ASMSTART
	v_perm_b32 v28, v28, v34, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v29, v36, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v32, v38, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v33, v40, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v30, v42, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v31, v44, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v26, v46, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v27, v48, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v24, v50, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v25, v52, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v22, v54, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v21, v23, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v19, v20, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v17, v18, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v15, v16, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v12, v14, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v3, v2, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v5, v4, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v7, v6, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v9, v8, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v11, v10, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v13, v86, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v84, v85, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v82, v83, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v80, v81, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v78, v79, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v76, v77, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v74, v75, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v72, v73, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v70, v71, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v68, v69, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v66, v67, s79
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v34, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v34, 0x100, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_164
	s_barrier
.LBB0_164:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v37, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v34, 3, v37
	v_bfe_u32 v36, v37, 6, 3
	v_lshlrev_b32_e32 v35, 7, v37
	v_and_b32_e32 v34, 4, v34
	v_and_or_b32 v34, v35, s47, v34
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_166
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
.LBB0_166:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v39, 0x1ff, v37
	s_lshl_b32 s0, s82, 11
	v_lshlrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v37, 56, v37
	s_ashr_i32 s1, s0, 31
	v_xor_b32_e32 v37, v40, v37
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v37, 1, v37
	s_add_u32 s68, s88, s0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v39, 7, v39
	ds_read_b128 v[42:45], v37
	s_addc_u32 s0, s89, s1
	s_lshl_b32 s4, s76, 7
	v_and_b32_e32 v39, 0xf800, v39
	v_and_b32_e32 v38, 0x78, v40
	v_add_u32_e32 v40, s4, v39
	v_or_b32_e32 v40, v40, v38
	s_lshl_b32 s70, s75, 12
	s_and_b32 s69, s0, 0xffff
	v_lshlrev_b32_e32 v40, 1, v40
	v_cmp_eq_u32_e32 vcc, 1, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[42:45], v40, s[68:71], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_168
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
.LBB0_168:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s4, s4, 0x10000
	v_or_b32_e32 v38, v38, v39
	v_add_lshl_u32 v39, s4, v38, 1
	v_cmp_eq_u32_e32 vcc, 2, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[68:71], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_170
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
.LBB0_170:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x10000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 3, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[68:71], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_172
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
.LBB0_172:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x20000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 4, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[68:71], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_174
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
.LBB0_174:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x30000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 5, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[68:71], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_176
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
.LBB0_176:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x40000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 6, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[68:71], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_178
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
.LBB0_178:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x50000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 7, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[68:71], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_180
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
.LBB0_180:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[4:7], v37
	s_add_i32 s4, s4, 0x60000
	v_add_lshl_u32 v2, s4, v38, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v2, s[68:71], 0 offen
	s_barrier
	;;#ASMSTART
	s_mov_b32 s0, s58
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_184
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_183
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v3, s8
	global_atomic_add v3, v156, v3, s[90:91] sc0
.LBB0_183:
	s_or_b64 exec, exec, s[6:7]
	s_lshl_b64 s[6:7], s[0:1], 2
	s_add_u32 s6, s90, s6
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v3
	s_addc_u32 s7, s91, s7
	s_nop 0
	v_add_u32_e32 v2, s8, v2
	global_store_dword v156, v2, s[6:7]
	s_waitcnt vmcnt(0)
.LBB0_184:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s90, s0
	s_addc_u32 s1, s91, s1
	s_barrier
	global_load_dword v2, v156, s[0:1]
	s_sub_i32 s0, s33, s2
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s2, v2
	s_add_i32 s33, s0, s2
	s_cmp_ge_i32 s74, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s98
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_187
	s_branch .LBB0_12
.LBB0_185:
	s_mov_b32 s74, s5
.LBB0_186:
	s_sub_i32 s33, s33, s98
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s76, 0, s4
	s_cmp_ge_i32 s74, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s98, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_187:
	s_add_i32 s4, s76, 1
	s_cmp_gt_i32 s4, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s4, 16
	s_cbranch_scc1 .LBB0_190
	s_add_i32 s5, s74, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s98
	s_cbranch_scc1 .LBB0_185
	s_ashr_i32 s75, s74, 31
	s_lshl_b64 s[6:7], s[74:75], 2
	s_add_u32 s6, s34, s6
	s_addc_u32 s7, s35, s7
	global_load_dwordx2 v[2:3], v156, s[6:7] offset:4
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
	s_branch .LBB0_185
.LBB0_190:
	s_mov_b32 s6, s98
	s_branch .LBB0_186
.LBB0_191:
	s_endpgm
.Lfunc_end0:
	.size	attn_kernel_0, .Lfunc_end0-attn_kernel_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_kernel_0
		.amdhsa_group_segment_fixed_size 24576
		.amdhsa_private_segment_fixed_size 76
		.amdhsa_kernarg_size 204
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 256
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

	.set .Lattn_kernel_0.num_vgpr, 256
	.set .Lattn_kernel_0.num_agpr, 0
	.set .Lattn_kernel_0.numbered_sgpr, 100
	.set .Lattn_kernel_0.num_named_barrier, 0
	.set .Lattn_kernel_0.private_seg_size, 76
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
      - .offset:         120
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     global_buffer
      - .offset:         152
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     global_buffer
      - .offset:         168
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         176
        .size:           8
        .value_kind:     global_buffer
      - .offset:         184
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         192
        .size:           8
        .value_kind:     global_buffer
      - .offset:         200
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 24576
    .kernarg_segment_align: 8
    .kernarg_segment_size: 204
    .max_flat_workgroup_size: 512
    .name:           attn_kernel_0
    .private_segment_fixed_size: 76
    .reqd_workgroup_size:
      - 512
      - 1
      - 1
    .sgpr_count:     106
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 170
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
