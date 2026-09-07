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
	s_mov_b32 s84, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s74, s12
.LBB0_3:
	s_sub_i32 s33, s33, s10
	s_and_b64 s[8:9], s[8:9], exec
	s_cselect_b32 s84, 0, s11
	s_cmp_ge_i32 s74, s3
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s33, s78
	s_cselect_b64 s[10:11], -1, 0
	s_or_b64 s[8:9], s[8:9], s[10:11]
	s_and_b64 vcc, exec, s[8:9]
	s_mov_b32 s10, s78
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s11, s84, 1
	s_cmp_gt_i32 s11, 15
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s11, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s12, s74, 1
	s_cmp_ge_i32 s12, s3
	s_mov_b32 s78, s10
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
	s_subb_u32 s78, s18, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s78, s10
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s78, s10
	s_mov_b32 s84, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s91, s[4:5], 0x0
	s_load_dword s79, s[6:7], 0x0
	s_cmp_ge_i32 s74, s3
	s_cbranch_scc1 .LBB0_128
	v_lshrrev_b32_e32 v3, 6, v0
	v_lshrrev_b32_e32 v4, 2, v0
	v_mul_u32_u24_e32 v3, 0x18000, v3
	s_load_dwordx2 s[46:47], s[0:1], 0x0
	s_load_dwordx2 s[48:49], s[0:1], 0x10
	s_load_dwordx2 s[80:81], s[0:1], 0x20
	s_load_dwordx2 s[50:51], s[0:1], 0x50
	s_load_dwordx2 s[52:53], s[0:1], 0x60
	s_load_dwordx2 s[54:55], s[0:1], 0x70
	s_load_dwordx2 s[56:57], s[0:1], 0xa0
	s_load_dwordx2 s[4:5], s[0:1], 0xb0
	s_load_dwordx2 s[94:95], s[0:1], 0xc0
	v_and_b32_e32 v2, 31, v0
	v_and_or_b32 v3, v4, 8, v3
	s_movk_i32 s0, 0xc00
	v_lshrrev_b32_e32 v1, 1, v0
	s_movk_i32 s58, 0xe0
	v_mad_u32_u24 v85, v2, s0, v3
	s_mov_b32 s0, 0xaaaaaab
	v_and_or_b32 v1, v1, s58, v2
	v_mul_hi_u32 v2, v0, s0
	s_mov_b32 s0, 0x2aaaaab
	v_mul_hi_u32 v4, v0, s0
	s_mov_b32 s0, 0x1555556
	v_mul_u32_u24_e32 v3, 24, v2
	v_mul_hi_u32 v5, v0, s0
	v_sub_u32_e32 v3, v0, v3
	v_lshlrev_b32_e32 v4, 5, v4
	v_lshlrev_b32_e32 v5, 4, v5
	v_and_b32_e32 v4, 32, v4
	v_lshl_or_b32 v3, v3, 8, v5
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_u32_e32 v3, v3, v4
	v_and_or_b32 v116, v2, 12, v3
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v118, v2, v3
	v_and_b32_e32 v2, 63, v0
	s_waitcnt lgkmcnt(0)
	v_writelane_b32 v240, s4, 0
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v1, 6, v1
	v_writelane_b32 v240, s5, 1
	v_or_b32_e32 v117, 0x80, v116
	v_xor_b32_e32 v119, 0x80, v2
	v_mov_b32_e32 v120, 0
	s_mov_b32 s71, 0x27000
	s_movk_i32 s82, 0x180
	v_mov_b32_e32 v121, 0x40e00000
	v_mov_b32_e32 v122, 1.0
	s_mov_b32 s83, 0x7060302
	s_movk_i32 s86, 0x1000
	s_movk_i32 s59, 0xf80
	s_mov_b32 s60, s2
	v_mov_b32_e32 v123, 0xff800000
	v_mov_b32_e32 v124, 0x42000000
	s_mov_b32 s36, 0
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s78, s6
.LBB0_12:
	s_mov_b32 s2, s4
	s_cmp_ge_i32 s74, s3
	s_cbranch_scc1 .LBB0_128
.LBB0_13:
	s_ashr_i32 s75, s74, 31
	s_lshl_b32 s6, s33, 8
	s_lshl_b64 s[0:1], s[74:75], 2
	s_add_u32 s4, s34, s0
	s_addc_u32 s5, s35, s1
	global_load_dwordx2 v[2:3], v120, s[4:5]
	s_mul_i32 s4, s84, 0xc0
	v_add_lshl_u32 v7, s4, v85, 1
	s_mov_b32 s7, s71
	v_lshl_add_u32 v6, s84, 2, v1
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s4, v2
	s_add_i32 s96, s4, s6
	v_readfirstlane_b32 s5, v3
	s_add_i32 s6, s96, 0x100
	s_min_i32 s5, s6, s5
	s_sub_i32 s75, s5, s96
	s_add_u32 s8, s50, s0
	s_addc_u32 s9, s51, s1
	s_mul_i32 s4, s96, 0xc00
	s_add_u32 s0, s72, s0
	s_addc_u32 s1, s73, s1
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s10, s96, 4
	global_load_dwordx2 v[4:5], v120, s[8:9]
	global_load_dword v3, v120, s[0:1]
	s_lshl_b64 s[0:1], s[4:5], 1
	s_add_u32 s68, s46, s0
	s_addc_u32 s0, s47, s1
	s_mul_i32 s70, s75, 0x1800
	s_and_b32 s69, s0, 0xffff
	buffer_load_dwordx4 v[144:147], v7, s[68:71], 0 offen
	buffer_load_dwordx4 v[148:151], v7, s[68:71], 0 offen offset:32
	buffer_load_dwordx4 v[152:155], v7, s[68:71], 0 offen offset:64
	buffer_load_dwordx4 v[156:159], v7, s[68:71], 0 offen offset:96
	buffer_load_dwordx4 v[160:163], v7, s[68:71], 0 offen offset:128
	buffer_load_dwordx4 v[164:167], v7, s[68:71], 0 offen offset:160
	buffer_load_dwordx4 v[168:171], v7, s[68:71], 0 offen offset:192
	buffer_load_dwordx4 v[172:175], v7, s[68:71], 0 offen offset:224
	buffer_load_dwordx4 v[176:179], v7, s[68:71], 0 offen offset:256
	buffer_load_dwordx4 v[180:183], v7, s[68:71], 0 offen offset:288
	s_ashr_i32 s11, s10, 31
	s_lshl_b64 s[0:1], s[10:11], 2
	s_add_u32 s4, s54, s0
	s_addc_u32 s0, s55, s1
	s_lshl_b32 s6, s75, 6
	s_and_b32 s5, s0, 0xffff
	buffer_load_dword v2, v6, s[4:7], 0 offen
	buffer_load_dwordx4 v[184:187], v7, s[68:71], 0 offen offset:320
	buffer_load_dwordx4 v[188:191], v7, s[68:71], 0 offen offset:352
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
	s_ashr_i32 s85, s84, 31
	s_sub_i32 s98, s1, s0
	s_lshr_b32 s1, s85, 28
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
	s_add_u32 s92, s48, s8
	s_addc_u32 s93, s49, s9
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s68, s52, s0
	s_addc_u32 s0, s53, s1
	s_lshl_b32 s70, s98, 2
	s_and_b32 s69, s0, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[68:71], 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s90, v4
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
	buffer_load_dword v3, off, s[68:71], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s97, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_mul_i32 s5, s90, 0x1800
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s82, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_17
	v_add_u32_e32 v4, s5, v116
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[92:93]
	global_load_dwordx4 v[128:131], v[4:5], off
	global_load_dwordx4 v[132:135], v[4:5], off offset:256
.LBB0_17:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	buffer_load_dword v3, off, s[68:71], 0 offset:12
	;;#ASMSTART
	v_mov_b32 v4, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s89, v3
	v_and_b32_e32 v4, 0x180, v4
	v_cmp_ne_u32_e32 vcc, s82, v4
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_19
	v_add_u32_e32 v4, s5, v117
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[92:93]
	global_load_dwordx4 v[136:139], v[4:5], off
	global_load_dwordx4 v[140:143], v[4:5], off offset:256
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
	v_cmp_ne_u32_e32 vcc, s82, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_21
	ds_write_b128 v118, v[128:131] offset:12288
	ds_write_b128 v118, v[132:135] offset:18432
.LBB0_21:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s82, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_23
	s_mul_i32 s5, s88, 0x1800
	v_add_u32_e32 v4, s5, v116
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[92:93]
	global_load_dwordx4 v[128:131], v[4:5], off
	global_load_dwordx4 v[132:135], v[4:5], off offset:256
.LBB0_23:
	s_or_b64 exec, exec, s[0:1]
	v_mul_f32_e32 v2, s91, v2
	s_add_i32 s7, s98, -1
	v_mul_f32_e32 v82, 0x3dd53b95, v2
	s_lshl_b32 s87, s4, 13
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
	ds_read_b128 v[192:195], v3 offset:12288
	v_add_u32_e32 v3, 0x1810, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[196:199], v3
	v_add_u32_e32 v3, 0x1820, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[200:203], v3
	v_add_u32_e32 v3, 0x1830, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[204:207], v3
	v_add_u32_e32 v3, 0x1840, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[208:211], v3
	v_add_u32_e32 v3, 0x1850, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[212:215], v3
	v_add_u32_e32 v3, 0x1860, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[216:219], v3
	v_add_u32_e32 v3, 0x1870, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[220:223], v3
	v_add_u32_e32 v3, 0x1880, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[224:227], v3
	v_add_u32_e32 v3, 0x1890, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[228:231], v3
	v_add_u32_e32 v3, 0x18a0, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_u32_e32 v2, 0x18b0, v2
	ds_read_b128 v[232:235], v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[236:239], v2
	s_add_i32 s0, s98, -2
	s_bitcmp0_b32 s98, 0
	s_cselect_b32 s76, s0, s7
	s_ashr_i32 s77, s76, 31
	s_cmp_lt_i32 s76, 1
	s_cbranch_scc1 .LBB0_59
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v83, v82
	v_mov_b32_e32 v86, v82
	v_mov_b32_e32 v87, v82
	v_mov_b32_e32 v88, v82
	v_mov_b32_e32 v89, v82
	v_mov_b32_e32 v90, v82
	v_mov_b32_e32 v91, v82
	v_mov_b32_e32 v92, v82
	v_mov_b32_e32 v93, v82
	v_mov_b32_e32 v94, v82
	v_mov_b32_e32 v95, v82
	v_mov_b32_e32 v96, v82
	v_mov_b32_e32 v97, v82
	v_mov_b32_e32 v98, v82
	v_mov_b32_e32 v99, v82
	s_mov_b64 s[0:1], 0
	s_mov_b32 s8, 20
	v_mov_b32_e32 v84, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v125
	v_mov_b32_e32 v4, v125
	v_mov_b32_e32 v5, v125
	v_mov_b32_e32 v6, v125
	v_mov_b32_e32 v7, v125
	v_mov_b32_e32 v8, v125
	v_mov_b32_e32 v9, v125
	v_mov_b32_e32 v10, v125
	v_mov_b32_e32 v11, v125
	v_mov_b32_e32 v12, v125
	v_mov_b32_e32 v13, v125
	v_mov_b32_e32 v14, v125
	v_mov_b32_e32 v15, v125
	v_mov_b32_e32 v16, v125
	v_mov_b32_e32 v17, v125
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v125
	v_mov_b32_e32 v20, v125
	v_mov_b32_e32 v21, v125
	v_mov_b32_e32 v22, v125
	v_mov_b32_e32 v23, v125
	v_mov_b32_e32 v24, v125
	v_mov_b32_e32 v25, v125
	v_mov_b32_e32 v26, v125
	v_mov_b32_e32 v27, v125
	v_mov_b32_e32 v28, v125
	v_mov_b32_e32 v29, v125
	v_mov_b32_e32 v30, v125
	v_mov_b32_e32 v31, v125
	v_mov_b32_e32 v32, v125
	v_mov_b32_e32 v33, v125
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v125
	v_mov_b32_e32 v36, v125
	v_mov_b32_e32 v37, v125
	v_mov_b32_e32 v38, v125
	v_mov_b32_e32 v39, v125
	v_mov_b32_e32 v40, v125
	v_mov_b32_e32 v41, v125
	v_mov_b32_e32 v42, v125
	v_mov_b32_e32 v43, v125
	v_mov_b32_e32 v44, v125
	v_mov_b32_e32 v45, v125
	v_mov_b32_e32 v46, v125
	v_mov_b32_e32 v47, v125
	v_mov_b32_e32 v48, v125
	v_mov_b32_e32 v49, v125
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v125
	v_mov_b32_e32 v52, v125
	v_mov_b32_e32 v53, v125
	v_mov_b32_e32 v54, v125
	v_mov_b32_e32 v55, v125
	v_mov_b32_e32 v56, v125
	v_mov_b32_e32 v57, v125
	v_mov_b32_e32 v58, v125
	v_mov_b32_e32 v59, v125
	v_mov_b32_e32 v60, v125
	v_mov_b32_e32 v61, v125
	v_mov_b32_e32 v62, v125
	v_mov_b32_e32 v63, v125
	v_mov_b32_e32 v64, v125
	v_mov_b32_e32 v65, v125
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
	v_perm_b32 v66, v67, v66, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s83
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s11, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s11
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[80:81]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[100:103], v[74:75], off offset:512
	global_load_dwordx4 v[104:107], v[74:75], off offset:1024
	global_load_dwordx4 v[108:111], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[100:101], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[104:105], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s86, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[102:103], v[68:69], v[18:33]
	global_load_dwordx4 v[100:103], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[106:107], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[110:111], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[100:101], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[102:103], v[72:73], v[34:49]
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
	ds_read_b128 v[192:195], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[196:199], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[232:235], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[236:239], v66
	s_add_u32 s0, s0, 2
	s_addc_u32 s1, s1, 0
	v_mov_b64_e32 v[66:67], s[76:77]
	v_cmp_lt_i64_e32 vcc, s[0:1], v[66:67]
	s_add_i32 s8, s8, 8
	s_mov_b32 s88, s10
	s_mov_b32 s90, s9
	s_cbranch_vccz .LBB0_58
.LBB0_26:
	s_add_i32 s4, s8, -4
	v_mov_b32_e32 v66, s4
	buffer_load_dword v66, v66, s[68:71], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_mov_b32 s9, s97
	v_and_b32_e32 v67, 0x180, v67
	s_mov_b32 s10, s89
	v_cmp_ne_u32_e32 vcc, s82, v67
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s97, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_28
	ds_write_b128 v118, v[136:139]
	ds_write_b128 v118, v[140:143] offset:6144
.LBB0_28:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_30
	s_mul_i32 s11, s88, 0x1800
	v_add_u32_e32 v66, s11, v117
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[92:93]
	global_load_dwordx4 v[136:139], v[66:67], off
	global_load_dwordx4 v[140:143], v[66:67], off offset:256
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[144:145], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[178:179], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[180:181], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[182:183], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[184:185], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[186:187], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[188:189], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[190:191], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v100, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v100, v100, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v100, v100, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v100, v100, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v100, v100, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v100, v100, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v100, v100, v78, v79
	v_max3_f32 v100, v100, v80, v81
	ds_bpermute_b32 v101, v119, v100
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v101, v100, v101
	;;#ASMSTART
	v_add_f32 v100, v84, v121
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v101, v100
	v_mov_b32_e32 v100, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v101, v101, v122
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v101
	v_exp_f32_e32 v100, v84
	v_mov_b32_e32 v84, v101
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
	v_add_f32_e32 v101, 0, v66
	v_add_f32_e32 v101, v101, v67
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
	v_add_f32_e32 v101, v101, v68
	v_add_f32_e32 v101, v101, v69
	v_add_f32_e32 v101, v101, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v71
	v_add_f32_e32 v101, v101, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v73
	v_add_f32_e32 v101, v101, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v75
	v_add_f32_e32 v101, v101, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v77
	v_add_f32_e32 v101, v101, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v100
	v_add_f32_e32 v101, v101, v79
	v_add_f32_e32 v101, v101, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v101, v101, v81
	;;#ASMSTART
	v_fma_f32 v125, v125, v100, v101
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_34
	;;#ASMSTART
	v_mul_f32 v2, v2, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v100
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
	v_perm_b32 v66, v67, v66, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s83
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_lshl_b32 s12, s90, 13
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	s_add_i32 s12, s12, s87
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s12
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[80:81]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[100:103], v[74:75], off offset:512
	global_load_dwordx4 v[104:107], v[74:75], off offset:1024
	global_load_dwordx4 v[108:111], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[100:101], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[104:105], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s86, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[102:103], v[68:69], v[18:33]
	global_load_dwordx4 v[100:103], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[106:107], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[110:111], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[100:101], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[102:103], v[72:73], v[34:49]
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
	ds_read_b128 v[220:223], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[216:219], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[212:215], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[196:199], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[192:195], v67
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
	ds_read_b128 v[108:111], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[104:107], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[100:103], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_36
	ds_write_b128 v118, v[128:131] offset:12288
	ds_write_b128 v118, v[132:135] offset:18432
.LBB0_36:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s11, s9, 0x1800
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_38
	v_add_u32_e32 v66, s11, v116
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[92:93]
	global_load_dwordx4 v[128:131], v[66:67], off
	global_load_dwordx4 v[132:135], v[66:67], off offset:256
.LBB0_38:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[144:145], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[114:115], v[178:179], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[180:181], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[182:183], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[184:185], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[186:187], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[188:189], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[190:191], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v100, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v100, v100, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v100, v100, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v100, v100, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v100, v100, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v100, v100, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v100, v100, v78, v79
	v_max3_f32 v100, v100, v80, v81
	ds_bpermute_b32 v101, v119, v100
	v_mov_b32_e32 v126, 1.0
	v_mov_b32_e32 v102, v84
	v_mov_b32_e32 v103, v84
	v_mov_b32_e32 v104, v84
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v127, v100, v101
	;;#ASMSTART
	v_add_f32 v100, v84, v121
	;;#ASMEND
	v_mov_b32_e32 v101, v84
	v_cmp_gt_f32_e32 vcc, v127, v100
	v_mov_b32_e32 v100, v84
	v_mov_b32_e32 v105, v84
	v_mov_b32_e32 v106, v84
	v_mov_b32_e32 v107, v84
	v_mov_b32_e32 v108, v84
	v_mov_b32_e32 v109, v84
	v_mov_b32_e32 v110, v84
	v_mov_b32_e32 v111, v84
	v_mov_b32_e32 v112, v84
	v_mov_b32_e32 v113, v84
	v_mov_b32_e32 v114, v84
	v_mov_b32_e32 v115, v84
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_40
	;;#ASMSTART
	v_add_f32 v100, v127, v122
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v126, v84
	v_mov_b32_e32 v84, v100
	v_mov_b32_e32 v101, v100
	v_mov_b32_e32 v102, v100
	v_mov_b32_e32 v103, v100
	v_mov_b32_e32 v104, v100
	v_mov_b32_e32 v105, v100
	v_mov_b32_e32 v106, v100
	v_mov_b32_e32 v107, v100
	v_mov_b32_e32 v108, v100
	v_mov_b32_e32 v109, v100
	v_mov_b32_e32 v110, v100
	v_mov_b32_e32 v111, v100
	v_mov_b32_e32 v112, v100
	v_mov_b32_e32 v113, v100
	v_mov_b32_e32 v114, v100
	v_mov_b32_e32 v115, v100
.LBB0_40:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, 0, v66
	v_add_f32_e32 v127, v127, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v68
	v_add_f32_e32 v127, v127, v69
	v_add_f32_e32 v127, v127, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v71
	v_add_f32_e32 v127, v127, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v73
	v_add_f32_e32 v127, v127, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v75
	v_add_f32_e32 v127, v127, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v77
	v_add_f32_e32 v127, v127, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v126
	v_add_f32_e32 v127, v127, v79
	v_add_f32_e32 v127, v127, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v127, v127, v81
	;;#ASMSTART
	v_fma_f32 v125, v125, v126, v127
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_42
	;;#ASMSTART
	v_mul_f32 v2, v2, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v126
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
	v_perm_b32 v66, v67, v66, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s83
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s12, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s12
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[80:81]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[192:195], v[74:75], off offset:512
	global_load_dwordx4 v[196:199], v[74:75], off offset:1024
	global_load_dwordx4 v[200:203], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s86, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[194:195], v[68:69], v[18:33]
	global_load_dwordx4 v[192:195], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[202:203], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[72:73], v[34:49]
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
	ds_read_b128 v[236:239], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[232:235], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[196:199], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[192:195], v66
	v_mov_b32_e32 v66, s8
	buffer_load_dword v66, v66, s[68:71], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s89, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s82, v67
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_44
	ds_write_b128 v118, v[136:139]
	ds_write_b128 v118, v[140:143] offset:6144
.LBB0_44:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_46
	v_add_u32_e32 v66, s11, v117
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[92:93]
	global_load_dwordx4 v[136:139], v[66:67], off
	global_load_dwordx4 v[140:143], v[66:67], off offset:256
.LBB0_46:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[144:145], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[178:179], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[180:181], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[182:183], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[184:185], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[186:187], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[188:189], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[190:191], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v126, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v126, v126, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v126, v126, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v126, v126, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v126, v126, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v126, v126, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v126, v126, v78, v79
	v_max3_f32 v126, v126, v80, v81
	ds_bpermute_b32 v127, v119, v126
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v127, v127, v127
	v_max_f32_e32 v127, v126, v127
	;;#ASMSTART
	v_add_f32 v126, v84, v121
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v127, v126
	v_mov_b32_e32 v126, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_48
	;;#ASMSTART
	v_add_f32 v100, v127, v122
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v126, v84
	v_mov_b32_e32 v84, v100
	v_mov_b32_e32 v101, v100
	v_mov_b32_e32 v102, v100
	v_mov_b32_e32 v103, v100
	v_mov_b32_e32 v104, v100
	v_mov_b32_e32 v105, v100
	v_mov_b32_e32 v106, v100
	v_mov_b32_e32 v107, v100
	v_mov_b32_e32 v108, v100
	v_mov_b32_e32 v109, v100
	v_mov_b32_e32 v110, v100
	v_mov_b32_e32 v111, v100
	v_mov_b32_e32 v112, v100
	v_mov_b32_e32 v113, v100
	v_mov_b32_e32 v114, v100
	v_mov_b32_e32 v115, v100
.LBB0_48:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, 0, v66
	v_add_f32_e32 v127, v127, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v68
	v_add_f32_e32 v127, v127, v69
	v_add_f32_e32 v127, v127, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v71
	v_add_f32_e32 v127, v127, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v73
	v_add_f32_e32 v127, v127, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v75
	v_add_f32_e32 v127, v127, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v77
	v_add_f32_e32 v127, v127, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v126
	v_add_f32_e32 v127, v127, v79
	v_add_f32_e32 v127, v127, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v127, v127, v81
	;;#ASMSTART
	v_fma_f32 v125, v125, v126, v127
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_50
	;;#ASMSTART
	v_mul_f32 v2, v2, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v126
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
	v_perm_b32 v66, v67, v66, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s83
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_lshl_b32 s11, s88, 13
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	s_add_i32 s11, s11, s87
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s11
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[80:81]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[192:195], v[74:75], off offset:512
	global_load_dwordx4 v[196:199], v[74:75], off offset:1024
	global_load_dwordx4 v[200:203], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s86, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[194:195], v[68:69], v[18:33]
	global_load_dwordx4 v[192:195], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[202:203], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[72:73], v[34:49]
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
	ds_read_b128 v[236:239], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[232:235], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[228:231], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[196:199], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[192:195], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_52
	ds_write_b128 v118, v[128:131] offset:12288
	ds_write_b128 v118, v[132:135] offset:18432
.LBB0_52:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_54
	s_mul_i32 s12, s10, 0x1800
	v_add_u32_e32 v66, s12, v116
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[92:93]
	global_load_dwordx4 v[128:131], v[66:67], off
	global_load_dwordx4 v[132:135], v[66:67], off offset:256
.LBB0_54:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[144:145], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[178:179], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[180:181], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[182:183], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[184:185], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[186:187], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[188:189], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[190:191], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v126, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v126, v126, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v126, v126, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v126, v126, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v126, v126, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v126, v126, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v126, v126, v78, v79
	v_max3_f32 v126, v126, v80, v81
	ds_bpermute_b32 v127, v119, v126
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v127, v127, v127
	v_max_f32_e32 v127, v126, v127
	;;#ASMSTART
	v_add_f32 v126, v84, v121
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v127, v126
	v_mov_b32_e32 v126, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_56
	;;#ASMSTART
	v_add_f32 v100, v127, v122
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v126, v84
	v_mov_b32_e32 v84, v100
	v_mov_b32_e32 v101, v100
	v_mov_b32_e32 v102, v100
	v_mov_b32_e32 v103, v100
	v_mov_b32_e32 v104, v100
	v_mov_b32_e32 v105, v100
	v_mov_b32_e32 v106, v100
	v_mov_b32_e32 v107, v100
	v_mov_b32_e32 v108, v100
	v_mov_b32_e32 v109, v100
	v_mov_b32_e32 v110, v100
	v_mov_b32_e32 v111, v100
	v_mov_b32_e32 v112, v100
	v_mov_b32_e32 v113, v100
	v_mov_b32_e32 v114, v100
	v_mov_b32_e32 v115, v100
.LBB0_56:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, 0, v66
	v_add_f32_e32 v100, v100, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, v100, v68
	v_add_f32_e32 v100, v100, v69
	v_add_f32_e32 v100, v100, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, v100, v71
	v_add_f32_e32 v100, v100, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, v100, v73
	v_add_f32_e32 v100, v100, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, v100, v75
	v_add_f32_e32 v100, v100, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, v100, v77
	v_add_f32_e32 v100, v100, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v126
	v_add_f32_e32 v100, v100, v79
	v_add_f32_e32 v100, v100, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v100, v100, v81
	;;#ASMSTART
	v_fma_f32 v125, v125, v126, v100
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_25
	;;#ASMSTART
	v_mul_f32 v2, v2, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v126
	;;#ASMEND
	s_branch .LBB0_25
.LBB0_58:
	s_mov_b32 s88, s10
	s_mov_b32 s90, s9
	s_cmp_ge_i32 s76, s98
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
	s_mov_b64 s[10:11], s[52:53]
	s_mov_b32 s52, s36
	s_mov_b32 s53, s36
	s_mov_b64 s[12:13], s[54:55]
	s_mov_b32 s54, s36
	s_mov_b32 s55, s36
	s_mov_b64 s[14:15], s[56:57]
	s_mov_b32 s56, s36
	s_mov_b32 s57, s36
	s_mov_b32 s58, s36
	s_mov_b32 s59, s36
	s_mov_b32 s16, s60
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
	v_mov_b32_e32 v125, 0
	s_mov_b32 s60, s16
	s_movk_i32 s59, 0xf80
	s_movk_i32 s58, 0xe0
	s_mov_b64 s[56:57], s[14:15]
	s_mov_b64 s[54:55], s[12:13]
	s_mov_b64 s[52:53], s[10:11]
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
	s_cmp_ge_i32 s76, s98
	s_cbranch_scc1 .LBB0_97
.LBB0_60:
	s_lshl_b32 s37, s7, 6
	s_lshl_b32 s0, s76, 6
	s_ashr_i32 s99, s98, 31
	s_add_i32 s37, s37, s6
	v_mov_b32_e32 v83, v82
	v_mov_b32_e32 v86, v82
	v_mov_b32_e32 v87, v82
	v_mov_b32_e32 v88, v82
	v_mov_b32_e32 v89, v82
	v_mov_b32_e32 v90, v82
	v_mov_b32_e32 v91, v82
	v_mov_b32_e32 v92, v82
	v_mov_b32_e32 v93, v82
	v_mov_b32_e32 v94, v82
	v_mov_b32_e32 v95, v82
	v_mov_b32_e32 v96, v82
	v_mov_b32_e32 v97, v82
	v_mov_b32_e32 v98, v82
	v_mov_b32_e32 v99, v82
	s_add_i32 s40, s0, 0x77
	s_add_i32 s41, s76, 1
	s_lshl2_add_u32 s42, s76, 20
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
	v_perm_b32 v66, v67, v66, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s83
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[80:81]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[100:103], v[74:75], off offset:512
	global_load_dwordx4 v[104:107], v[74:75], off offset:1024
	global_load_dwordx4 v[108:111], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[100:101], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[104:105], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s86, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[102:103], v[68:69], v[18:33]
	global_load_dwordx4 v[100:103], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[106:107], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[110:111], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[100:101], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[102:103], v[72:73], v[34:49]
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
	ds_read_b128 v[192:195], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[196:199], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[232:235], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[236:239], v66
.LBB0_62:
	s_and_b64 s[0:1], s[38:39], exec
	s_cselect_b32 s90, s97, s88
	s_cselect_b32 s88, s89, s97
	s_cselect_b32 s97, s43, s89
	s_add_u32 s76, s76, 2
	s_addc_u32 s77, s77, 0
	v_mov_b64_e32 v[66:67], s[98:99]
	v_cmp_lt_i64_e32 vcc, s[76:77], v[66:67]
	s_addk_i32 s40, 0x80
	s_add_i32 s41, s41, 2
	s_add_i32 s42, s42, 8
	s_mov_b32 s89, s44
	s_cbranch_vccz .LBB0_97
.LBB0_63:
	s_add_i32 s0, s42, -4
	v_mov_b32_e32 v66, s0
	buffer_load_dword v66, v66, s[68:71], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s82, v67
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_65
	ds_write_b128 v118, v[136:139]
	ds_write_b128 v118, v[140:143] offset:6144
.LBB0_65:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_67
	s_mul_i32 s4, s88, 0x1800
	v_add_u32_e32 v66, s4, v117
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[92:93]
	global_load_dwordx4 v[136:139], v[66:67], off
	global_load_dwordx4 v[140:143], v[66:67], off offset:256
.LBB0_67:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[144:145], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[178:179], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[180:181], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[182:183], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[184:185], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[186:187], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[188:189], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[190:191], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v100, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v100
	v_and_b32_e32 v100, 8, v100
	v_add_u32_e32 v100, s40, v100
	v_add_u32_e32 v101, 0xffffff89, v100
	v_cmp_gt_i32_e32 vcc, s37, v101
	v_add_u32_e32 v101, 0xffffff8a, v100
	v_cmp_gt_i32_e64 s[0:1], s37, v101
	v_add_u32_e32 v101, 0xffffff8b, v100
	v_cmp_gt_i32_e64 s[4:5], s37, v101
	v_add_u32_e32 v101, 0xffffff8c, v100
	v_cmp_gt_i32_e64 s[6:7], s37, v101
	v_add_u32_e32 v101, 0xffffff8d, v100
	v_cmp_gt_i32_e64 s[8:9], s37, v101
	v_add_u32_e32 v101, 0xffffff8e, v100
	v_cmp_gt_i32_e64 s[10:11], s37, v101
	v_add_u32_e32 v101, 0xffffff8f, v100
	v_cmp_gt_i32_e64 s[12:13], s37, v101
	v_add_u32_e32 v101, 0xffffff90, v100
	v_cmp_gt_i32_e64 s[14:15], s37, v101
	v_add_u32_e32 v101, 0xffffff99, v100
	v_cmp_gt_i32_e64 s[16:17], s37, v101
	v_add_u32_e32 v101, 0xffffff9a, v100
	v_cmp_gt_i32_e64 s[18:19], s37, v101
	v_add_u32_e32 v101, 0xffffff9b, v100
	v_cmp_gt_i32_e64 s[20:21], s37, v101
	v_add_u32_e32 v101, 0xffffff9c, v100
	v_cmp_gt_i32_e64 s[22:23], s37, v101
	v_add_u32_e32 v101, 0xffffff9d, v100
	v_cmp_gt_i32_e64 s[24:25], s37, v101
	v_add_u32_e32 v101, 0xffffff9e, v100
	v_cmp_gt_i32_e64 s[26:27], s37, v101
	v_add_u32_e32 v101, 0xffffff9f, v100
	v_add_u32_e32 v100, 0xffffffa0, v100
	v_cmp_gt_i32_e64 s[28:29], s37, v101
	v_cmp_gt_i32_e64 s[30:31], s37, v100
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
	v_cndmask_b32_e64 v79, v123, v79, s[26:27]
	v_cndmask_b32_e64 v78, v123, v78, s[24:25]
	v_cndmask_b32_e64 v105, v123, v67, s[0:1]
	v_cndmask_b32_e32 v104, v123, v66, vcc
	v_cndmask_b32_e64 v77, v123, v77, s[22:23]
	v_cndmask_b32_e64 v76, v123, v76, s[20:21]
	v_cndmask_b32_e64 v75, v123, v75, s[18:19]
	v_cndmask_b32_e64 v74, v123, v74, s[16:17]
	v_cndmask_b32_e64 v101, v123, v71, s[10:11]
	v_cndmask_b32_e64 v100, v123, v70, s[8:9]
	v_cndmask_b32_e64 v103, v123, v69, s[6:7]
	v_cndmask_b32_e64 v102, v123, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[96:97], v[78:79]
	v_pk_mul_f32 v[78:79], v[82:83], v[104:105]
	v_pk_mul_f32 v[68:69], v[94:95], v[76:77]
	v_pk_mul_f32 v[70:71], v[92:93], v[74:75]
	v_pk_mul_f32 v[74:75], v[88:89], v[100:101]
	v_pk_mul_f32 v[76:77], v[86:87], v[102:103]
	v_max_f32_e32 v100, v78, v79
	v_cndmask_b32_e64 v73, v123, v73, s[14:15]
	v_cndmask_b32_e64 v72, v123, v72, s[12:13]
	v_max3_f32 v100, v100, v76, v77
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v100, v100, v74, v75
	v_max3_f32 v100, v100, v72, v73
	v_max3_f32 v100, v100, v70, v71
	v_cndmask_b32_e64 v81, v123, v81, s[30:31]
	v_cndmask_b32_e64 v80, v123, v80, s[28:29]
	v_max3_f32 v100, v100, v68, v69
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v100, v100, v66, v67
	v_max3_f32 v100, v100, v80, v81
	ds_bpermute_b32 v101, v119, v100
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v101, v100, v101
	;;#ASMSTART
	v_add_f32 v100, v84, v121
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v101, v100
	v_mov_b32_e32 v100, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_69
	;;#ASMSTART
	v_add_f32 v101, v101, v122
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v101
	v_exp_f32_e32 v100, v84
	v_mov_b32_e32 v84, v101
.LBB0_69:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[102:103], v[66:67], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67], v[78:79], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[104:105], v[68:69], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	v_pk_add_f32 v[68:69], v[76:77], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, 0, v66
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[70:71], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v67
	v_add_f32_e32 v101, v101, v68
	v_pk_add_f32 v[70:71], v[74:75], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v69
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
	v_exp_f32 v74, v106
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v75, v107
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v101, v101, v70
	v_add_f32_e32 v101, v101, v71
	v_add_f32_e32 v101, v101, v72
	v_add_f32_e32 v101, v101, v73
	v_add_f32_e32 v101, v101, v74
	v_add_f32_e32 v101, v101, v75
	;;#ASMSTART
	v_exp_f32 v76, v104
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v77, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v102
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v76
	v_add_f32_e32 v101, v101, v77
	v_add_f32_e32 v101, v101, v78
	;;#ASMSTART
	v_exp_f32 v79, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v100
	v_add_f32_e32 v101, v101, v79
	v_add_f32_e32 v101, v101, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v101, v101, v81
	;;#ASMSTART
	v_fma_f32 v125, v125, v100, v101
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_71
	;;#ASMSTART
	v_mul_f32 v2, v2, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v100
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
	v_perm_b32 v66, v67, v66, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s83
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_lshl_b32 s38, s90, 13
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	s_add_i32 s38, s38, s87
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s38
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[80:81]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[100:103], v[74:75], off offset:512
	global_load_dwordx4 v[104:107], v[74:75], off offset:1024
	global_load_dwordx4 v[108:111], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[100:101], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[104:105], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s86, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[102:103], v[68:69], v[18:33]
	global_load_dwordx4 v[100:103], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[106:107], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[110:111], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[100:101], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[102:103], v[72:73], v[34:49]
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
	ds_read_b128 v[220:223], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[216:219], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[212:215], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[196:199], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[192:195], v67
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
	ds_read_b128 v[108:111], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[104:107], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[100:103], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_73
	ds_write_b128 v118, v[128:131] offset:12288
	ds_write_b128 v118, v[132:135] offset:18432
.LBB0_73:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s45, s97, 0x1800
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_75
	v_add_u32_e32 v66, s45, v116
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[92:93]
	global_load_dwordx4 v[128:131], v[66:67], off
	global_load_dwordx4 v[132:135], v[66:67], off offset:256
.LBB0_75:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[144:145], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[114:115], v[178:179], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[180:181], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[182:183], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[184:185], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[186:187], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[188:189], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[190:191], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v100, v0
	;;#ASMEND
	v_mov_b32_e32 v126, 1.0
	v_lshrrev_b32_e32 v100, 2, v100
	v_and_b32_e32 v100, 8, v100
	v_add_u32_e32 v100, s40, v100
	v_add_u32_e32 v101, 0xffffffa9, v100
	v_cmp_gt_i32_e32 vcc, s37, v101
	v_add_u32_e32 v101, 0xffffffaa, v100
	v_cmp_gt_i32_e64 s[0:1], s37, v101
	v_add_u32_e32 v101, 0xffffffab, v100
	v_cmp_gt_i32_e64 s[4:5], s37, v101
	v_add_u32_e32 v101, 0xffffffac, v100
	v_cmp_gt_i32_e64 s[6:7], s37, v101
	v_add_u32_e32 v101, 0xffffffad, v100
	v_cmp_gt_i32_e64 s[8:9], s37, v101
	v_add_u32_e32 v101, 0xffffffae, v100
	v_cmp_gt_i32_e64 s[10:11], s37, v101
	v_add_u32_e32 v101, 0xffffffaf, v100
	v_cmp_gt_i32_e64 s[12:13], s37, v101
	v_add_u32_e32 v101, 0xffffffb0, v100
	v_cmp_gt_i32_e64 s[14:15], s37, v101
	v_add_u32_e32 v101, 0xffffffb9, v100
	v_cmp_gt_i32_e64 s[16:17], s37, v101
	v_add_u32_e32 v101, 0xffffffba, v100
	v_cmp_gt_i32_e64 s[18:19], s37, v101
	v_add_u32_e32 v101, 0xffffffbb, v100
	v_cmp_gt_i32_e64 s[20:21], s37, v101
	v_add_u32_e32 v101, 0xffffffbc, v100
	v_cmp_gt_i32_e64 s[22:23], s37, v101
	v_add_u32_e32 v101, 0xffffffbd, v100
	v_cmp_gt_i32_e64 s[24:25], s37, v101
	v_add_u32_e32 v101, 0xffffffbe, v100
	v_cmp_gt_i32_e64 s[26:27], s37, v101
	v_add_u32_e32 v101, 0xffffffbf, v100
	v_subrev_u32_e32 v100, 64, v100
	v_cmp_gt_i32_e64 s[28:29], s37, v101
	v_cmp_gt_i32_e64 s[30:31], s37, v100
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
	v_cndmask_b32_e64 v67, v123, v67, s[0:1]
	v_cndmask_b32_e32 v66, v123, v66, vcc
	v_cndmask_b32_e64 v69, v123, v69, s[6:7]
	v_cndmask_b32_e64 v68, v123, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e64 v71, v123, v71, s[10:11]
	v_cndmask_b32_e64 v70, v123, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v100, v66, v67
	v_cndmask_b32_e64 v73, v123, v73, s[14:15]
	v_cndmask_b32_e64 v72, v123, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v100, v100, v68, v69
	v_cndmask_b32_e64 v75, v123, v75, s[18:19]
	v_cndmask_b32_e64 v74, v123, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v100, v100, v70, v71
	v_cndmask_b32_e64 v77, v123, v77, s[22:23]
	v_cndmask_b32_e64 v76, v123, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v100, v100, v72, v73
	v_cndmask_b32_e64 v79, v123, v79, s[26:27]
	v_cndmask_b32_e64 v78, v123, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v100, v100, v74, v75
	v_cndmask_b32_e64 v81, v123, v81, s[30:31]
	v_cndmask_b32_e64 v80, v123, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v100, v100, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v100, v100, v78, v79
	v_max3_f32 v100, v100, v80, v81
	ds_bpermute_b32 v101, v119, v100
	v_mov_b32_e32 v102, v84
	v_mov_b32_e32 v103, v84
	v_mov_b32_e32 v104, v84
	v_mov_b32_e32 v105, v84
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v127, v100, v101
	;;#ASMSTART
	v_add_f32 v100, v84, v121
	;;#ASMEND
	v_mov_b32_e32 v101, v84
	v_cmp_gt_f32_e32 vcc, v127, v100
	v_mov_b32_e32 v100, v84
	v_mov_b32_e32 v106, v84
	v_mov_b32_e32 v107, v84
	v_mov_b32_e32 v108, v84
	v_mov_b32_e32 v109, v84
	v_mov_b32_e32 v110, v84
	v_mov_b32_e32 v111, v84
	v_mov_b32_e32 v112, v84
	v_mov_b32_e32 v113, v84
	v_mov_b32_e32 v114, v84
	v_mov_b32_e32 v115, v84
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_77
	;;#ASMSTART
	v_add_f32 v100, v127, v122
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v126, v84
	v_mov_b32_e32 v84, v100
	v_mov_b32_e32 v101, v100
	v_mov_b32_e32 v102, v100
	v_mov_b32_e32 v103, v100
	v_mov_b32_e32 v104, v100
	v_mov_b32_e32 v105, v100
	v_mov_b32_e32 v106, v100
	v_mov_b32_e32 v107, v100
	v_mov_b32_e32 v108, v100
	v_mov_b32_e32 v109, v100
	v_mov_b32_e32 v110, v100
	v_mov_b32_e32 v111, v100
	v_mov_b32_e32 v112, v100
	v_mov_b32_e32 v113, v100
	v_mov_b32_e32 v114, v100
	v_mov_b32_e32 v115, v100
.LBB0_77:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, 0, v66
	v_add_f32_e32 v127, v127, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v68
	v_add_f32_e32 v127, v127, v69
	v_add_f32_e32 v127, v127, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v71
	v_add_f32_e32 v127, v127, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v73
	v_add_f32_e32 v127, v127, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v75
	v_add_f32_e32 v127, v127, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v77
	v_add_f32_e32 v127, v127, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v126
	v_add_f32_e32 v127, v127, v79
	v_add_f32_e32 v127, v127, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v127, v127, v81
	;;#ASMSTART
	v_fma_f32 v125, v125, v126, v127
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_79
	;;#ASMSTART
	v_mul_f32 v2, v2, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v126
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
	v_perm_b32 v66, v67, v66, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s83
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_addk_i32 s38, 0x1000
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s38
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[80:81]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[192:195], v[74:75], off offset:512
	global_load_dwordx4 v[196:199], v[74:75], off offset:1024
	global_load_dwordx4 v[200:203], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s86, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[194:195], v[68:69], v[18:33]
	global_load_dwordx4 v[192:195], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[202:203], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[72:73], v[34:49]
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
	ds_read_b128 v[192:195], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[196:199], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x1860, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[232:235], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[236:239], v66
	s_cmp_gt_i32 s98, s41
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_le_i32 s98, s41
	s_cbranch_scc1 .LBB0_96
	v_mov_b32_e32 v66, s42
	buffer_load_dword v66, v66, s[68:71], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s82, v67
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_82
	ds_write_b128 v118, v[136:139]
	ds_write_b128 v118, v[140:143] offset:6144
.LBB0_82:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_84
	v_add_u32_e32 v66, s45, v117
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[92:93]
	global_load_dwordx4 v[136:139], v[66:67], off
	global_load_dwordx4 v[140:143], v[66:67], off offset:256
.LBB0_84:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[144:145], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[178:179], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[180:181], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[182:183], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[184:185], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[186:187], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[188:189], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[190:191], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v126, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v126, 2, v126
	v_and_b32_e32 v126, 8, v126
	v_add_u32_e32 v126, s40, v126
	v_subrev_u32_e32 v127, 55, v126
	v_cmp_gt_i32_e32 vcc, s37, v127
	v_subrev_u32_e32 v127, 54, v126
	v_cmp_gt_i32_e64 s[0:1], s37, v127
	v_subrev_u32_e32 v127, 53, v126
	v_cmp_gt_i32_e64 s[4:5], s37, v127
	v_subrev_u32_e32 v127, 52, v126
	v_cmp_gt_i32_e64 s[6:7], s37, v127
	v_subrev_u32_e32 v127, 51, v126
	v_cmp_gt_i32_e64 s[8:9], s37, v127
	v_subrev_u32_e32 v127, 50, v126
	v_cmp_gt_i32_e64 s[10:11], s37, v127
	v_subrev_u32_e32 v127, 49, v126
	v_cmp_gt_i32_e64 s[12:13], s37, v127
	v_subrev_u32_e32 v127, 48, v126
	v_cmp_gt_i32_e64 s[14:15], s37, v127
	v_subrev_u32_e32 v127, 39, v126
	v_cmp_gt_i32_e64 s[16:17], s37, v127
	v_subrev_u32_e32 v127, 38, v126
	v_cmp_gt_i32_e64 s[18:19], s37, v127
	v_subrev_u32_e32 v127, 37, v126
	v_cmp_gt_i32_e64 s[20:21], s37, v127
	v_subrev_u32_e32 v127, 36, v126
	v_cmp_gt_i32_e64 s[22:23], s37, v127
	v_subrev_u32_e32 v127, 35, v126
	v_cmp_gt_i32_e64 s[24:25], s37, v127
	v_subrev_u32_e32 v127, 34, v126
	v_cmp_gt_i32_e64 s[26:27], s37, v127
	v_subrev_u32_e32 v127, 33, v126
	v_subrev_u32_e32 v126, 32, v126
	v_cmp_gt_i32_e64 s[28:29], s37, v127
	v_cmp_gt_i32_e64 s[30:31], s37, v126
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
	v_cndmask_b32_e64 v67, v123, v67, s[0:1]
	v_cndmask_b32_e32 v66, v123, v66, vcc
	v_cndmask_b32_e64 v69, v123, v69, s[6:7]
	v_cndmask_b32_e64 v68, v123, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e64 v71, v123, v71, s[10:11]
	v_cndmask_b32_e64 v70, v123, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v126, v66, v67
	v_cndmask_b32_e64 v73, v123, v73, s[14:15]
	v_cndmask_b32_e64 v72, v123, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v126, v126, v68, v69
	v_cndmask_b32_e64 v75, v123, v75, s[18:19]
	v_cndmask_b32_e64 v74, v123, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v126, v126, v70, v71
	v_cndmask_b32_e64 v77, v123, v77, s[22:23]
	v_cndmask_b32_e64 v76, v123, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v126, v126, v72, v73
	v_cndmask_b32_e64 v79, v123, v79, s[26:27]
	v_cndmask_b32_e64 v78, v123, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v126, v126, v74, v75
	v_cndmask_b32_e64 v81, v123, v81, s[30:31]
	v_cndmask_b32_e64 v80, v123, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v126, v126, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v126, v126, v78, v79
	v_max3_f32 v126, v126, v80, v81
	ds_bpermute_b32 v127, v119, v126
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v127, v127, v127
	v_max_f32_e32 v127, v126, v127
	;;#ASMSTART
	v_add_f32 v126, v84, v121
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v127, v126
	v_mov_b32_e32 v126, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_86
	;;#ASMSTART
	v_add_f32 v100, v127, v122
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v126, v84
	v_mov_b32_e32 v84, v100
	v_mov_b32_e32 v101, v100
	v_mov_b32_e32 v102, v100
	v_mov_b32_e32 v103, v100
	v_mov_b32_e32 v104, v100
	v_mov_b32_e32 v105, v100
	v_mov_b32_e32 v106, v100
	v_mov_b32_e32 v107, v100
	v_mov_b32_e32 v108, v100
	v_mov_b32_e32 v109, v100
	v_mov_b32_e32 v110, v100
	v_mov_b32_e32 v111, v100
	v_mov_b32_e32 v112, v100
	v_mov_b32_e32 v113, v100
	v_mov_b32_e32 v114, v100
	v_mov_b32_e32 v115, v100
.LBB0_86:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, 0, v66
	v_add_f32_e32 v127, v127, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v68
	v_add_f32_e32 v127, v127, v69
	v_add_f32_e32 v127, v127, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v71
	v_add_f32_e32 v127, v127, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v73
	v_add_f32_e32 v127, v127, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v75
	v_add_f32_e32 v127, v127, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v127, v127, v77
	v_add_f32_e32 v127, v127, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v126
	v_add_f32_e32 v127, v127, v79
	v_add_f32_e32 v127, v127, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v127, v127, v81
	;;#ASMSTART
	v_fma_f32 v125, v125, v126, v127
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_88
	;;#ASMSTART
	v_mul_f32 v2, v2, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v126
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
	v_perm_b32 v66, v67, v66, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s83
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v74, v0
	;;#ASMEND
	s_lshl_b32 s45, s88, 13
	v_lshlrev_b32_e32 v75, 3, v74
	v_lshlrev_b32_e32 v74, 5, v74
	s_add_i32 s45, s45, s87
	v_and_b32_e32 v75, 0xf8, v75
	v_and_b32_e32 v74, 0x400, v74
	v_or3_b32 v74, v75, v74, s45
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[80:81]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[192:195], v[74:75], off offset:512
	global_load_dwordx4 v[196:199], v[74:75], off offset:1024
	global_load_dwordx4 v[200:203], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s86, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[194:195], v[68:69], v[18:33]
	global_load_dwordx4 v[192:195], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[202:203], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[72:73], v[34:49]
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
	ds_read_b128 v[236:239], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[232:235], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[228:231], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[196:199], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[192:195], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_90
	ds_write_b128 v118, v[128:131] offset:12288
	ds_write_b128 v118, v[132:135] offset:18432
.LBB0_90:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s82, v66
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_92
	s_mul_i32 s4, s89, 0x1800
	v_add_u32_e32 v66, s4, v116
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[92:93]
	global_load_dwordx4 v[128:131], v[66:67], off
	global_load_dwordx4 v[132:135], v[66:67], off offset:256
.LBB0_92:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[144:145], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[178:179], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[180:181], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[182:183], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[184:185], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[186:187], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[188:189], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[190:191], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v126, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v126, 2, v126
	v_and_b32_e32 v126, 8, v126
	v_add_u32_e32 v126, s40, v126
	v_subrev_u32_e32 v127, 23, v126
	v_cmp_gt_i32_e32 vcc, s37, v127
	v_subrev_u32_e32 v127, 22, v126
	v_cmp_gt_i32_e64 s[0:1], s37, v127
	v_subrev_u32_e32 v127, 21, v126
	v_cmp_gt_i32_e64 s[4:5], s37, v127
	v_subrev_u32_e32 v127, 20, v126
	v_cmp_gt_i32_e64 s[6:7], s37, v127
	v_subrev_u32_e32 v127, 19, v126
	v_cmp_gt_i32_e64 s[8:9], s37, v127
	v_subrev_u32_e32 v127, 18, v126
	v_cmp_gt_i32_e64 s[10:11], s37, v127
	v_subrev_u32_e32 v127, 17, v126
	v_cmp_gt_i32_e64 s[12:13], s37, v127
	v_add_u32_e32 v127, -16, v126
	v_cmp_gt_i32_e64 s[14:15], s37, v127
	v_add_u32_e32 v127, -7, v126
	v_cmp_gt_i32_e64 s[16:17], s37, v127
	v_add_u32_e32 v127, -6, v126
	v_cmp_gt_i32_e64 s[18:19], s37, v127
	v_add_u32_e32 v127, -5, v126
	v_cmp_gt_i32_e64 s[20:21], s37, v127
	v_add_u32_e32 v127, -4, v126
	v_cmp_gt_i32_e64 s[22:23], s37, v127
	v_add_u32_e32 v127, -3, v126
	v_cmp_gt_i32_e64 s[24:25], s37, v127
	v_add_u32_e32 v127, -2, v126
	v_cmp_gt_i32_e64 s[26:27], s37, v127
	v_add_u32_e32 v127, -1, v126
	v_cmp_gt_i32_e64 s[28:29], s37, v127
	v_cmp_gt_i32_e64 s[30:31], s37, v126
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
	v_cndmask_b32_e64 v67, v123, v67, s[0:1]
	v_cndmask_b32_e32 v66, v123, v66, vcc
	v_cndmask_b32_e64 v69, v123, v69, s[6:7]
	v_cndmask_b32_e64 v68, v123, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e64 v71, v123, v71, s[10:11]
	v_cndmask_b32_e64 v70, v123, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v126, v66, v67
	v_cndmask_b32_e64 v73, v123, v73, s[14:15]
	v_cndmask_b32_e64 v72, v123, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v126, v126, v68, v69
	v_cndmask_b32_e64 v75, v123, v75, s[18:19]
	v_cndmask_b32_e64 v74, v123, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v126, v126, v70, v71
	v_cndmask_b32_e64 v77, v123, v77, s[22:23]
	v_cndmask_b32_e64 v76, v123, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v126, v126, v72, v73
	v_cndmask_b32_e64 v79, v123, v79, s[26:27]
	v_cndmask_b32_e64 v78, v123, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v126, v126, v74, v75
	v_cndmask_b32_e64 v81, v123, v81, s[30:31]
	v_cndmask_b32_e64 v80, v123, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v126, v126, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v126, v126, v78, v79
	v_max3_f32 v126, v126, v80, v81
	ds_bpermute_b32 v127, v119, v126
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v127, v127, v127
	v_max_f32_e32 v127, v126, v127
	;;#ASMSTART
	v_add_f32 v126, v84, v121
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v127, v126
	v_mov_b32_e32 v126, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_94
	;;#ASMSTART
	v_add_f32 v100, v127, v122
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v126, v84
	v_mov_b32_e32 v84, v100
	v_mov_b32_e32 v101, v100
	v_mov_b32_e32 v102, v100
	v_mov_b32_e32 v103, v100
	v_mov_b32_e32 v104, v100
	v_mov_b32_e32 v105, v100
	v_mov_b32_e32 v106, v100
	v_mov_b32_e32 v107, v100
	v_mov_b32_e32 v108, v100
	v_mov_b32_e32 v109, v100
	v_mov_b32_e32 v110, v100
	v_mov_b32_e32 v111, v100
	v_mov_b32_e32 v112, v100
	v_mov_b32_e32 v113, v100
	v_mov_b32_e32 v114, v100
	v_mov_b32_e32 v115, v100
.LBB0_94:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, 0, v66
	v_add_f32_e32 v100, v100, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, v100, v68
	v_add_f32_e32 v100, v100, v69
	v_add_f32_e32 v100, v100, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, v100, v71
	v_add_f32_e32 v100, v100, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, v100, v73
	v_add_f32_e32 v100, v100, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, v100, v75
	v_add_f32_e32 v100, v100, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v100, v100, v77
	v_add_f32_e32 v100, v100, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v126
	v_add_f32_e32 v100, v100, v79
	v_add_f32_e32 v100, v100, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v100, v100, v81
	;;#ASMSTART
	v_fma_f32 v125, v125, v126, v100
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_61
	;;#ASMSTART
	v_mul_f32 v2, v2, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v126
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v126
	;;#ASMEND
	s_branch .LBB0_61
.LBB0_96:
	s_mov_b32 s44, s43
	s_branch .LBB0_62
.LBB0_97:
	;;#ASMSTART
	v_mov_b32 v68, v0
	;;#ASMEND
	ds_bpermute_b32 v66, v119, v125
	v_lshrrev_b32_e32 v67, 1, v68
	v_and_b32_e32 v69, 31, v68
	v_and_or_b32 v67, v67, s58, v69
	v_and_b32_e32 v68, 32, v68
	v_cmp_eq_u32_e32 vcc, 0, v68
	v_cmp_gt_i32_e64 s[0:1], s75, v67
	s_and_b64 s[4:5], vcc, s[0:1]
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v66, v125, v66
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_99
	s_mov_b32 s4, 0x800000
	v_cmp_gt_f32_e32 vcc, s4, v66
	s_ashr_i32 s97, s96, 31
	s_lshl_b64 s[4:5], s[96:97], 6
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v66, v69
	v_log_f32_e32 v69, v69
	v_cndmask_b32_e32 v68, 0, v124, vcc
	v_readlane_b32 s6, v240, 0
	v_readlane_b32 s7, v240, 1
	v_sub_f32_e32 v68, v69, v68
	s_add_u32 s6, s6, s4
	v_add_f32_e32 v68, v84, v68
	s_addc_u32 s7, s7, s5
	s_lshl_b64 s[4:5], s[84:85], 2
	v_mul_f32_e32 v68, 0x3f317218, v68
	v_cmp_lt_f32_e32 vcc, 0, v66
	s_add_u32 s4, s6, s4
	s_addc_u32 s5, s7, s5
	v_cndmask_b32_e32 v68, v123, v68, vcc
	v_lshlrev_b32_e32 v67, 6, v67
	global_store_dword v67, v68, s[4:5]
.LBB0_99:
	s_or_b64 exec, exec, s[0:1]
	v_div_scale_f32 v67, s[0:1], v66, v66, s79
	v_rcp_f32_e32 v68, v67
	v_div_scale_f32 v69, vcc, s79, v66, s79
	v_fma_f32 v70, -v67, v68, 1.0
	v_fmac_f32_e32 v68, v70, v68
	v_mul_f32_e32 v70, v69, v68
	v_fma_f32 v71, -v67, v70, v69
	v_fmac_f32_e32 v70, v71, v68
	v_fma_f32 v67, -v67, v70, v69
	v_div_fmas_f32 v67, v67, v68, v70
	v_div_fixup_f32 v66, v67, v66, s79
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
	v_perm_b32 v28, v3, v2, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v5, v4, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v7, v6, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v9, v8, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v11, v10, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v13, v12, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v15, v14, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v17, v16, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v19, v18, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v21, v20, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v23, v22, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v74, v75, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v72, v73, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v70, v71, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v68, v69, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v66, v67, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v35, v34, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v37, v36, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v39, v38, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v41, v40, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v43, v42, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v45, v44, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v47, v46, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v49, v48, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v51, v50, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v53, v52, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v55, v54, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v57, v56, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v59, v58, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v61, v60, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v63, v62, s83
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v65, v64, s83
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v34, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v34, 0x100, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_101
	s_barrier
.LBB0_101:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v37, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v34, 3, v37
	v_bfe_u32 v36, v37, 6, 3
	v_lshlrev_b32_e32 v35, 7, v37
	v_and_b32_e32 v34, 4, v34
	v_and_or_b32 v34, v35, s59, v34
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_103
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
.LBB0_103:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v39, 0x1ff, v37
	s_lshl_b32 s0, s96, 11
	v_lshlrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v37, 56, v37
	s_ashr_i32 s1, s0, 31
	v_xor_b32_e32 v37, v40, v37
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v37, 1, v37
	s_add_u32 s68, s56, s0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v39, 7, v39
	ds_read_b128 v[42:45], v37
	s_addc_u32 s0, s57, s1
	s_lshl_b32 s4, s84, 7
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
	s_cbranch_execz .LBB0_105
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
.LBB0_105:
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
	s_add_i32 s0, s4, 0x10000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 3, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[68:71], 0 offen
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
	s_add_i32 s0, s4, 0x20000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 4, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[68:71], 0 offen
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
	s_add_i32 s0, s4, 0x30000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 5, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[68:71], 0 offen
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
	s_add_i32 s0, s4, 0x40000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 6, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[68:71], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_115
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
.LBB0_115:
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
	s_cbranch_execz .LBB0_117
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
.LBB0_117:
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
	s_mov_b32 s0, s60
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_121
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_120
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v3, s8
	global_atomic_add v3, v120, v3, s[94:95] sc0
.LBB0_120:
	s_or_b64 exec, exec, s[6:7]
	s_lshl_b64 s[6:7], s[0:1], 2
	s_add_u32 s6, s94, s6
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v3
	s_addc_u32 s7, s95, s7
	s_nop 0
	v_add_u32_e32 v2, s8, v2
	global_store_dword v120, v2, s[6:7]
	s_waitcnt vmcnt(0)
.LBB0_121:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s94, s0
	s_addc_u32 s1, s95, s1
	s_barrier
	global_load_dword v2, v120, s[0:1]
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s4, v2
	s_add_i32 s0, s4, s33
	s_sub_i32 s33, s0, s2
	s_cmp_ge_i32 s74, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s78
	s_cselect_b64 s[6:7], -1, 0
	s_or_b64 s[0:1], s[0:1], s[6:7]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_124
	s_branch .LBB0_12
.LBB0_122:
	s_mov_b32 s74, s5
.LBB0_123:
	s_sub_i32 s33, s33, s78
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s84, 0, s2
	s_cmp_ge_i32 s74, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s78, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_124:
	s_add_i32 s2, s84, 1
	s_cmp_gt_i32 s2, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s2, 16
	s_cbranch_scc1 .LBB0_127
	s_add_i32 s5, s74, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s78
	s_cbranch_scc1 .LBB0_122
	s_ashr_i32 s75, s74, 31
	s_lshl_b64 s[6:7], s[74:75], 2
	s_add_u32 s6, s34, s6
	s_addc_u32 s7, s35, s7
	global_load_dwordx2 v[2:3], v120, s[6:7] offset:4
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
	s_branch .LBB0_122
.LBB0_127:
	s_mov_b32 s6, s78
	s_branch .LBB0_123
.LBB0_128:
	s_endpgm
.Lfunc_end0:
	.size	attn_kernel_0, .Lfunc_end0-attn_kernel_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_kernel_0
		.amdhsa_group_segment_fixed_size 24576
		.amdhsa_private_segment_fixed_size 0
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
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 241
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 244
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

	.set .Lattn_kernel_0.num_vgpr, 241
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
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 512
      - 1
      - 1
    .sgpr_count:     106
    .sgpr_spill_count: 2
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     241
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
