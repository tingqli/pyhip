	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	attn_kernel_0
	.p2align	8
	.type	attn_kernel_0,@function
attn_kernel_0:
	s_load_dwordx2 s[8:9], s[0:1], 0x30
	s_load_dword s3, s[0:1], 0x38
	s_load_dwordx2 s[10:11], s[0:1], 0x90
	s_load_dwordx4 s[4:7], s[0:1], 0x80
	s_mov_b32 s12, 0
	s_waitcnt lgkmcnt(0)
	s_load_dwordx2 s[14:15], s[8:9], 0x0
	s_add_i32 s3, s3, -1
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s13, s15, s14
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
	s_subb_u32 s16, s18, 0
	s_cmp_lt_i32 s3, 1
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_lt_i32 s2, s16
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[14:15], s[14:15], s[18:19]
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB0_8
	s_mov_b32 s22, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s12, s18
.LBB0_3:
	s_sub_i32 s33, s33, s16
	s_and_b64 s[14:15], s[14:15], exec
	s_cselect_b32 s22, 0, s17
	s_cmp_ge_i32 s12, s3
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_lt_i32 s33, s72
	s_cselect_b64 s[16:17], -1, 0
	s_or_b64 s[14:15], s[14:15], s[16:17]
	s_and_b64 vcc, exec, s[14:15]
	s_mov_b32 s16, s72
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s17, s22, 1
	s_cmp_gt_i32 s17, 15
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_lt_i32 s17, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s18, s12, 1
	s_cmp_ge_i32 s18, s3
	s_mov_b32 s72, s16
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s12, s8, s12
	s_addc_u32 s13, s9, s13
	s_load_dwordx2 s[20:21], s[12:13], 0x4
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s12, s21, s20
	s_add_i32 s19, s12, 0xff
	s_ashr_i32 s12, s19, 31
	s_lshr_b32 s12, s12, 24
	s_add_i32 s12, s19, s12
	s_ashr_i32 s22, s12, 8
	s_and_b32 s12, s12, 0xffffff00
	s_cmp_lg_u32 s19, s12
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s19, 0
	s_cselect_b64 s[20:21], -1, 0
	s_and_b64 s[12:13], s[20:21], s[12:13]
	s_subb_u32 s72, s22, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s72, s16
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s72, s16
	s_mov_b32 s22, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s73, s[4:5], 0x0
	s_load_dword s74, s[6:7], 0x0
	s_cmp_ge_i32 s12, s3
	s_cbranch_scc1 .LBB0_161
	v_lshrrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v1, 31, v0
	s_movk_i32 s75, 0xe0
	v_and_or_b32 v2, v2, s75, v1
	v_lshlrev_b32_e32 v85, 6, v2
	v_lshrrev_b32_e32 v2, 6, v0
	v_lshrrev_b32_e32 v3, 2, v0
	v_mul_u32_u24_e32 v2, 0x18000, v2
	s_load_dwordx2 s[14:15], s[0:1], 0x0
	s_load_dwordx2 s[16:17], s[0:1], 0x10
	s_load_dwordx2 s[18:19], s[0:1], 0x20
	s_load_dwordx2 s[20:21], s[0:1], 0x50
	s_load_dwordx2 s[24:25], s[0:1], 0x60
	s_load_dwordx2 s[26:27], s[0:1], 0x70
	s_load_dwordx2 s[28:29], s[0:1], 0xa0
	s_load_dwordx2 s[30:31], s[0:1], 0xb0
	s_load_dwordx2 s[34:35], s[0:1], 0xc0
	v_and_or_b32 v2, v3, 8, v2
	s_movk_i32 s0, 0xc00
	v_mad_u32_u24 v116, v1, s0, v2
	s_mov_b32 s0, 0xaaaaaab
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
	v_lshl_or_b32 v3, v3, 9, v5
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_u32_e32 v3, v3, v4
	v_and_or_b32 v117, v2, 12, v3
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v119, v2, v3
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_or_b32_e32 v118, 0x80, v117
	v_or_b32_e32 v120, 0x100, v117
	s_movk_i32 s76, 0x180
	v_or_b32_e32 v121, 0x180, v117
	v_xor_b32_e32 v122, 0x80, v2
	v_mov_b32_e32 v123, 0
	s_mov_b32 s7, 0x27000
	v_mov_b32_e32 v124, 0x40e00000
	v_mov_b32_e32 v125, 1.0
	s_mov_b32 s77, 0x7060302
	s_movk_i32 s78, 0x1000
	s_mov_b32 s79, 0x800000
	s_movk_i32 s80, 0xf80
	s_mov_b32 s81, s2
	v_mov_b32_e32 v126, 0xff800000
	v_mov_b32_e32 v127, 0x42000000
	s_mov_b32 s36, 0
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s72, s6
.LBB0_12:
	s_cmp_ge_i32 s12, s3
	s_cbranch_scc1 .LBB0_161
.LBB0_13:
	s_ashr_i32 s13, s12, 31
	s_lshl_b32 s86, s33, 8
	s_lshl_b64 s[0:1], s[12:13], 2
	s_add_u32 s4, s8, s0
	s_addc_u32 s5, s9, s1
	global_load_dwordx2 v[2:3], v123, s[4:5]
	s_mul_i32 s4, s22, 0xc0
	v_add_lshl_u32 v7, s4, v116, 1
	s_mov_b32 s47, s7
	v_lshl_add_u32 v6, s22, 2, v85
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s40, v2
	s_add_i32 s68, s40, s86
	v_readfirstlane_b32 s41, v3
	s_add_i32 s5, s68, 0x100
	s_min_i32 s5, s5, s41
	s_sub_i32 s13, s5, s68
	s_waitcnt lgkmcnt(0)
	s_add_u32 s38, s20, s0
	s_addc_u32 s39, s21, s1
	s_mul_i32 s4, s68, 0xc00
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s42, s68, 4
	global_load_dwordx2 v[4:5], v123, s[38:39]
	global_load_dword v3, v123, s[0:1]
	s_lshl_b64 s[0:1], s[4:5], 1
	s_add_u32 s4, s14, s0
	s_addc_u32 s0, s15, s1
	s_mul_i32 s6, s13, 0x1800
	s_and_b32 s5, s0, 0xffff
	buffer_load_dwordx4 v[132:135], v7, s[4:7], 0 offen
	buffer_load_dwordx4 v[136:139], v7, s[4:7], 0 offen offset:32
	buffer_load_dwordx4 v[140:143], v7, s[4:7], 0 offen offset:64
	buffer_load_dwordx4 v[144:147], v7, s[4:7], 0 offen offset:96
	buffer_load_dwordx4 v[148:151], v7, s[4:7], 0 offen offset:128
	buffer_load_dwordx4 v[152:155], v7, s[4:7], 0 offen offset:160
	buffer_load_dwordx4 v[156:159], v7, s[4:7], 0 offen offset:192
	buffer_load_dwordx4 v[160:163], v7, s[4:7], 0 offen offset:224
	buffer_load_dwordx4 v[164:167], v7, s[4:7], 0 offen offset:256
	buffer_load_dwordx4 v[168:171], v7, s[4:7], 0 offen offset:288
	s_ashr_i32 s43, s42, 31
	s_lshl_b64 s[0:1], s[42:43], 2
	s_add_u32 s44, s26, s0
	s_addc_u32 s0, s27, s1
	s_lshl_b32 s46, s13, 6
	s_and_b32 s45, s0, 0xffff
	buffer_load_dword v2, v6, s[44:47], 0 offen
	buffer_load_dwordx4 v[172:175], v7, s[4:7], 0 offen offset:320
	buffer_load_dwordx4 v[176:179], v7, s[4:7], 0 offen offset:352
	s_waitcnt vmcnt(14)
	v_readfirstlane_b32 s4, v4
	s_waitcnt vmcnt(13)
	v_readfirstlane_b32 s37, v3
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	v_readfirstlane_b32 s5, v5
	v_and_b32_e32 v3, 0x100, v3
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_15
	s_barrier
.LBB0_15:
	s_or_b64 exec, exec, s[0:1]
	s_ashr_i32 s23, s22, 31
	s_lshr_b32 s0, s23, 28
	s_add_i32 s0, s22, s0
	s_sub_i32 s87, s5, s4
	s_ashr_i32 s5, s0, 4
	s_and_b32 s0, s0, -16
	s_cmp_lg_u32 s22, s0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s22, 0
	s_cselect_b64 s[38:39], -1, 0
	s_and_b64 s[0:1], s[38:39], s[0:1]
	s_subb_u32 s42, s5, 0
	s_mul_i32 s0, s42, 0x6000
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s16, s0
	s_addc_u32 s1, s17, s1
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s4, s24, s4
	s_addc_u32 s5, s25, s5
	s_lshl_b32 s6, s87, 2
	s_and_b32 s5, s5, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[4:7], 0
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
	buffer_load_dword v3, off, s[4:7], 0 offset:8
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
	buffer_load_dword v3, off, s[4:7], 0 offset:12
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
	v_cmp_ne_u32_e32 vcc, s76, v3
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_17
	v_add_u32_e32 v4, s43, v117
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[4:5], off
	global_load_dwordx4 v[184:187], v[4:5], off offset:256
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
	v_cmp_ne_u32_e32 vcc, s76, v3
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_19
	v_add_u32_e32 v4, s43, v118
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[0:1]
	global_load_dwordx4 v[188:191], v[4:5], off
	global_load_dwordx4 v[192:195], v[4:5], off offset:256
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
	v_cmp_ne_u32_e32 vcc, s76, v3
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_21
	s_waitcnt vmcnt(1)
	ds_write_b128 v119, v[180:183] offset:12288
	s_waitcnt vmcnt(0)
	ds_write_b128 v119, v[184:187] offset:18432
.LBB0_21:
	s_or_b64 exec, exec, s[38:39]
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s76, v3
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_23
	v_add_u32_e32 v4, s43, v120
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[4:5], off
	global_load_dwordx4 v[184:187], v[4:5], off offset:256
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
	ds_read_b128 v[196:199], v3 offset:12288
	v_add_u32_e32 v3, 0x1810, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[200:203], v3
	v_add_u32_e32 v3, 0x1820, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[204:207], v3
	v_add_u32_e32 v3, 0x1830, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[208:211], v3
	v_add_u32_e32 v3, 0x1840, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[212:215], v3
	v_add_u32_e32 v3, 0x1850, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[216:219], v3
	v_add_u32_e32 v3, 0x1860, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[220:223], v3
	v_add_u32_e32 v3, 0x1870, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[224:227], v3
	v_add_u32_e32 v3, 0x1880, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[228:231], v3
	v_add_u32_e32 v3, 0x1890, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	ds_read_b128 v[232:235], v3
	v_add_u32_e32 v3, 0x18a0, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_u32_e32 v2, 0x18b0, v2
	ds_read_b128 v[236:239], v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[240:243], v2
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
	v_mov_b32_e32 v129, 0
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
	s_mov_b64 s[38:39], 0
	s_mov_b32 s37, 20
	v_mov_b32_e32 v84, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v129
	v_mov_b32_e32 v4, v129
	v_mov_b32_e32 v5, v129
	v_mov_b32_e32 v6, v129
	v_mov_b32_e32 v7, v129
	v_mov_b32_e32 v8, v129
	v_mov_b32_e32 v9, v129
	v_mov_b32_e32 v10, v129
	v_mov_b32_e32 v11, v129
	v_mov_b32_e32 v12, v129
	v_mov_b32_e32 v13, v129
	v_mov_b32_e32 v14, v129
	v_mov_b32_e32 v15, v129
	v_mov_b32_e32 v16, v129
	v_mov_b32_e32 v17, v129
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v129
	v_mov_b32_e32 v20, v129
	v_mov_b32_e32 v21, v129
	v_mov_b32_e32 v22, v129
	v_mov_b32_e32 v23, v129
	v_mov_b32_e32 v24, v129
	v_mov_b32_e32 v25, v129
	v_mov_b32_e32 v26, v129
	v_mov_b32_e32 v27, v129
	v_mov_b32_e32 v28, v129
	v_mov_b32_e32 v29, v129
	v_mov_b32_e32 v30, v129
	v_mov_b32_e32 v31, v129
	v_mov_b32_e32 v32, v129
	v_mov_b32_e32 v33, v129
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v129
	v_mov_b32_e32 v36, v129
	v_mov_b32_e32 v37, v129
	v_mov_b32_e32 v38, v129
	v_mov_b32_e32 v39, v129
	v_mov_b32_e32 v40, v129
	v_mov_b32_e32 v41, v129
	v_mov_b32_e32 v42, v129
	v_mov_b32_e32 v43, v129
	v_mov_b32_e32 v44, v129
	v_mov_b32_e32 v45, v129
	v_mov_b32_e32 v46, v129
	v_mov_b32_e32 v47, v129
	v_mov_b32_e32 v48, v129
	v_mov_b32_e32 v49, v129
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v129
	v_mov_b32_e32 v52, v129
	v_mov_b32_e32 v53, v129
	v_mov_b32_e32 v54, v129
	v_mov_b32_e32 v55, v129
	v_mov_b32_e32 v56, v129
	v_mov_b32_e32 v57, v129
	v_mov_b32_e32 v58, v129
	v_mov_b32_e32 v59, v129
	v_mov_b32_e32 v60, v129
	v_mov_b32_e32 v61, v129
	v_mov_b32_e32 v62, v129
	v_mov_b32_e32 v63, v129
	v_mov_b32_e32 v64, v129
	v_mov_b32_e32 v65, v129
	s_branch .LBB0_26
.LBB0_25:
	s_or_b64 exec, exec, s[40:41]
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
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v100, v100, v79
	v_add_f32_e32 v100, v100, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v100, v100, v81
	;;#ASMSTART
	v_fma_f32 v129, v128, v130, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v130
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
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
	v_add_co_u32_e32 v66, vcc, s78, v74
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
	ds_read_b128 v[196:199], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
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
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[232:235], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[236:239], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[240:243], v66
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
	buffer_load_dword v66, v66, s[4:7], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_mov_b32 s42, s69
	v_and_b32_e32 v67, 0x180, v67
	s_mov_b32 s43, s85
	v_cmp_ne_u32_e32 vcc, s76, v67
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s69, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_28
	ds_write_b128 v119, v[188:191]
	ds_write_b128 v119, v[192:195] offset:6144
.LBB0_28:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_30
	s_mul_i32 s44, s90, 0x3000
	v_add_u32_e32 v66, s44, v121
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[188:191], v[66:67], off
	global_load_dwordx4 v[192:195], v[66:67], off offset:256
.LBB0_30:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[178:179], v[66:81]
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
	ds_bpermute_b32 v101, v122, v100
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v101, v100, v101
	;;#ASMSTART
	v_add_f32 v100, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v101, v100
	v_mov_b32_e32 v100, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v101, v101, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v101
	v_exp_f32_e32 v100, v84
	v_mov_b32_e32 v84, v101
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
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v101, v101, v79
	v_add_f32_e32 v101, v101, v80
	v_add_f32_e32 v101, v101, v81
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	;;#ASMSTART
	v_fma_f32 v128, v129, v100, v101
	;;#ASMEND
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
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
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
	v_add_co_u32_e32 v66, vcc, s78, v74
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
	ds_read_b128 v[224:227], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[220:223], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[216:219], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[196:199], v67
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
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_34
	ds_write_b128 v119, v[180:183] offset:12288
	ds_write_b128 v119, v[184:187] offset:18432
.LBB0_34:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s44, s84, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_36
	v_add_u32_e32 v66, s44, v117
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[66:67], off
	global_load_dwordx4 v[184:187], v[66:67], off offset:256
.LBB0_36:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[114:115], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[178:179], v[66:81]
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
	ds_bpermute_b32 v101, v122, v100
	v_mov_b32_e32 v129, 1.0
	v_mov_b32_e32 v102, v84
	v_mov_b32_e32 v103, v84
	v_mov_b32_e32 v104, v84
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v130, v100, v101
	;;#ASMSTART
	v_add_f32 v100, v84, v124
	;;#ASMEND
	v_mov_b32_e32 v101, v84
	v_cmp_gt_f32_e32 vcc, v130, v100
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
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_38
	;;#ASMSTART
	v_add_f32 v100, v130, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v129, v84
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
.LBB0_38:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, 0, v66
	v_add_f32_e32 v130, v130, v67
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
	v_add_f32_e32 v130, v130, v68
	v_add_f32_e32 v130, v130, v69
	v_add_f32_e32 v130, v130, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v71
	v_add_f32_e32 v130, v130, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v73
	v_add_f32_e32 v130, v130, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v75
	v_add_f32_e32 v130, v130, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v77
	v_add_f32_e32 v130, v130, v78
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
	v_add_f32_e32 v130, v130, v79
	v_add_f32_e32 v130, v130, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v130, v130, v81
	;;#ASMSTART
	v_fma_f32 v128, v128, v129, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v129
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[236:239], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[232:235], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
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
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_40
	ds_write_b128 v119, v[188:191]
	ds_write_b128 v119, v[192:195] offset:6144
.LBB0_40:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_42
	v_add_u32_e32 v66, s44, v118
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[188:191], v[66:67], off
	global_load_dwordx4 v[192:195], v[66:67], off offset:256
.LBB0_42:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v129, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v129, v129, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v129, v129, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v129, v129, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v129, v129, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v129, v129, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v129, v129, v78, v79
	v_max3_f32 v129, v129, v80, v81
	ds_bpermute_b32 v130, v122, v129
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v130, v130, v130
	v_max_f32_e32 v130, v129, v130
	;;#ASMSTART
	v_add_f32 v129, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v130, v129
	v_mov_b32_e32 v129, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_44
	;;#ASMSTART
	v_add_f32 v100, v130, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v129, v84
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
.LBB0_44:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, 0, v66
	v_add_f32_e32 v130, v130, v67
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
	v_add_f32_e32 v130, v130, v68
	v_add_f32_e32 v130, v130, v69
	v_add_f32_e32 v130, v130, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v71
	v_add_f32_e32 v130, v130, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v73
	v_add_f32_e32 v130, v130, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v75
	v_add_f32_e32 v130, v130, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v77
	v_add_f32_e32 v130, v130, v78
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
	v_add_f32_e32 v130, v130, v79
	v_add_f32_e32 v130, v130, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v130, v130, v81
	;;#ASMSTART
	v_fma_f32 v128, v128, v129, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v129
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[236:239], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[232:235], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_46
	ds_write_b128 v119, v[180:183] offset:12288
	ds_write_b128 v119, v[184:187] offset:18432
.LBB0_46:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_48
	v_add_u32_e32 v66, s44, v120
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[66:67], off
	global_load_dwordx4 v[184:187], v[66:67], off offset:256
.LBB0_48:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v129, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v129, v129, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v129, v129, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v129, v129, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v129, v129, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v129, v129, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v129, v129, v78, v79
	v_max3_f32 v129, v129, v80, v81
	ds_bpermute_b32 v130, v122, v129
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v130, v130, v130
	v_max_f32_e32 v130, v129, v130
	;;#ASMSTART
	v_add_f32 v129, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v130, v129
	v_mov_b32_e32 v129, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_50
	;;#ASMSTART
	v_add_f32 v100, v130, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v129, v84
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
.LBB0_50:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, 0, v66
	v_add_f32_e32 v130, v130, v67
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
	v_add_f32_e32 v130, v130, v68
	v_add_f32_e32 v130, v130, v69
	v_add_f32_e32 v130, v130, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v71
	v_add_f32_e32 v130, v130, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v73
	v_add_f32_e32 v130, v130, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v75
	v_add_f32_e32 v130, v130, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v77
	v_add_f32_e32 v130, v130, v78
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
	v_add_f32_e32 v130, v130, v79
	v_add_f32_e32 v130, v130, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v130, v130, v81
	;;#ASMSTART
	v_fma_f32 v128, v128, v129, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v129
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[236:239], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[232:235], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
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
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	v_mov_b32_e32 v66, s37
	buffer_load_dword v66, v66, s[4:7], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s85, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s76, v67
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_52
	ds_write_b128 v119, v[188:191]
	ds_write_b128 v119, v[192:195] offset:6144
.LBB0_52:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_54
	v_add_u32_e32 v66, s44, v121
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[188:191], v[66:67], off
	global_load_dwordx4 v[192:195], v[66:67], off offset:256
.LBB0_54:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v129, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v129, v129, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v129, v129, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v129, v129, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v129, v129, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v129, v129, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v129, v129, v78, v79
	v_max3_f32 v129, v129, v80, v81
	ds_bpermute_b32 v130, v122, v129
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v130, v130, v130
	v_max_f32_e32 v130, v129, v130
	;;#ASMSTART
	v_add_f32 v129, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v130, v129
	v_mov_b32_e32 v129, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_56
	;;#ASMSTART
	v_add_f32 v100, v130, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v129, v84
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
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, 0, v66
	v_add_f32_e32 v130, v130, v67
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
	v_add_f32_e32 v130, v130, v68
	v_add_f32_e32 v130, v130, v69
	v_add_f32_e32 v130, v130, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v71
	v_add_f32_e32 v130, v130, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v73
	v_add_f32_e32 v130, v130, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v75
	v_add_f32_e32 v130, v130, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v77
	v_add_f32_e32 v130, v130, v78
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
	v_add_f32_e32 v130, v130, v79
	v_add_f32_e32 v130, v130, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v130, v130, v81
	;;#ASMSTART
	v_fma_f32 v128, v128, v129, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v129
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[236:239], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[232:235], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_58
	ds_write_b128 v119, v[180:183] offset:12288
	ds_write_b128 v119, v[184:187] offset:18432
.LBB0_58:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s44, s42, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_60
	v_add_u32_e32 v66, s44, v117
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[66:67], off
	global_load_dwordx4 v[184:187], v[66:67], off offset:256
.LBB0_60:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v129, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v129, v129, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v129, v129, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v129, v129, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v129, v129, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v129, v129, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v129, v129, v78, v79
	v_max3_f32 v129, v129, v80, v81
	ds_bpermute_b32 v130, v122, v129
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v130, v130, v130
	v_max_f32_e32 v130, v129, v130
	;;#ASMSTART
	v_add_f32 v129, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v130, v129
	v_mov_b32_e32 v129, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_62
	;;#ASMSTART
	v_add_f32 v100, v130, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v129, v84
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
.LBB0_62:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, 0, v66
	v_add_f32_e32 v130, v130, v67
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
	v_add_f32_e32 v130, v130, v68
	v_add_f32_e32 v130, v130, v69
	v_add_f32_e32 v130, v130, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v71
	v_add_f32_e32 v130, v130, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v73
	v_add_f32_e32 v130, v130, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v75
	v_add_f32_e32 v130, v130, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v77
	v_add_f32_e32 v130, v130, v78
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
	v_add_f32_e32 v130, v130, v79
	v_add_f32_e32 v130, v130, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v130, v130, v81
	;;#ASMSTART
	v_fma_f32 v128, v128, v129, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v129
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[236:239], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[232:235], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
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
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_64
	ds_write_b128 v119, v[188:191]
	ds_write_b128 v119, v[192:195] offset:6144
.LBB0_64:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_66
	v_add_u32_e32 v66, s44, v118
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[188:191], v[66:67], off
	global_load_dwordx4 v[192:195], v[66:67], off offset:256
.LBB0_66:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v129, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v129, v129, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v129, v129, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v129, v129, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v129, v129, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v129, v129, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v129, v129, v78, v79
	v_max3_f32 v129, v129, v80, v81
	ds_bpermute_b32 v130, v122, v129
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v130, v130, v130
	v_max_f32_e32 v130, v129, v130
	;;#ASMSTART
	v_add_f32 v129, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v130, v129
	v_mov_b32_e32 v129, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_68
	;;#ASMSTART
	v_add_f32 v100, v130, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v129, v84
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
.LBB0_68:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, 0, v66
	v_add_f32_e32 v130, v130, v67
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
	v_add_f32_e32 v130, v130, v68
	v_add_f32_e32 v130, v130, v69
	v_add_f32_e32 v130, v130, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v71
	v_add_f32_e32 v130, v130, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v73
	v_add_f32_e32 v130, v130, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v75
	v_add_f32_e32 v130, v130, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v130, v130, v77
	v_add_f32_e32 v130, v130, v78
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
	v_add_f32_e32 v130, v130, v79
	v_add_f32_e32 v130, v130, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v130, v130, v81
	;;#ASMSTART
	v_fma_f32 v128, v128, v129, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v129
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v129
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[236:239], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[232:235], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_70
	ds_write_b128 v119, v[180:183] offset:12288
	ds_write_b128 v119, v[184:187] offset:18432
.LBB0_70:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_72
	v_add_u32_e32 v66, s44, v120
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[66:67], off
	global_load_dwordx4 v[184:187], v[66:67], off offset:256
.LBB0_72:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v129, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v129, v129, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v129, v129, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v129, v129, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v129, v129, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v129, v129, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v129, v129, v78, v79
	v_max3_f32 v129, v129, v80, v81
	ds_bpermute_b32 v130, v122, v129
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v130, v130, v130
	v_max_f32_e32 v129, v129, v130
	;;#ASMSTART
	v_add_f32 v130, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v129, v130
	v_mov_b32_e32 v130, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_25
	;;#ASMSTART
	v_add_f32 v100, v129, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v130, v84
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
	v_mov_b32_e32 v129, 0
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
	s_add_i32 s37, s13, s89
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
	v_or_b32_e32 v128, s86, v1
	s_addk_i32 s37, 0xf7
	s_lshl3_add_u32 s44, s88, 20
	s_branch .LBB0_80
.LBB0_78:
	s_or_b64 exec, exec, s[42:43]
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
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_add_u32_e32 v73, 0x8000, v73
	v_add_f32_e32 v100, v100, v79
	v_add_f32_e32 v100, v100, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v100, v100, v81
	;;#ASMSTART
	v_fma_f32 v129, v129, v130, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v130
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
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
	v_add_co_u32_e32 v66, vcc, s78, v74
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
	ds_read_b128 v[196:199], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
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
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[232:235], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[236:239], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[240:243], v66
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
	buffer_load_dword v66, v66, s[4:7], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s76, v67
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_82
	ds_write_b128 v119, v[188:191]
	ds_write_b128 v119, v[192:195] offset:6144
.LBB0_82:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_84
	s_mul_i32 s42, s90, 0x3000
	v_add_u32_e32 v66, s42, v121
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[188:191], v[66:67], off
	global_load_dwordx4 v[192:195], v[66:67], off offset:256
.LBB0_84:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v100, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v101, 2, v100
	v_and_b32_e32 v101, 8, v101
	v_lshrrev_b32_e32 v100, 1, v100
	v_and_or_b32 v100, v100, s75, v128
	v_add_u32_e32 v109, s37, v101
	v_add_u32_e32 v108, s83, v100
	v_add_u32_e32 v100, 0xffffff09, v109
	v_cmp_lt_i32_e32 vcc, v100, v108
	s_nop 1
	v_cndmask_b32_e32 v101, v126, v67, vcc
	v_cmp_le_i32_e32 vcc, v100, v108
	v_add_u32_e32 v67, 0xffffff20, v109
	s_nop 0
	v_cndmask_b32_e32 v100, v126, v66, vcc
	v_add_u32_e32 v66, 0xffffff0b, v109
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff0c, v109
	s_nop 0
	v_cndmask_b32_e32 v102, v126, v68, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff0d, v109
	s_nop 0
	v_cndmask_b32_e32 v103, v126, v69, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff0e, v109
	s_nop 0
	v_cndmask_b32_e32 v104, v126, v70, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff0f, v109
	s_nop 0
	v_cndmask_b32_e32 v105, v126, v71, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff10, v109
	s_nop 0
	v_cndmask_b32_e32 v106, v126, v72, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff19, v109
	s_nop 0
	v_cndmask_b32_e32 v107, v126, v73, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff1a, v109
	s_nop 0
	v_cndmask_b32_e32 v72, v126, v74, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff1b, v109
	s_nop 0
	v_cndmask_b32_e32 v73, v126, v75, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff1c, v109
	v_pk_mul_f32 v[74:75], v[90:91], v[106:107]
	v_cndmask_b32_e32 v70, v126, v76, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff1d, v109
	v_pk_mul_f32 v[72:73], v[92:93], v[72:73]
	v_cndmask_b32_e32 v71, v126, v77, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff1e, v109
	v_pk_mul_f32 v[76:77], v[88:89], v[104:105]
	v_cndmask_b32_e32 v68, v126, v78, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_add_u32_e32 v66, 0xffffff1f, v109
	v_pk_mul_f32 v[70:71], v[94:95], v[70:71]
	v_cndmask_b32_e32 v69, v126, v79, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_pk_mul_f32 v[78:79], v[86:87], v[102:103]
	v_pk_mul_f32 v[68:69], v[96:97], v[68:69]
	v_cndmask_b32_e32 v66, v126, v80, vcc
	v_cmp_le_i32_e32 vcc, v67, v108
	s_nop 1
	v_cndmask_b32_e32 v67, v126, v81, vcc
	v_pk_mul_f32 v[80:81], v[82:83], v[100:101]
	v_pk_mul_f32 v[66:67], v[98:99], v[66:67]
	v_max_f32_e32 v100, v80, v81
	v_max3_f32 v100, v100, v78, v79
	v_max3_f32 v100, v100, v76, v77
	v_max3_f32 v100, v100, v74, v75
	v_max3_f32 v100, v100, v72, v73
	v_max3_f32 v100, v100, v70, v71
	v_max3_f32 v100, v100, v68, v69
	v_max3_f32 v100, v100, v66, v67
	ds_bpermute_b32 v101, v122, v100
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v101, v100, v101
	;;#ASMSTART
	v_add_f32 v100, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v101, v100
	v_mov_b32_e32 v100, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_86
	;;#ASMSTART
	v_add_f32 v101, v101, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v101
	v_exp_f32_e32 v100, v84
	v_mov_b32_e32 v84, v101
.LBB0_86:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[80:81], v[80:81], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[78:79], v[78:79], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, 0, v80
	v_add_f32_e32 v101, v101, v81
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v78
	v_add_f32_e32 v101, v101, v79
	v_add_f32_e32 v101, v101, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v77
	v_add_f32_e32 v101, v101, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v75
	v_add_f32_e32 v101, v101, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[68:69], v[68:69], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v73
	v_add_f32_e32 v101, v101, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	v_pk_add_f32 v[66:67], v[66:67], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v71
	v_add_f32_e32 v101, v101, v68
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v100
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v100
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v101, v101, v69
	v_add_f32_e32 v101, v101, v66
	v_add_f32_e32 v101, v101, v67
	;;#ASMSTART
	v_fma_f32 v129, v129, v100, v101
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
	v_add_u32_e32 v100, 0x8000, v67
	v_add_u32_e32 v101, 0x8000, v66
	v_add_u32_e32 v102, 0x8000, v69
	v_add_u32_e32 v103, 0x8000, v68
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v104, 0x8000, v70
	v_add_u32_e32 v70, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v69, 0x8000, v75
	v_add_u32_e32 v73, 0x8000, v74
	v_add_u32_e32 v68, 0x8000, v77
	v_add_u32_e32 v67, 0x8000, v79
	v_add_u32_e32 v66, 0x8000, v81
	v_add_u32_e32 v74, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v78
	v_add_u32_e32 v76, 0x8000, v80
	;;#ASMSTART
	v_perm_b32 v66, v66, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v67, v75, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v68, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v69, v73, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v70, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v71, v104, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v102, v103, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v100, v101, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
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
	v_add_co_u32_e32 v66, vcc, s78, v74
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
	ds_read_b128 v[224:227], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[220:223], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[216:219], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[196:199], v67
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
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_88
	ds_write_b128 v119, v[180:183] offset:12288
	ds_write_b128 v119, v[184:187] offset:18432
.LBB0_88:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s47, s84, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_90
	v_add_u32_e32 v66, s47, v117
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[66:67], off
	global_load_dwordx4 v[184:187], v[66:67], off offset:256
.LBB0_90:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[114:115], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v100, v0
	;;#ASMEND
	v_mov_b32_e32 v130, 1.0
	v_lshrrev_b32_e32 v101, 2, v100
	v_and_b32_e32 v101, 8, v101
	v_lshrrev_b32_e32 v100, 1, v100
	v_and_or_b32 v100, v100, s75, v128
	v_add_u32_e32 v101, s37, v101
	v_add_u32_e32 v100, s83, v100
	v_add_u32_e32 v102, 0xffffff29, v101
	v_cmp_lt_i32_e32 vcc, v102, v100
	v_mov_b32_e32 v103, v84
	v_mov_b32_e32 v104, v84
	v_cndmask_b32_e32 v67, v126, v67, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff2b, v101
	v_mov_b32_e32 v105, v84
	v_cndmask_b32_e32 v66, v126, v66, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff2c, v101
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e32 v68, v126, v68, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff2d, v101
	v_mov_b32_e32 v106, v84
	v_cndmask_b32_e32 v69, v126, v69, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff2e, v101
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_cndmask_b32_e32 v70, v126, v70, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff2f, v101
	v_mov_b32_e32 v107, v84
	v_cndmask_b32_e32 v71, v126, v71, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff30, v101
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_cndmask_b32_e32 v72, v126, v72, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff39, v101
	v_mov_b32_e32 v108, v84
	v_cndmask_b32_e32 v73, v126, v73, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff3a, v101
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_cndmask_b32_e32 v74, v126, v74, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff3b, v101
	v_mov_b32_e32 v109, v84
	v_cndmask_b32_e32 v75, v126, v75, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff3c, v101
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_cndmask_b32_e32 v76, v126, v76, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff3d, v101
	v_mov_b32_e32 v110, v84
	v_cndmask_b32_e32 v77, v126, v77, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff3e, v101
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_cndmask_b32_e32 v78, v126, v78, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, 0xffffff3f, v101
	v_add_u32_e32 v101, 0xffffff40, v101
	v_cndmask_b32_e32 v79, v126, v79, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_mov_b32_e32 v102, v84
	v_cndmask_b32_e32 v80, v126, v80, vcc
	v_cmp_le_i32_e32 vcc, v101, v100
	v_max_f32_e32 v100, v66, v67
	v_max3_f32 v100, v100, v68, v69
	v_max3_f32 v100, v100, v70, v71
	v_max3_f32 v100, v100, v72, v73
	v_max3_f32 v100, v100, v74, v75
	v_cndmask_b32_e32 v81, v126, v81, vcc
	v_max3_f32 v100, v100, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v100, v100, v78, v79
	v_max3_f32 v100, v100, v80, v81
	ds_bpermute_b32 v101, v122, v100
	v_mov_b32_e32 v111, v84
	v_mov_b32_e32 v112, v84
	v_mov_b32_e32 v113, v84
	v_mov_b32_e32 v114, v84
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v131, v100, v101
	;;#ASMSTART
	v_add_f32 v100, v84, v124
	;;#ASMEND
	v_mov_b32_e32 v101, v84
	v_cmp_gt_f32_e32 vcc, v131, v100
	v_mov_b32_e32 v100, v84
	v_mov_b32_e32 v115, v84
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_92
	;;#ASMSTART
	v_add_f32 v100, v131, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v130, v84
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
.LBB0_92:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, 0, v66
	v_add_f32_e32 v131, v131, v67
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
	v_add_f32_e32 v131, v131, v68
	v_add_f32_e32 v131, v131, v69
	v_add_f32_e32 v131, v131, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v71
	v_add_f32_e32 v131, v131, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v73
	v_add_f32_e32 v131, v131, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v75
	v_add_f32_e32 v131, v131, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v77
	v_add_f32_e32 v131, v131, v78
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
	v_add_f32_e32 v131, v131, v79
	v_add_f32_e32 v131, v131, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v131, v131, v81
	;;#ASMSTART
	v_fma_f32 v129, v129, v130, v131
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v130
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[236:239], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[232:235], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
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
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_94
	ds_write_b128 v119, v[188:191]
	ds_write_b128 v119, v[192:195] offset:6144
.LBB0_94:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_96
	v_add_u32_e32 v66, s47, v118
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[188:191], v[66:67], off
	global_load_dwordx4 v[192:195], v[66:67], off offset:256
.LBB0_96:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v130, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v131, 2, v130
	v_and_b32_e32 v131, 8, v131
	v_lshrrev_b32_e32 v130, 1, v130
	v_and_or_b32 v130, v130, s75, v128
	v_add_u32_e32 v131, s37, v131
	v_add_u32_e32 v130, s83, v130
	v_add_u32_e32 v196, 0xffffff49, v131
	v_cmp_lt_i32_e32 vcc, v196, v130
	s_nop 1
	v_cndmask_b32_e32 v67, v126, v67, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff4b, v131
	s_nop 0
	v_cndmask_b32_e32 v66, v126, v66, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff4c, v131
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e32 v68, v126, v68, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff4d, v131
	s_nop 0
	v_cndmask_b32_e32 v69, v126, v69, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff4e, v131
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_cndmask_b32_e32 v70, v126, v70, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff4f, v131
	s_nop 0
	v_cndmask_b32_e32 v71, v126, v71, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff50, v131
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_cndmask_b32_e32 v72, v126, v72, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff59, v131
	s_nop 0
	v_cndmask_b32_e32 v73, v126, v73, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff5a, v131
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_cndmask_b32_e32 v74, v126, v74, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff5b, v131
	s_nop 0
	v_cndmask_b32_e32 v75, v126, v75, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff5c, v131
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_cndmask_b32_e32 v76, v126, v76, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff5d, v131
	s_nop 0
	v_cndmask_b32_e32 v77, v126, v77, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff5e, v131
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_cndmask_b32_e32 v78, v126, v78, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff5f, v131
	v_add_u32_e32 v131, 0xffffff60, v131
	v_cndmask_b32_e32 v79, v126, v79, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	s_nop 0
	v_cndmask_b32_e32 v80, v126, v80, vcc
	v_cmp_le_i32_e32 vcc, v131, v130
	v_max_f32_e32 v130, v66, v67
	v_max3_f32 v130, v130, v68, v69
	v_max3_f32 v130, v130, v70, v71
	v_max3_f32 v130, v130, v72, v73
	v_max3_f32 v130, v130, v74, v75
	v_cndmask_b32_e32 v81, v126, v81, vcc
	v_max3_f32 v130, v130, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v130, v130, v78, v79
	v_max3_f32 v130, v130, v80, v81
	ds_bpermute_b32 v131, v122, v130
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v131, v131, v131
	v_max_f32_e32 v131, v130, v131
	;;#ASMSTART
	v_add_f32 v130, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v131, v130
	v_mov_b32_e32 v130, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_98
	;;#ASMSTART
	v_add_f32 v100, v131, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v130, v84
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
.LBB0_98:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, 0, v66
	v_add_f32_e32 v131, v131, v67
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
	v_add_f32_e32 v131, v131, v68
	v_add_f32_e32 v131, v131, v69
	v_add_f32_e32 v131, v131, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v71
	v_add_f32_e32 v131, v131, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v73
	v_add_f32_e32 v131, v131, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v75
	v_add_f32_e32 v131, v131, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v77
	v_add_f32_e32 v131, v131, v78
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
	v_add_f32_e32 v131, v131, v79
	v_add_f32_e32 v131, v131, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v131, v131, v81
	;;#ASMSTART
	v_fma_f32 v129, v129, v130, v131
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v130
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[236:239], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[232:235], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_100
	ds_write_b128 v119, v[180:183] offset:12288
	ds_write_b128 v119, v[184:187] offset:18432
.LBB0_100:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_102
	v_add_u32_e32 v66, s47, v120
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[66:67], off
	global_load_dwordx4 v[184:187], v[66:67], off offset:256
.LBB0_102:
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v130, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v131, 2, v130
	v_and_b32_e32 v131, 8, v131
	v_lshrrev_b32_e32 v130, 1, v130
	v_and_or_b32 v130, v130, s75, v128
	v_add_u32_e32 v131, s37, v131
	v_add_u32_e32 v130, s83, v130
	v_add_u32_e32 v196, 0xffffff69, v131
	v_cmp_lt_i32_e32 vcc, v196, v130
	s_nop 1
	v_cndmask_b32_e32 v67, v126, v67, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff6b, v131
	s_nop 0
	v_cndmask_b32_e32 v66, v126, v66, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff6c, v131
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e32 v68, v126, v68, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff6d, v131
	s_nop 0
	v_cndmask_b32_e32 v69, v126, v69, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff6e, v131
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_cndmask_b32_e32 v70, v126, v70, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff6f, v131
	s_nop 0
	v_cndmask_b32_e32 v71, v126, v71, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff70, v131
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_cndmask_b32_e32 v72, v126, v72, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff79, v131
	s_nop 0
	v_cndmask_b32_e32 v73, v126, v73, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff7a, v131
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_cndmask_b32_e32 v74, v126, v74, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff7b, v131
	s_nop 0
	v_cndmask_b32_e32 v75, v126, v75, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff7c, v131
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_cndmask_b32_e32 v76, v126, v76, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff7d, v131
	s_nop 0
	v_cndmask_b32_e32 v77, v126, v77, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff7e, v131
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_cndmask_b32_e32 v78, v126, v78, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff7f, v131
	v_add_u32_e32 v131, 0xffffff80, v131
	v_cndmask_b32_e32 v79, v126, v79, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	s_nop 0
	v_cndmask_b32_e32 v80, v126, v80, vcc
	v_cmp_le_i32_e32 vcc, v131, v130
	v_max_f32_e32 v130, v66, v67
	v_max3_f32 v130, v130, v68, v69
	v_max3_f32 v130, v130, v70, v71
	v_max3_f32 v130, v130, v72, v73
	v_max3_f32 v130, v130, v74, v75
	v_cndmask_b32_e32 v81, v126, v81, vcc
	v_max3_f32 v130, v130, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v130, v130, v78, v79
	v_max3_f32 v130, v130, v80, v81
	ds_bpermute_b32 v131, v122, v130
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v131, v131, v131
	v_max_f32_e32 v131, v130, v131
	;;#ASMSTART
	v_add_f32 v130, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v131, v130
	v_mov_b32_e32 v130, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_104
	;;#ASMSTART
	v_add_f32 v100, v131, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v130, v84
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
.LBB0_104:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, 0, v66
	v_add_f32_e32 v131, v131, v67
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
	v_add_f32_e32 v131, v131, v68
	v_add_f32_e32 v131, v131, v69
	v_add_f32_e32 v131, v131, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v71
	v_add_f32_e32 v131, v131, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v73
	v_add_f32_e32 v131, v131, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v75
	v_add_f32_e32 v131, v131, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v77
	v_add_f32_e32 v131, v131, v78
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
	v_add_f32_e32 v131, v131, v79
	v_add_f32_e32 v131, v131, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v131, v131, v81
	;;#ASMSTART
	v_fma_f32 v129, v129, v130, v131
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v130
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[196:199], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[200:203], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
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
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x1870, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[232:235], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[236:239], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[240:243], v66
	s_add_i32 s42, s70, 1
	s_cmp_gt_i32 s38, s42
	s_cselect_b64 s[40:41], -1, 0
	s_cmp_le_i32 s38, s42
	s_cbranch_scc1 .LBB0_129
	v_mov_b32_e32 v66, s44
	buffer_load_dword v66, v66, s[4:7], 0 offen
	;;#ASMSTART
	v_mov_b32 v67, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s46, v66
	v_and_b32_e32 v67, 0x180, v67
	v_cmp_ne_u32_e32 vcc, s76, v67
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_107
	ds_write_b128 v119, v[188:191]
	ds_write_b128 v119, v[192:195] offset:6144
.LBB0_107:
	s_or_b64 exec, exec, s[42:43]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_109
	v_add_u32_e32 v66, s47, v121
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[188:191], v[66:67], off
	global_load_dwordx4 v[192:195], v[66:67], off offset:256
.LBB0_109:
	s_or_b64 exec, exec, s[42:43]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v130, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v131, 2, v130
	v_and_b32_e32 v131, 8, v131
	v_lshrrev_b32_e32 v130, 1, v130
	v_and_or_b32 v130, v130, s75, v128
	v_add_u32_e32 v131, s37, v131
	v_add_u32_e32 v130, s83, v130
	v_add_u32_e32 v196, 0xffffff89, v131
	v_cmp_lt_i32_e32 vcc, v196, v130
	s_nop 1
	v_cndmask_b32_e32 v67, v126, v67, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff8b, v131
	s_nop 0
	v_cndmask_b32_e32 v66, v126, v66, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff8c, v131
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e32 v68, v126, v68, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff8d, v131
	s_nop 0
	v_cndmask_b32_e32 v69, v126, v69, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff8e, v131
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_cndmask_b32_e32 v70, v126, v70, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff8f, v131
	s_nop 0
	v_cndmask_b32_e32 v71, v126, v71, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff90, v131
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_cndmask_b32_e32 v72, v126, v72, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff99, v131
	s_nop 0
	v_cndmask_b32_e32 v73, v126, v73, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff9a, v131
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_cndmask_b32_e32 v74, v126, v74, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff9b, v131
	s_nop 0
	v_cndmask_b32_e32 v75, v126, v75, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff9c, v131
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_cndmask_b32_e32 v76, v126, v76, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff9d, v131
	s_nop 0
	v_cndmask_b32_e32 v77, v126, v77, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff9e, v131
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_cndmask_b32_e32 v78, v126, v78, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffff9f, v131
	v_add_u32_e32 v131, 0xffffffa0, v131
	v_cndmask_b32_e32 v79, v126, v79, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	s_nop 0
	v_cndmask_b32_e32 v80, v126, v80, vcc
	v_cmp_le_i32_e32 vcc, v131, v130
	v_max_f32_e32 v130, v66, v67
	v_max3_f32 v130, v130, v68, v69
	v_max3_f32 v130, v130, v70, v71
	v_max3_f32 v130, v130, v72, v73
	v_max3_f32 v130, v130, v74, v75
	v_cndmask_b32_e32 v81, v126, v81, vcc
	v_max3_f32 v130, v130, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v130, v130, v78, v79
	v_max3_f32 v130, v130, v80, v81
	ds_bpermute_b32 v131, v122, v130
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v131, v131, v131
	v_max_f32_e32 v131, v130, v131
	;;#ASMSTART
	v_add_f32 v130, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v131, v130
	v_mov_b32_e32 v130, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_111
	;;#ASMSTART
	v_add_f32 v100, v131, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v130, v84
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
.LBB0_111:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, 0, v66
	v_add_f32_e32 v131, v131, v67
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
	v_add_f32_e32 v131, v131, v68
	v_add_f32_e32 v131, v131, v69
	v_add_f32_e32 v131, v131, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v71
	v_add_f32_e32 v131, v131, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v73
	v_add_f32_e32 v131, v131, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v75
	v_add_f32_e32 v131, v131, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v77
	v_add_f32_e32 v131, v131, v78
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
	v_add_f32_e32 v131, v131, v79
	v_add_f32_e32 v131, v131, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v131, v131, v81
	;;#ASMSTART
	v_fma_f32 v129, v129, v130, v131
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v130
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[236:239], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[232:235], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_113
	ds_write_b128 v119, v[180:183] offset:12288
	ds_write_b128 v119, v[184:187] offset:18432
.LBB0_113:
	s_or_b64 exec, exec, s[42:43]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_mul_i32 s47, s69, 0x3000
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_115
	v_add_u32_e32 v66, s47, v117
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[66:67], off
	global_load_dwordx4 v[184:187], v[66:67], off offset:256
.LBB0_115:
	s_or_b64 exec, exec, s[42:43]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v130, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v131, 2, v130
	v_and_b32_e32 v131, 8, v131
	v_lshrrev_b32_e32 v130, 1, v130
	v_and_or_b32 v130, v130, s75, v128
	v_add_u32_e32 v131, s37, v131
	v_add_u32_e32 v130, s83, v130
	v_add_u32_e32 v196, 0xffffffa9, v131
	v_cmp_lt_i32_e32 vcc, v196, v130
	s_nop 1
	v_cndmask_b32_e32 v67, v126, v67, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffab, v131
	s_nop 0
	v_cndmask_b32_e32 v66, v126, v66, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffac, v131
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e32 v68, v126, v68, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffad, v131
	s_nop 0
	v_cndmask_b32_e32 v69, v126, v69, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffae, v131
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_cndmask_b32_e32 v70, v126, v70, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffaf, v131
	s_nop 0
	v_cndmask_b32_e32 v71, v126, v71, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffb0, v131
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_cndmask_b32_e32 v72, v126, v72, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffb9, v131
	s_nop 0
	v_cndmask_b32_e32 v73, v126, v73, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffba, v131
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_cndmask_b32_e32 v74, v126, v74, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffbb, v131
	s_nop 0
	v_cndmask_b32_e32 v75, v126, v75, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffbc, v131
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_cndmask_b32_e32 v76, v126, v76, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffbd, v131
	s_nop 0
	v_cndmask_b32_e32 v77, v126, v77, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffbe, v131
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_cndmask_b32_e32 v78, v126, v78, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, 0xffffffbf, v131
	v_subrev_u32_e32 v131, 64, v131
	v_cndmask_b32_e32 v79, v126, v79, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	s_nop 0
	v_cndmask_b32_e32 v80, v126, v80, vcc
	v_cmp_le_i32_e32 vcc, v131, v130
	v_max_f32_e32 v130, v66, v67
	v_max3_f32 v130, v130, v68, v69
	v_max3_f32 v130, v130, v70, v71
	v_max3_f32 v130, v130, v72, v73
	v_max3_f32 v130, v130, v74, v75
	v_cndmask_b32_e32 v81, v126, v81, vcc
	v_max3_f32 v130, v130, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v130, v130, v78, v79
	v_max3_f32 v130, v130, v80, v81
	ds_bpermute_b32 v131, v122, v130
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v131, v131, v131
	v_max_f32_e32 v131, v130, v131
	;;#ASMSTART
	v_add_f32 v130, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v131, v130
	v_mov_b32_e32 v130, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_117
	;;#ASMSTART
	v_add_f32 v100, v131, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v130, v84
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
.LBB0_117:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, 0, v66
	v_add_f32_e32 v131, v131, v67
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
	v_add_f32_e32 v131, v131, v68
	v_add_f32_e32 v131, v131, v69
	v_add_f32_e32 v131, v131, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v71
	v_add_f32_e32 v131, v131, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v73
	v_add_f32_e32 v131, v131, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v75
	v_add_f32_e32 v131, v131, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v77
	v_add_f32_e32 v131, v131, v78
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
	v_add_f32_e32 v131, v131, v79
	v_add_f32_e32 v131, v131, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v131, v131, v81
	;;#ASMSTART
	v_fma_f32 v129, v129, v130, v131
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v130
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v67 offset:12288
	v_add_u32_e32 v67, 0x1810, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[236:239], v67
	v_add_u32_e32 v67, 0x1820, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[232:235], v67
	v_add_u32_e32 v67, 0x1830, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 0x1840, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x1850, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
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
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x1880, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x1890, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0x18a0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0x18b0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_119
	ds_write_b128 v119, v[188:191]
	ds_write_b128 v119, v[192:195] offset:6144
.LBB0_119:
	s_or_b64 exec, exec, s[42:43]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_121
	v_add_u32_e32 v66, s47, v118
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[188:191], v[66:67], off
	global_load_dwordx4 v[192:195], v[66:67], off offset:256
.LBB0_121:
	s_or_b64 exec, exec, s[42:43]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v130, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v131, 2, v130
	v_and_b32_e32 v131, 8, v131
	v_lshrrev_b32_e32 v130, 1, v130
	v_and_or_b32 v130, v130, s75, v128
	v_add_u32_e32 v131, s37, v131
	v_add_u32_e32 v130, s83, v130
	v_subrev_u32_e32 v196, 55, v131
	v_cmp_lt_i32_e32 vcc, v196, v130
	s_nop 1
	v_cndmask_b32_e32 v67, v126, v67, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 53, v131
	s_nop 0
	v_cndmask_b32_e32 v66, v126, v66, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 52, v131
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e32 v68, v126, v68, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 51, v131
	s_nop 0
	v_cndmask_b32_e32 v69, v126, v69, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 50, v131
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_cndmask_b32_e32 v70, v126, v70, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 49, v131
	s_nop 0
	v_cndmask_b32_e32 v71, v126, v71, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 48, v131
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_cndmask_b32_e32 v72, v126, v72, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 39, v131
	s_nop 0
	v_cndmask_b32_e32 v73, v126, v73, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 38, v131
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_cndmask_b32_e32 v74, v126, v74, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 37, v131
	s_nop 0
	v_cndmask_b32_e32 v75, v126, v75, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 36, v131
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_cndmask_b32_e32 v76, v126, v76, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 35, v131
	s_nop 0
	v_cndmask_b32_e32 v77, v126, v77, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 34, v131
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_cndmask_b32_e32 v78, v126, v78, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 33, v131
	v_subrev_u32_e32 v131, 32, v131
	v_cndmask_b32_e32 v79, v126, v79, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	s_nop 0
	v_cndmask_b32_e32 v80, v126, v80, vcc
	v_cmp_le_i32_e32 vcc, v131, v130
	v_max_f32_e32 v130, v66, v67
	v_max3_f32 v130, v130, v68, v69
	v_max3_f32 v130, v130, v70, v71
	v_max3_f32 v130, v130, v72, v73
	v_max3_f32 v130, v130, v74, v75
	v_cndmask_b32_e32 v81, v126, v81, vcc
	v_max3_f32 v130, v130, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v130, v130, v78, v79
	v_max3_f32 v130, v130, v80, v81
	ds_bpermute_b32 v131, v122, v130
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v131, v131, v131
	v_max_f32_e32 v131, v130, v131
	;;#ASMSTART
	v_add_f32 v130, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v131, v130
	v_mov_b32_e32 v130, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_123
	;;#ASMSTART
	v_add_f32 v100, v131, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v130, v84
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
.LBB0_123:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, 0, v66
	v_add_f32_e32 v131, v131, v67
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
	v_add_f32_e32 v131, v131, v68
	v_add_f32_e32 v131, v131, v69
	v_add_f32_e32 v131, v131, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v71
	v_add_f32_e32 v131, v131, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v73
	v_add_f32_e32 v131, v131, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v75
	v_add_f32_e32 v131, v131, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v131, v131, v77
	v_add_f32_e32 v131, v131, v78
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
	v_add_f32_e32 v131, v131, v79
	v_add_f32_e32 v131, v131, v80
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	v_add_f32_e32 v131, v131, v81
	;;#ASMSTART
	v_fma_f32 v129, v129, v130, v131
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v130
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v130
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
	v_perm_b32 v66, v67, v66, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s77
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
	v_lshl_add_u64 v[74:75], v[74:75], 1, s[18:19]
	global_load_dwordx4 v[76:79], v[74:75], off
	global_load_dwordx4 v[196:199], v[74:75], off offset:512
	global_load_dwordx4 v[200:203], v[74:75], off offset:1024
	global_load_dwordx4 v[204:207], v[74:75], off offset:1536
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[66:67], v[2:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[196:197], v[66:67], v[18:33]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[200:201], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[66:67], v[50:65]
	v_add_co_u32_e32 v66, vcc, s78, v74
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v75, vcc
	global_load_dwordx4 v[74:77], v[66:67], off
	v_mfma_f32_32x32x8_bf16 v[2:17], v[78:79], v[68:69], v[2:17]
	global_load_dwordx4 v[78:81], v[66:67], off offset:512
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[68:69], v[18:33]
	global_load_dwordx4 v[196:199], v[66:67], off offset:1024
	v_mfma_f32_32x32x8_bf16 v[34:49], v[202:203], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[68:69], v[50:65]
	global_load_dwordx4 v[66:69], v[66:67], off offset:1536
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[74:75], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[78:79], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[66:67], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[76:77], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[80:81], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[72:73], v[34:49]
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
	ds_read_b128 v[240:243], v68
	v_or_b32_e32 v68, 16, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[236:239], v68
	v_or_b32_e32 v68, 32, v66
	v_xor_b32_e32 v68, v68, v67
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[232:235], v68
	v_or_b32_e32 v68, 48, v66
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[228:231], v67
	v_add_u32_e32 v67, 64, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[224:227], v67
	v_add_u32_e32 v67, 0x50, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[220:223], v67
	v_add_u32_e32 v67, 0x60, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[216:219], v67
	v_add_u32_e32 v67, 0x70, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[212:215], v67
	v_add_u32_e32 v67, 0x80, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[208:211], v67
	v_add_u32_e32 v67, 0x90, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	ds_read_b128 v[204:207], v67
	v_add_u32_e32 v67, 0xa0, v66
	v_lshrrev_b32_e32 v68, 3, v67
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v67, v68, v67
	v_lshlrev_b32_e32 v67, 1, v67
	v_add_u32_e32 v66, 0xb0, v66
	ds_read_b128 v[200:203], v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[196:199], v66
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_125
	ds_write_b128 v119, v[180:183] offset:12288
	ds_write_b128 v119, v[184:187] offset:18432
.LBB0_125:
	s_or_b64 exec, exec, s[42:43]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v66, 0x180, v66
	v_cmp_ne_u32_e32 vcc, s76, v66
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_127
	v_add_u32_e32 v66, s47, v120
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[66:67], off
	global_load_dwordx4 v[184:187], v[66:67], off offset:256
.LBB0_127:
	s_or_b64 exec, exec, s[42:43]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[132:133], 0
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[134:135], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[138:139], v[66:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[142:143], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[146:147], v[66:81]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[170:171], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[174:175], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[178:179], v[66:81]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v130, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v131, 2, v130
	v_and_b32_e32 v131, 8, v131
	v_lshrrev_b32_e32 v130, 1, v130
	v_and_or_b32 v130, v130, s75, v128
	v_add_u32_e32 v131, s37, v131
	v_add_u32_e32 v130, s83, v130
	v_subrev_u32_e32 v196, 23, v131
	v_cmp_lt_i32_e32 vcc, v196, v130
	s_nop 1
	v_cndmask_b32_e32 v67, v126, v67, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 21, v131
	s_nop 0
	v_cndmask_b32_e32 v66, v126, v66, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 20, v131
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e32 v68, v126, v68, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 19, v131
	s_nop 0
	v_cndmask_b32_e32 v69, v126, v69, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 18, v131
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_cndmask_b32_e32 v70, v126, v70, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_subrev_u32_e32 v196, 17, v131
	s_nop 0
	v_cndmask_b32_e32 v71, v126, v71, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, -16, v131
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_cndmask_b32_e32 v72, v126, v72, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, -7, v131
	s_nop 0
	v_cndmask_b32_e32 v73, v126, v73, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, -6, v131
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_cndmask_b32_e32 v74, v126, v74, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, -5, v131
	s_nop 0
	v_cndmask_b32_e32 v75, v126, v75, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, -4, v131
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_cndmask_b32_e32 v76, v126, v76, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, -3, v131
	s_nop 0
	v_cndmask_b32_e32 v77, v126, v77, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, -2, v131
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_cndmask_b32_e32 v78, v126, v78, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_add_u32_e32 v196, -1, v131
	s_nop 0
	v_cndmask_b32_e32 v79, v126, v79, vcc
	v_cmp_le_i32_e32 vcc, v196, v130
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	s_nop 0
	v_cndmask_b32_e32 v80, v126, v80, vcc
	v_cmp_le_i32_e32 vcc, v131, v130
	v_max_f32_e32 v130, v66, v67
	v_max3_f32 v130, v130, v68, v69
	v_max3_f32 v130, v130, v70, v71
	v_max3_f32 v130, v130, v72, v73
	v_max3_f32 v130, v130, v74, v75
	v_cndmask_b32_e32 v81, v126, v81, vcc
	v_max3_f32 v130, v130, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v130, v130, v78, v79
	v_max3_f32 v130, v130, v80, v81
	ds_bpermute_b32 v131, v122, v130
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v131, v131, v131
	v_max_f32_e32 v131, v130, v131
	;;#ASMSTART
	v_add_f32 v130, v84, v124
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v131, v130
	v_mov_b32_e32 v130, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_78
	;;#ASMSTART
	v_add_f32 v100, v131, v125
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v130, v84
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
	s_branch .LBB0_78
.LBB0_129:
	s_mov_b32 s46, s45
	s_branch .LBB0_79
.LBB0_130:
	;;#ASMSTART
	v_mov_b32 v68, v0
	;;#ASMEND
	ds_bpermute_b32 v66, v122, v129
	v_lshrrev_b32_e32 v67, 1, v68
	v_and_b32_e32 v69, 31, v68
	v_and_or_b32 v67, v67, s75, v69
	v_and_b32_e32 v68, 32, v68
	v_cmp_eq_u32_e32 vcc, 0, v68
	v_cmp_gt_i32_e64 s[0:1], s13, v67
	s_and_b64 s[4:5], vcc, s[0:1]
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v66, v129, v66
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_132
	v_cmp_gt_f32_e32 vcc, s79, v66
	s_ashr_i32 s69, s68, 31
	s_lshl_b64 s[4:5], s[68:69], 6
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v66, v69
	v_log_f32_e32 v69, v69
	v_cndmask_b32_e32 v68, 0, v127, vcc
	s_add_u32 s6, s30, s4
	s_addc_u32 s37, s31, s5
	v_sub_f32_e32 v68, v69, v68
	v_add_f32_e32 v68, v84, v68
	s_lshl_b64 s[4:5], s[22:23], 2
	v_mul_f32_e32 v68, 0x3f317218, v68
	v_cmp_lt_f32_e32 vcc, 0, v66
	s_add_u32 s4, s6, s4
	s_addc_u32 s5, s37, s5
	v_cndmask_b32_e32 v68, v126, v68, vcc
	v_lshlrev_b32_e32 v67, 6, v67
	global_store_dword v67, v68, s[4:5]
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
	v_perm_b32 v28, v3, v2, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v5, v4, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v7, v6, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v9, v8, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v11, v10, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v13, v12, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v15, v14, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v17, v16, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v19, v18, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v21, v20, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v23, v22, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v74, v75, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v72, v73, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v70, v71, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v68, v69, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v66, v67, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v35, v34, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v37, v36, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v39, v38, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v41, v40, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v43, v42, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v45, v44, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v47, v46, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v49, v48, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v51, v50, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v53, v52, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v55, v54, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v57, v56, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v59, v58, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v61, v60, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v63, v62, s77
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v65, v64, s77
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
	s_add_u32 s4, s28, s0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v39, 7, v39
	ds_read_b128 v[42:45], v37
	s_addc_u32 s0, s29, s1
	s_lshl_b32 s6, s13, 12
	s_lshl_b32 s13, s22, 7
	v_and_b32_e32 v39, 0xf800, v39
	v_and_b32_e32 v38, 0x78, v40
	v_add_u32_e32 v40, s13, v39
	v_or_b32_e32 v40, v40, v38
	s_and_b32 s5, s0, 0xffff
	v_lshlrev_b32_e32 v40, 1, v40
	v_cmp_eq_u32_e32 vcc, 1, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[42:45], v40, s[4:7], 0 offen
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
	s_add_i32 s13, s13, 0x10000
	v_or_b32_e32 v38, v38, v39
	v_add_lshl_u32 v39, s13, v38, 1
	v_cmp_eq_u32_e32 vcc, 2, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[4:7], 0 offen
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
	s_add_i32 s0, s13, 0x10000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 3, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[4:7], 0 offen
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
	s_add_i32 s0, s13, 0x20000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 4, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[4:7], 0 offen
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
	s_add_i32 s0, s13, 0x30000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 5, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[4:7], 0 offen
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
	s_add_i32 s0, s13, 0x40000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 6, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[4:7], 0 offen
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
	s_add_i32 s0, s13, 0x50000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 7, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[4:7], 0 offen
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
	s_add_i32 s13, s13, 0x60000
	v_add_lshl_u32 v2, s13, v38, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v2, s[4:7], 0 offen
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
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_154
	s_mov_b64 s[40:41], exec
	v_mbcnt_lo_u32_b32 v2, s40, 0
	v_mbcnt_hi_u32_b32 v2, s41, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_153
	s_bcnt1_i32_b64 s6, s[40:41]
	v_mov_b32_e32 v3, s6
	global_atomic_add v3, v123, v3, s[34:35] sc0
.LBB0_153:
	s_or_b64 exec, exec, s[38:39]
	s_lshl_b64 s[38:39], s[0:1], 2
	s_add_u32 s38, s34, s38
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s6, v3
	s_addc_u32 s39, s35, s39
	s_nop 0
	v_add_u32_e32 v2, s6, v2
	global_store_dword v123, v2, s[38:39]
	s_waitcnt vmcnt(0)
.LBB0_154:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s34, s0
	s_addc_u32 s1, s35, s1
	s_barrier
	global_load_dword v2, v123, s[0:1]
	s_sub_i32 s0, s33, s2
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s2, v2
	s_add_i32 s33, s0, s2
	s_cmp_ge_i32 s12, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s72
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_157
	s_branch .LBB0_12
.LBB0_155:
	s_mov_b32 s12, s5
.LBB0_156:
	s_sub_i32 s33, s33, s72
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s22, 0, s4
	s_cmp_ge_i32 s12, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s72, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_157:
	s_add_i32 s4, s22, 1
	s_cmp_gt_i32 s4, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s4, 16
	s_cbranch_scc1 .LBB0_160
	s_add_i32 s5, s12, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s72
	s_cbranch_scc1 .LBB0_155
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s12, s8, s12
	s_addc_u32 s13, s9, s13
	global_load_dwordx2 v[2:3], v123, s[12:13] offset:4
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s6, v3
	v_readfirstlane_b32 s12, v2
	s_sub_i32 s6, s6, s12
	s_addk_i32 s6, 0xff
	s_ashr_i32 s12, s6, 31
	s_lshr_b32 s12, s12, 24
	s_add_i32 s12, s6, s12
	s_ashr_i32 s37, s12, 8
	s_and_b32 s12, s12, 0xffffff00
	s_cmp_lg_u32 s6, s12
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s6, 0
	s_cselect_b64 s[22:23], -1, 0
	s_and_b64 s[12:13], s[22:23], s[12:13]
	s_subb_u32 s6, s37, 0
	s_branch .LBB0_155
.LBB0_160:
	s_mov_b32 s6, s72
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
		.amdhsa_next_free_vgpr 244
		.amdhsa_next_free_sgpr 96
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

	.set .Lattn_kernel_0.num_vgpr, 244
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
    .sgpr_count:     97
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     244
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
