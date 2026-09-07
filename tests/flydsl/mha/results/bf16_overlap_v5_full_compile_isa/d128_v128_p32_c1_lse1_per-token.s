	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	attn_kernel_0
	.p2align	8
	.type	attn_kernel_0,@function
attn_kernel_0:
	s_load_dwordx2 s[12:13], s[0:1], 0x30
	s_load_dword s3, s[0:1], 0x38
	s_load_dwordx2 s[14:15], s[0:1], 0x90
	s_load_dwordx4 s[4:7], s[0:1], 0x80
	s_mov_b32 s16, 0
	s_waitcnt lgkmcnt(0)
	s_load_dwordx2 s[8:9], s[12:13], 0x0
	s_add_i32 s3, s3, -1
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s8, s9, s8
	s_add_i32 s10, s8, 0xff
	s_ashr_i32 s8, s10, 31
	s_lshr_b32 s8, s8, 24
	s_add_i32 s8, s10, s8
	s_ashr_i32 s17, s8, 8
	s_and_b32 s8, s8, 0xffffff00
	s_cmp_lg_u32 s10, s8
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s10, 0
	s_cselect_b64 s[10:11], -1, 0
	s_and_b64 s[8:9], s[10:11], s[8:9]
	s_subb_u32 s10, s17, 0
	s_cmp_lt_i32 s3, 1
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s2, s10
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[8:9], s[8:9], s[18:19]
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_8
	s_mov_b32 s26, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s16, s18
.LBB0_3:
	s_sub_i32 s33, s33, s10
	s_and_b64 s[8:9], s[8:9], exec
	s_cselect_b32 s26, 0, s11
	s_cmp_ge_i32 s16, s3
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s33, s76
	s_cselect_b64 s[10:11], -1, 0
	s_or_b64 s[8:9], s[8:9], s[10:11]
	s_and_b64 vcc, exec, s[8:9]
	s_mov_b32 s10, s76
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s11, s26, 1
	s_cmp_gt_i32 s11, 15
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s11, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s18, s16, 1
	s_cmp_ge_i32 s18, s3
	s_mov_b32 s76, s10
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[16:17], s[16:17], 2
	s_add_u32 s16, s12, s16
	s_addc_u32 s17, s13, s17
	s_load_dwordx2 s[20:21], s[16:17], 0x4
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s16, s21, s20
	s_add_i32 s19, s16, 0xff
	s_ashr_i32 s16, s19, 31
	s_lshr_b32 s16, s16, 24
	s_add_i32 s16, s19, s16
	s_ashr_i32 s22, s16, 8
	s_and_b32 s16, s16, 0xffffff00
	s_cmp_lg_u32 s19, s16
	s_cselect_b64 s[16:17], -1, 0
	s_cmp_lt_i32 s19, 0
	s_cselect_b64 s[20:21], -1, 0
	s_and_b64 s[16:17], s[20:21], s[16:17]
	s_subb_u32 s76, s22, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s76, s10
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s76, s10
	s_mov_b32 s26, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s77, s[4:5], 0x0
	s_load_dword s78, s[6:7], 0x0
	s_cmp_ge_i32 s16, s3
	s_cbranch_scc1 .LBB0_73
	v_lshrrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v1, 31, v0
	s_movk_i32 s79, 0xe0
	v_and_or_b32 v3, v2, s79, v1
	v_lshrrev_b32_e32 v4, 2, v0
	v_lshlrev_b32_e32 v6, 10, v0
	v_lshlrev_b32_e32 v85, 6, v3
	v_lshlrev_b32_e32 v3, 11, v1
	v_and_b32_e32 v5, 8, v4
	v_and_b32_e32 v6, 0x70000, v6
	v_lshrrev_b32_e32 v7, 3, v0
	v_or3_b32 v116, v6, v3, v5
	v_lshlrev_b32_e32 v3, 7, v0
	v_and_b32_e32 v6, 12, v4
	v_and_b32_e32 v2, 32, v2
	v_and_b32_e32 v7, 16, v7
	v_and_b32_e32 v3, 0x780, v3
	v_or3_b32 v2, v6, v2, v7
	v_and_b32_e32 v4, 64, v4
	v_or3_b32 v117, v2, v4, v3
	v_lshlrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v2, 0x70, v2
	v_lshlrev_b32_e32 v3, 4, v0
	v_xor_b32_e32 v118, v3, v2
	v_lshl_or_b32 v2, v1, 7, v5
	v_and_b32_e32 v3, 48, v3
	v_or_b32_e32 v4, v2, v3
	v_lshlrev_b32_e32 v119, 1, v4
	v_or_b32_e32 v4, 16, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v120, 1, v4
	v_or_b32_e32 v4, 32, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v121, 1, v4
	v_or_b32_e32 v4, 48, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v122, 1, v4
	v_or_b32_e32 v4, 64, v2
	v_lshrrev_b32_e32 v5, 3, v4
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v4, v5, v4
	v_lshlrev_b32_e32 v123, 1, v4
	v_or_b32_e32 v4, 0x50, v2
	v_lshrrev_b32_e32 v5, 3, v4
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v4, v5, v4
	v_lshlrev_b32_e32 v124, 1, v4
	v_or_b32_e32 v4, 0x60, v2
	v_lshrrev_b32_e32 v5, 3, v4
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v4, v5, v4
	v_lshlrev_b32_e32 v125, 1, v4
	v_or_b32_e32 v4, 0x70, v2
	v_lshrrev_b32_e32 v5, 3, v4
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v4, v5, v4
	v_lshlrev_b32_e32 v126, 1, v4
	v_or_b32_e32 v4, 0x1010, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v127, 1, v4
	v_or_b32_e32 v4, 0x1020, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v128, 1, v4
	v_or_b32_e32 v4, 0x1030, v2
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v129, 1, v3
	v_or_b32_e32 v3, 0x1040, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v130, 1, v3
	v_or_b32_e32 v3, 0x1050, v2
	s_load_dwordx2 s[18:19], s[0:1], 0x0
	s_load_dwordx2 s[20:21], s[0:1], 0x10
	s_load_dwordx2 s[22:23], s[0:1], 0x20
	s_load_dwordx2 s[24:25], s[0:1], 0x50
	s_load_dwordx2 s[28:29], s[0:1], 0x60
	s_load_dwordx2 s[30:31], s[0:1], 0x70
	s_load_dwordx2 s[34:35], s[0:1], 0xa0
	s_load_dwordx2 s[68:69], s[0:1], 0xb0
	s_load_dwordx2 s[70:71], s[0:1], 0xc0
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v131, 1, v3
	v_or_b32_e32 v3, 0x1060, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_or_b32_e32 v2, 0x1070, v2
	v_lshlrev_b32_e32 v132, 1, v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v133, 1, v2
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_xor_b32_e32 v134, 0x80, v2
	v_mov_b32_e32 v135, 0
	s_mov_b32 s7, 0x27000
	s_movk_i32 s80, 0x1000
	v_mov_b32_e32 v136, 0x40e00000
	v_mov_b32_e32 v137, 1.0
	s_mov_b32 s81, 0x7060302
	s_mov_b32 s82, 0x800000
	s_movk_i32 s83, 0xf80
	s_mov_b32 s84, s2
	v_mov_b32_e32 v138, 0xff800000
	v_mov_b32_e32 v139, 0x42000000
	s_mov_b32 s36, 0
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s76, s6
.LBB0_12:
	s_mov_b32 s2, s4
	s_cmp_ge_i32 s16, s3
	s_cbranch_scc1 .LBB0_73
.LBB0_13:
	s_ashr_i32 s17, s16, 31
	s_lshl_b32 s90, s33, 8
	s_lshl_b64 s[0:1], s[16:17], 2
	s_add_u32 s4, s12, s0
	s_addc_u32 s5, s13, s1
	global_load_dwordx2 v[2:3], v135, s[4:5]
	s_mov_b32 s11, s7
	v_lshl_add_u32 v6, s26, 2, v85
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s27, v2
	s_add_i32 s74, s27, s90
	v_readfirstlane_b32 s37, v3
	s_add_i32 s4, s74, 0x100
	s_min_i32 s4, s4, s37
	s_sub_i32 s75, s4, s74
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s24, s0
	s_addc_u32 s5, s25, s1
	s_add_u32 s0, s14, s0
	s_addc_u32 s1, s15, s1
	s_lshl_b32 s8, s74, 11
	s_ashr_i32 s9, s8, 31
	s_lshl_b32 s38, s74, 4
	s_lshl_b64 s[72:73], s[8:9], 1
	s_add_u32 s8, s18, s72
	global_load_dwordx2 v[4:5], v135, s[4:5]
	global_load_dword v3, v135, s[0:1]
	s_addc_u32 s0, s19, s73
	s_lshl_b32 s17, s26, 7
	s_ashr_i32 s39, s38, 31
	s_lshl_b32 s10, s75, 12
	s_and_b32 s9, s0, 0xffff
	s_lshl_b64 s[0:1], s[38:39], 2
	v_add_lshl_u32 v7, s17, v116, 1
	s_add_u32 s4, s30, s0
	buffer_load_dwordx4 v[144:147], v7, s[8:11], 0 offen
	buffer_load_dwordx4 v[148:151], v7, s[8:11], 0 offen offset:32
	buffer_load_dwordx4 v[152:155], v7, s[8:11], 0 offen offset:64
	buffer_load_dwordx4 v[156:159], v7, s[8:11], 0 offen offset:96
	buffer_load_dwordx4 v[160:163], v7, s[8:11], 0 offen offset:128
	buffer_load_dwordx4 v[164:167], v7, s[8:11], 0 offen offset:160
	s_addc_u32 s0, s31, s1
	s_lshl_b32 s6, s75, 6
	s_and_b32 s5, s0, 0xffff
	buffer_load_dword v2, v6, s[4:7], 0 offen
	buffer_load_dwordx4 v[168:171], v7, s[8:11], 0 offen offset:192
	buffer_load_dwordx4 v[172:175], v7, s[8:11], 0 offen offset:224
	s_waitcnt vmcnt(10)
	v_readfirstlane_b32 s4, v4
	s_waitcnt vmcnt(9)
	v_readfirstlane_b32 s8, v3
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
	s_sub_i32 s9, s27, s37
	s_ashr_i32 s27, s26, 31
	s_lshr_b32 s0, s27, 28
	s_sub_i32 s91, s5, s4
	s_add_i32 s0, s26, s0
	s_lshl_b32 s37, s91, 5
	s_ashr_i32 s5, s0, 4
	s_and_b32 s0, s0, -16
	s_cmp_lg_u32 s26, s0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s26, 0
	s_cselect_b64 s[38:39], -1, 0
	s_and_b64 s[0:1], s[38:39], s[0:1]
	s_subb_u32 s11, s5, 0
	s_lshl_b32 s0, s11, 12
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s20, s0
	s_addc_u32 s1, s21, s1
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s4, s28, s4
	s_addc_u32 s5, s29, s5
	s_lshl_b32 s6, s91, 2
	s_and_b32 s5, s5, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[4:7], 0
	s_waitcnt vmcnt(3)
	v_mul_f32_e32 v2, s77, v2
	v_mul_f32_e32 v82, 0x3e0293ee, v2
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s89, v4
	v_readfirstlane_b32 s88, v5
	s_nop 0
	v_lshl_or_b32 v6, s89, 11, v117
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[0:1]
	global_load_dwordx4 v[6:9], v[6:7], off
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	v_lshl_or_b32 v2, s88, 11, v117
	v_ashrrev_i32_e32 v3, 31, v2
	buffer_load_dword v4, off, s[4:7], 0 offset:8
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[0:1]
	global_load_dwordx4 v[176:179], v[2:3], off
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s85, v4
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	v_lshl_or_b32 v2, s85, 11, v117
	v_ashrrev_i32_e32 v3, 31, v2
	buffer_load_dword v4, off, s[4:7], 0 offset:12
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[2:3], off
	ds_write_b128 v118, v[6:9]
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s87, v4
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	ds_read_b128 v[200:203], v119
	ds_read_b128 v[196:199], v120
	ds_read_b128 v[192:195], v121
	ds_read_b128 v[184:187], v122
	ds_read_b128 v[188:191], v123
	ds_read_b128 v[106:109], v124
	ds_read_b128 v[110:113], v125
	ds_read_b128 v[102:105], v126
	s_add_i32 s9, s9, s37
	s_add_i32 s9, s9, s8
	s_sub_i32 s86, s9, 32
	s_add_i32 s93, s86, s90
	s_add_i32 s37, s93, 1
	s_ashr_i32 s8, s37, 31
	s_lshr_b32 s8, s8, 27
	s_add_i32 s8, s37, s8
	s_ashr_i32 s40, s8, 5
	s_andn2_b32 s8, s8, 31
	s_cmp_lg_u32 s37, s8
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s37, 0
	s_cselect_b64 s[38:39], -1, 0
	s_and_b64 s[8:9], s[38:39], s[8:9]
	s_subb_u32 s37, s40, 0
	s_lshr_b32 s8, s37, 31
	s_add_i32 s8, s37, s8
	s_ashr_i32 s40, s8, 1
	s_and_b32 s8, s8, -2
	s_cmp_lg_u32 s37, s8
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s37, 0
	s_cselect_b64 s[38:39], -1, 0
	s_and_b64 s[8:9], s[38:39], s[8:9]
	s_subb_u32 s92, s40, 0
	s_lshl_b32 s8, s92, 1
	s_ashr_i32 s9, s8, 31
	s_cmp_lt_i32 s92, 1
	s_cbranch_scc1 .LBB0_27
	v_mov_b32_e32 v140, 0
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
	v_mov_b32_e32 v100, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v140
	v_mov_b32_e32 v4, v140
	v_mov_b32_e32 v5, v140
	v_mov_b32_e32 v6, v140
	v_mov_b32_e32 v7, v140
	v_mov_b32_e32 v8, v140
	v_mov_b32_e32 v9, v140
	v_mov_b32_e32 v10, v140
	v_mov_b32_e32 v11, v140
	v_mov_b32_e32 v12, v140
	v_mov_b32_e32 v13, v140
	v_mov_b32_e32 v14, v140
	v_mov_b32_e32 v15, v140
	v_mov_b32_e32 v16, v140
	v_mov_b32_e32 v17, v140
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v140
	v_mov_b32_e32 v20, v140
	v_mov_b32_e32 v21, v140
	v_mov_b32_e32 v22, v140
	v_mov_b32_e32 v23, v140
	v_mov_b32_e32 v24, v140
	v_mov_b32_e32 v25, v140
	v_mov_b32_e32 v26, v140
	v_mov_b32_e32 v27, v140
	v_mov_b32_e32 v28, v140
	v_mov_b32_e32 v29, v140
	v_mov_b32_e32 v30, v140
	v_mov_b32_e32 v31, v140
	v_mov_b32_e32 v32, v140
	v_mov_b32_e32 v33, v140
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v140
	v_mov_b32_e32 v36, v140
	v_mov_b32_e32 v37, v140
	v_mov_b32_e32 v38, v140
	v_mov_b32_e32 v39, v140
	v_mov_b32_e32 v40, v140
	v_mov_b32_e32 v41, v140
	v_mov_b32_e32 v42, v140
	v_mov_b32_e32 v43, v140
	v_mov_b32_e32 v44, v140
	v_mov_b32_e32 v45, v140
	v_mov_b32_e32 v46, v140
	v_mov_b32_e32 v47, v140
	v_mov_b32_e32 v48, v140
	v_mov_b32_e32 v49, v140
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v140
	v_mov_b32_e32 v52, v140
	v_mov_b32_e32 v53, v140
	v_mov_b32_e32 v54, v140
	v_mov_b32_e32 v55, v140
	v_mov_b32_e32 v56, v140
	v_mov_b32_e32 v57, v140
	v_mov_b32_e32 v58, v140
	v_mov_b32_e32 v59, v140
	v_mov_b32_e32 v60, v140
	v_mov_b32_e32 v61, v140
	v_mov_b32_e32 v62, v140
	v_mov_b32_e32 v63, v140
	v_mov_b32_e32 v64, v140
	v_mov_b32_e32 v65, v140
	s_branch .LBB0_18
.LBB0_17:
	s_or_b64 exec, exec, s[40:41]
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
	v_perm_b32 v66, v67, v66, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s81
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[200:203], v119
	ds_read_b128 v[196:199], v120
	v_mfma_f32_32x32x8_bf16 v[2:17], v[186:187], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[216:217], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[208:209], v[66:67], v[34:49]
	ds_read_b128 v[192:195], v121
	ds_read_b128 v[184:187], v122
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[188:189], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[218:219], v[68:69], v[18:33]
	ds_read_b128 v[188:191], v123
	ds_read_b128 v[106:109], v124
	v_mfma_f32_32x32x8_bf16 v[34:49], v[210:211], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[212:213], v[70:71], v[2:17]
	ds_read_b128 v[110:113], v125
	ds_read_b128 v[102:105], v126
	v_mfma_f32_32x32x8_bf16 v[18:33], v[224:225], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[228:229], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[214:215], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[72:73], v[50:65]
	s_add_u32 s38, s38, 2
	s_addc_u32 s39, s39, 0
	v_mov_b64_e32 v[66:67], s[8:9]
	v_cmp_lt_i64_e32 vcc, s[38:39], v[66:67]
	s_add_i32 s37, s37, 8
	s_mov_b32 s88, s42
	v_mov_b32_e32 v100, v84
	s_cbranch_vccz .LBB0_26
.LBB0_18:
	ds_write_b128 v118, v[176:179] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[144:145], 0
	s_add_i32 s40, s37, -4
	v_mov_b32_e32 v84, s40
	s_mov_b32 s42, s87
	v_lshl_or_b32 v114, s42, 11, v117
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 2, s[0:1]
	s_add_i32 s89, s89, s11
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[146:147], v[66:81]
	buffer_load_dword v84, v84, s[4:7], 0 offen
	s_lshl_b32 s40, s89, 12
	s_mov_b32 s89, s85
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s85, v84
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[152:153], v[66:81]
	global_load_dwordx4 v[176:179], v[114:115], off
	;;#ASMSTART
	v_mov_b32 v101, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v114, 3, v101
	v_lshlrev_b32_e32 v101, 5, v101
	v_and_b32_e32 v114, 0xf8, v114
	v_and_b32_e32 v101, 0x400, v101
	v_or3_b32 v114, v114, v101, s40
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, s[22:23]
	global_load_dwordx4 v[196:199], v[114:115], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	global_load_dwordx4 v[192:195], v[114:115], off offset:512
	global_load_dwordx4 v[184:187], v[114:115], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[164:165], v[66:81]
	v_add_co_u32_e32 v106, vcc, s80, v114
	global_load_dwordx4 v[200:203], v[114:115], off offset:1536
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v115, vcc
	global_load_dwordx4 v[188:191], v[106:107], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[170:171], v[66:81]
	global_load_dwordx4 v[110:113], v[106:107], off offset:512
	global_load_dwordx4 v[204:207], v[106:107], off offset:1024
	s_nop 0
	global_load_dwordx4 v[106:109], v[106:107], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[174:175], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v84, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v84, v84, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v84, v84, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v84, v84, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v84, v84, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v84, v84, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v84, v84, v78, v79
	v_max3_f32 v84, v84, v80, v81
	ds_bpermute_b32 v101, v134, v84
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v101, v84, v101
	;;#ASMSTART
	v_add_f32 v84, v100, v136
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v101, v84
	v_mov_b32_e32 v84, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_20
	;;#ASMSTART
	v_add_f32 v101, v101, v137
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v100, v101
	v_exp_f32_e32 v84, v84
	v_mov_b32_e32 v100, v101
.LBB0_20:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[66:67], v[66:67], v[100:101] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[80:81], v[80:81], v[100:101] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[78:79], v[78:79], v[100:101] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[76:77], v[76:77], v[100:101] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[74:75], v[74:75], v[100:101] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[72:73], v[72:73], v[100:101] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[70:71], v[70:71], v[100:101] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[100:101] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	v_add_f32_e32 v101, 0, v66
	v_add_f32_e32 v101, v101, v67
	v_add_f32_e32 v101, v101, v68
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v101, v101, v69
	v_add_f32_e32 v101, v101, v70
	v_add_f32_e32 v101, v101, v71
	v_add_f32_e32 v101, v101, v72
	v_add_f32_e32 v101, v101, v73
	v_add_f32_e32 v101, v101, v74
	v_add_f32_e32 v101, v101, v75
	v_add_f32_e32 v101, v101, v76
	v_add_f32_e32 v101, v101, v77
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v84
	v_add_f32_e32 v101, v101, v78
	v_add_f32_e32 v101, v101, v79
	v_add_f32_e32 v101, v101, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v101, v101, v81
	;;#ASMSTART
	v_fma_f32 v140, v140, v84, v101
	;;#ASMEND
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_22
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
	v_mul_f32 v16, v16, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v84
	;;#ASMEND
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
	v_mul_f32 v48, v48, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v84
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v84
	;;#ASMEND
.LBB0_22:
	s_or_b64 exec, exec, s[40:41]
	v_add_u32_e32 v81, 0x8000, v81
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v79, 0x8000, v79
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v77, 0x8000, v77
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v75, 0x8000, v75
	v_add_u32_e32 v74, 0x8000, v74
	v_add_u32_e32 v73, 0x8000, v73
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v71, 0x8000, v71
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v69, 0x8000, v69
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v67, 0x8000, v67
	v_add_u32_e32 v66, 0x8000, v66
	;;#ASMSTART
	v_perm_b32 v66, v67, v66, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s81
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[102:105], v119 offset:8192
	ds_read_b128 v[208:211], v127
	v_mfma_f32_32x32x8_bf16 v[2:17], v[196:197], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[66:67], v[34:49]
	ds_read_b128 v[212:215], v128
	ds_read_b128 v[216:219], v129
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[198:199], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[194:195], v[68:69], v[18:33]
	ds_read_b128 v[192:195], v130
	ds_read_b128 v[196:199], v131
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[202:203], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[188:189], v[70:71], v[2:17]
	ds_read_b128 v[200:203], v132
	ds_read_b128 v[232:235], v133
	v_mfma_f32_32x32x8_bf16 v[18:33], v[110:111], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[204:205], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[190:191], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[112:113], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[206:207], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[72:73], v[50:65]
	ds_write_b128 v118, v[180:183]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[144:145], 0
	v_mov_b32_e32 v84, s37
	v_lshl_or_b32 v102, s85, 11, v117
	v_ashrrev_i32_e32 v103, 31, v102
	v_lshl_add_u64 v[102:103], v[102:103], 2, s[0:1]
	s_add_i32 s40, s88, s11
	s_lshl_b32 s40, s40, 12
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[146:147], v[66:81]
	buffer_load_dword v84, v84, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s87, v84
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[152:153], v[66:81]
	global_load_dwordx4 v[180:183], v[102:103], off
	;;#ASMSTART
	v_mov_b32 v101, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v102, 3, v101
	v_lshlrev_b32_e32 v101, 5, v101
	v_and_b32_e32 v102, 0xf8, v102
	v_and_b32_e32 v101, 0x400, v101
	v_or3_b32 v102, v102, v101, s40
	v_ashrrev_i32_e32 v103, 31, v102
	v_lshl_add_u64 v[102:103], v[102:103], 1, s[22:23]
	global_load_dwordx4 v[186:189], v[102:103], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[158:159], v[66:81]
	global_load_dwordx4 v[216:219], v[102:103], off offset:512
	global_load_dwordx4 v[208:211], v[102:103], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[164:165], v[66:81]
	global_load_dwordx4 v[220:223], v[102:103], off offset:1536
	v_add_co_u32_e32 v102, vcc, s80, v102
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	global_load_dwordx4 v[212:215], v[102:103], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[170:171], v[66:81]
	global_load_dwordx4 v[224:227], v[102:103], off offset:512
	global_load_dwordx4 v[228:231], v[102:103], off offset:1024
	global_load_dwordx4 v[204:207], v[102:103], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[174:175], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_max_f32_e32 v84, v66, v67
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_max3_f32 v84, v84, v68, v69
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_max3_f32 v84, v84, v70, v71
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_max3_f32 v84, v84, v72, v73
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_max3_f32 v84, v84, v74, v75
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_max3_f32 v84, v84, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v84, v84, v78, v79
	v_max3_f32 v84, v84, v80, v81
	ds_bpermute_b32 v101, v134, v84
	v_mov_b32_e32 v141, 1.0
	v_mov_b32_e32 v102, v100
	v_mov_b32_e32 v103, v100
	v_mov_b32_e32 v104, v100
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v142, v84, v101
	;;#ASMSTART
	v_add_f32 v84, v100, v136
	;;#ASMEND
	v_mov_b32_e32 v101, v100
	v_cmp_gt_f32_e32 vcc, v142, v84
	v_mov_b32_e32 v84, v100
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
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_24
	;;#ASMSTART
	v_add_f32 v84, v142, v137
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v100, v100, v84
	v_exp_f32_e32 v141, v100
	v_mov_b32_e32 v100, v84
	v_mov_b32_e32 v101, v84
	v_mov_b32_e32 v102, v84
	v_mov_b32_e32 v103, v84
	v_mov_b32_e32 v104, v84
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
.LBB0_24:
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
	v_cmp_gt_f32_e32 vcc, 1.0, v141
	v_add_f32_e32 v100, v100, v79
	v_add_f32_e32 v100, v100, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v100, v100, v81
	;;#ASMSTART
	v_fma_f32 v140, v140, v141, v100
	;;#ASMEND
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_17
	;;#ASMSTART
	v_mul_f32 v2, v2, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v141
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v141
	;;#ASMEND
	s_branch .LBB0_17
.LBB0_26:
	s_mov_b32 s88, s42
	s_branch .LBB0_28
.LBB0_27:
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
	v_mov_b32_e32 v140, 0
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
.LBB0_28:
	s_add_i32 s37, s75, s93
	s_add_i32 s37, s37, 31
	s_ashr_i32 s38, s37, 31
	s_lshr_b32 s38, s38, 27
	s_add_i32 s38, s37, s38
	s_ashr_i32 s42, s38, 5
	s_andn2_b32 s38, s38, 31
	s_cmp_lg_u32 s37, s38
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_lt_i32 s37, 0
	s_cselect_b64 s[40:41], -1, 0
	s_and_b64 s[38:39], s[40:41], s[38:39]
	s_subb_u32 s37, s42, 0
	s_min_i32 s38, s37, s91
	s_cmp_ge_i32 s8, s38
	s_cbranch_scc1 .LBB0_42
	s_lshl_b32 s37, s92, 6
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
	v_or_b32_e32 v141, s90, v1
	s_or_b32 s37, s37, 55
	s_lshl3_add_u32 s44, s92, 20
	s_branch .LBB0_32
.LBB0_30:
	s_or_b64 exec, exec, s[42:43]
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
	v_perm_b32 v66, v67, v66, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s81
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[200:203], v119
	ds_read_b128 v[196:199], v120
	v_mfma_f32_32x32x8_bf16 v[2:17], v[208:209], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[220:221], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[66:67], v[34:49]
	ds_read_b128 v[192:195], v121
	ds_read_b128 v[184:187], v122
	v_mfma_f32_32x32x8_bf16 v[50:65], v[224:225], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[210:211], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[222:223], v[68:69], v[18:33]
	ds_read_b128 v[188:191], v123
	ds_read_b128 v[106:109], v124
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[226:227], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[216:217], v[70:71], v[2:17]
	ds_read_b128 v[110:113], v125
	ds_read_b128 v[102:105], v126
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[218:219], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[72:73], v[50:65]
.LBB0_31:
	s_and_b64 s[40:41], s[40:41], exec
	s_cselect_b32 s89, s85, s88
	s_cselect_b32 s88, s87, s85
	s_cselect_b32 s85, s45, s87
	s_add_u32 s8, s8, 2
	s_addc_u32 s9, s9, 0
	v_mov_b64_e32 v[66:67], s[38:39]
	v_cmp_lt_i64_e32 vcc, s[8:9], v[66:67]
	s_add_i32 s37, s37, 64
	s_add_i32 s44, s44, 8
	s_mov_b32 s87, s46
	s_cbranch_vccz .LBB0_42
.LBB0_32:
	ds_write_b128 v118, v[176:179] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[144:145], 0
	s_add_i32 s40, s44, -4
	v_mov_b32_e32 v100, s40
	s_add_i32 s40, s89, s11
	s_lshl_b32 s40, s40, 12
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[146:147], v[66:81]
	buffer_load_dword v114, v100, s[4:7], 0 offen
	v_lshl_or_b32 v100, s87, 11, v117
	v_ashrrev_i32_e32 v101, 31, v100
	v_lshl_add_u64 v[100:101], v[100:101], 2, s[0:1]
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v114
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[152:153], v[66:81]
	global_load_dwordx4 v[176:179], v[100:101], off
	;;#ASMSTART
	v_mov_b32 v100, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v101, 3, v100
	v_lshlrev_b32_e32 v100, 5, v100
	v_and_b32_e32 v101, 0xf8, v101
	v_and_b32_e32 v100, 0x400, v100
	v_or3_b32 v100, v101, v100, s40
	v_ashrrev_i32_e32 v101, 31, v100
	v_lshl_add_u64 v[100:101], v[100:101], 1, s[22:23]
	global_load_dwordx4 v[208:211], v[100:101], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	global_load_dwordx4 v[220:223], v[100:101], off offset:512
	global_load_dwordx4 v[212:215], v[100:101], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[164:165], v[66:81]
	global_load_dwordx4 v[224:227], v[100:101], off offset:1536
	v_add_co_u32_e32 v100, vcc, s80, v100
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	global_load_dwordx4 v[216:219], v[100:101], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[170:171], v[66:81]
	global_load_dwordx4 v[228:231], v[100:101], off offset:512
	global_load_dwordx4 v[232:235], v[100:101], off offset:1024
	global_load_dwordx4 v[204:207], v[100:101], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[174:175], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v100, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v101, 2, v100
	v_and_b32_e32 v101, 8, v101
	v_lshrrev_b32_e32 v100, 1, v100
	v_and_or_b32 v100, v100, s79, v141
	v_add_u32_e32 v109, s37, v101
	v_add_u32_e32 v108, s86, v100
	v_subrev_u32_e32 v100, 55, v109
	v_cmp_lt_i32_e32 vcc, v100, v108
	s_nop 1
	v_cndmask_b32_e32 v101, v138, v67, vcc
	v_cmp_le_i32_e32 vcc, v100, v108
	v_subrev_u32_e32 v67, 32, v109
	s_nop 0
	v_cndmask_b32_e32 v100, v138, v66, vcc
	v_subrev_u32_e32 v66, 53, v109
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 52, v109
	s_nop 0
	v_cndmask_b32_e32 v102, v138, v68, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 51, v109
	s_nop 0
	v_cndmask_b32_e32 v103, v138, v69, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 50, v109
	s_nop 0
	v_cndmask_b32_e32 v104, v138, v70, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 49, v109
	s_nop 0
	v_cndmask_b32_e32 v105, v138, v71, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 48, v109
	s_nop 0
	v_cndmask_b32_e32 v106, v138, v72, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 39, v109
	s_nop 0
	v_cndmask_b32_e32 v107, v138, v73, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 38, v109
	s_nop 0
	v_cndmask_b32_e32 v72, v138, v74, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 37, v109
	s_nop 0
	v_cndmask_b32_e32 v73, v138, v75, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 36, v109
	v_pk_mul_f32 v[74:75], v[90:91], v[106:107]
	v_cndmask_b32_e32 v70, v138, v76, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 35, v109
	v_pk_mul_f32 v[72:73], v[92:93], v[72:73]
	v_cndmask_b32_e32 v71, v138, v77, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 34, v109
	v_pk_mul_f32 v[76:77], v[88:89], v[104:105]
	v_cndmask_b32_e32 v68, v138, v78, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_subrev_u32_e32 v66, 33, v109
	v_pk_mul_f32 v[70:71], v[94:95], v[70:71]
	v_cndmask_b32_e32 v69, v138, v79, vcc
	v_cmp_le_i32_e32 vcc, v66, v108
	v_pk_mul_f32 v[78:79], v[86:87], v[102:103]
	v_pk_mul_f32 v[68:69], v[96:97], v[68:69]
	v_cndmask_b32_e32 v66, v138, v80, vcc
	v_cmp_le_i32_e32 vcc, v67, v108
	s_nop 1
	v_cndmask_b32_e32 v67, v138, v81, vcc
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
	ds_bpermute_b32 v101, v134, v100
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v101, v100, v101
	;;#ASMSTART
	v_add_f32 v100, v84, v136
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v101, v100
	v_mov_b32_e32 v100, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_34
	;;#ASMSTART
	v_add_f32 v101, v101, v137
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v101
	v_exp_f32_e32 v100, v84
	v_mov_b32_e32 v84, v101
.LBB0_34:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[102:103], v[66:67], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67], v[80:81], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[104:105], v[68:69], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	v_pk_add_f32 v[68:69], v[78:79], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[70:71], v[76:77], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[72:73], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v69
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[74:75], v[84:85] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v101, v101, v70
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
	v_exp_f32 v74, v108
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v75, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v106
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v101, v101, v71
	v_add_f32_e32 v101, v101, v72
	v_add_f32_e32 v101, v101, v73
	v_add_f32_e32 v101, v101, v74
	v_add_f32_e32 v101, v101, v75
	v_add_f32_e32 v101, v101, v76
	;;#ASMSTART
	v_exp_f32 v77, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v104
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v79, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v102
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v100
	v_add_f32_e32 v101, v101, v77
	v_add_f32_e32 v101, v101, v78
	v_add_f32_e32 v101, v101, v79
	v_add_f32_e32 v101, v101, v80
	;;#ASMSTART
	v_exp_f32 v81, v103
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v101, v101, v81
	;;#ASMSTART
	v_fma_f32 v140, v140, v100, v101
	;;#ASMEND
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_36
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
.LBB0_36:
	s_or_b64 exec, exec, s[40:41]
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
	v_perm_b32 v66, v67, v66, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s81
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[200:203], v119 offset:8192
	ds_read_b128 v[196:199], v127
	v_mfma_f32_32x32x8_bf16 v[2:17], v[208:209], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[220:221], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[66:67], v[34:49]
	ds_read_b128 v[192:195], v128
	ds_read_b128 v[184:187], v129
	v_mfma_f32_32x32x8_bf16 v[50:65], v[224:225], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[210:211], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[222:223], v[68:69], v[18:33]
	ds_read_b128 v[188:191], v130
	ds_read_b128 v[106:109], v131
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[226:227], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[216:217], v[70:71], v[2:17]
	ds_read_b128 v[110:113], v132
	ds_read_b128 v[102:105], v133
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[218:219], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[72:73], v[50:65]
	s_add_i32 s42, s8, 1
	s_cmp_gt_i32 s38, s42
	s_cselect_b64 s[40:41], -1, 0
	s_cmp_le_i32 s38, s42
	s_cbranch_scc1 .LBB0_41
	ds_write_b128 v118, v[180:183]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[144:145], 0
	v_mov_b32_e32 v100, s44
	s_add_i32 s42, s88, s11
	s_lshl_b32 s42, s42, 12
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[146:147], v[66:81]
	buffer_load_dword v114, v100, s[4:7], 0 offen
	v_lshl_or_b32 v100, s45, 11, v117
	v_ashrrev_i32_e32 v101, 31, v100
	v_lshl_add_u64 v[100:101], v[100:101], 2, s[0:1]
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s46, v114
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[150:151], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[152:153], v[66:81]
	global_load_dwordx4 v[180:183], v[100:101], off
	;;#ASMSTART
	v_mov_b32 v100, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v101, 3, v100
	v_lshlrev_b32_e32 v100, 5, v100
	v_and_b32_e32 v101, 0xf8, v101
	v_and_b32_e32 v100, 0x400, v100
	v_or3_b32 v100, v101, v100, s42
	v_ashrrev_i32_e32 v101, 31, v100
	v_lshl_add_u64 v[100:101], v[100:101], 1, s[22:23]
	global_load_dwordx4 v[208:211], v[100:101], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[154:155], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[158:159], v[66:81]
	global_load_dwordx4 v[220:223], v[100:101], off offset:512
	global_load_dwordx4 v[212:215], v[100:101], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[162:163], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[164:165], v[66:81]
	global_load_dwordx4 v[224:227], v[100:101], off offset:1536
	v_add_co_u32_e32 v100, vcc, s80, v100
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	global_load_dwordx4 v[216:219], v[100:101], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[166:167], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[170:171], v[66:81]
	global_load_dwordx4 v[228:231], v[100:101], off offset:512
	global_load_dwordx4 v[232:235], v[100:101], off offset:1024
	global_load_dwordx4 v[204:207], v[100:101], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[174:175], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v100, v0
	;;#ASMEND
	v_mov_b32_e32 v142, 1.0
	v_lshrrev_b32_e32 v101, 2, v100
	v_and_b32_e32 v101, 8, v101
	v_lshrrev_b32_e32 v100, 1, v100
	v_and_or_b32 v100, v100, s79, v141
	v_add_u32_e32 v101, s37, v101
	v_add_u32_e32 v100, s86, v100
	v_subrev_u32_e32 v102, 23, v101
	v_cmp_lt_i32_e32 vcc, v102, v100
	v_mov_b32_e32 v103, v84
	v_mov_b32_e32 v104, v84
	v_cndmask_b32_e32 v67, v138, v67, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_subrev_u32_e32 v102, 21, v101
	v_mov_b32_e32 v105, v84
	v_cndmask_b32_e32 v66, v138, v66, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_subrev_u32_e32 v102, 20, v101
	v_pk_mul_f32 v[66:67], v[82:83], v[66:67]
	v_cndmask_b32_e32 v68, v138, v68, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_subrev_u32_e32 v102, 19, v101
	v_mov_b32_e32 v106, v84
	v_cndmask_b32_e32 v69, v138, v69, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_subrev_u32_e32 v102, 18, v101
	v_pk_mul_f32 v[68:69], v[86:87], v[68:69]
	v_cndmask_b32_e32 v70, v138, v70, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_subrev_u32_e32 v102, 17, v101
	v_mov_b32_e32 v107, v84
	v_cndmask_b32_e32 v71, v138, v71, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, -16, v101
	v_pk_mul_f32 v[70:71], v[88:89], v[70:71]
	v_cndmask_b32_e32 v72, v138, v72, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, -7, v101
	v_mov_b32_e32 v108, v84
	v_cndmask_b32_e32 v73, v138, v73, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, -6, v101
	v_pk_mul_f32 v[72:73], v[90:91], v[72:73]
	v_cndmask_b32_e32 v74, v138, v74, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, -5, v101
	v_mov_b32_e32 v109, v84
	v_cndmask_b32_e32 v75, v138, v75, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, -4, v101
	v_pk_mul_f32 v[74:75], v[92:93], v[74:75]
	v_cndmask_b32_e32 v76, v138, v76, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, -3, v101
	v_mov_b32_e32 v110, v84
	v_cndmask_b32_e32 v77, v138, v77, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, -2, v101
	v_pk_mul_f32 v[76:77], v[94:95], v[76:77]
	v_cndmask_b32_e32 v78, v138, v78, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_add_u32_e32 v102, -1, v101
	v_mov_b32_e32 v111, v84
	v_cndmask_b32_e32 v79, v138, v79, vcc
	v_cmp_le_i32_e32 vcc, v102, v100
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79]
	v_mov_b32_e32 v102, v84
	v_cndmask_b32_e32 v80, v138, v80, vcc
	v_cmp_le_i32_e32 vcc, v101, v100
	v_max_f32_e32 v100, v66, v67
	v_max3_f32 v100, v100, v68, v69
	v_max3_f32 v100, v100, v70, v71
	v_max3_f32 v100, v100, v72, v73
	v_max3_f32 v100, v100, v74, v75
	v_cndmask_b32_e32 v81, v138, v81, vcc
	v_max3_f32 v100, v100, v76, v77
	v_pk_mul_f32 v[80:81], v[98:99], v[80:81]
	v_max3_f32 v100, v100, v78, v79
	v_max3_f32 v100, v100, v80, v81
	ds_bpermute_b32 v101, v134, v100
	v_mov_b32_e32 v112, v84
	v_mov_b32_e32 v113, v84
	v_mov_b32_e32 v114, v84
	v_mov_b32_e32 v115, v84
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v101, v101, v101
	v_max_f32_e32 v143, v100, v101
	;;#ASMSTART
	v_add_f32 v100, v84, v136
	;;#ASMEND
	v_mov_b32_e32 v101, v84
	v_cmp_gt_f32_e32 vcc, v143, v100
	v_mov_b32_e32 v100, v84
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_39
	;;#ASMSTART
	v_add_f32 v100, v143, v137
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v84, v84, v100
	v_exp_f32_e32 v142, v84
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
.LBB0_39:
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
	v_cmp_gt_f32_e32 vcc, 1.0, v142
	v_add_f32_e32 v100, v100, v79
	v_add_f32_e32 v100, v100, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v100, v100, v81
	;;#ASMSTART
	v_fma_f32 v140, v140, v142, v100
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_30
	;;#ASMSTART
	v_mul_f32 v2, v2, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v142
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v142
	;;#ASMEND
	s_branch .LBB0_30
.LBB0_41:
	s_mov_b32 s46, s45
	s_branch .LBB0_31
.LBB0_42:
	;;#ASMSTART
	v_mov_b32 v68, v0
	;;#ASMEND
	ds_bpermute_b32 v66, v134, v140
	v_lshrrev_b32_e32 v67, 1, v68
	v_and_b32_e32 v69, 31, v68
	v_and_or_b32 v67, v67, s79, v69
	v_and_b32_e32 v68, 32, v68
	v_cmp_eq_u32_e32 vcc, 0, v68
	v_cmp_gt_i32_e64 s[0:1], s75, v67
	s_and_b64 s[4:5], vcc, s[0:1]
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v66, v140, v66
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_44
	v_cmp_gt_f32_e32 vcc, s82, v66
	s_ashr_i32 s75, s74, 31
	s_lshl_b64 s[4:5], s[74:75], 6
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v66, v69
	v_log_f32_e32 v69, v69
	v_cndmask_b32_e32 v68, 0, v139, vcc
	s_add_u32 s6, s68, s4
	s_addc_u32 s8, s69, s5
	v_sub_f32_e32 v68, v69, v68
	v_add_f32_e32 v68, v84, v68
	s_lshl_b64 s[4:5], s[26:27], 2
	v_mul_f32_e32 v68, 0x3f317218, v68
	v_cmp_lt_f32_e32 vcc, 0, v66
	s_add_u32 s4, s6, s4
	s_addc_u32 s5, s8, s5
	v_cndmask_b32_e32 v68, v138, v68, vcc
	v_lshlrev_b32_e32 v67, 6, v67
	global_store_dword v67, v68, s[4:5]
.LBB0_44:
	s_or_b64 exec, exec, s[0:1]
	v_div_scale_f32 v67, s[0:1], v66, v66, s78
	v_rcp_f32_e32 v68, v67
	v_div_scale_f32 v69, vcc, s78, v66, s78
	v_fma_f32 v70, -v67, v68, 1.0
	v_fmac_f32_e32 v68, v70, v68
	v_mul_f32_e32 v70, v69, v68
	v_fma_f32 v71, -v67, v70, v69
	v_fmac_f32_e32 v70, v71, v68
	v_fma_f32 v67, -v67, v70, v69
	v_div_fmas_f32 v67, v67, v68, v70
	v_div_fixup_f32 v66, v67, v66, s78
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
	v_perm_b32 v28, v3, v2, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v5, v4, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v7, v6, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v9, v8, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v11, v10, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v13, v12, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v15, v14, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v17, v16, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v19, v18, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v21, v20, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v23, v22, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v74, v75, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v72, v73, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v70, v71, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v68, v69, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v66, v67, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v35, v34, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v37, v36, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v39, v38, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v41, v40, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v43, v42, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v45, v44, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v47, v46, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v49, v48, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v51, v50, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v53, v52, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v55, v54, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v57, v56, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v59, v58, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v61, v60, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v63, v62, s81
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v65, v64, s81
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v34, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v34, 0x100, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_46
	s_barrier
.LBB0_46:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v37, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v34, 3, v37
	v_bfe_u32 v36, v37, 6, 3
	v_lshlrev_b32_e32 v35, 7, v37
	v_and_b32_e32 v34, 4, v34
	v_and_or_b32 v34, v35, s83, v34
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_48
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
.LBB0_48:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v39, 0x1ff, v37
	v_lshlrev_b32_e32 v40, 3, v39
	v_and_b32_e32 v37, 56, v37
	v_xor_b32_e32 v37, v40, v37
	v_lshlrev_b32_e32 v37, 1, v37
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v39, 7, v39
	ds_read_b128 v[42:45], v37
	v_and_b32_e32 v39, 0xf800, v39
	s_add_u32 s8, s34, s72
	v_and_b32_e32 v38, 0x78, v40
	v_add_u32_e32 v40, s17, v39
	s_addc_u32 s0, s35, s73
	v_or_b32_e32 v40, v40, v38
	s_and_b32 s9, s0, 0xffff
	s_mov_b32 s11, s7
	v_lshlrev_b32_e32 v40, 1, v40
	v_cmp_eq_u32_e32 vcc, 1, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[42:45], v40, s[8:11], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_50
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
.LBB0_50:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s4, s17, 0x10000
	v_or_b32_e32 v38, v38, v39
	v_add_lshl_u32 v39, s4, v38, 1
	v_cmp_eq_u32_e32 vcc, 2, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[8:11], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_52
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
.LBB0_52:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x10000
	v_add_lshl_u32 v39, s0, v38, 1
	s_mov_b32 s11, s7
	v_cmp_eq_u32_e32 vcc, 3, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[8:11], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_54
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
.LBB0_54:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x20000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 4, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[8:11], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_56
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
.LBB0_56:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x30000
	v_add_lshl_u32 v39, s0, v38, 1
	s_mov_b32 s11, s7
	v_cmp_eq_u32_e32 vcc, 5, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[8:11], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_58
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
.LBB0_58:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x40000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 6, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[8:11], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_60
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
.LBB0_60:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x50000
	v_add_lshl_u32 v39, s0, v38, 1
	s_mov_b32 s11, s7
	v_cmp_eq_u32_e32 vcc, 7, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[8:11], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_62
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
.LBB0_62:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[4:7], v37
	s_add_i32 s4, s4, 0x60000
	v_add_lshl_u32 v2, s4, v38, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v2, s[8:11], 0 offen
	s_barrier
	;;#ASMSTART
	s_mov_b32 s0, s84
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_66
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_65
	s_bcnt1_i32_b64 s6, s[10:11]
	v_mov_b32_e32 v3, s6
	global_atomic_add v3, v135, v3, s[70:71] sc0
.LBB0_65:
	s_or_b64 exec, exec, s[8:9]
	s_lshl_b64 s[8:9], s[0:1], 2
	s_add_u32 s8, s70, s8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s6, v3
	s_addc_u32 s9, s71, s9
	s_nop 0
	v_add_u32_e32 v2, s6, v2
	global_store_dword v135, v2, s[8:9]
	s_waitcnt vmcnt(0)
.LBB0_66:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s70, s0
	s_addc_u32 s1, s71, s1
	s_barrier
	global_load_dword v2, v135, s[0:1]
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s4, v2
	s_add_i32 s0, s4, s33
	s_sub_i32 s33, s0, s2
	s_cmp_ge_i32 s16, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s76
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_69
	s_branch .LBB0_12
.LBB0_67:
	s_mov_b32 s16, s5
.LBB0_68:
	s_sub_i32 s33, s33, s76
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s26, 0, s2
	s_cmp_ge_i32 s16, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s76, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_69:
	s_add_i32 s2, s26, 1
	s_cmp_gt_i32 s2, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s2, 16
	s_cbranch_scc1 .LBB0_72
	s_add_i32 s5, s16, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s76
	s_cbranch_scc1 .LBB0_67
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[8:9], s[16:17], 2
	s_add_u32 s8, s12, s8
	s_addc_u32 s9, s13, s9
	global_load_dwordx2 v[2:3], v135, s[8:9] offset:4
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s6, v3
	v_readfirstlane_b32 s8, v2
	s_sub_i32 s6, s6, s8
	s_addk_i32 s6, 0xff
	s_ashr_i32 s8, s6, 31
	s_lshr_b32 s8, s8, 24
	s_add_i32 s8, s6, s8
	s_ashr_i32 s16, s8, 8
	s_and_b32 s8, s8, 0xffffff00
	s_cmp_lg_u32 s6, s8
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s6, 0
	s_cselect_b64 s[10:11], -1, 0
	s_and_b64 s[8:9], s[10:11], s[8:9]
	s_subb_u32 s6, s16, 0
	s_branch .LBB0_67
.LBB0_72:
	s_mov_b32 s6, s76
	s_branch .LBB0_68
.LBB0_73:
	s_endpgm
.Lfunc_end0:
	.size	attn_kernel_0, .Lfunc_end0-attn_kernel_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_kernel_0
		.amdhsa_group_segment_fixed_size 16384
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
		.amdhsa_next_free_vgpr 236
		.amdhsa_next_free_sgpr 94
		.amdhsa_accum_offset 236
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

	.set .Lattn_kernel_0.num_vgpr, 236
	.set .Lattn_kernel_0.num_agpr, 0
	.set .Lattn_kernel_0.numbered_sgpr, 94
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
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 204
    .max_flat_workgroup_size: 512
    .name:           attn_kernel_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 512
      - 1
      - 1
    .sgpr_count:     100
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     236
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
