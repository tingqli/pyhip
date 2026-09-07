	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	attn_kernel_0
	.p2align	8
	.type	attn_kernel_0,@function
attn_kernel_0:
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dword s3, s[0:1], 0x38
	s_load_dwordx2 s[46:47], s[0:1], 0x90
	s_load_dwordx4 s[4:7], s[0:1], 0x80
	s_mov_b32 s78, 0
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
	s_mov_b32 s88, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s78, s12
.LBB0_3:
	s_sub_i32 s33, s33, s10
	s_and_b64 s[8:9], s[8:9], exec
	s_cselect_b32 s88, 0, s11
	s_cmp_ge_i32 s78, s3
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s33, s92
	s_cselect_b64 s[10:11], -1, 0
	s_or_b64 s[8:9], s[8:9], s[10:11]
	s_and_b64 vcc, exec, s[8:9]
	s_mov_b32 s10, s92
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s11, s88, 1
	s_cmp_gt_i32 s11, 15
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s11, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s12, s78, 1
	s_cmp_ge_i32 s12, s3
	s_mov_b32 s92, s10
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s79, s78, 31
	s_lshl_b64 s[14:15], s[78:79], 2
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
	s_subb_u32 s92, s18, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s92, s10
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s92, s10
	s_mov_b32 s88, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s77, s[4:5], 0x0
	s_load_dword s93, s[6:7], 0x0
	s_cmp_ge_i32 s78, s3
	s_cbranch_scc1 .LBB0_88
	v_and_b32_e32 v3, 31, v0
	v_lshrrev_b32_e32 v5, 2, v0
	v_lshlrev_b32_e32 v7, 10, v0
	v_lshrrev_b32_e32 v2, 1, v0
	s_movk_i32 s60, 0xe0
	v_lshlrev_b32_e32 v4, 11, v3
	v_and_b32_e32 v6, 8, v5
	v_and_b32_e32 v7, 0x70000, v7
	v_lshrrev_b32_e32 v8, 3, v0
	v_and_or_b32 v1, v2, s60, v3
	v_or3_b32 v87, v7, v4, v6
	v_lshlrev_b32_e32 v4, 8, v0
	v_and_b32_e32 v7, 12, v5
	v_and_b32_e32 v2, 32, v2
	v_and_b32_e32 v8, 16, v8
	v_and_b32_e32 v4, 0xf00, v4
	v_or3_b32 v2, v7, v2, v8
	v_and_b32_e32 v5, 64, v5
	v_or3_b32 v82, v2, v5, v4
	v_lshlrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v2, 0x70, v2
	v_lshlrev_b32_e32 v4, 4, v0
	v_xor_b32_e32 v118, v4, v2
	v_lshl_or_b32 v2, v3, 7, v6
	v_and_b32_e32 v3, 48, v4
	v_or_b32_e32 v4, v2, v3
	v_lshlrev_b32_e32 v119, 1, v4
	v_or_b32_e32 v4, 0x1010, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v120, 1, v4
	v_or_b32_e32 v4, 0x1020, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v121, 1, v4
	v_or_b32_e32 v4, 0x1030, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v122, 1, v4
	v_or_b32_e32 v4, 0x1040, v2
	v_lshrrev_b32_e32 v5, 3, v4
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v4, v5, v4
	v_lshlrev_b32_e32 v123, 1, v4
	v_or_b32_e32 v4, 0x1050, v2
	v_lshrrev_b32_e32 v5, 3, v4
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v4, v5, v4
	v_lshlrev_b32_e32 v124, 1, v4
	v_or_b32_e32 v4, 0x1060, v2
	v_lshrrev_b32_e32 v5, 3, v4
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v4, v5, v4
	v_lshlrev_b32_e32 v125, 1, v4
	v_or_b32_e32 v4, 0x1070, v2
	v_lshrrev_b32_e32 v5, 3, v4
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v4, v5, v4
	v_lshlrev_b32_e32 v126, 1, v4
	v_or_b32_e32 v4, 16, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v127, 1, v4
	v_or_b32_e32 v4, 32, v2
	v_xor_b32_e32 v4, v4, v3
	v_lshlrev_b32_e32 v128, 1, v4
	v_or_b32_e32 v4, 48, v2
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v129, 1, v3
	v_or_b32_e32 v3, 64, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v130, 1, v3
	v_or_b32_e32 v3, 0x50, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v131, 1, v3
	v_or_b32_e32 v3, 0x60, v2
	v_lshrrev_b32_e32 v4, 3, v3
	v_and_b32_e32 v4, 56, v4
	s_load_dwordx2 s[48:49], s[0:1], 0x0
	s_load_dwordx2 s[50:51], s[0:1], 0x10
	s_load_dwordx2 s[84:85], s[0:1], 0x20
	s_load_dwordx2 s[52:53], s[0:1], 0x50
	s_load_dwordx2 s[54:55], s[0:1], 0x60
	s_load_dwordx2 s[56:57], s[0:1], 0x70
	s_load_dwordx2 s[58:59], s[0:1], 0xa0
	s_load_dwordx2 s[4:5], s[0:1], 0xb0
	v_xor_b32_e32 v3, v4, v3
	v_or_b32_e32 v2, 0x70, v2
	v_lshlrev_b32_e32 v132, 1, v3
	v_lshrrev_b32_e32 v3, 3, v2
	s_load_dwordx2 s[98:99], s[0:1], 0xc0
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v133, 1, v2
	v_and_b32_e32 v2, 63, v0
	s_waitcnt lgkmcnt(0)
	v_writelane_b32 v246, s4, 0
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v1, 6, v1
	v_writelane_b32 v246, s5, 1
	v_mov_b32_e32 v83, 0
	v_xor_b32_e32 v134, 0x80, v2
	s_mov_b32 s71, 0x27000
	s_movk_i32 s94, 0x1000
	v_mov_b32_e32 v135, 0x40e00000
	v_mov_b32_e32 v136, 1.0
	s_mov_b32 s95, 0x7060302
	s_movk_i32 s61, 0xf80
	s_mov_b32 s62, s2
	v_mov_b32_e32 v137, 0xff800000
	v_mov_b32_e32 v138, 0x42000000
	s_mov_b32 s36, 0
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s92, s6
.LBB0_12:
	s_mov_b32 s2, s4
	s_cmp_ge_i32 s78, s3
	s_cbranch_scc1 .LBB0_88
.LBB0_13:
	s_ashr_i32 s79, s78, 31
	s_lshl_b32 s6, s33, 8
	s_lshl_b64 s[0:1], s[78:79], 2
	s_add_u32 s4, s34, s0
	s_addc_u32 s5, s35, s1
	global_load_dwordx2 v[2:3], v83, s[4:5]
	s_mov_b32 s75, s71
	v_lshl_add_u32 v6, s88, 2, v1
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s4, v2
	s_add_i32 s80, s4, s6
	v_readfirstlane_b32 s5, v3
	s_add_i32 s4, s80, 0x100
	s_min_i32 s4, s4, s5
	s_sub_i32 s81, s4, s80
	s_add_u32 s4, s52, s0
	s_addc_u32 s5, s53, s1
	s_add_u32 s0, s46, s0
	s_addc_u32 s1, s47, s1
	s_lshl_b32 s8, s80, 11
	s_ashr_i32 s9, s8, 31
	s_lshl_b32 s6, s80, 4
	s_lshl_b64 s[96:97], s[8:9], 1
	s_add_u32 s72, s48, s96
	global_load_dwordx2 v[4:5], v83, s[4:5]
	global_load_dword v3, v83, s[0:1]
	s_addc_u32 s0, s49, s97
	s_lshl_b32 s79, s88, 7
	s_ashr_i32 s7, s6, 31
	s_lshl_b32 s74, s81, 12
	s_and_b32 s73, s0, 0xffff
	s_lshl_b64 s[0:1], s[6:7], 2
	v_add_lshl_u32 v7, s79, v87, 1
	s_add_u32 s68, s56, s0
	buffer_load_dwordx4 v[142:145], v7, s[72:75], 0 offen
	buffer_load_dwordx4 v[146:149], v7, s[72:75], 0 offen offset:32
	buffer_load_dwordx4 v[150:153], v7, s[72:75], 0 offen offset:64
	buffer_load_dwordx4 v[154:157], v7, s[72:75], 0 offen offset:96
	buffer_load_dwordx4 v[158:161], v7, s[72:75], 0 offen offset:128
	buffer_load_dwordx4 v[162:165], v7, s[72:75], 0 offen offset:160
	s_addc_u32 s0, s57, s1
	s_lshl_b32 s70, s81, 6
	s_and_b32 s69, s0, 0xffff
	buffer_load_dword v2, v6, s[68:71], 0 offen
	buffer_load_dwordx4 v[166:169], v7, s[72:75], 0 offen offset:192
	buffer_load_dwordx4 v[170:173], v7, s[72:75], 0 offen offset:224
	s_waitcnt vmcnt(10)
	v_readfirstlane_b32 s0, v4
	s_waitcnt vmcnt(9)
	v_readfirstlane_b32 s9, v3
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
	s_ashr_i32 s89, s88, 31
	s_sub_i32 s72, s1, s0
	s_lshr_b32 s1, s89, 28
	s_add_i32 s1, s88, s1
	s_add_i32 s10, s72, -1
	s_ashr_i32 s8, s1, 4
	s_and_b32 s1, s1, -16
	s_cmp_lg_u32 s88, s1
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s88, 0
	s_cselect_b64 s[6:7], -1, 0
	s_and_b64 s[4:5], s[6:7], s[4:5]
	s_subb_u32 s1, s8, 0
	s_lshl_b32 s82, s1, 13
	s_ashr_i32 s83, s82, 31
	s_lshl_b64 s[4:5], s[82:83], 1
	s_add_u32 s86, s50, s4
	s_addc_u32 s87, s51, s5
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s68, s54, s0
	s_addc_u32 s0, s55, s1
	s_lshl_b32 s70, s72, 2
	s_and_b32 s69, s0, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[68:71], 0
	s_waitcnt vmcnt(3)
	v_mul_f32_e32 v2, s77, v2
	v_mul_f32_e32 v84, 0x3e0293ee, v2
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v4
	v_readfirstlane_b32 s76, v5
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
	buffer_load_dword v2, off, s[68:71], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s75, v2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_lshl_b32 s0, s8, 12
	v_or_b32_e32 v2, s0, v82
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 2, s[86:87]
	global_load_dwordx4 v[6:9], v[4:5], off
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_ashr_i32 s0, s0, 31
	v_mov_b32_e32 v3, s0
	buffer_load_dword v4, off, s[68:71], 0 offset:12
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[86:87]
	global_load_dwordx4 v[174:177], v[2:3], off offset:512
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s83, v4
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	v_lshl_or_b32 v2, s76, 12, v82
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[86:87]
	global_load_dwordx4 v[178:181], v[2:3], off
	ds_write_b128 v118, v[6:9] offset:8192
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	ds_read_b128 v[210:213], v119 offset:8192
	ds_read_b128 v[206:209], v120
	ds_read_b128 v[202:205], v121
	ds_read_b128 v[194:197], v122
	ds_read_b128 v[198:201], v123
	ds_read_b128 v[186:189], v124
	ds_read_b128 v[190:193], v125
	ds_read_b128 v[182:185], v126
	s_add_i32 s0, s72, -2
	s_bitcmp0_b32 s72, 0
	s_cselect_b32 s90, s0, s10
	s_ashr_i32 s91, s90, 31
	s_cmp_lt_i32 s90, 1
	s_cbranch_scc1 .LBB0_35
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v85, v84
	v_mov_b32_e32 v88, v84
	v_mov_b32_e32 v89, v84
	v_mov_b32_e32 v90, v84
	v_mov_b32_e32 v91, v84
	v_mov_b32_e32 v92, v84
	v_mov_b32_e32 v93, v84
	v_mov_b32_e32 v94, v84
	v_mov_b32_e32 v95, v84
	v_mov_b32_e32 v96, v84
	v_mov_b32_e32 v97, v84
	v_mov_b32_e32 v98, v84
	v_mov_b32_e32 v99, v84
	v_mov_b32_e32 v100, v84
	v_mov_b32_e32 v101, v84
	s_mov_b64 s[0:1], 0
	s_mov_b32 s11, 20
	v_mov_b32_e32 v86, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v139
	v_mov_b32_e32 v4, v139
	v_mov_b32_e32 v5, v139
	v_mov_b32_e32 v6, v139
	v_mov_b32_e32 v7, v139
	v_mov_b32_e32 v8, v139
	v_mov_b32_e32 v9, v139
	v_mov_b32_e32 v10, v139
	v_mov_b32_e32 v11, v139
	v_mov_b32_e32 v12, v139
	v_mov_b32_e32 v13, v139
	v_mov_b32_e32 v14, v139
	v_mov_b32_e32 v15, v139
	v_mov_b32_e32 v16, v139
	v_mov_b32_e32 v17, v139
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v139
	v_mov_b32_e32 v20, v139
	v_mov_b32_e32 v21, v139
	v_mov_b32_e32 v22, v139
	v_mov_b32_e32 v23, v139
	v_mov_b32_e32 v24, v139
	v_mov_b32_e32 v25, v139
	v_mov_b32_e32 v26, v139
	v_mov_b32_e32 v27, v139
	v_mov_b32_e32 v28, v139
	v_mov_b32_e32 v29, v139
	v_mov_b32_e32 v30, v139
	v_mov_b32_e32 v31, v139
	v_mov_b32_e32 v32, v139
	v_mov_b32_e32 v33, v139
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v139
	v_mov_b32_e32 v36, v139
	v_mov_b32_e32 v37, v139
	v_mov_b32_e32 v38, v139
	v_mov_b32_e32 v39, v139
	v_mov_b32_e32 v40, v139
	v_mov_b32_e32 v41, v139
	v_mov_b32_e32 v42, v139
	v_mov_b32_e32 v43, v139
	v_mov_b32_e32 v44, v139
	v_mov_b32_e32 v45, v139
	v_mov_b32_e32 v46, v139
	v_mov_b32_e32 v47, v139
	v_mov_b32_e32 v48, v139
	v_mov_b32_e32 v49, v139
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v139
	v_mov_b32_e32 v52, v139
	v_mov_b32_e32 v53, v139
	v_mov_b32_e32 v54, v139
	v_mov_b32_e32 v55, v139
	v_mov_b32_e32 v56, v139
	v_mov_b32_e32 v57, v139
	v_mov_b32_e32 v58, v139
	v_mov_b32_e32 v59, v139
	v_mov_b32_e32 v60, v139
	v_mov_b32_e32 v61, v139
	v_mov_b32_e32 v62, v139
	v_mov_b32_e32 v63, v139
	v_mov_b32_e32 v64, v139
	v_mov_b32_e32 v65, v139
	s_branch .LBB0_18
.LBB0_17:
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
	v_perm_b32 v66, v67, v66, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s95
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[210:213], v119 offset:8192
	ds_read_b128 v[206:209], v120
	v_mfma_f32_32x32x8_bf16 v[2:17], v[182:183], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[186:187], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[190:191], v[66:67], v[34:49]
	ds_read_b128 v[202:205], v121
	ds_read_b128 v[194:197], v122
	v_mfma_f32_32x32x8_bf16 v[50:65], v[218:219], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[184:185], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[188:189], v[68:69], v[18:33]
	ds_read_b128 v[198:201], v123
	ds_read_b128 v[186:189], v124
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[222:223], v[70:71], v[2:17]
	ds_read_b128 v[190:193], v125
	ds_read_b128 v[182:185], v126
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[214:215], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[224:225], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[216:217], v[72:73], v[50:65]
	s_add_u32 s0, s0, 2
	s_addc_u32 s1, s1, 0
	v_mov_b64_e32 v[66:67], s[90:91]
	v_cmp_lt_i64_e32 vcc, s[0:1], v[66:67]
	s_add_i32 s11, s11, 8
	s_mov_b32 s76, s12
	s_cbranch_vccz .LBB0_34
.LBB0_18:
	ds_write_b128 v118, v[174:177]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[142:143], 0
	s_add_i32 s4, s11, -4
	v_mov_b32_e32 v102, s4
	s_lshl_b32 s4, s76, 12
	s_ashr_i32 s5, s4, 31
	v_mov_b32_e32 v103, s5
	s_lshl_b32 s6, s8, 13
	s_add_i32 s6, s6, s82
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[144:145], v[66:81]
	buffer_load_dword v116, v102, s[68:71], 0 offen
	v_or_b32_e32 v102, s4, v82
	v_lshl_add_u64 v[102:103], v[102:103], 2, s[86:87]
	s_mov_b32 s8, s75
	s_mov_b32 s12, s83
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s75, v116
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[150:151], v[66:81]
	global_load_dwordx4 v[174:177], v[102:103], off offset:512
	;;#ASMSTART
	v_mov_b32 v102, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v103, 3, v102
	v_lshlrev_b32_e32 v102, 5, v102
	v_and_b32_e32 v103, 0xf8, v103
	v_and_b32_e32 v102, 0x400, v102
	v_or3_b32 v102, v103, v102, s6
	v_ashrrev_i32_e32 v103, 31, v102
	v_lshl_add_u64 v[102:103], v[102:103], 1, s[84:85]
	global_load_dwordx4 v[108:111], v[102:103], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[156:157], v[66:81]
	global_load_dwordx4 v[202:205], v[102:103], off offset:512
	global_load_dwordx4 v[112:115], v[102:103], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[162:163], v[66:81]
	global_load_dwordx4 v[198:201], v[102:103], off offset:1536
	v_add_co_u32_e32 v102, vcc, s94, v102
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	global_load_dwordx4 v[194:197], v[102:103], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[168:169], v[66:81]
	global_load_dwordx4 v[186:189], v[102:103], off offset:512
	global_load_dwordx4 v[190:193], v[102:103], off offset:1024
	global_load_dwordx4 v[104:107], v[102:103], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[172:173], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[84:85], v[66:67]
	v_pk_mul_f32 v[68:69], v[88:89], v[68:69]
	v_max_f32_e32 v102, v66, v67
	v_pk_mul_f32 v[70:71], v[90:91], v[70:71]
	v_max3_f32 v102, v102, v68, v69
	v_pk_mul_f32 v[72:73], v[92:93], v[72:73]
	v_max3_f32 v102, v102, v70, v71
	v_pk_mul_f32 v[74:75], v[94:95], v[74:75]
	v_max3_f32 v102, v102, v72, v73
	v_pk_mul_f32 v[76:77], v[96:97], v[76:77]
	v_max3_f32 v102, v102, v74, v75
	v_pk_mul_f32 v[78:79], v[98:99], v[78:79]
	v_max3_f32 v102, v102, v76, v77
	v_pk_mul_f32 v[80:81], v[100:101], v[80:81]
	v_max3_f32 v102, v102, v78, v79
	v_max3_f32 v102, v102, v80, v81
	ds_bpermute_b32 v103, v134, v102
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v103, v103, v103
	v_max_f32_e32 v103, v102, v103
	;;#ASMSTART
	v_add_f32 v102, v86, v135
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v103, v102
	v_mov_b32_e32 v102, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_20
	;;#ASMSTART
	v_add_f32 v103, v103, v136
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v86, v86, v103
	v_exp_f32_e32 v102, v86
	v_mov_b32_e32 v86, v103
.LBB0_20:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v103, 0, v66
	v_add_f32_e32 v103, v103, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v103, v103, v68
	v_add_f32_e32 v103, v103, v69
	v_add_f32_e32 v103, v103, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v103, v103, v71
	v_add_f32_e32 v103, v103, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v103, v103, v73
	v_add_f32_e32 v103, v103, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v103, v103, v75
	v_add_f32_e32 v103, v103, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v103, v103, v77
	v_add_f32_e32 v103, v103, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v102
	v_add_f32_e32 v103, v103, v79
	v_add_f32_e32 v103, v103, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v103, v103, v81
	;;#ASMSTART
	v_fma_f32 v139, v139, v102, v103
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_22
	;;#ASMSTART
	v_mul_f32 v2, v2, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v102
	;;#ASMEND
.LBB0_22:
	s_or_b64 exec, exec, s[4:5]
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
	v_perm_b32 v66, v67, v66, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s95
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[182:185], v119
	ds_read_b128 v[206:209], v127
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[202:203], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[112:113], v[66:67], v[34:49]
	ds_read_b128 v[210:213], v128
	ds_read_b128 v[214:217], v129
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[204:205], v[68:69], v[18:33]
	ds_read_b128 v[108:111], v130
	ds_read_b128 v[218:221], v131
	v_mfma_f32_32x32x8_bf16 v[34:49], v[114:115], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[194:195], v[70:71], v[2:17]
	ds_read_b128 v[112:115], v132
	ds_read_b128 v[222:225], v133
	v_mfma_f32_32x32x8_bf16 v[18:33], v[186:187], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[190:191], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[104:105], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[196:197], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[188:189], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[72:73], v[50:65]
	ds_write_b128 v118, v[178:181] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[142:143], 0
	s_lshl_b32 s4, s8, 12
	v_or_b32_e32 v102, s4, v82
	v_ashrrev_i32_e32 v103, 31, v102
	v_lshl_add_u64 v[102:103], v[102:103], 2, s[86:87]
	s_addk_i32 s6, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[144:145], v[66:81]
	global_load_dwordx4 v[178:181], v[102:103], off
	;;#ASMSTART
	v_mov_b32 v102, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v103, 3, v102
	v_lshlrev_b32_e32 v102, 5, v102
	v_and_b32_e32 v103, 0xf8, v103
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[146:147], v[66:81]
	v_and_b32_e32 v102, 0x400, v102
	v_or3_b32 v102, v103, v102, s6
	v_ashrrev_i32_e32 v103, 31, v102
	v_lshl_add_u64 v[102:103], v[102:103], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[150:151], v[66:81]
	global_load_dwordx4 v[186:189], v[102:103], off
	global_load_dwordx4 v[190:193], v[102:103], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	global_load_dwordx4 v[194:197], v[102:103], off offset:1024
	global_load_dwordx4 v[198:201], v[102:103], off offset:1536
	v_add_co_u32_e32 v102, vcc, s94, v102
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[162:163], v[66:81]
	global_load_dwordx4 v[202:205], v[102:103], off
	global_load_dwordx4 v[206:209], v[102:103], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[114:115], v[168:169], v[66:81]
	global_load_dwordx4 v[210:213], v[102:103], off offset:1024
	global_load_dwordx4 v[182:185], v[102:103], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[172:173], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[84:85], v[66:67]
	v_pk_mul_f32 v[68:69], v[88:89], v[68:69]
	v_max_f32_e32 v102, v66, v67
	v_pk_mul_f32 v[70:71], v[90:91], v[70:71]
	v_max3_f32 v102, v102, v68, v69
	v_pk_mul_f32 v[72:73], v[92:93], v[72:73]
	v_max3_f32 v102, v102, v70, v71
	v_pk_mul_f32 v[74:75], v[94:95], v[74:75]
	v_max3_f32 v102, v102, v72, v73
	v_pk_mul_f32 v[76:77], v[96:97], v[76:77]
	v_max3_f32 v102, v102, v74, v75
	v_pk_mul_f32 v[78:79], v[98:99], v[78:79]
	v_max3_f32 v102, v102, v76, v77
	v_pk_mul_f32 v[80:81], v[100:101], v[80:81]
	v_max3_f32 v102, v102, v78, v79
	v_max3_f32 v102, v102, v80, v81
	ds_bpermute_b32 v103, v134, v102
	v_mov_b32_e32 v140, 1.0
	v_mov_b32_e32 v104, v86
	v_mov_b32_e32 v105, v86
	v_mov_b32_e32 v106, v86
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v103, v103, v103
	v_max_f32_e32 v141, v102, v103
	;;#ASMSTART
	v_add_f32 v102, v86, v135
	;;#ASMEND
	v_mov_b32_e32 v103, v86
	v_cmp_gt_f32_e32 vcc, v141, v102
	v_mov_b32_e32 v102, v86
	v_mov_b32_e32 v107, v86
	v_mov_b32_e32 v108, v86
	v_mov_b32_e32 v109, v86
	v_mov_b32_e32 v110, v86
	v_mov_b32_e32 v111, v86
	v_mov_b32_e32 v112, v86
	v_mov_b32_e32 v113, v86
	v_mov_b32_e32 v114, v86
	v_mov_b32_e32 v115, v86
	v_mov_b32_e32 v116, v86
	v_mov_b32_e32 v117, v86
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_24
	;;#ASMSTART
	v_add_f32 v102, v141, v136
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v86, v86, v102
	v_exp_f32_e32 v140, v86
	v_mov_b32_e32 v86, v102
	v_mov_b32_e32 v103, v102
	v_mov_b32_e32 v104, v102
	v_mov_b32_e32 v105, v102
	v_mov_b32_e32 v106, v102
	v_mov_b32_e32 v107, v102
	v_mov_b32_e32 v108, v102
	v_mov_b32_e32 v109, v102
	v_mov_b32_e32 v110, v102
	v_mov_b32_e32 v111, v102
	v_mov_b32_e32 v112, v102
	v_mov_b32_e32 v113, v102
	v_mov_b32_e32 v114, v102
	v_mov_b32_e32 v115, v102
	v_mov_b32_e32 v116, v102
	v_mov_b32_e32 v117, v102
.LBB0_24:
	s_or_b64 exec, exec, s[6:7]
	v_pk_add_f32 v[66:67], v[66:67], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, 0, v66
	v_add_f32_e32 v141, v141, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v68
	v_add_f32_e32 v141, v141, v69
	v_add_f32_e32 v141, v141, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v71
	v_add_f32_e32 v141, v141, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v73
	v_add_f32_e32 v141, v141, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v75
	v_add_f32_e32 v141, v141, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[116:117] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v77
	v_add_f32_e32 v141, v141, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v140
	v_add_f32_e32 v141, v141, v79
	v_add_f32_e32 v141, v141, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v141, v141, v81
	;;#ASMSTART
	v_fma_f32 v139, v139, v140, v141
	;;#ASMEND
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_26
	;;#ASMSTART
	v_mul_f32 v2, v2, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v140
	;;#ASMEND
.LBB0_26:
	s_or_b64 exec, exec, s[6:7]
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
	v_perm_b32 v66, v67, v66, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s95
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[214:217], v119 offset:8192
	ds_read_b128 v[218:221], v120
	v_mfma_f32_32x32x8_bf16 v[2:17], v[186:187], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[66:67], v[34:49]
	ds_read_b128 v[222:225], v121
	ds_read_b128 v[226:229], v122
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[188:189], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[68:69], v[18:33]
	ds_read_b128 v[230:233], v123
	ds_read_b128 v[234:237], v124
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[202:203], v[70:71], v[2:17]
	ds_read_b128 v[238:241], v125
	ds_read_b128 v[242:245], v126
	v_mfma_f32_32x32x8_bf16 v[18:33], v[206:207], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[210:211], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[182:183], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[184:185], v[72:73], v[50:65]
	ds_write_b128 v118, v[174:177]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[142:143], 0
	v_mov_b32_e32 v140, s11
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s6, s76, 13
	s_add_i32 s6, s6, s82
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[144:145], v[66:81]
	buffer_load_dword v214, v140, s[68:71], 0 offen
	v_lshl_add_u64 v[140:141], s[4:5], 0, v[82:83]
	v_lshl_add_u64 v[140:141], v[140:141], 2, s[86:87]
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s83, v214
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[150:151], v[66:81]
	global_load_dwordx4 v[174:177], v[140:141], off offset:512
	;;#ASMSTART
	v_mov_b32 v140, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v141, 3, v140
	v_lshlrev_b32_e32 v140, 5, v140
	v_and_b32_e32 v141, 0xf8, v141
	v_and_b32_e32 v140, 0x400, v140
	v_or3_b32 v140, v141, v140, s6
	v_ashrrev_i32_e32 v141, 31, v140
	v_lshl_add_u64 v[140:141], v[140:141], 1, s[84:85]
	global_load_dwordx4 v[186:189], v[140:141], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[156:157], v[66:81]
	global_load_dwordx4 v[198:201], v[140:141], off offset:512
	global_load_dwordx4 v[190:193], v[140:141], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[162:163], v[66:81]
	global_load_dwordx4 v[202:205], v[140:141], off offset:1536
	v_add_co_u32_e32 v140, vcc, s94, v140
	s_nop 1
	v_addc_co_u32_e32 v141, vcc, 0, v141, vcc
	global_load_dwordx4 v[194:197], v[140:141], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[168:169], v[66:81]
	global_load_dwordx4 v[206:209], v[140:141], off offset:512
	global_load_dwordx4 v[210:213], v[140:141], off offset:1024
	global_load_dwordx4 v[182:185], v[140:141], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[244:245], v[172:173], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[84:85], v[66:67]
	v_pk_mul_f32 v[68:69], v[88:89], v[68:69]
	v_max_f32_e32 v140, v66, v67
	v_pk_mul_f32 v[70:71], v[90:91], v[70:71]
	v_max3_f32 v140, v140, v68, v69
	v_pk_mul_f32 v[72:73], v[92:93], v[72:73]
	v_max3_f32 v140, v140, v70, v71
	v_pk_mul_f32 v[74:75], v[94:95], v[74:75]
	v_max3_f32 v140, v140, v72, v73
	v_pk_mul_f32 v[76:77], v[96:97], v[76:77]
	v_max3_f32 v140, v140, v74, v75
	v_pk_mul_f32 v[78:79], v[98:99], v[78:79]
	v_max3_f32 v140, v140, v76, v77
	v_pk_mul_f32 v[80:81], v[100:101], v[80:81]
	v_max3_f32 v140, v140, v78, v79
	v_max3_f32 v140, v140, v80, v81
	ds_bpermute_b32 v141, v134, v140
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v141, v141, v141
	v_max_f32_e32 v141, v140, v141
	;;#ASMSTART
	v_add_f32 v140, v86, v135
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v141, v140
	v_mov_b32_e32 v140, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_28
	;;#ASMSTART
	v_add_f32 v102, v141, v136
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v86, v86, v102
	v_exp_f32_e32 v140, v86
	v_mov_b32_e32 v86, v102
	v_mov_b32_e32 v103, v102
	v_mov_b32_e32 v104, v102
	v_mov_b32_e32 v105, v102
	v_mov_b32_e32 v106, v102
	v_mov_b32_e32 v107, v102
	v_mov_b32_e32 v108, v102
	v_mov_b32_e32 v109, v102
	v_mov_b32_e32 v110, v102
	v_mov_b32_e32 v111, v102
	v_mov_b32_e32 v112, v102
	v_mov_b32_e32 v113, v102
	v_mov_b32_e32 v114, v102
	v_mov_b32_e32 v115, v102
	v_mov_b32_e32 v116, v102
	v_mov_b32_e32 v117, v102
.LBB0_28:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, 0, v66
	v_add_f32_e32 v141, v141, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v68
	v_add_f32_e32 v141, v141, v69
	v_add_f32_e32 v141, v141, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v71
	v_add_f32_e32 v141, v141, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v73
	v_add_f32_e32 v141, v141, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v75
	v_add_f32_e32 v141, v141, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[116:117] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v77
	v_add_f32_e32 v141, v141, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v140
	v_add_f32_e32 v141, v141, v79
	v_add_f32_e32 v141, v141, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v141, v141, v81
	;;#ASMSTART
	v_fma_f32 v139, v139, v140, v141
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_30
	;;#ASMSTART
	v_mul_f32 v2, v2, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v140
	;;#ASMEND
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]
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
	v_perm_b32 v66, v67, v66, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s95
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[214:217], v119
	ds_read_b128 v[218:221], v127
	v_mfma_f32_32x32x8_bf16 v[2:17], v[186:187], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[198:199], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[190:191], v[66:67], v[34:49]
	ds_read_b128 v[222:225], v128
	ds_read_b128 v[226:229], v129
	v_mfma_f32_32x32x8_bf16 v[50:65], v[202:203], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[188:189], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[200:201], v[68:69], v[18:33]
	ds_read_b128 v[198:201], v130
	ds_read_b128 v[230:233], v131
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[194:195], v[70:71], v[2:17]
	ds_read_b128 v[202:205], v132
	ds_read_b128 v[234:237], v133
	v_mfma_f32_32x32x8_bf16 v[18:33], v[206:207], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[210:211], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[182:183], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[196:197], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[184:185], v[72:73], v[50:65]
	ds_write_b128 v118, v[178:181] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[142:143], 0
	v_lshl_or_b32 v140, s12, 12, v82
	v_ashrrev_i32_e32 v141, 31, v140
	v_lshl_add_u64 v[140:141], v[140:141], 2, s[86:87]
	s_addk_i32 s6, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[144:145], v[66:81]
	global_load_dwordx4 v[178:181], v[140:141], off
	;;#ASMSTART
	v_mov_b32 v140, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v141, 3, v140
	v_lshlrev_b32_e32 v140, 5, v140
	v_and_b32_e32 v141, 0xf8, v141
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[146:147], v[66:81]
	v_and_b32_e32 v140, 0x400, v140
	v_or3_b32 v140, v141, v140, s6
	v_ashrrev_i32_e32 v141, 31, v140
	v_lshl_add_u64 v[140:141], v[140:141], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[150:151], v[66:81]
	global_load_dwordx4 v[182:185], v[140:141], off
	global_load_dwordx4 v[186:189], v[140:141], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[156:157], v[66:81]
	global_load_dwordx4 v[190:193], v[140:141], off offset:1024
	global_load_dwordx4 v[218:221], v[140:141], off offset:1536
	v_add_co_u32_e32 v140, vcc, s94, v140
	s_nop 1
	v_addc_co_u32_e32 v141, vcc, 0, v141, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[230:231], v[162:163], v[66:81]
	global_load_dwordx4 v[222:225], v[140:141], off
	global_load_dwordx4 v[226:229], v[140:141], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[232:233], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[168:169], v[66:81]
	global_load_dwordx4 v[230:233], v[140:141], off offset:1024
	global_load_dwordx4 v[214:217], v[140:141], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[234:235], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[172:173], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[84:85], v[66:67]
	v_pk_mul_f32 v[68:69], v[88:89], v[68:69]
	v_max_f32_e32 v140, v66, v67
	v_pk_mul_f32 v[70:71], v[90:91], v[70:71]
	v_max3_f32 v140, v140, v68, v69
	v_pk_mul_f32 v[72:73], v[92:93], v[72:73]
	v_max3_f32 v140, v140, v70, v71
	v_pk_mul_f32 v[74:75], v[94:95], v[74:75]
	v_max3_f32 v140, v140, v72, v73
	v_pk_mul_f32 v[76:77], v[96:97], v[76:77]
	v_max3_f32 v140, v140, v74, v75
	v_pk_mul_f32 v[78:79], v[98:99], v[78:79]
	v_max3_f32 v140, v140, v76, v77
	v_pk_mul_f32 v[80:81], v[100:101], v[80:81]
	v_max3_f32 v140, v140, v78, v79
	v_max3_f32 v140, v140, v80, v81
	ds_bpermute_b32 v141, v134, v140
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v141, v141, v141
	v_max_f32_e32 v141, v140, v141
	;;#ASMSTART
	v_add_f32 v140, v86, v135
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v141, v140
	v_mov_b32_e32 v140, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v102, v141, v136
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v86, v86, v102
	v_exp_f32_e32 v140, v86
	v_mov_b32_e32 v86, v102
	v_mov_b32_e32 v103, v102
	v_mov_b32_e32 v104, v102
	v_mov_b32_e32 v105, v102
	v_mov_b32_e32 v106, v102
	v_mov_b32_e32 v107, v102
	v_mov_b32_e32 v108, v102
	v_mov_b32_e32 v109, v102
	v_mov_b32_e32 v110, v102
	v_mov_b32_e32 v111, v102
	v_mov_b32_e32 v112, v102
	v_mov_b32_e32 v113, v102
	v_mov_b32_e32 v114, v102
	v_mov_b32_e32 v115, v102
	v_mov_b32_e32 v116, v102
	v_mov_b32_e32 v117, v102
.LBB0_32:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, 0, v66
	v_add_f32_e32 v102, v102, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, v102, v68
	v_add_f32_e32 v102, v102, v69
	v_add_f32_e32 v102, v102, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, v102, v71
	v_add_f32_e32 v102, v102, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, v102, v73
	v_add_f32_e32 v102, v102, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, v102, v75
	v_add_f32_e32 v102, v102, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[116:117] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, v102, v77
	v_add_f32_e32 v102, v102, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v140
	v_add_f32_e32 v102, v102, v79
	v_add_f32_e32 v102, v102, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v102, v102, v81
	;;#ASMSTART
	v_fma_f32 v139, v139, v140, v102
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_17
	;;#ASMSTART
	v_mul_f32 v2, v2, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v140
	;;#ASMEND
	s_branch .LBB0_17
.LBB0_34:
	s_mov_b32 s76, s12
	s_cmp_ge_i32 s90, s72
	s_cbranch_scc0 .LBB0_36
	s_branch .LBB0_57
.LBB0_35:
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
	s_mov_b64 s[6:7], s[50:51]
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	s_mov_b64 s[12:13], s[52:53]
	s_mov_b32 s52, s36
	s_mov_b32 s53, s36
	s_mov_b64 s[14:15], s[54:55]
	s_mov_b32 s54, s36
	s_mov_b32 s55, s36
	s_mov_b64 s[16:17], s[56:57]
	s_mov_b32 s56, s36
	s_mov_b32 s57, s36
	s_mov_b64 s[18:19], s[58:59]
	s_mov_b32 s58, s36
	s_mov_b32 s59, s36
	s_mov_b32 s60, s36
	s_mov_b32 s61, s36
	s_mov_b32 s11, s62
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
	v_mov_b32_e32 v86, 0xff800000
	v_mov_b32_e32 v139, 0
	s_mov_b32 s62, s11
	s_movk_i32 s61, 0xf80
	s_movk_i32 s60, 0xe0
	s_mov_b64 s[58:59], s[18:19]
	s_mov_b64 s[56:57], s[16:17]
	s_mov_b64 s[54:55], s[14:15]
	s_mov_b64 s[52:53], s[12:13]
	s_mov_b64 s[50:51], s[6:7]
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
	s_cmp_ge_i32 s90, s72
	s_cbranch_scc1 .LBB0_57
.LBB0_36:
	s_lshl_b32 s37, s10, 6
	s_lshl_b32 s0, s90, 6
	s_ashr_i32 s73, s72, 31
	s_add_i32 s37, s37, s9
	v_mov_b32_e32 v85, v84
	v_mov_b32_e32 v88, v84
	v_mov_b32_e32 v89, v84
	v_mov_b32_e32 v90, v84
	v_mov_b32_e32 v91, v84
	v_mov_b32_e32 v92, v84
	v_mov_b32_e32 v93, v84
	v_mov_b32_e32 v94, v84
	v_mov_b32_e32 v95, v84
	v_mov_b32_e32 v96, v84
	v_mov_b32_e32 v97, v84
	v_mov_b32_e32 v98, v84
	v_mov_b32_e32 v99, v84
	v_mov_b32_e32 v100, v84
	v_mov_b32_e32 v101, v84
	s_add_i32 s42, s0, 0x77
	s_add_i32 s43, s90, 1
	s_lshl2_add_u32 s44, s90, 20
	s_branch .LBB0_39
.LBB0_37:
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
	v_perm_b32 v66, v67, v66, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s95
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[210:213], v119 offset:8192
	ds_read_b128 v[206:209], v120
	v_mfma_f32_32x32x8_bf16 v[2:17], v[182:183], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[186:187], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[190:191], v[66:67], v[34:49]
	ds_read_b128 v[202:205], v121
	ds_read_b128 v[194:197], v122
	v_mfma_f32_32x32x8_bf16 v[50:65], v[218:219], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[184:185], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[188:189], v[68:69], v[18:33]
	ds_read_b128 v[198:201], v123
	ds_read_b128 v[186:189], v124
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[222:223], v[70:71], v[2:17]
	ds_read_b128 v[190:193], v125
	ds_read_b128 v[182:185], v126
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[214:215], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[224:225], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[216:217], v[72:73], v[50:65]
.LBB0_38:
	s_and_b64 s[0:1], s[38:39], exec
	s_cselect_b32 s8, s75, s76
	s_cselect_b32 s76, s83, s75
	s_cselect_b32 s75, s45, s83
	s_add_u32 s90, s90, 2
	s_addc_u32 s91, s91, 0
	v_mov_b64_e32 v[66:67], s[72:73]
	v_cmp_lt_i64_e32 vcc, s[90:91], v[66:67]
	s_addk_i32 s42, 0x80
	s_add_i32 s43, s43, 2
	s_add_i32 s44, s44, 8
	s_mov_b32 s83, s40
	s_cbranch_vccz .LBB0_57
.LBB0_39:
	ds_write_b128 v118, v[174:177]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[142:143], 0
	s_add_i32 s0, s44, -4
	v_mov_b32_e32 v102, s0
	s_lshl_b32 s0, s76, 12
	s_ashr_i32 s1, s0, 31
	v_mov_b32_e32 v103, s1
	s_lshl_b32 s38, s8, 13
	s_add_i32 s38, s38, s82
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[144:145], v[66:81]
	buffer_load_dword v116, v102, s[68:71], 0 offen
	v_or_b32_e32 v102, s0, v82
	v_lshl_add_u64 v[102:103], v[102:103], 2, s[86:87]
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v116
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[150:151], v[66:81]
	global_load_dwordx4 v[174:177], v[102:103], off offset:512
	;;#ASMSTART
	v_mov_b32 v102, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v103, 3, v102
	v_lshlrev_b32_e32 v102, 5, v102
	v_and_b32_e32 v103, 0xf8, v103
	v_and_b32_e32 v102, 0x400, v102
	v_or3_b32 v102, v103, v102, s38
	v_ashrrev_i32_e32 v103, 31, v102
	v_lshl_add_u64 v[102:103], v[102:103], 1, s[84:85]
	global_load_dwordx4 v[108:111], v[102:103], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[156:157], v[66:81]
	global_load_dwordx4 v[202:205], v[102:103], off offset:512
	global_load_dwordx4 v[112:115], v[102:103], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[162:163], v[66:81]
	global_load_dwordx4 v[198:201], v[102:103], off offset:1536
	v_add_co_u32_e32 v102, vcc, s94, v102
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	global_load_dwordx4 v[194:197], v[102:103], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[168:169], v[66:81]
	global_load_dwordx4 v[186:189], v[102:103], off offset:512
	global_load_dwordx4 v[190:193], v[102:103], off offset:1024
	global_load_dwordx4 v[104:107], v[102:103], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[172:173], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v102, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v102, 2, v102
	v_and_b32_e32 v102, 8, v102
	v_add_u32_e32 v102, s42, v102
	v_add_u32_e32 v103, 0xffffff89, v102
	v_cmp_gt_i32_e32 vcc, s37, v103
	v_add_u32_e32 v103, 0xffffff8a, v102
	v_cmp_gt_i32_e64 s[0:1], s37, v103
	v_add_u32_e32 v103, 0xffffff8b, v102
	v_cmp_gt_i32_e64 s[4:5], s37, v103
	v_add_u32_e32 v103, 0xffffff8c, v102
	v_cmp_gt_i32_e64 s[6:7], s37, v103
	v_add_u32_e32 v103, 0xffffff8d, v102
	v_cmp_gt_i32_e64 s[8:9], s37, v103
	v_add_u32_e32 v103, 0xffffff8e, v102
	v_cmp_gt_i32_e64 s[10:11], s37, v103
	v_add_u32_e32 v103, 0xffffff8f, v102
	v_cmp_gt_i32_e64 s[12:13], s37, v103
	v_add_u32_e32 v103, 0xffffff90, v102
	v_cmp_gt_i32_e64 s[14:15], s37, v103
	v_add_u32_e32 v103, 0xffffff99, v102
	v_cmp_gt_i32_e64 s[16:17], s37, v103
	v_add_u32_e32 v103, 0xffffff9a, v102
	v_cmp_gt_i32_e64 s[18:19], s37, v103
	v_add_u32_e32 v103, 0xffffff9b, v102
	v_cmp_gt_i32_e64 s[20:21], s37, v103
	v_add_u32_e32 v103, 0xffffff9c, v102
	v_cmp_gt_i32_e64 s[22:23], s37, v103
	v_add_u32_e32 v103, 0xffffff9d, v102
	v_cmp_gt_i32_e64 s[24:25], s37, v103
	v_add_u32_e32 v103, 0xffffff9e, v102
	v_cmp_gt_i32_e64 s[26:27], s37, v103
	v_add_u32_e32 v103, 0xffffff9f, v102
	v_add_u32_e32 v102, 0xffffffa0, v102
	v_cmp_gt_i32_e64 s[28:29], s37, v103
	v_cmp_gt_i32_e64 s[30:31], s37, v102
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
	v_cndmask_b32_e64 v79, v137, v79, s[26:27]
	v_cndmask_b32_e64 v78, v137, v78, s[24:25]
	v_cndmask_b32_e64 v141, v137, v67, s[0:1]
	v_cndmask_b32_e32 v140, v137, v66, vcc
	v_cndmask_b32_e64 v77, v137, v77, s[22:23]
	v_cndmask_b32_e64 v76, v137, v76, s[20:21]
	v_cndmask_b32_e64 v75, v137, v75, s[18:19]
	v_cndmask_b32_e64 v74, v137, v74, s[16:17]
	v_cndmask_b32_e64 v103, v137, v71, s[10:11]
	v_cndmask_b32_e64 v102, v137, v70, s[8:9]
	v_cndmask_b32_e64 v117, v137, v69, s[6:7]
	v_cndmask_b32_e64 v116, v137, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[98:99], v[78:79]
	v_pk_mul_f32 v[78:79], v[84:85], v[140:141]
	v_pk_mul_f32 v[68:69], v[96:97], v[76:77]
	v_pk_mul_f32 v[70:71], v[94:95], v[74:75]
	v_pk_mul_f32 v[74:75], v[90:91], v[102:103]
	v_pk_mul_f32 v[76:77], v[88:89], v[116:117]
	v_max_f32_e32 v102, v78, v79
	v_cndmask_b32_e64 v73, v137, v73, s[14:15]
	v_cndmask_b32_e64 v72, v137, v72, s[12:13]
	v_max3_f32 v102, v102, v76, v77
	v_pk_mul_f32 v[72:73], v[92:93], v[72:73]
	v_max3_f32 v102, v102, v74, v75
	v_max3_f32 v102, v102, v72, v73
	v_max3_f32 v102, v102, v70, v71
	v_cndmask_b32_e64 v81, v137, v81, s[30:31]
	v_cndmask_b32_e64 v80, v137, v80, s[28:29]
	v_max3_f32 v102, v102, v68, v69
	v_pk_mul_f32 v[80:81], v[100:101], v[80:81]
	v_max3_f32 v102, v102, v66, v67
	v_max3_f32 v102, v102, v80, v81
	ds_bpermute_b32 v103, v134, v102
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v103, v103, v103
	v_max_f32_e32 v103, v102, v103
	;;#ASMSTART
	v_add_f32 v102, v86, v135
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v103, v102
	v_mov_b32_e32 v102, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_41
	;;#ASMSTART
	v_add_f32 v103, v103, v136
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v86, v86, v103
	v_exp_f32_e32 v102, v86
	v_mov_b32_e32 v86, v103
.LBB0_41:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[116:117], v[66:67], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67], v[78:79], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[140:141], v[68:69], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	v_pk_add_f32 v[68:69], v[76:77], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v103, 0, v66
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	v_pk_add_f32 v[182:183], v[70:71], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v103, v103, v67
	v_add_f32_e32 v103, v103, v68
	v_pk_add_f32 v[70:71], v[74:75], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v103, v103, v69
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
	v_exp_f32 v74, v182
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v75, v183
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v103, v103, v70
	v_add_f32_e32 v103, v103, v71
	v_add_f32_e32 v103, v103, v72
	v_add_f32_e32 v103, v103, v73
	v_add_f32_e32 v103, v103, v74
	v_add_f32_e32 v103, v103, v75
	;;#ASMSTART
	v_exp_f32 v76, v140
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v77, v141
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v116
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[86:87] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v103, v103, v76
	v_add_f32_e32 v103, v103, v77
	v_add_f32_e32 v103, v103, v78
	;;#ASMSTART
	v_exp_f32 v79, v117
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v102
	v_add_f32_e32 v103, v103, v79
	v_add_f32_e32 v103, v103, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v103, v103, v81
	;;#ASMSTART
	v_fma_f32 v139, v139, v102, v103
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_43
	;;#ASMSTART
	v_mul_f32 v2, v2, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v102
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v102
	;;#ASMEND
.LBB0_43:
	s_or_b64 exec, exec, s[0:1]
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
	v_perm_b32 v66, v67, v66, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s95
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[182:185], v119
	ds_read_b128 v[206:209], v127
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[202:203], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[112:113], v[66:67], v[34:49]
	ds_read_b128 v[210:213], v128
	ds_read_b128 v[214:217], v129
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[204:205], v[68:69], v[18:33]
	ds_read_b128 v[108:111], v130
	ds_read_b128 v[202:205], v131
	v_mfma_f32_32x32x8_bf16 v[34:49], v[114:115], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[200:201], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[194:195], v[70:71], v[2:17]
	ds_read_b128 v[112:115], v132
	ds_read_b128 v[198:201], v133
	v_mfma_f32_32x32x8_bf16 v[18:33], v[186:187], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[190:191], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[104:105], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[196:197], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[188:189], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[72:73], v[50:65]
	ds_write_b128 v118, v[178:181] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[142:143], 0
	s_lshl_b32 s40, s75, 12
	v_or_b32_e32 v102, s40, v82
	v_ashrrev_i32_e32 v103, 31, v102
	v_lshl_add_u64 v[102:103], v[102:103], 2, s[86:87]
	s_addk_i32 s38, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[144:145], v[66:81]
	global_load_dwordx4 v[178:181], v[102:103], off
	;;#ASMSTART
	v_mov_b32 v102, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v103, 3, v102
	v_lshlrev_b32_e32 v102, 5, v102
	v_and_b32_e32 v103, 0xf8, v103
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[146:147], v[66:81]
	v_and_b32_e32 v102, 0x400, v102
	v_or3_b32 v102, v103, v102, s38
	v_ashrrev_i32_e32 v103, 31, v102
	v_lshl_add_u64 v[102:103], v[102:103], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[150:151], v[66:81]
	global_load_dwordx4 v[182:185], v[102:103], off
	global_load_dwordx4 v[186:189], v[102:103], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[156:157], v[66:81]
	global_load_dwordx4 v[190:193], v[102:103], off offset:1024
	global_load_dwordx4 v[218:221], v[102:103], off offset:1536
	v_add_co_u32_e32 v102, vcc, s94, v102
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[162:163], v[66:81]
	global_load_dwordx4 v[222:225], v[102:103], off
	global_load_dwordx4 v[226:229], v[102:103], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[114:115], v[168:169], v[66:81]
	global_load_dwordx4 v[230:233], v[102:103], off offset:1024
	global_load_dwordx4 v[214:217], v[102:103], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[172:173], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v102, v0
	;;#ASMEND
	v_mov_b32_e32 v140, 1.0
	v_lshrrev_b32_e32 v102, 2, v102
	v_and_b32_e32 v102, 8, v102
	v_add_u32_e32 v102, s42, v102
	v_add_u32_e32 v103, 0xffffffa9, v102
	v_cmp_gt_i32_e32 vcc, s37, v103
	v_add_u32_e32 v103, 0xffffffaa, v102
	v_cmp_gt_i32_e64 s[0:1], s37, v103
	v_add_u32_e32 v103, 0xffffffab, v102
	v_cmp_gt_i32_e64 s[4:5], s37, v103
	v_add_u32_e32 v103, 0xffffffac, v102
	v_cmp_gt_i32_e64 s[6:7], s37, v103
	v_add_u32_e32 v103, 0xffffffad, v102
	v_cmp_gt_i32_e64 s[8:9], s37, v103
	v_add_u32_e32 v103, 0xffffffae, v102
	v_cmp_gt_i32_e64 s[10:11], s37, v103
	v_add_u32_e32 v103, 0xffffffaf, v102
	v_cmp_gt_i32_e64 s[12:13], s37, v103
	v_add_u32_e32 v103, 0xffffffb0, v102
	v_cmp_gt_i32_e64 s[14:15], s37, v103
	v_add_u32_e32 v103, 0xffffffb9, v102
	v_cmp_gt_i32_e64 s[16:17], s37, v103
	v_add_u32_e32 v103, 0xffffffba, v102
	v_cmp_gt_i32_e64 s[18:19], s37, v103
	v_add_u32_e32 v103, 0xffffffbb, v102
	v_cmp_gt_i32_e64 s[20:21], s37, v103
	v_add_u32_e32 v103, 0xffffffbc, v102
	v_cmp_gt_i32_e64 s[22:23], s37, v103
	v_add_u32_e32 v103, 0xffffffbd, v102
	v_cmp_gt_i32_e64 s[24:25], s37, v103
	v_add_u32_e32 v103, 0xffffffbe, v102
	v_cmp_gt_i32_e64 s[26:27], s37, v103
	v_add_u32_e32 v103, 0xffffffbf, v102
	v_subrev_u32_e32 v102, 64, v102
	v_cmp_gt_i32_e64 s[28:29], s37, v103
	v_cmp_gt_i32_e64 s[30:31], s37, v102
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
	v_cndmask_b32_e64 v67, v137, v67, s[0:1]
	v_cndmask_b32_e32 v66, v137, v66, vcc
	v_cndmask_b32_e64 v69, v137, v69, s[6:7]
	v_cndmask_b32_e64 v68, v137, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[84:85], v[66:67]
	v_cndmask_b32_e64 v71, v137, v71, s[10:11]
	v_cndmask_b32_e64 v70, v137, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[88:89], v[68:69]
	v_max_f32_e32 v102, v66, v67
	v_cndmask_b32_e64 v73, v137, v73, s[14:15]
	v_cndmask_b32_e64 v72, v137, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[90:91], v[70:71]
	v_max3_f32 v102, v102, v68, v69
	v_cndmask_b32_e64 v75, v137, v75, s[18:19]
	v_cndmask_b32_e64 v74, v137, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[92:93], v[72:73]
	v_max3_f32 v102, v102, v70, v71
	v_cndmask_b32_e64 v77, v137, v77, s[22:23]
	v_cndmask_b32_e64 v76, v137, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[94:95], v[74:75]
	v_max3_f32 v102, v102, v72, v73
	v_cndmask_b32_e64 v79, v137, v79, s[26:27]
	v_cndmask_b32_e64 v78, v137, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[96:97], v[76:77]
	v_max3_f32 v102, v102, v74, v75
	v_cndmask_b32_e64 v81, v137, v81, s[30:31]
	v_cndmask_b32_e64 v80, v137, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[98:99], v[78:79]
	v_max3_f32 v102, v102, v76, v77
	v_pk_mul_f32 v[80:81], v[100:101], v[80:81]
	v_max3_f32 v102, v102, v78, v79
	v_max3_f32 v102, v102, v80, v81
	ds_bpermute_b32 v103, v134, v102
	v_mov_b32_e32 v104, v86
	v_mov_b32_e32 v105, v86
	v_mov_b32_e32 v106, v86
	v_mov_b32_e32 v107, v86
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v103, v103, v103
	v_max_f32_e32 v141, v102, v103
	;;#ASMSTART
	v_add_f32 v102, v86, v135
	;;#ASMEND
	v_mov_b32_e32 v103, v86
	v_cmp_gt_f32_e32 vcc, v141, v102
	v_mov_b32_e32 v102, v86
	v_mov_b32_e32 v108, v86
	v_mov_b32_e32 v109, v86
	v_mov_b32_e32 v110, v86
	v_mov_b32_e32 v111, v86
	v_mov_b32_e32 v112, v86
	v_mov_b32_e32 v113, v86
	v_mov_b32_e32 v114, v86
	v_mov_b32_e32 v115, v86
	v_mov_b32_e32 v116, v86
	v_mov_b32_e32 v117, v86
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_45
	;;#ASMSTART
	v_add_f32 v102, v141, v136
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v86, v86, v102
	v_exp_f32_e32 v140, v86
	v_mov_b32_e32 v86, v102
	v_mov_b32_e32 v103, v102
	v_mov_b32_e32 v104, v102
	v_mov_b32_e32 v105, v102
	v_mov_b32_e32 v106, v102
	v_mov_b32_e32 v107, v102
	v_mov_b32_e32 v108, v102
	v_mov_b32_e32 v109, v102
	v_mov_b32_e32 v110, v102
	v_mov_b32_e32 v111, v102
	v_mov_b32_e32 v112, v102
	v_mov_b32_e32 v113, v102
	v_mov_b32_e32 v114, v102
	v_mov_b32_e32 v115, v102
	v_mov_b32_e32 v116, v102
	v_mov_b32_e32 v117, v102
.LBB0_45:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, 0, v66
	v_add_f32_e32 v141, v141, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v68
	v_add_f32_e32 v141, v141, v69
	v_add_f32_e32 v141, v141, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v71
	v_add_f32_e32 v141, v141, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v73
	v_add_f32_e32 v141, v141, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v75
	v_add_f32_e32 v141, v141, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[116:117] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v77
	v_add_f32_e32 v141, v141, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v140
	v_add_f32_e32 v141, v141, v79
	v_add_f32_e32 v141, v141, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v141, v141, v81
	;;#ASMSTART
	v_fma_f32 v139, v139, v140, v141
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_47
	;;#ASMSTART
	v_mul_f32 v2, v2, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v140
	;;#ASMEND
.LBB0_47:
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
	v_perm_b32 v66, v67, v66, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s95
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[210:213], v119 offset:8192
	ds_read_b128 v[206:209], v120
	v_mfma_f32_32x32x8_bf16 v[2:17], v[182:183], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[186:187], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[190:191], v[66:67], v[34:49]
	ds_read_b128 v[202:205], v121
	ds_read_b128 v[194:197], v122
	v_mfma_f32_32x32x8_bf16 v[50:65], v[218:219], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[184:185], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[188:189], v[68:69], v[18:33]
	ds_read_b128 v[198:201], v123
	ds_read_b128 v[186:189], v124
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[222:223], v[70:71], v[2:17]
	ds_read_b128 v[190:193], v125
	ds_read_b128 v[182:185], v126
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[214:215], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[224:225], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[216:217], v[72:73], v[50:65]
	s_cmp_gt_i32 s72, s43
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_le_i32 s72, s43
	s_cbranch_scc1 .LBB0_56
	ds_write_b128 v118, v[174:177]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[142:143], 0
	v_mov_b32_e32 v140, s44
	s_ashr_i32 s41, s40, 31
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[144:145], v[66:81]
	buffer_load_dword v218, v140, s[68:71], 0 offen
	v_lshl_add_u64 v[140:141], s[40:41], 0, v[82:83]
	v_lshl_add_u64 v[140:141], v[140:141], 2, s[86:87]
	s_lshl_b32 s41, s76, 13
	s_add_i32 s41, s41, s82
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s40, v218
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[150:151], v[66:81]
	global_load_dwordx4 v[174:177], v[140:141], off offset:512
	;;#ASMSTART
	v_mov_b32 v140, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v141, 3, v140
	v_lshlrev_b32_e32 v140, 5, v140
	v_and_b32_e32 v141, 0xf8, v141
	v_and_b32_e32 v140, 0x400, v140
	v_or3_b32 v140, v141, v140, s41
	v_ashrrev_i32_e32 v141, 31, v140
	v_lshl_add_u64 v[140:141], v[140:141], 1, s[84:85]
	global_load_dwordx4 v[206:209], v[140:141], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[156:157], v[66:81]
	global_load_dwordx4 v[202:205], v[140:141], off offset:512
	global_load_dwordx4 v[194:197], v[140:141], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[198:199], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[186:187], v[162:163], v[66:81]
	global_load_dwordx4 v[210:213], v[140:141], off offset:1536
	v_add_co_u32_e32 v140, vcc, s94, v140
	s_nop 1
	v_addc_co_u32_e32 v141, vcc, 0, v141, vcc
	global_load_dwordx4 v[198:201], v[140:141], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[192:193], v[168:169], v[66:81]
	global_load_dwordx4 v[190:193], v[140:141], off offset:512
	global_load_dwordx4 v[214:217], v[140:141], off offset:1024
	global_load_dwordx4 v[186:189], v[140:141], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[172:173], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v140, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v140, 2, v140
	v_and_b32_e32 v140, 8, v140
	v_add_u32_e32 v140, s42, v140
	v_subrev_u32_e32 v141, 55, v140
	v_cmp_gt_i32_e32 vcc, s37, v141
	v_subrev_u32_e32 v141, 54, v140
	v_cmp_gt_i32_e64 s[0:1], s37, v141
	v_subrev_u32_e32 v141, 53, v140
	v_cmp_gt_i32_e64 s[4:5], s37, v141
	v_subrev_u32_e32 v141, 52, v140
	v_cmp_gt_i32_e64 s[6:7], s37, v141
	v_subrev_u32_e32 v141, 51, v140
	v_cmp_gt_i32_e64 s[8:9], s37, v141
	v_subrev_u32_e32 v141, 50, v140
	v_cmp_gt_i32_e64 s[10:11], s37, v141
	v_subrev_u32_e32 v141, 49, v140
	v_cmp_gt_i32_e64 s[12:13], s37, v141
	v_subrev_u32_e32 v141, 48, v140
	v_cmp_gt_i32_e64 s[14:15], s37, v141
	v_subrev_u32_e32 v141, 39, v140
	v_cmp_gt_i32_e64 s[16:17], s37, v141
	v_subrev_u32_e32 v141, 38, v140
	v_cmp_gt_i32_e64 s[18:19], s37, v141
	v_subrev_u32_e32 v141, 37, v140
	v_cmp_gt_i32_e64 s[20:21], s37, v141
	v_subrev_u32_e32 v141, 36, v140
	v_cmp_gt_i32_e64 s[22:23], s37, v141
	v_subrev_u32_e32 v141, 35, v140
	v_cmp_gt_i32_e64 s[24:25], s37, v141
	v_subrev_u32_e32 v141, 34, v140
	v_cmp_gt_i32_e64 s[26:27], s37, v141
	v_subrev_u32_e32 v141, 33, v140
	v_subrev_u32_e32 v140, 32, v140
	v_cmp_gt_i32_e64 s[28:29], s37, v141
	v_cmp_gt_i32_e64 s[30:31], s37, v140
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
	v_cndmask_b32_e64 v67, v137, v67, s[0:1]
	v_cndmask_b32_e32 v66, v137, v66, vcc
	v_cndmask_b32_e64 v69, v137, v69, s[6:7]
	v_cndmask_b32_e64 v68, v137, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[84:85], v[66:67]
	v_cndmask_b32_e64 v71, v137, v71, s[10:11]
	v_cndmask_b32_e64 v70, v137, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[88:89], v[68:69]
	v_max_f32_e32 v140, v66, v67
	v_cndmask_b32_e64 v73, v137, v73, s[14:15]
	v_cndmask_b32_e64 v72, v137, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[90:91], v[70:71]
	v_max3_f32 v140, v140, v68, v69
	v_cndmask_b32_e64 v75, v137, v75, s[18:19]
	v_cndmask_b32_e64 v74, v137, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[92:93], v[72:73]
	v_max3_f32 v140, v140, v70, v71
	v_cndmask_b32_e64 v77, v137, v77, s[22:23]
	v_cndmask_b32_e64 v76, v137, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[94:95], v[74:75]
	v_max3_f32 v140, v140, v72, v73
	v_cndmask_b32_e64 v79, v137, v79, s[26:27]
	v_cndmask_b32_e64 v78, v137, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[96:97], v[76:77]
	v_max3_f32 v140, v140, v74, v75
	v_cndmask_b32_e64 v81, v137, v81, s[30:31]
	v_cndmask_b32_e64 v80, v137, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[98:99], v[78:79]
	v_max3_f32 v140, v140, v76, v77
	v_pk_mul_f32 v[80:81], v[100:101], v[80:81]
	v_max3_f32 v140, v140, v78, v79
	v_max3_f32 v140, v140, v80, v81
	ds_bpermute_b32 v141, v134, v140
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v141, v141, v141
	v_max_f32_e32 v141, v140, v141
	;;#ASMSTART
	v_add_f32 v140, v86, v135
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v141, v140
	v_mov_b32_e32 v140, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_50
	;;#ASMSTART
	v_add_f32 v102, v141, v136
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v86, v86, v102
	v_exp_f32_e32 v140, v86
	v_mov_b32_e32 v86, v102
	v_mov_b32_e32 v103, v102
	v_mov_b32_e32 v104, v102
	v_mov_b32_e32 v105, v102
	v_mov_b32_e32 v106, v102
	v_mov_b32_e32 v107, v102
	v_mov_b32_e32 v108, v102
	v_mov_b32_e32 v109, v102
	v_mov_b32_e32 v110, v102
	v_mov_b32_e32 v111, v102
	v_mov_b32_e32 v112, v102
	v_mov_b32_e32 v113, v102
	v_mov_b32_e32 v114, v102
	v_mov_b32_e32 v115, v102
	v_mov_b32_e32 v116, v102
	v_mov_b32_e32 v117, v102
.LBB0_50:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, 0, v66
	v_add_f32_e32 v141, v141, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v68
	v_add_f32_e32 v141, v141, v69
	v_add_f32_e32 v141, v141, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v71
	v_add_f32_e32 v141, v141, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v73
	v_add_f32_e32 v141, v141, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v75
	v_add_f32_e32 v141, v141, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[116:117] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v77
	v_add_f32_e32 v141, v141, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v140
	v_add_f32_e32 v141, v141, v79
	v_add_f32_e32 v141, v141, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v141, v141, v81
	;;#ASMSTART
	v_fma_f32 v139, v139, v140, v141
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_52
	;;#ASMSTART
	v_mul_f32 v2, v2, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v140
	;;#ASMEND
.LBB0_52:
	s_or_b64 exec, exec, s[0:1]
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
	v_perm_b32 v66, v67, v66, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s95
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(0)
	ds_read_b128 v[182:185], v119
	ds_read_b128 v[218:221], v127
	v_mfma_f32_32x32x8_bf16 v[2:17], v[206:207], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[202:203], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[66:67], v[34:49]
	ds_read_b128 v[222:225], v128
	ds_read_b128 v[226:229], v129
	v_mfma_f32_32x32x8_bf16 v[50:65], v[210:211], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[208:209], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[204:205], v[68:69], v[18:33]
	ds_read_b128 v[202:205], v130
	ds_read_b128 v[206:209], v131
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[212:213], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[198:199], v[70:71], v[2:17]
	ds_read_b128 v[194:197], v132
	ds_read_b128 v[210:213], v133
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[186:187], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[200:201], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[188:189], v[72:73], v[50:65]
	ds_write_b128 v118, v[178:181] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[182:183], v[142:143], 0
	v_lshl_or_b32 v140, s83, 12, v82
	v_ashrrev_i32_e32 v141, 31, v140
	v_lshl_add_u64 v[140:141], v[140:141], 2, s[86:87]
	s_addk_i32 s41, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[184:185], v[144:145], v[66:81]
	global_load_dwordx4 v[178:181], v[140:141], off
	;;#ASMSTART
	v_mov_b32 v140, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v141, 3, v140
	v_lshlrev_b32_e32 v140, 5, v140
	v_and_b32_e32 v141, 0xf8, v141
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[146:147], v[66:81]
	v_and_b32_e32 v140, 0x400, v140
	v_or3_b32 v140, v141, v140, s41
	v_ashrrev_i32_e32 v141, 31, v140
	v_lshl_add_u64 v[140:141], v[140:141], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[148:149], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[150:151], v[66:81]
	global_load_dwordx4 v[182:185], v[140:141], off
	global_load_dwordx4 v[186:189], v[140:141], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[152:153], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[156:157], v[66:81]
	global_load_dwordx4 v[190:193], v[140:141], off offset:1024
	global_load_dwordx4 v[218:221], v[140:141], off offset:1536
	v_add_co_u32_e32 v140, vcc, s94, v140
	s_nop 1
	v_addc_co_u32_e32 v141, vcc, 0, v141, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[158:159], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[204:205], v[160:161], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[162:163], v[66:81]
	global_load_dwordx4 v[222:225], v[140:141], off
	global_load_dwordx4 v[226:229], v[140:141], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[164:165], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[194:195], v[166:167], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[196:197], v[168:169], v[66:81]
	global_load_dwordx4 v[230:233], v[140:141], off offset:1024
	global_load_dwordx4 v[214:217], v[140:141], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[170:171], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[172:173], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v140, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v140, 2, v140
	v_and_b32_e32 v140, 8, v140
	v_add_u32_e32 v140, s42, v140
	v_subrev_u32_e32 v141, 23, v140
	v_cmp_gt_i32_e32 vcc, s37, v141
	v_subrev_u32_e32 v141, 22, v140
	v_cmp_gt_i32_e64 s[0:1], s37, v141
	v_subrev_u32_e32 v141, 21, v140
	v_cmp_gt_i32_e64 s[4:5], s37, v141
	v_subrev_u32_e32 v141, 20, v140
	v_cmp_gt_i32_e64 s[6:7], s37, v141
	v_subrev_u32_e32 v141, 19, v140
	v_cmp_gt_i32_e64 s[8:9], s37, v141
	v_subrev_u32_e32 v141, 18, v140
	v_cmp_gt_i32_e64 s[10:11], s37, v141
	v_subrev_u32_e32 v141, 17, v140
	v_cmp_gt_i32_e64 s[12:13], s37, v141
	v_add_u32_e32 v141, -16, v140
	v_cmp_gt_i32_e64 s[14:15], s37, v141
	v_add_u32_e32 v141, -7, v140
	v_cmp_gt_i32_e64 s[16:17], s37, v141
	v_add_u32_e32 v141, -6, v140
	v_cmp_gt_i32_e64 s[18:19], s37, v141
	v_add_u32_e32 v141, -5, v140
	v_cmp_gt_i32_e64 s[20:21], s37, v141
	v_add_u32_e32 v141, -4, v140
	v_cmp_gt_i32_e64 s[22:23], s37, v141
	v_add_u32_e32 v141, -3, v140
	v_cmp_gt_i32_e64 s[24:25], s37, v141
	v_add_u32_e32 v141, -2, v140
	v_cmp_gt_i32_e64 s[26:27], s37, v141
	v_add_u32_e32 v141, -1, v140
	v_cmp_gt_i32_e64 s[28:29], s37, v141
	v_cmp_gt_i32_e64 s[30:31], s37, v140
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
	v_cndmask_b32_e64 v67, v137, v67, s[0:1]
	v_cndmask_b32_e32 v66, v137, v66, vcc
	v_cndmask_b32_e64 v69, v137, v69, s[6:7]
	v_cndmask_b32_e64 v68, v137, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[84:85], v[66:67]
	v_cndmask_b32_e64 v71, v137, v71, s[10:11]
	v_cndmask_b32_e64 v70, v137, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[88:89], v[68:69]
	v_max_f32_e32 v140, v66, v67
	v_cndmask_b32_e64 v73, v137, v73, s[14:15]
	v_cndmask_b32_e64 v72, v137, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[90:91], v[70:71]
	v_max3_f32 v140, v140, v68, v69
	v_cndmask_b32_e64 v75, v137, v75, s[18:19]
	v_cndmask_b32_e64 v74, v137, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[92:93], v[72:73]
	v_max3_f32 v140, v140, v70, v71
	v_cndmask_b32_e64 v77, v137, v77, s[22:23]
	v_cndmask_b32_e64 v76, v137, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[94:95], v[74:75]
	v_max3_f32 v140, v140, v72, v73
	v_cndmask_b32_e64 v79, v137, v79, s[26:27]
	v_cndmask_b32_e64 v78, v137, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[96:97], v[76:77]
	v_max3_f32 v140, v140, v74, v75
	v_cndmask_b32_e64 v81, v137, v81, s[30:31]
	v_cndmask_b32_e64 v80, v137, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[98:99], v[78:79]
	v_max3_f32 v140, v140, v76, v77
	v_pk_mul_f32 v[80:81], v[100:101], v[80:81]
	v_max3_f32 v140, v140, v78, v79
	v_max3_f32 v140, v140, v80, v81
	ds_bpermute_b32 v141, v134, v140
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v141, v141, v141
	v_max_f32_e32 v141, v140, v141
	;;#ASMSTART
	v_add_f32 v140, v86, v135
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v141, v140
	v_mov_b32_e32 v140, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_54
	;;#ASMSTART
	v_add_f32 v102, v141, v136
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v86, v86, v102
	v_exp_f32_e32 v140, v86
	v_mov_b32_e32 v86, v102
	v_mov_b32_e32 v103, v102
	v_mov_b32_e32 v104, v102
	v_mov_b32_e32 v105, v102
	v_mov_b32_e32 v106, v102
	v_mov_b32_e32 v107, v102
	v_mov_b32_e32 v108, v102
	v_mov_b32_e32 v109, v102
	v_mov_b32_e32 v110, v102
	v_mov_b32_e32 v111, v102
	v_mov_b32_e32 v112, v102
	v_mov_b32_e32 v113, v102
	v_mov_b32_e32 v114, v102
	v_mov_b32_e32 v115, v102
	v_mov_b32_e32 v116, v102
	v_mov_b32_e32 v117, v102
.LBB0_54:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, 0, v66
	v_add_f32_e32 v102, v102, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, v102, v68
	v_add_f32_e32 v102, v102, v69
	v_add_f32_e32 v102, v102, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, v102, v71
	v_add_f32_e32 v102, v102, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, v102, v73
	v_add_f32_e32 v102, v102, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[114:115] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, v102, v75
	v_add_f32_e32 v102, v102, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[116:117] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v102, v102, v77
	v_add_f32_e32 v102, v102, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v140
	v_add_f32_e32 v102, v102, v79
	v_add_f32_e32 v102, v102, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v102, v102, v81
	;;#ASMSTART
	v_fma_f32 v139, v139, v140, v102
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_37
	;;#ASMSTART
	v_mul_f32 v2, v2, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v140
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v140
	;;#ASMEND
	s_branch .LBB0_37
.LBB0_56:
	s_mov_b32 s40, s45
	s_branch .LBB0_38
.LBB0_57:
	;;#ASMSTART
	v_mov_b32 v68, v0
	;;#ASMEND
	ds_bpermute_b32 v66, v134, v139
	v_lshrrev_b32_e32 v67, 1, v68
	v_and_b32_e32 v69, 31, v68
	v_and_or_b32 v67, v67, s60, v69
	v_and_b32_e32 v68, 32, v68
	v_cmp_eq_u32_e32 vcc, 0, v68
	v_cmp_gt_i32_e64 s[0:1], s81, v67
	s_and_b64 s[4:5], vcc, s[0:1]
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v66, v139, v66
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_59
	s_mov_b32 s4, 0x800000
	v_cmp_gt_f32_e32 vcc, s4, v66
	s_ashr_i32 s81, s80, 31
	s_lshl_b64 s[4:5], s[80:81], 6
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v66, v69
	v_log_f32_e32 v69, v69
	v_cndmask_b32_e32 v68, 0, v138, vcc
	v_readlane_b32 s6, v246, 0
	v_readlane_b32 s7, v246, 1
	v_sub_f32_e32 v68, v69, v68
	s_add_u32 s6, s6, s4
	v_add_f32_e32 v68, v86, v68
	s_addc_u32 s7, s7, s5
	s_lshl_b64 s[4:5], s[88:89], 2
	v_mul_f32_e32 v68, 0x3f317218, v68
	v_cmp_lt_f32_e32 vcc, 0, v66
	s_add_u32 s4, s6, s4
	s_addc_u32 s5, s7, s5
	v_cndmask_b32_e32 v68, v137, v68, vcc
	v_lshlrev_b32_e32 v67, 6, v67
	global_store_dword v67, v68, s[4:5]
.LBB0_59:
	s_or_b64 exec, exec, s[0:1]
	v_div_scale_f32 v67, s[0:1], v66, v66, s93
	v_rcp_f32_e32 v68, v67
	v_div_scale_f32 v69, vcc, s93, v66, s93
	v_fma_f32 v70, -v67, v68, 1.0
	v_fmac_f32_e32 v68, v70, v68
	v_mul_f32_e32 v70, v69, v68
	v_fma_f32 v71, -v67, v70, v69
	v_fmac_f32_e32 v70, v71, v68
	v_fma_f32 v67, -v67, v70, v69
	v_div_fmas_f32 v67, v67, v68, v70
	v_div_fixup_f32 v66, v67, v66, s93
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
	v_perm_b32 v28, v3, v2, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v5, v4, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v7, v6, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v9, v8, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v11, v10, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v13, v12, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v15, v14, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v17, v16, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v19, v18, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v21, v20, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v23, v22, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v74, v75, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v72, v73, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v70, v71, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v68, v69, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v66, v67, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v35, v34, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v37, v36, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v39, v38, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v41, v40, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v43, v42, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v45, v44, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v47, v46, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v49, v48, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v51, v50, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v53, v52, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v55, v54, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v57, v56, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v59, v58, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v61, v60, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v63, v62, s95
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v65, v64, s95
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v34, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v34, 0x100, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_61
	s_barrier
.LBB0_61:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v37, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v34, 3, v37
	v_bfe_u32 v36, v37, 6, 3
	v_lshlrev_b32_e32 v35, 7, v37
	v_and_b32_e32 v34, 4, v34
	v_and_or_b32 v34, v35, s61, v34
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_63
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
.LBB0_63:
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
	s_add_u32 s72, s58, s96
	v_and_b32_e32 v38, 0x78, v40
	v_add_u32_e32 v40, s79, v39
	s_addc_u32 s0, s59, s97
	v_or_b32_e32 v40, v40, v38
	s_and_b32 s73, s0, 0xffff
	s_mov_b32 s75, s71
	v_lshlrev_b32_e32 v40, 1, v40
	v_cmp_eq_u32_e32 vcc, 1, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[42:45], v40, s[72:75], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_65
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
.LBB0_65:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s4, s79, 0x10000
	v_or_b32_e32 v38, v38, v39
	v_add_lshl_u32 v39, s4, v38, 1
	v_cmp_eq_u32_e32 vcc, 2, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[72:75], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_67
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
.LBB0_67:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x10000
	v_add_lshl_u32 v39, s0, v38, 1
	s_mov_b32 s75, s71
	v_cmp_eq_u32_e32 vcc, 3, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[72:75], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_69
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
.LBB0_69:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x20000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 4, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[72:75], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_71
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
.LBB0_71:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x30000
	v_add_lshl_u32 v39, s0, v38, 1
	s_mov_b32 s75, s71
	v_cmp_eq_u32_e32 vcc, 5, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[72:75], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_73
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
.LBB0_73:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x40000
	v_add_lshl_u32 v39, s0, v38, 1
	v_cmp_eq_u32_e32 vcc, 6, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[72:75], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_75
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
.LBB0_75:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[40:43], v37
	s_add_i32 s0, s4, 0x50000
	v_add_lshl_u32 v39, s0, v38, 1
	s_mov_b32 s75, s71
	v_cmp_eq_u32_e32 vcc, 7, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[72:75], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_77
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
.LBB0_77:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[4:7], v37
	s_add_i32 s4, s4, 0x60000
	v_add_lshl_u32 v2, s4, v38, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v2, s[72:75], 0 offen
	s_barrier
	;;#ASMSTART
	s_mov_b32 s0, s62
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_81
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_80
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v3, s8
	global_atomic_add v3, v83, v3, s[98:99] sc0
.LBB0_80:
	s_or_b64 exec, exec, s[6:7]
	s_lshl_b64 s[6:7], s[0:1], 2
	s_add_u32 s6, s98, s6
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v3
	s_addc_u32 s7, s99, s7
	s_nop 0
	v_add_u32_e32 v2, s8, v2
	global_store_dword v83, v2, s[6:7]
	s_waitcnt vmcnt(0)
.LBB0_81:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s98, s0
	s_addc_u32 s1, s99, s1
	s_barrier
	global_load_dword v2, v83, s[0:1]
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s4, v2
	s_add_i32 s0, s4, s33
	s_sub_i32 s33, s0, s2
	s_cmp_ge_i32 s78, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s92
	s_cselect_b64 s[6:7], -1, 0
	s_or_b64 s[0:1], s[0:1], s[6:7]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_84
	s_branch .LBB0_12
.LBB0_82:
	s_mov_b32 s78, s5
.LBB0_83:
	s_sub_i32 s33, s33, s92
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s88, 0, s2
	s_cmp_ge_i32 s78, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s92, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_84:
	s_add_i32 s2, s88, 1
	s_cmp_gt_i32 s2, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s2, 16
	s_cbranch_scc1 .LBB0_87
	s_add_i32 s5, s78, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s92
	s_cbranch_scc1 .LBB0_82
	s_ashr_i32 s79, s78, 31
	s_lshl_b64 s[6:7], s[78:79], 2
	s_add_u32 s6, s34, s6
	s_addc_u32 s7, s35, s7
	global_load_dwordx2 v[2:3], v83, s[6:7] offset:4
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
	s_branch .LBB0_82
.LBB0_87:
	s_mov_b32 s6, s92
	s_branch .LBB0_83
.LBB0_88:
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
		.amdhsa_next_free_vgpr 247
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 248
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

	.set .Lattn_kernel_0.num_vgpr, 247
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
    .sgpr_count:     106
    .sgpr_spill_count: 2
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     247
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
