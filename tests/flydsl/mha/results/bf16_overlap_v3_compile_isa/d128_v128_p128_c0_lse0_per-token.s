	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	attn_kernel_0
	.p2align	8
	.type	attn_kernel_0,@function
attn_kernel_0:
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dword s3, s[0:1], 0x38
	s_load_dwordx2 s[76:77], s[0:1], 0x90
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
	s_cmp_lt_i32 s33, s86
	s_cselect_b64 s[10:11], -1, 0
	s_or_b64 s[8:9], s[8:9], s[10:11]
	s_and_b64 vcc, exec, s[8:9]
	s_mov_b32 s10, s86
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s11, s88, 1
	s_cmp_gt_i32 s11, 15
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s11, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s12, s78, 1
	s_cmp_ge_i32 s12, s3
	s_mov_b32 s86, s10
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
	s_subb_u32 s86, s18, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s86, s10
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s86, s10
	s_mov_b32 s88, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s93, s[4:5], 0x0
	s_load_dword s87, s[6:7], 0x0
	s_cmp_ge_i32 s78, s3
	s_cbranch_scc1 .LBB0_118
	s_load_dwordx2 s[48:49], s[0:1], 0x0
	s_load_dwordx2 s[50:51], s[0:1], 0x10
	s_load_dwordx2 s[84:85], s[0:1], 0x20
	s_load_dwordx2 s[52:53], s[0:1], 0x50
	s_load_dwordx2 s[54:55], s[0:1], 0x60
	s_load_dwordx2 s[56:57], s[0:1], 0x70
	s_load_dwordx2 s[58:59], s[0:1], 0xa0
	s_load_dwordx2 s[94:95], s[0:1], 0xc0
	v_lshrrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v3, 31, v0
	s_movk_i32 s0, 0xe0
	v_lshrrev_b32_e32 v4, 2, v0
	v_lshlrev_b32_e32 v6, 10, v0
	v_and_or_b32 v1, v2, s0, v3
	v_lshlrev_b32_e32 v3, 11, v3
	v_and_b32_e32 v5, 8, v4
	v_and_b32_e32 v6, 0x70000, v6
	v_or3_b32 v121, v5, v3, v6
	v_lshrrev_b32_e32 v6, 3, v0
	v_lshlrev_b32_e32 v3, 9, v0
	v_and_b32_e32 v5, 12, v4
	v_and_b32_e32 v2, 32, v2
	v_and_b32_e32 v6, 16, v6
	v_and_b32_e32 v3, 0x1e00, v3
	v_or3_b32 v2, v5, v2, v6
	v_and_b32_e32 v4, 64, v4
	v_or3_b32 v116, v2, v4, v3
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v152, v2, v3
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v1, 6, v1
	v_mov_b32_e32 v117, 0
	v_xor_b32_e32 v153, 0x80, v2
	s_mov_b32 s71, 0x27000
	s_movk_i32 s89, 0xf80
	s_movk_i32 s90, 0x1000
	v_mov_b32_e32 v154, 0x40e00000
	v_mov_b32_e32 v155, 1.0
	s_mov_b32 s91, 0x7060302
	s_mov_b32 s60, s2
	v_mov_b32_e32 v156, 0xff800000
	s_mov_b32 s36, 0
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s86, s6
.LBB0_12:
	s_mov_b32 s2, s4
	s_cmp_ge_i32 s78, s3
	s_cbranch_scc1 .LBB0_118
.LBB0_13:
	s_ashr_i32 s79, s78, 31
	s_lshl_b32 s6, s33, 8
	s_lshl_b64 s[0:1], s[78:79], 2
	s_add_u32 s4, s34, s0
	s_addc_u32 s5, s35, s1
	global_load_dwordx2 v[2:3], v117, s[4:5]
	s_mov_b32 s75, s71
	v_lshl_add_u32 v6, s88, 2, v1
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s4, v2
	s_add_i32 s7, s4, s6
	v_readfirstlane_b32 s5, v3
	s_add_i32 s4, s7, 0x100
	s_min_i32 s4, s4, s5
	s_sub_i32 s10, s4, s7
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s52, s0
	s_addc_u32 s5, s53, s1
	s_add_u32 s0, s76, s0
	s_addc_u32 s1, s77, s1
	s_lshl_b32 s8, s7, 11
	s_ashr_i32 s9, s8, 31
	s_lshl_b32 s6, s7, 4
	s_lshl_b64 s[96:97], s[8:9], 1
	s_add_u32 s72, s48, s96
	global_load_dwordx2 v[4:5], v117, s[4:5]
	global_load_dword v3, v117, s[0:1]
	s_addc_u32 s0, s49, s97
	s_lshl_b32 s79, s88, 7
	s_ashr_i32 s7, s6, 31
	s_lshl_b32 s74, s10, 12
	s_and_b32 s73, s0, 0xffff
	s_lshl_b64 s[0:1], s[6:7], 2
	v_add_lshl_u32 v7, s79, v121, 1
	s_add_u32 s68, s56, s0
	buffer_load_dwordx4 v[174:177], v7, s[72:75], 0 offen
	buffer_load_dwordx4 v[178:181], v7, s[72:75], 0 offen offset:32
	buffer_load_dwordx4 v[182:185], v7, s[72:75], 0 offen offset:64
	buffer_load_dwordx4 v[186:189], v7, s[72:75], 0 offen offset:96
	buffer_load_dwordx4 v[190:193], v7, s[72:75], 0 offen offset:128
	buffer_load_dwordx4 v[194:197], v7, s[72:75], 0 offen offset:160
	s_addc_u32 s0, s57, s1
	s_lshl_b32 s70, s10, 6
	s_and_b32 s69, s0, 0xffff
	buffer_load_dword v2, v6, s[68:71], 0 offen
	buffer_load_dwordx4 v[198:201], v7, s[72:75], 0 offen offset:192
	buffer_load_dwordx4 v[202:205], v7, s[72:75], 0 offen offset:224
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
	s_sub_i32 s72, s1, s0
	s_ashr_i32 s1, s88, 31
	s_lshr_b32 s1, s1, 28
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
	s_lshl_b32 s98, s1, 14
	s_ashr_i32 s99, s98, 31
	s_lshl_b64 s[4:5], s[98:99], 1
	s_add_u32 s80, s50, s4
	s_addc_u32 s81, s51, s5
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s68, s54, s0
	s_addc_u32 s0, s55, s1
	s_lshl_b32 s70, s72, 2
	s_and_b32 s69, s0, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[68:71], 0
	s_waitcnt vmcnt(3)
	v_mul_f32_e32 v2, s93, v2
	v_mul_f32_e32 v118, 0x3e0293ee, v2
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v4
	v_readfirstlane_b32 s92, v5
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
	buffer_load_dword v2, off, s[68:71], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s75, v2
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
	buffer_load_dword v2, off, s[68:71], 0 offset:12
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s99, v2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_lshl_b32 s0, s8, 13
	v_or_b32_e32 v2, s0, v116
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 2, s[80:81]
	global_load_dwordx4 v[4:7], v[4:5], off
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_ashr_i32 s0, s0, 31
	v_mov_b32_e32 v3, s0
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[80:81]
	global_load_dwordx4 v[158:161], v[2:3], off offset:512
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	global_load_dwordx4 v[162:165], v[2:3], off offset:1024
	ds_write_b128 v152, v[4:7] offset:8192
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v4, 2, v2
	v_lshlrev_b32_e32 v3, 7, v2
	v_and_b32_e32 v4, 8, v4
	v_lshlrev_b32_e32 v2, 4, v2
	v_and_or_b32 v3, v3, s89, v4
	v_and_b32_e32 v2, 48, v2
	v_or_b32_e32 v4, v3, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[226:229], v4 offset:8192
	v_or_b32_e32 v4, 0x1010, v3
	v_xor_b32_e32 v4, v4, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[210:213], v4
	v_or_b32_e32 v4, 0x1020, v3
	v_xor_b32_e32 v4, v4, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[166:169], v4
	v_or_b32_e32 v4, 0x1030, v3
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[206:209], v2
	v_or_b32_e32 v2, 0x1040, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[214:217], v2
	v_or_b32_e32 v2, 0x1050, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[218:221], v2
	v_or_b32_e32 v2, 0x1060, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[222:225], v2
	v_or_b32_e32 v2, 0x1070, v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[170:173], v2
	s_add_i32 s0, s72, -2
	s_bitcmp0_b32 s72, 0
	s_cselect_b32 s82, s0, s10
	s_ashr_i32 s83, s82, 31
	s_cmp_lt_i32 s82, 1
	s_cbranch_scc1 .LBB0_51
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v119, v118
	v_mov_b32_e32 v82, v118
	v_mov_b32_e32 v83, v118
	v_mov_b32_e32 v84, v118
	v_mov_b32_e32 v85, v118
	v_mov_b32_e32 v86, v118
	v_mov_b32_e32 v87, v118
	v_mov_b32_e32 v88, v118
	v_mov_b32_e32 v89, v118
	v_mov_b32_e32 v90, v118
	v_mov_b32_e32 v91, v118
	v_mov_b32_e32 v92, v118
	v_mov_b32_e32 v93, v118
	v_mov_b32_e32 v94, v118
	v_mov_b32_e32 v95, v118
	s_mov_b64 s[0:1], 0
	s_mov_b32 s11, 20
	v_mov_b32_e32 v120, 0xff800000
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v157
	v_mov_b32_e32 v36, v157
	v_mov_b32_e32 v37, v157
	v_mov_b32_e32 v38, v157
	v_mov_b32_e32 v39, v157
	v_mov_b32_e32 v40, v157
	v_mov_b32_e32 v41, v157
	v_mov_b32_e32 v42, v157
	v_mov_b32_e32 v43, v157
	v_mov_b32_e32 v44, v157
	v_mov_b32_e32 v45, v157
	v_mov_b32_e32 v46, v157
	v_mov_b32_e32 v47, v157
	v_mov_b32_e32 v48, v157
	v_mov_b32_e32 v49, v157
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v157
	v_mov_b32_e32 v52, v157
	v_mov_b32_e32 v53, v157
	v_mov_b32_e32 v54, v157
	v_mov_b32_e32 v55, v157
	v_mov_b32_e32 v56, v157
	v_mov_b32_e32 v57, v157
	v_mov_b32_e32 v58, v157
	v_mov_b32_e32 v59, v157
	v_mov_b32_e32 v60, v157
	v_mov_b32_e32 v61, v157
	v_mov_b32_e32 v62, v157
	v_mov_b32_e32 v63, v157
	v_mov_b32_e32 v64, v157
	v_mov_b32_e32 v65, v157
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v157
	v_mov_b32_e32 v4, v157
	v_mov_b32_e32 v5, v157
	v_mov_b32_e32 v6, v157
	v_mov_b32_e32 v7, v157
	v_mov_b32_e32 v8, v157
	v_mov_b32_e32 v9, v157
	v_mov_b32_e32 v10, v157
	v_mov_b32_e32 v11, v157
	v_mov_b32_e32 v12, v157
	v_mov_b32_e32 v13, v157
	v_mov_b32_e32 v14, v157
	v_mov_b32_e32 v15, v157
	v_mov_b32_e32 v16, v157
	v_mov_b32_e32 v17, v157
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v157
	v_mov_b32_e32 v20, v157
	v_mov_b32_e32 v21, v157
	v_mov_b32_e32 v22, v157
	v_mov_b32_e32 v23, v157
	v_mov_b32_e32 v24, v157
	v_mov_b32_e32 v25, v157
	v_mov_b32_e32 v26, v157
	v_mov_b32_e32 v27, v157
	v_mov_b32_e32 v28, v157
	v_mov_b32_e32 v29, v157
	v_mov_b32_e32 v30, v157
	v_mov_b32_e32 v31, v157
	v_mov_b32_e32 v32, v157
	v_mov_b32_e32 v33, v157
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[166:167], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[146:147], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[138:139], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[168:169], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[144:145], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[148:149], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[140:141], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[134:135], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[126:127], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[130:131], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[136:137], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[132:133], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[226:229], v68 offset:8192
	v_or_b32_e32 v68, 0x1010, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[210:213], v68
	v_or_b32_e32 v68, 0x1020, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[166:169], v68
	v_or_b32_e32 v68, 0x1030, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	v_or_b32_e32 v66, 0x1040, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	v_or_b32_e32 v66, 0x1050, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[218:221], v66
	v_or_b32_e32 v66, 0x1060, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[222:225], v66
	v_or_b32_e32 v66, 0x1070, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	s_add_u32 s0, s0, 2
	s_addc_u32 s1, s1, 0
	v_mov_b64_e32 v[66:67], s[82:83]
	v_cmp_lt_i64_e32 vcc, s[0:1], v[66:67]
	s_add_i32 s11, s11, 8
	s_mov_b32 s92, s12
	s_cbranch_vccz .LBB0_50
.LBB0_18:
	ds_write_b128 v152, v[158:161]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[174:175], 0
	s_add_i32 s4, s11, -4
	v_mov_b32_e32 v96, s4
	s_lshl_b32 s4, s8, 13
	s_ashr_i32 s5, s4, 31
	v_mov_b32_e32 v97, s5
	s_lshl_b32 s13, s8, 14
	s_add_i32 s13, s13, s98
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[176:177], v[66:81]
	buffer_load_dword v110, v96, s[68:71], 0 offen
	v_or_b32_e32 v96, s4, v116
	v_lshl_add_u64 v[96:97], v[96:97], 2, s[80:81]
	s_mov_b32 s8, s75
	s_mov_b32 s12, s99
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s75, v110
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[178:179], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[182:183], v[66:81]
	global_load_dwordx4 v[128:131], v[96:97], off offset:1536
	;;#ASMSTART
	v_mov_b32 v96, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v97, 3, v96
	v_lshlrev_b32_e32 v96, 5, v96
	v_and_b32_e32 v97, 0xf8, v97
	v_and_b32_e32 v96, 0x400, v96
	v_or3_b32 v96, v97, v96, s13
	v_ashrrev_i32_e32 v97, 31, v96
	v_lshl_add_u64 v[96:97], v[96:97], 1, s[84:85]
	global_load_dwordx4 v[140:143], v[96:97], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[188:189], v[66:81]
	global_load_dwordx4 v[144:147], v[96:97], off offset:512
	global_load_dwordx4 v[132:135], v[96:97], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[194:195], v[66:81]
	global_load_dwordx4 v[136:139], v[96:97], off offset:1536
	v_add_co_u32_e32 v96, vcc, s90, v96
	s_nop 1
	v_addc_co_u32_e32 v97, vcc, 0, v97, vcc
	global_load_dwordx4 v[106:109], v[96:97], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[200:201], v[66:81]
	global_load_dwordx4 v[122:125], v[96:97], off offset:512
	global_load_dwordx4 v[102:105], v[96:97], off offset:1024
	global_load_dwordx4 v[98:101], v[96:97], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
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
	ds_bpermute_b32 v97, v153, v96
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v97, v97, v97
	v_max_f32_e32 v97, v96, v97
	;;#ASMSTART
	v_add_f32 v96, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v97, v96
	v_mov_b32_e32 v96, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_20
	;;#ASMSTART
	v_add_f32 v97, v97, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v96, v120, v97
	v_exp_f32_e32 v96, v96
	v_mov_b32_e32 v120, v97
.LBB0_20:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[66:67], v[66:67], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[72:73], v[72:73], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v97, v97, v68
	v_add_f32_e32 v97, v97, v69
	v_add_f32_e32 v97, v97, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v97, v97, v71
	v_add_f32_e32 v97, v97, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v97, v97, v73
	v_add_f32_e32 v97, v97, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v97, v97, v75
	v_add_f32_e32 v97, v97, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
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
	v_fma_f32 v112, v157, v96, v97
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_22
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[140:141], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[144:145], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[132:133], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[136:137], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[142:143], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[146:147], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[134:135], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[138:139], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[106:107], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[122:123], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[98:99], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[108:109], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[124:125], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[104:105], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[100:101], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[96:99], v68
	v_or_b32_e32 v68, 16, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[100:103], v68
	v_or_b32_e32 v68, 32, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[104:107], v68
	v_or_b32_e32 v68, 48, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[108:111], v66
	v_or_b32_e32 v66, 64, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[132:135], v66
	v_or_b32_e32 v66, 0x50, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[138:141], v66
	v_or_b32_e32 v66, 0x60, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	v_or_b32_e32 v66, 0x70, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	ds_write_b128 v152, v[162:165] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[174:175], 0
	s_lshl_b32 s4, s92, 13
	v_or_b32_e32 v96, s4, v116
	v_ashrrev_i32_e32 v97, 31, v96
	v_lshl_add_u64 v[96:97], v[96:97], 2, s[80:81]
	s_addk_i32 s13, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[176:177], v[66:81]
	global_load_dwordx4 v[124:127], v[96:97], off
	;;#ASMSTART
	v_mov_b32 v96, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v97, 3, v96
	v_lshlrev_b32_e32 v96, 5, v96
	v_and_b32_e32 v97, 0xf8, v97
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[178:179], v[66:81]
	v_and_b32_e32 v96, 0x400, v96
	v_or3_b32 v96, v97, v96, s13
	v_ashrrev_i32_e32 v97, 31, v96
	v_lshl_add_u64 v[96:97], v[96:97], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[182:183], v[66:81]
	global_load_dwordx4 v[166:169], v[96:97], off
	global_load_dwordx4 v[158:161], v[96:97], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[188:189], v[66:81]
	global_load_dwordx4 v[162:165], v[96:97], off offset:1024
	global_load_dwordx4 v[148:151], v[96:97], off offset:1536
	v_add_co_u32_e32 v96, vcc, s90, v96
	s_nop 1
	v_addc_co_u32_e32 v97, vcc, 0, v97, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[194:195], v[66:81]
	global_load_dwordx4 v[144:147], v[96:97], off
	global_load_dwordx4 v[136:139], v[96:97], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[140:141], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[200:201], v[66:81]
	global_load_dwordx4 v[140:143], v[96:97], off offset:1024
	global_load_dwordx4 v[132:135], v[96:97], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
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
	ds_bpermute_b32 v97, v153, v96
	v_mov_b32_e32 v113, 1.0
	v_mov_b32_e32 v98, v120
	v_mov_b32_e32 v99, v120
	v_mov_b32_e32 v100, v120
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v97, v97, v97
	v_max_f32_e32 v114, v96, v97
	;;#ASMSTART
	v_add_f32 v96, v120, v154
	;;#ASMEND
	v_mov_b32_e32 v97, v120
	v_cmp_gt_f32_e32 vcc, v114, v96
	v_mov_b32_e32 v96, v120
	v_mov_b32_e32 v101, v120
	v_mov_b32_e32 v102, v120
	v_mov_b32_e32 v103, v120
	v_mov_b32_e32 v104, v120
	v_mov_b32_e32 v105, v120
	v_mov_b32_e32 v106, v120
	v_mov_b32_e32 v107, v120
	v_mov_b32_e32 v108, v120
	v_mov_b32_e32 v109, v120
	v_mov_b32_e32 v110, v120
	v_mov_b32_e32 v111, v120
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_24
	;;#ASMSTART
	v_add_f32 v96, v114, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v120, v96
	v_exp_f32_e32 v113, v97
	v_mov_b32_e32 v120, v96
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
.LBB0_24:
	s_or_b64 exec, exec, s[6:7]
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
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v113
	v_add_f32_e32 v114, v114, v79
	v_add_f32_e32 v114, v114, v80
	v_add_f32_e32 v114, v114, v81
	;;#ASMSTART
	v_fma_f32 v114, v112, v113, v114
	;;#ASMEND
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_26
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[166:167], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[158:159], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[162:163], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[148:149], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[168:169], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[160:161], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[164:165], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[150:151], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[136:137], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[140:141], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[142:143], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[132:135], v68 offset:8192
	v_or_b32_e32 v68, 0x1010, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[136:139], v68
	v_or_b32_e32 v68, 0x1020, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[140:143], v68
	v_or_b32_e32 v68, 0x1030, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[144:147], v66
	v_or_b32_e32 v66, 0x1040, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	v_or_b32_e32 v66, 0x1050, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	v_or_b32_e32 v66, 0x1060, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[210:213], v66
	v_or_b32_e32 v66, 0x1070, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	ds_write_b128 v152, v[128:131]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[174:175], 0
	s_ashr_i32 s5, s4, 31
	v_lshl_add_u64 v[112:113], s[4:5], 0, v[116:117]
	v_lshl_add_u64 v[112:113], v[112:113], 2, s[80:81]
	s_add_i32 s4, s13, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[176:177], v[66:81]
	global_load_dwordx4 v[128:131], v[112:113], off offset:512
	;;#ASMSTART
	v_mov_b32 v115, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v122, 3, v115
	v_lshlrev_b32_e32 v115, 5, v115
	v_and_b32_e32 v122, 0xf8, v122
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], v[66:81]
	v_and_b32_e32 v115, 0x400, v115
	v_or3_b32 v122, v122, v115, s4
	v_ashrrev_i32_e32 v123, 31, v122
	v_lshl_add_u64 v[122:123], v[122:123], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[140:141], v[182:183], v[66:81]
	global_load_dwordx4 v[166:169], v[122:123], off
	global_load_dwordx4 v[158:161], v[122:123], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[188:189], v[66:81]
	global_load_dwordx4 v[162:165], v[122:123], off offset:1024
	global_load_dwordx4 v[148:151], v[122:123], off offset:1536
	v_add_co_u32_e32 v122, vcc, s90, v122
	s_nop 1
	v_addc_co_u32_e32 v123, vcc, 0, v123, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[194:195], v[66:81]
	global_load_dwordx4 v[144:147], v[122:123], off
	global_load_dwordx4 v[136:139], v[122:123], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[200:201], v[66:81]
	global_load_dwordx4 v[140:143], v[122:123], off offset:1024
	global_load_dwordx4 v[132:135], v[122:123], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v115, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v115, v115, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v115, v115, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v115, v115, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v115, v115, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v115, v115, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v115, v115, v78, v79
	v_max3_f32 v115, v115, v80, v81
	ds_bpermute_b32 v122, v153, v115
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v122, v122, v122
	v_max_f32_e32 v122, v115, v122
	;;#ASMSTART
	v_add_f32 v115, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v122, v115
	v_mov_b32_e32 v115, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_28
	;;#ASMSTART
	v_add_f32 v96, v122, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v120, v96
	v_exp_f32_e32 v115, v97
	v_mov_b32_e32 v120, v96
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
.LBB0_28:
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
	v_add_f32_e32 v122, 0, v66
	v_add_f32_e32 v122, v122, v67
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
	v_add_f32_e32 v122, v122, v68
	v_add_f32_e32 v122, v122, v69
	v_add_f32_e32 v122, v122, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v71
	v_add_f32_e32 v122, v122, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v73
	v_add_f32_e32 v122, v122, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v75
	v_add_f32_e32 v122, v122, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v77
	v_add_f32_e32 v122, v122, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v115
	v_add_f32_e32 v122, v122, v79
	v_add_f32_e32 v122, v122, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v122, v122, v81
	;;#ASMSTART
	v_fma_f32 v114, v114, v115, v122
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_30
	;;#ASMSTART
	v_mul_f32 v34, v34, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v115
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[166:167], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[158:159], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[162:163], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[148:149], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[168:169], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[160:161], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[164:165], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[150:151], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[136:137], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[140:141], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[142:143], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[132:135], v68
	v_or_b32_e32 v68, 16, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[136:139], v68
	v_or_b32_e32 v68, 32, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[140:143], v68
	v_or_b32_e32 v68, 48, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[144:147], v66
	v_or_b32_e32 v66, 64, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	v_or_b32_e32 v66, 0x50, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	v_or_b32_e32 v66, 0x60, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[210:213], v66
	v_or_b32_e32 v66, 0x70, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	ds_write_b128 v152, v[124:127] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[174:175], 0
	s_addk_i32 s13, 0x2000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[176:177], v[66:81]
	global_load_dwordx4 v[124:127], v[112:113], off offset:1024
	;;#ASMSTART
	v_mov_b32 v115, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v122, 3, v115
	v_lshlrev_b32_e32 v115, 5, v115
	v_and_b32_e32 v122, 0xf8, v122
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], v[66:81]
	v_and_b32_e32 v115, 0x400, v115
	v_or3_b32 v122, v122, v115, s13
	v_ashrrev_i32_e32 v123, 31, v122
	v_lshl_add_u64 v[122:123], v[122:123], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[140:141], v[182:183], v[66:81]
	global_load_dwordx4 v[166:169], v[122:123], off
	global_load_dwordx4 v[158:161], v[122:123], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[188:189], v[66:81]
	global_load_dwordx4 v[162:165], v[122:123], off offset:1024
	global_load_dwordx4 v[148:151], v[122:123], off offset:1536
	v_add_co_u32_e32 v122, vcc, s90, v122
	s_nop 1
	v_addc_co_u32_e32 v123, vcc, 0, v123, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[194:195], v[66:81]
	global_load_dwordx4 v[144:147], v[122:123], off
	global_load_dwordx4 v[136:139], v[122:123], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[200:201], v[66:81]
	global_load_dwordx4 v[140:143], v[122:123], off offset:1024
	global_load_dwordx4 v[132:135], v[122:123], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v115, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v115, v115, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v115, v115, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v115, v115, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v115, v115, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v115, v115, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v115, v115, v78, v79
	v_max3_f32 v115, v115, v80, v81
	ds_bpermute_b32 v122, v153, v115
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v122, v122, v122
	v_max_f32_e32 v122, v115, v122
	;;#ASMSTART
	v_add_f32 v115, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v122, v115
	v_mov_b32_e32 v115, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v96, v122, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v120, v96
	v_exp_f32_e32 v115, v97
	v_mov_b32_e32 v120, v96
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
.LBB0_32:
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
	v_add_f32_e32 v122, 0, v66
	v_add_f32_e32 v122, v122, v67
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
	v_add_f32_e32 v122, v122, v68
	v_add_f32_e32 v122, v122, v69
	v_add_f32_e32 v122, v122, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v71
	v_add_f32_e32 v122, v122, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v73
	v_add_f32_e32 v122, v122, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v75
	v_add_f32_e32 v122, v122, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v77
	v_add_f32_e32 v122, v122, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v115
	v_add_f32_e32 v122, v122, v79
	v_add_f32_e32 v122, v122, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v122, v122, v81
	;;#ASMSTART
	v_fma_f32 v114, v114, v115, v122
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_34
	;;#ASMSTART
	v_mul_f32 v34, v34, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v115
	;;#ASMEND
.LBB0_34:
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[166:167], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[158:159], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[162:163], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[148:149], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[168:169], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[160:161], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[164:165], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[150:151], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[136:137], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[140:141], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[142:143], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[132:135], v68 offset:8192
	v_or_b32_e32 v68, 0x1010, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[136:139], v68
	v_or_b32_e32 v68, 0x1020, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[140:143], v68
	v_or_b32_e32 v68, 0x1030, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[144:147], v66
	v_or_b32_e32 v66, 0x1040, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[158:161], v66
	v_or_b32_e32 v66, 0x1050, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	v_or_b32_e32 v66, 0x1060, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	v_or_b32_e32 v66, 0x1070, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[210:213], v66
	ds_write_b128 v152, v[128:131]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[174:175], 0
	v_mov_b32_e32 v115, s11
	s_lshl_b32 s13, s92, 14
	s_add_i32 s13, s13, s98
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[176:177], v[66:81]
	buffer_load_dword v115, v115, s[68:71], 0 offen
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s99, v115
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[140:141], v[182:183], v[66:81]
	global_load_dwordx4 v[128:131], v[112:113], off offset:1536
	;;#ASMSTART
	v_mov_b32 v112, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v113, 3, v112
	v_lshlrev_b32_e32 v112, 5, v112
	v_and_b32_e32 v113, 0xf8, v113
	v_and_b32_e32 v112, 0x400, v112
	v_or3_b32 v112, v113, v112, s13
	v_ashrrev_i32_e32 v113, 31, v112
	v_lshl_add_u64 v[112:113], v[112:113], 1, s[84:85]
	global_load_dwordx4 v[162:165], v[112:113], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[188:189], v[66:81]
	global_load_dwordx4 v[166:169], v[112:113], off offset:512
	global_load_dwordx4 v[148:151], v[112:113], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[158:159], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[194:195], v[66:81]
	global_load_dwordx4 v[158:161], v[112:113], off offset:1536
	v_add_co_u32_e32 v112, vcc, s90, v112
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	global_load_dwordx4 v[140:143], v[112:113], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[200:201], v[66:81]
	global_load_dwordx4 v[144:147], v[112:113], off offset:512
	global_load_dwordx4 v[136:139], v[112:113], off offset:1024
	global_load_dwordx4 v[132:135], v[112:113], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v112, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v112, v112, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v112, v112, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v112, v112, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v112, v112, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v112, v112, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v112, v112, v78, v79
	v_max3_f32 v112, v112, v80, v81
	ds_bpermute_b32 v113, v153, v112
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v113, v113, v113
	v_max_f32_e32 v112, v112, v113
	;;#ASMSTART
	v_add_f32 v113, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v112, v113
	v_mov_b32_e32 v113, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_36
	;;#ASMSTART
	v_add_f32 v96, v112, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v120, v96
	v_exp_f32_e32 v113, v97
	v_mov_b32_e32 v120, v96
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
.LBB0_36:
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
	v_pk_add_f32 v[72:73], v[72:73], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v68
	v_add_f32_e32 v112, v112, v69
	v_add_f32_e32 v112, v112, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v71
	v_add_f32_e32 v112, v112, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v73
	v_add_f32_e32 v112, v112, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v75
	v_add_f32_e32 v112, v112, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v112, v112, v77
	v_add_f32_e32 v112, v112, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v113
	v_add_f32_e32 v112, v112, v79
	v_add_f32_e32 v112, v112, v80
	v_add_f32_e32 v112, v112, v81
	;;#ASMSTART
	v_fma_f32 v112, v114, v113, v112
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_38
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
.LBB0_38:
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[162:163], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[166:167], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[148:149], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[158:159], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[164:165], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[168:169], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[150:151], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[160:161], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[140:141], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[144:145], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[136:137], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[142:143], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[146:147], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[138:139], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[132:135], v68
	v_or_b32_e32 v68, 16, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[136:139], v68
	v_or_b32_e32 v68, 32, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[140:143], v68
	v_or_b32_e32 v68, 48, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[144:147], v66
	v_or_b32_e32 v66, 64, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	v_or_b32_e32 v66, 0x50, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	v_or_b32_e32 v66, 0x60, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[210:213], v66
	v_or_b32_e32 v66, 0x70, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	ds_write_b128 v152, v[124:127] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[174:175], 0
	s_lshl_b32 s4, s8, 13
	v_or_b32_e32 v114, s4, v116
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 2, s[80:81]
	s_addk_i32 s13, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[176:177], v[66:81]
	global_load_dwordx4 v[124:127], v[114:115], off
	;;#ASMSTART
	v_mov_b32 v113, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v114, 3, v113
	v_lshlrev_b32_e32 v113, 5, v113
	v_and_b32_e32 v114, 0xf8, v114
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], v[66:81]
	v_and_b32_e32 v113, 0x400, v113
	v_or3_b32 v114, v114, v113, s13
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[140:141], v[182:183], v[66:81]
	global_load_dwordx4 v[166:169], v[114:115], off
	global_load_dwordx4 v[158:161], v[114:115], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[188:189], v[66:81]
	global_load_dwordx4 v[162:165], v[114:115], off offset:1024
	global_load_dwordx4 v[148:151], v[114:115], off offset:1536
	v_add_co_u32_e32 v114, vcc, s90, v114
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[194:195], v[66:81]
	global_load_dwordx4 v[144:147], v[114:115], off
	global_load_dwordx4 v[136:139], v[114:115], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[200:201], v[66:81]
	global_load_dwordx4 v[140:143], v[114:115], off offset:1024
	global_load_dwordx4 v[132:135], v[114:115], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
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
	ds_bpermute_b32 v114, v153, v113
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v114, v114, v114
	v_max_f32_e32 v114, v113, v114
	;;#ASMSTART
	v_add_f32 v113, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v114, v113
	v_mov_b32_e32 v113, 1.0
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_40
	;;#ASMSTART
	v_add_f32 v96, v114, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v120, v96
	v_exp_f32_e32 v113, v97
	v_mov_b32_e32 v120, v96
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
	s_or_b64 exec, exec, s[6:7]
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
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v113
	v_add_f32_e32 v114, v114, v79
	v_add_f32_e32 v114, v114, v80
	v_add_f32_e32 v114, v114, v81
	;;#ASMSTART
	v_fma_f32 v114, v112, v113, v114
	;;#ASMEND
	s_and_saveexec_b64 s[6:7], vcc
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[166:167], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[158:159], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[162:163], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[148:149], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[168:169], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[160:161], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[164:165], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[150:151], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[136:137], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[140:141], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[146:147], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[142:143], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[132:135], v68 offset:8192
	v_or_b32_e32 v68, 0x1010, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[136:139], v68
	v_or_b32_e32 v68, 0x1020, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[140:143], v68
	v_or_b32_e32 v68, 0x1030, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[144:147], v66
	v_or_b32_e32 v66, 0x1040, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	v_or_b32_e32 v66, 0x1050, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	v_or_b32_e32 v66, 0x1060, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[210:213], v66
	v_or_b32_e32 v66, 0x1070, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	ds_write_b128 v152, v[128:131]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[174:175], 0
	s_ashr_i32 s5, s4, 31
	v_lshl_add_u64 v[112:113], s[4:5], 0, v[116:117]
	v_lshl_add_u64 v[112:113], v[112:113], 2, s[80:81]
	s_add_i32 s4, s13, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[176:177], v[66:81]
	global_load_dwordx4 v[158:161], v[112:113], off offset:512
	;;#ASMSTART
	v_mov_b32 v115, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v122, 3, v115
	v_lshlrev_b32_e32 v115, 5, v115
	v_and_b32_e32 v122, 0xf8, v122
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[178:179], v[66:81]
	v_and_b32_e32 v115, 0x400, v115
	v_or3_b32 v122, v122, v115, s4
	v_ashrrev_i32_e32 v123, 31, v122
	v_lshl_add_u64 v[122:123], v[122:123], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[140:141], v[182:183], v[66:81]
	global_load_dwordx4 v[166:169], v[122:123], off
	global_load_dwordx4 v[148:151], v[122:123], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[188:189], v[66:81]
	global_load_dwordx4 v[162:165], v[122:123], off offset:1024
	global_load_dwordx4 v[144:147], v[122:123], off offset:1536
	v_add_co_u32_e32 v122, vcc, s90, v122
	s_nop 1
	v_addc_co_u32_e32 v123, vcc, 0, v123, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[194:195], v[66:81]
	global_load_dwordx4 v[140:143], v[122:123], off
	global_load_dwordx4 v[132:135], v[122:123], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[200:201], v[66:81]
	global_load_dwordx4 v[136:139], v[122:123], off offset:1024
	global_load_dwordx4 v[128:131], v[122:123], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v115, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v115, v115, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v115, v115, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v115, v115, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v115, v115, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v115, v115, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v115, v115, v78, v79
	v_max3_f32 v115, v115, v80, v81
	ds_bpermute_b32 v122, v153, v115
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v122, v122, v122
	v_max_f32_e32 v122, v115, v122
	;;#ASMSTART
	v_add_f32 v115, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v122, v115
	v_mov_b32_e32 v115, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_44
	;;#ASMSTART
	v_add_f32 v96, v122, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v120, v96
	v_exp_f32_e32 v115, v97
	v_mov_b32_e32 v120, v96
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
.LBB0_44:
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
	v_add_f32_e32 v122, 0, v66
	v_add_f32_e32 v122, v122, v67
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
	v_add_f32_e32 v122, v122, v68
	v_add_f32_e32 v122, v122, v69
	v_add_f32_e32 v122, v122, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v71
	v_add_f32_e32 v122, v122, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[106:107] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v73
	v_add_f32_e32 v122, v122, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[108:109] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v75
	v_add_f32_e32 v122, v122, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v77
	v_add_f32_e32 v122, v122, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v115
	v_add_f32_e32 v122, v122, v79
	v_add_f32_e32 v122, v122, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v122, v122, v81
	;;#ASMSTART
	v_fma_f32 v114, v114, v115, v122
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_46
	;;#ASMSTART
	v_mul_f32 v34, v34, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v115
	;;#ASMEND
.LBB0_46:
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[166:167], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[162:163], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[144:145], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[168:169], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[164:165], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[146:147], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[140:141], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[136:137], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[128:129], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[142:143], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[138:139], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[130:131], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[128:131], v68
	v_or_b32_e32 v68, 16, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[132:135], v68
	v_or_b32_e32 v68, 32, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[136:139], v68
	v_or_b32_e32 v68, 48, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[146:149], v66
	v_or_b32_e32 v66, 64, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	v_or_b32_e32 v66, 0x50, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	v_or_b32_e32 v66, 0x60, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[210:213], v66
	v_or_b32_e32 v66, 0x70, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	ds_write_b128 v152, v[124:127] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[128:129], v[174:175], 0
	s_addk_i32 s13, 0x2000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[130:131], v[176:177], v[66:81]
	global_load_dwordx4 v[162:165], v[112:113], off offset:1024
	;;#ASMSTART
	v_mov_b32 v112, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v113, 3, v112
	v_lshlrev_b32_e32 v112, 5, v112
	v_and_b32_e32 v113, 0xf8, v113
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[178:179], v[66:81]
	v_and_b32_e32 v112, 0x400, v112
	v_or3_b32 v112, v113, v112, s13
	v_ashrrev_i32_e32 v113, 31, v112
	v_lshl_add_u64 v[112:113], v[112:113], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[182:183], v[66:81]
	global_load_dwordx4 v[166:169], v[112:113], off
	global_load_dwordx4 v[142:145], v[112:113], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[148:149], v[188:189], v[66:81]
	global_load_dwordx4 v[146:149], v[112:113], off offset:1024
	global_load_dwordx4 v[138:141], v[112:113], off offset:1536
	v_add_co_u32_e32 v112, vcc, s90, v112
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[194:195], v[66:81]
	global_load_dwordx4 v[134:137], v[112:113], off
	global_load_dwordx4 v[126:129], v[112:113], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[200:201], v[66:81]
	global_load_dwordx4 v[130:133], v[112:113], off offset:1024
	global_load_dwordx4 v[122:125], v[112:113], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
	v_pk_mul_f32 v[68:69], v[82:83], v[68:69]
	v_max_f32_e32 v112, v66, v67
	v_pk_mul_f32 v[70:71], v[84:85], v[70:71]
	v_max3_f32 v112, v112, v68, v69
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_max3_f32 v112, v112, v70, v71
	v_pk_mul_f32 v[74:75], v[88:89], v[74:75]
	v_max3_f32 v112, v112, v72, v73
	v_pk_mul_f32 v[76:77], v[90:91], v[76:77]
	v_max3_f32 v112, v112, v74, v75
	v_pk_mul_f32 v[78:79], v[92:93], v[78:79]
	v_max3_f32 v112, v112, v76, v77
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	v_max3_f32 v112, v112, v78, v79
	v_max3_f32 v112, v112, v80, v81
	ds_bpermute_b32 v113, v153, v112
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v113, v113, v113
	v_max_f32_e32 v113, v112, v113
	;;#ASMSTART
	v_add_f32 v112, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v113, v112
	v_mov_b32_e32 v112, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_48
	;;#ASMSTART
	v_add_f32 v96, v113, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v97, v120, v96
	v_exp_f32_e32 v112, v97
	v_mov_b32_e32 v120, v96
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
	v_cmp_gt_f32_e32 vcc, 1.0, v112
	v_add_f32_e32 v96, v96, v79
	v_add_f32_e32 v96, v96, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v96, v96, v81
	;;#ASMSTART
	v_fma_f32 v157, v114, v112, v96
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_17
	;;#ASMSTART
	v_mul_f32 v34, v34, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v112
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v112
	;;#ASMEND
	s_branch .LBB0_17
.LBB0_50:
	s_mov_b32 s92, s12
	s_cmp_ge_i32 s82, s72
	s_cbranch_scc0 .LBB0_52
	s_branch .LBB0_89
.LBB0_51:
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
	s_mov_b64 s[6:7], s[52:53]
	s_mov_b32 s52, s36
	s_mov_b32 s53, s36
	s_mov_b64 s[12:13], s[54:55]
	s_mov_b32 s54, s36
	s_mov_b32 s55, s36
	s_mov_b64 s[14:15], s[56:57]
	s_mov_b32 s56, s36
	s_mov_b32 s57, s36
	s_mov_b64 s[16:17], s[58:59]
	s_mov_b32 s58, s36
	s_mov_b32 s59, s36
	s_mov_b32 s11, s60
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
	v_mov_b32_e32 v120, 0xff800000
	v_mov_b32_e32 v157, 0
	s_mov_b32 s60, s11
	s_mov_b64 s[58:59], s[16:17]
	s_mov_b64 s[56:57], s[14:15]
	s_mov_b64 s[54:55], s[12:13]
	s_mov_b64 s[52:53], s[6:7]
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
	s_cmp_ge_i32 s82, s72
	s_cbranch_scc1 .LBB0_89
.LBB0_52:
	s_lshl_b32 s37, s10, 7
	s_lshl_b32 s0, s82, 7
	s_ashr_i32 s73, s72, 31
	s_add_i32 s37, s37, s9
	v_mov_b32_e32 v119, v118
	v_mov_b32_e32 v122, v118
	v_mov_b32_e32 v123, v118
	v_mov_b32_e32 v124, v118
	v_mov_b32_e32 v125, v118
	v_mov_b32_e32 v126, v118
	v_mov_b32_e32 v127, v118
	v_mov_b32_e32 v128, v118
	v_mov_b32_e32 v129, v118
	v_mov_b32_e32 v130, v118
	v_mov_b32_e32 v131, v118
	v_mov_b32_e32 v132, v118
	v_mov_b32_e32 v133, v118
	v_mov_b32_e32 v134, v118
	v_mov_b32_e32 v135, v118
	s_add_i32 s42, s0, 0xf7
	s_add_i32 s43, s82, 1
	s_lshl2_add_u32 s44, s82, 20
	s_branch .LBB0_55
.LBB0_53:
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
	v_perm_b32 v2, v3, v2, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v5, v4, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v7, v6, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v9, v8, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v11, v10, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v13, v12, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v15, v14, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v17, v16, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[52:67], v[166:167], v[2:3], v[52:67]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[68:83], v[42:43], v[2:3], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[46:47], v[2:3], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[38:39], v[2:3], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[168:169], v[4:5], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[44:45], v[4:5], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[48:49], v[4:5], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[40:41], v[4:5], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[34:35], v[6:7], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[26:27], v[6:7], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[30:31], v[6:7], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[22:23], v[6:7], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[36:37], v[8:9], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[28:29], v[8:9], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[32:33], v[8:9], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[24:25], v[8:9], v[100:115]
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v4, 2, v2
	v_lshlrev_b32_e32 v3, 7, v2
	v_and_b32_e32 v4, 8, v4
	v_lshlrev_b32_e32 v2, 4, v2
	v_and_or_b32 v3, v3, s89, v4
	v_and_b32_e32 v2, 48, v2
	v_or_b32_e32 v4, v3, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[226:229], v4 offset:8192
	v_or_b32_e32 v4, 0x1010, v3
	v_xor_b32_e32 v4, v4, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[210:213], v4
	v_or_b32_e32 v4, 0x1020, v3
	v_xor_b32_e32 v4, v4, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[166:169], v4
	v_or_b32_e32 v4, 0x1030, v3
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[206:209], v2
	v_or_b32_e32 v2, 0x1040, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[214:217], v2
	v_or_b32_e32 v2, 0x1050, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[218:221], v2
	v_or_b32_e32 v2, 0x1060, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[222:225], v2
	v_or_b32_e32 v2, 0x1070, v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[170:173], v2
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
.LBB0_54:
	s_and_b64 s[0:1], s[38:39], exec
	s_cselect_b32 s8, s75, s92
	s_cselect_b32 s92, s99, s75
	s_cselect_b32 s75, s45, s99
	s_add_u32 s82, s82, 2
	s_addc_u32 s83, s83, 0
	v_mov_b64_e32 v[66:67], s[72:73]
	v_cmp_lt_i64_e32 vcc, s[82:83], v[66:67]
	s_addk_i32 s42, 0x100
	s_add_i32 s43, s43, 2
	s_add_i32 s44, s44, 8
	s_mov_b32 s99, s46
	s_cbranch_vccz .LBB0_89
.LBB0_55:
	ds_write_b128 v152, v[158:161]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[174:175], 0
	s_add_i32 s0, s44, -4
	v_mov_b32_e32 v82, s0
	s_lshl_b32 s0, s8, 13
	s_ashr_i32 s1, s0, 31
	v_mov_b32_e32 v83, s1
	s_lshl_b32 s40, s8, 14
	s_add_i32 s40, s40, s98
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[176:177], v[66:81]
	buffer_load_dword v140, v82, s[68:71], 0 offen
	v_or_b32_e32 v82, s0, v116
	v_lshl_add_u64 v[82:83], v[82:83], 2, s[80:81]
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v140
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[178:179], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[182:183], v[66:81]
	global_load_dwordx4 v[92:95], v[82:83], off offset:1536
	;;#ASMSTART
	v_mov_b32 v82, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v83, 3, v82
	v_lshlrev_b32_e32 v82, 5, v82
	v_and_b32_e32 v83, 0xf8, v83
	v_and_b32_e32 v82, 0x400, v82
	v_or3_b32 v82, v83, v82, s40
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, s[84:85]
	global_load_dwordx4 v[112:115], v[82:83], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[188:189], v[66:81]
	global_load_dwordx4 v[136:139], v[82:83], off offset:512
	global_load_dwordx4 v[104:107], v[82:83], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[194:195], v[66:81]
	global_load_dwordx4 v[108:111], v[82:83], off offset:1536
	v_add_co_u32_e32 v82, vcc, s90, v82
	s_nop 1
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	global_load_dwordx4 v[96:99], v[82:83], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[200:201], v[66:81]
	global_load_dwordx4 v[100:103], v[82:83], off offset:512
	global_load_dwordx4 v[88:91], v[82:83], off offset:1024
	global_load_dwordx4 v[84:87], v[82:83], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v82, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v82, 2, v82
	v_and_b32_e32 v82, 8, v82
	v_add_u32_e32 v82, s42, v82
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
	v_cndmask_b32_e64 v79, v156, v79, s[26:27]
	v_cndmask_b32_e64 v78, v156, v78, s[24:25]
	v_cndmask_b32_e64 v143, v156, v67, s[0:1]
	v_cndmask_b32_e32 v142, v156, v66, vcc
	v_cndmask_b32_e64 v77, v156, v77, s[22:23]
	v_cndmask_b32_e64 v76, v156, v76, s[20:21]
	v_cndmask_b32_e64 v75, v156, v75, s[18:19]
	v_cndmask_b32_e64 v74, v156, v74, s[16:17]
	v_cndmask_b32_e64 v83, v156, v71, s[10:11]
	v_cndmask_b32_e64 v82, v156, v70, s[8:9]
	v_cndmask_b32_e64 v141, v156, v69, s[6:7]
	v_cndmask_b32_e64 v140, v156, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[132:133], v[78:79]
	v_pk_mul_f32 v[78:79], v[118:119], v[142:143]
	v_pk_mul_f32 v[68:69], v[130:131], v[76:77]
	v_pk_mul_f32 v[70:71], v[128:129], v[74:75]
	v_pk_mul_f32 v[74:75], v[124:125], v[82:83]
	v_pk_mul_f32 v[76:77], v[122:123], v[140:141]
	v_max_f32_e32 v82, v78, v79
	v_cndmask_b32_e64 v73, v156, v73, s[14:15]
	v_cndmask_b32_e64 v72, v156, v72, s[12:13]
	v_max3_f32 v82, v82, v76, v77
	v_pk_mul_f32 v[72:73], v[126:127], v[72:73]
	v_max3_f32 v82, v82, v74, v75
	v_max3_f32 v82, v82, v72, v73
	v_max3_f32 v82, v82, v70, v71
	v_cndmask_b32_e64 v81, v156, v81, s[30:31]
	v_cndmask_b32_e64 v80, v156, v80, s[28:29]
	v_max3_f32 v82, v82, v68, v69
	v_pk_mul_f32 v[80:81], v[134:135], v[80:81]
	v_max3_f32 v82, v82, v66, v67
	v_max3_f32 v82, v82, v80, v81
	ds_bpermute_b32 v83, v153, v82
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v83, v83, v83
	v_max_f32_e32 v82, v82, v83
	;;#ASMSTART
	v_add_f32 v83, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v82, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_57
	;;#ASMSTART
	v_add_f32 v82, v82, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v120, v82
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v120, v82
.LBB0_57:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[140:141], v[66:67], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67], v[78:79], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[142:143], v[68:69], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	v_pk_add_f32 v[68:69], v[76:77], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, 0, v66
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	v_pk_add_f32 v[144:145], v[70:71], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v67
	v_add_f32_e32 v82, v82, v68
	v_pk_add_f32 v[70:71], v[74:75], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
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
	v_exp_f32 v74, v144
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v75, v145
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v82, v82, v70
	v_add_f32_e32 v82, v82, v71
	v_add_f32_e32 v82, v82, v72
	v_add_f32_e32 v82, v82, v73
	v_add_f32_e32 v82, v82, v74
	v_add_f32_e32 v82, v82, v75
	;;#ASMSTART
	v_exp_f32 v76, v142
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v77, v143
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v140
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v76
	v_add_f32_e32 v82, v82, v77
	v_add_f32_e32 v82, v82, v78
	;;#ASMSTART
	v_exp_f32 v79, v141
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
	v_fma_f32 v82, v157, v83, v82
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_59
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
.LBB0_59:
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[112:113], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[136:137], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[104:105], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[114:115], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[106:107], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[110:111], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[96:97], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[100:101], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[88:89], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[84:85], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[98:99], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[102:103], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[90:91], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[86:87], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[84:87], v68
	v_or_b32_e32 v68, 16, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[96:99], v68
	v_or_b32_e32 v68, 32, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[100:103], v68
	v_or_b32_e32 v68, 48, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[104:107], v66
	v_or_b32_e32 v66, 64, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[108:111], v66
	v_or_b32_e32 v66, 0x50, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[136:139], v66
	v_or_b32_e32 v66, 0x60, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[140:143], v66
	v_or_b32_e32 v66, 0x70, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[144:147], v66
	ds_write_b128 v152, v[162:165] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[84:85], v[174:175], 0
	s_lshl_b32 s38, s92, 13
	v_or_b32_e32 v84, s38, v116
	v_ashrrev_i32_e32 v85, 31, v84
	v_lshl_add_u64 v[84:85], v[84:85], 2, s[80:81]
	s_addk_i32 s40, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[86:87], v[176:177], v[66:81]
	global_load_dwordx4 v[88:91], v[84:85], off
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v84, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	v_and_b32_e32 v84, 0xf8, v84
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[178:179], v[66:81]
	v_and_b32_e32 v83, 0x400, v83
	v_or3_b32 v84, v84, v83, s40
	v_ashrrev_i32_e32 v85, 31, v84
	v_lshl_add_u64 v[84:85], v[84:85], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[182:183], v[66:81]
	global_load_dwordx4 v[166:169], v[84:85], off
	global_load_dwordx4 v[158:161], v[84:85], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[188:189], v[66:81]
	global_load_dwordx4 v[162:165], v[84:85], off offset:1024
	global_load_dwordx4 v[112:115], v[84:85], off offset:1536
	v_add_co_u32_e32 v84, vcc, s90, v84
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[194:195], v[66:81]
	global_load_dwordx4 v[108:111], v[84:85], off
	global_load_dwordx4 v[100:103], v[84:85], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[140:141], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[200:201], v[66:81]
	global_load_dwordx4 v[104:107], v[84:85], off offset:1024
	global_load_dwordx4 v[96:99], v[84:85], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	v_mov_b32_e32 v136, v120
	v_lshrrev_b32_e32 v83, 2, v83
	v_and_b32_e32 v83, 8, v83
	v_add_u32_e32 v83, s42, v83
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
	v_cndmask_b32_e64 v67, v156, v67, s[0:1]
	v_cndmask_b32_e32 v66, v156, v66, vcc
	v_cndmask_b32_e64 v69, v156, v69, s[6:7]
	v_cndmask_b32_e64 v68, v156, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
	v_cndmask_b32_e64 v71, v156, v71, s[10:11]
	v_cndmask_b32_e64 v70, v156, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[122:123], v[68:69]
	v_max_f32_e32 v83, v66, v67
	v_cndmask_b32_e64 v73, v156, v73, s[14:15]
	v_cndmask_b32_e64 v72, v156, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[124:125], v[70:71]
	v_max3_f32 v83, v83, v68, v69
	v_cndmask_b32_e64 v75, v156, v75, s[18:19]
	v_cndmask_b32_e64 v74, v156, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[126:127], v[72:73]
	v_max3_f32 v83, v83, v70, v71
	v_cndmask_b32_e64 v77, v156, v77, s[22:23]
	v_cndmask_b32_e64 v76, v156, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[128:129], v[74:75]
	v_max3_f32 v83, v83, v72, v73
	v_cndmask_b32_e64 v79, v156, v79, s[26:27]
	v_cndmask_b32_e64 v78, v156, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[130:131], v[76:77]
	v_max3_f32 v83, v83, v74, v75
	v_cndmask_b32_e64 v81, v156, v81, s[30:31]
	v_cndmask_b32_e64 v80, v156, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[132:133], v[78:79]
	v_max3_f32 v83, v83, v76, v77
	v_pk_mul_f32 v[80:81], v[134:135], v[80:81]
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v84, v153, v83
	v_mov_b32_e32 v137, v120
	v_mov_b32_e32 v138, v120
	v_mov_b32_e32 v139, v120
	v_mov_b32_e32 v140, v120
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v84, v84, v84
	v_max_f32_e32 v84, v83, v84
	;;#ASMSTART
	v_add_f32 v83, v120, v154
	;;#ASMEND
	v_mov_b32_e32 v141, v120
	v_cmp_gt_f32_e32 vcc, v84, v83
	v_mov_b32_e32 v83, 1.0
	v_mov_b32_e32 v142, v120
	v_mov_b32_e32 v143, v120
	v_mov_b32_e32 v144, v120
	v_mov_b32_e32 v145, v120
	v_mov_b32_e32 v146, v120
	v_mov_b32_e32 v147, v120
	v_mov_b32_e32 v148, v120
	v_mov_b32_e32 v149, v120
	v_mov_b32_e32 v150, v120
	v_mov_b32_e32 v151, v120
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_61
	;;#ASMSTART
	v_add_f32 v136, v84, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v120, v136
	v_exp_f32_e32 v83, v83
	v_mov_b32_e32 v120, v136
	v_mov_b32_e32 v137, v136
	v_mov_b32_e32 v138, v136
	v_mov_b32_e32 v139, v136
	v_mov_b32_e32 v140, v136
	v_mov_b32_e32 v141, v136
	v_mov_b32_e32 v142, v136
	v_mov_b32_e32 v143, v136
	v_mov_b32_e32 v144, v136
	v_mov_b32_e32 v145, v136
	v_mov_b32_e32 v146, v136
	v_mov_b32_e32 v147, v136
	v_mov_b32_e32 v148, v136
	v_mov_b32_e32 v149, v136
	v_mov_b32_e32 v150, v136
	v_mov_b32_e32 v151, v136
.LBB0_61:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[72:73], v[72:73], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v68
	v_add_f32_e32 v84, v84, v69
	v_add_f32_e32 v84, v84, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v71
	v_add_f32_e32 v84, v84, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v73
	v_add_f32_e32 v84, v84, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v75
	v_add_f32_e32 v84, v84, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[150:151] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v84, v84, v77
	v_add_f32_e32 v84, v84, v78
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
	v_add_f32_e32 v84, v84, v79
	v_add_f32_e32 v84, v84, v80
	v_add_f32_e32 v84, v84, v81
	;;#ASMSTART
	v_fma_f32 v84, v82, v83, v84
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_63
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
.LBB0_63:
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[166:167], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[158:159], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[162:163], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[112:113], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[168:169], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[160:161], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[164:165], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[114:115], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[108:109], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[100:101], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[104:105], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[96:97], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[110:111], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[102:103], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[106:107], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[98:99], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[96:99], v68 offset:8192
	v_or_b32_e32 v68, 0x1010, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[100:103], v68
	v_or_b32_e32 v68, 0x1020, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[104:107], v68
	v_or_b32_e32 v68, 0x1030, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[108:111], v66
	v_or_b32_e32 v66, 0x1040, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	v_or_b32_e32 v66, 0x1050, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	v_or_b32_e32 v66, 0x1060, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[210:213], v66
	v_or_b32_e32 v66, 0x1070, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	ds_write_b128 v152, v[92:95]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[174:175], 0
	s_ashr_i32 s39, s38, 31
	v_lshl_add_u64 v[82:83], s[38:39], 0, v[116:117]
	v_lshl_add_u64 v[82:83], v[82:83], 2, s[80:81]
	s_add_i32 s0, s40, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[176:177], v[66:81]
	global_load_dwordx4 v[158:161], v[82:83], off offset:512
	;;#ASMSTART
	v_mov_b32 v85, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v86, 3, v85
	v_lshlrev_b32_e32 v85, 5, v85
	v_and_b32_e32 v86, 0xf8, v86
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[178:179], v[66:81]
	v_and_b32_e32 v85, 0x400, v85
	v_or3_b32 v86, v86, v85, s0
	v_ashrrev_i32_e32 v87, 31, v86
	v_lshl_add_u64 v[86:87], v[86:87], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[182:183], v[66:81]
	global_load_dwordx4 v[166:169], v[86:87], off
	global_load_dwordx4 v[112:115], v[86:87], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[188:189], v[66:81]
	global_load_dwordx4 v[162:165], v[86:87], off offset:1024
	global_load_dwordx4 v[108:111], v[86:87], off offset:1536
	v_add_co_u32_e32 v86, vcc, s90, v86
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[194:195], v[66:81]
	global_load_dwordx4 v[104:107], v[86:87], off
	global_load_dwordx4 v[96:99], v[86:87], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[200:201], v[66:81]
	global_load_dwordx4 v[100:103], v[86:87], off offset:1024
	global_load_dwordx4 v[92:95], v[86:87], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v85, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v85, 2, v85
	v_and_b32_e32 v85, 8, v85
	v_add_u32_e32 v85, s42, v85
	v_add_u32_e32 v86, 0xffffff49, v85
	v_cmp_gt_i32_e32 vcc, s37, v86
	v_add_u32_e32 v86, 0xffffff4a, v85
	v_cmp_gt_i32_e64 s[0:1], s37, v86
	v_add_u32_e32 v86, 0xffffff4b, v85
	v_cmp_gt_i32_e64 s[4:5], s37, v86
	v_add_u32_e32 v86, 0xffffff4c, v85
	v_cmp_gt_i32_e64 s[6:7], s37, v86
	v_add_u32_e32 v86, 0xffffff4d, v85
	v_cmp_gt_i32_e64 s[8:9], s37, v86
	v_add_u32_e32 v86, 0xffffff4e, v85
	v_cmp_gt_i32_e64 s[10:11], s37, v86
	v_add_u32_e32 v86, 0xffffff4f, v85
	v_cmp_gt_i32_e64 s[12:13], s37, v86
	v_add_u32_e32 v86, 0xffffff50, v85
	v_cmp_gt_i32_e64 s[14:15], s37, v86
	v_add_u32_e32 v86, 0xffffff59, v85
	v_cmp_gt_i32_e64 s[16:17], s37, v86
	v_add_u32_e32 v86, 0xffffff5a, v85
	v_cmp_gt_i32_e64 s[18:19], s37, v86
	v_add_u32_e32 v86, 0xffffff5b, v85
	v_cmp_gt_i32_e64 s[20:21], s37, v86
	v_add_u32_e32 v86, 0xffffff5c, v85
	v_cmp_gt_i32_e64 s[22:23], s37, v86
	v_add_u32_e32 v86, 0xffffff5d, v85
	v_cmp_gt_i32_e64 s[24:25], s37, v86
	v_add_u32_e32 v86, 0xffffff5e, v85
	v_cmp_gt_i32_e64 s[26:27], s37, v86
	v_add_u32_e32 v86, 0xffffff5f, v85
	v_add_u32_e32 v85, 0xffffff60, v85
	v_cmp_gt_i32_e64 s[28:29], s37, v86
	v_cmp_gt_i32_e64 s[30:31], s37, v85
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
	v_cndmask_b32_e64 v67, v156, v67, s[0:1]
	v_cndmask_b32_e32 v66, v156, v66, vcc
	v_cndmask_b32_e64 v69, v156, v69, s[6:7]
	v_cndmask_b32_e64 v68, v156, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
	v_cndmask_b32_e64 v71, v156, v71, s[10:11]
	v_cndmask_b32_e64 v70, v156, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[122:123], v[68:69]
	v_max_f32_e32 v85, v66, v67
	v_cndmask_b32_e64 v73, v156, v73, s[14:15]
	v_cndmask_b32_e64 v72, v156, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[124:125], v[70:71]
	v_max3_f32 v85, v85, v68, v69
	v_cndmask_b32_e64 v75, v156, v75, s[18:19]
	v_cndmask_b32_e64 v74, v156, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[126:127], v[72:73]
	v_max3_f32 v85, v85, v70, v71
	v_cndmask_b32_e64 v77, v156, v77, s[22:23]
	v_cndmask_b32_e64 v76, v156, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[128:129], v[74:75]
	v_max3_f32 v85, v85, v72, v73
	v_cndmask_b32_e64 v79, v156, v79, s[26:27]
	v_cndmask_b32_e64 v78, v156, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[130:131], v[76:77]
	v_max3_f32 v85, v85, v74, v75
	v_cndmask_b32_e64 v81, v156, v81, s[30:31]
	v_cndmask_b32_e64 v80, v156, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[132:133], v[78:79]
	v_max3_f32 v85, v85, v76, v77
	v_pk_mul_f32 v[80:81], v[134:135], v[80:81]
	v_max3_f32 v85, v85, v78, v79
	v_max3_f32 v85, v85, v80, v81
	ds_bpermute_b32 v86, v153, v85
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v86, v86, v86
	v_max_f32_e32 v86, v85, v86
	;;#ASMSTART
	v_add_f32 v85, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v86, v85
	v_mov_b32_e32 v85, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_65
	;;#ASMSTART
	v_add_f32 v136, v86, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v85, v120, v136
	v_exp_f32_e32 v85, v85
	v_mov_b32_e32 v120, v136
	v_mov_b32_e32 v137, v136
	v_mov_b32_e32 v138, v136
	v_mov_b32_e32 v139, v136
	v_mov_b32_e32 v140, v136
	v_mov_b32_e32 v141, v136
	v_mov_b32_e32 v142, v136
	v_mov_b32_e32 v143, v136
	v_mov_b32_e32 v144, v136
	v_mov_b32_e32 v145, v136
	v_mov_b32_e32 v146, v136
	v_mov_b32_e32 v147, v136
	v_mov_b32_e32 v148, v136
	v_mov_b32_e32 v149, v136
	v_mov_b32_e32 v150, v136
	v_mov_b32_e32 v151, v136
.LBB0_65:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, 0, v66
	v_add_f32_e32 v86, v86, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, v86, v68
	v_add_f32_e32 v86, v86, v69
	v_add_f32_e32 v86, v86, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, v86, v71
	v_add_f32_e32 v86, v86, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, v86, v73
	v_add_f32_e32 v86, v86, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, v86, v75
	v_add_f32_e32 v86, v86, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[150:151] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, v86, v77
	v_add_f32_e32 v86, v86, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v85
	v_add_f32_e32 v86, v86, v79
	v_add_f32_e32 v86, v86, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v86, v86, v81
	;;#ASMSTART
	v_fma_f32 v84, v84, v85, v86
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_67
	;;#ASMSTART
	v_mul_f32 v34, v34, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v85
	;;#ASMEND
.LBB0_67:
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[166:167], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[112:113], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[162:163], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[168:169], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[114:115], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[164:165], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[110:111], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[104:105], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[96:97], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[92:93], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[106:107], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[98:99], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[94:95], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[92:95], v68
	v_or_b32_e32 v68, 16, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[96:99], v68
	v_or_b32_e32 v68, 32, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[100:103], v68
	v_or_b32_e32 v68, 48, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[104:107], v66
	v_or_b32_e32 v66, 64, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	v_or_b32_e32 v66, 0x50, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	v_or_b32_e32 v66, 0x60, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[210:213], v66
	v_or_b32_e32 v66, 0x70, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	ds_write_b128 v152, v[88:91] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[92:93], v[174:175], 0
	s_addk_i32 s40, 0x2000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[94:95], v[176:177], v[66:81]
	global_load_dwordx4 v[162:165], v[82:83], off offset:1024
	;;#ASMSTART
	v_mov_b32 v85, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v86, 3, v85
	v_lshlrev_b32_e32 v85, 5, v85
	v_and_b32_e32 v86, 0xf8, v86
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[178:179], v[66:81]
	v_and_b32_e32 v85, 0x400, v85
	v_or3_b32 v86, v86, v85, s40
	v_ashrrev_i32_e32 v87, 31, v86
	v_lshl_add_u64 v[86:87], v[86:87], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[182:183], v[66:81]
	global_load_dwordx4 v[166:169], v[86:87], off
	global_load_dwordx4 v[108:111], v[86:87], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[188:189], v[66:81]
	global_load_dwordx4 v[112:115], v[86:87], off offset:1024
	global_load_dwordx4 v[104:107], v[86:87], off offset:1536
	v_add_co_u32_e32 v86, vcc, s90, v86
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[194:195], v[66:81]
	global_load_dwordx4 v[100:103], v[86:87], off
	global_load_dwordx4 v[92:95], v[86:87], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[200:201], v[66:81]
	global_load_dwordx4 v[96:99], v[86:87], off offset:1024
	global_load_dwordx4 v[88:91], v[86:87], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v85, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v85, 2, v85
	v_and_b32_e32 v85, 8, v85
	v_add_u32_e32 v85, s42, v85
	v_add_u32_e32 v86, 0xffffff69, v85
	v_cmp_gt_i32_e32 vcc, s37, v86
	v_add_u32_e32 v86, 0xffffff6a, v85
	v_cmp_gt_i32_e64 s[0:1], s37, v86
	v_add_u32_e32 v86, 0xffffff6b, v85
	v_cmp_gt_i32_e64 s[4:5], s37, v86
	v_add_u32_e32 v86, 0xffffff6c, v85
	v_cmp_gt_i32_e64 s[6:7], s37, v86
	v_add_u32_e32 v86, 0xffffff6d, v85
	v_cmp_gt_i32_e64 s[8:9], s37, v86
	v_add_u32_e32 v86, 0xffffff6e, v85
	v_cmp_gt_i32_e64 s[10:11], s37, v86
	v_add_u32_e32 v86, 0xffffff6f, v85
	v_cmp_gt_i32_e64 s[12:13], s37, v86
	v_add_u32_e32 v86, 0xffffff70, v85
	v_cmp_gt_i32_e64 s[14:15], s37, v86
	v_add_u32_e32 v86, 0xffffff79, v85
	v_cmp_gt_i32_e64 s[16:17], s37, v86
	v_add_u32_e32 v86, 0xffffff7a, v85
	v_cmp_gt_i32_e64 s[18:19], s37, v86
	v_add_u32_e32 v86, 0xffffff7b, v85
	v_cmp_gt_i32_e64 s[20:21], s37, v86
	v_add_u32_e32 v86, 0xffffff7c, v85
	v_cmp_gt_i32_e64 s[22:23], s37, v86
	v_add_u32_e32 v86, 0xffffff7d, v85
	v_cmp_gt_i32_e64 s[24:25], s37, v86
	v_add_u32_e32 v86, 0xffffff7e, v85
	v_cmp_gt_i32_e64 s[26:27], s37, v86
	v_add_u32_e32 v86, 0xffffff7f, v85
	v_add_u32_e32 v85, 0xffffff80, v85
	v_cmp_gt_i32_e64 s[28:29], s37, v86
	v_cmp_gt_i32_e64 s[30:31], s37, v85
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
	v_cndmask_b32_e64 v67, v156, v67, s[0:1]
	v_cndmask_b32_e32 v66, v156, v66, vcc
	v_cndmask_b32_e64 v69, v156, v69, s[6:7]
	v_cndmask_b32_e64 v68, v156, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
	v_cndmask_b32_e64 v71, v156, v71, s[10:11]
	v_cndmask_b32_e64 v70, v156, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[122:123], v[68:69]
	v_max_f32_e32 v85, v66, v67
	v_cndmask_b32_e64 v73, v156, v73, s[14:15]
	v_cndmask_b32_e64 v72, v156, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[124:125], v[70:71]
	v_max3_f32 v85, v85, v68, v69
	v_cndmask_b32_e64 v75, v156, v75, s[18:19]
	v_cndmask_b32_e64 v74, v156, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[126:127], v[72:73]
	v_max3_f32 v85, v85, v70, v71
	v_cndmask_b32_e64 v77, v156, v77, s[22:23]
	v_cndmask_b32_e64 v76, v156, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[128:129], v[74:75]
	v_max3_f32 v85, v85, v72, v73
	v_cndmask_b32_e64 v79, v156, v79, s[26:27]
	v_cndmask_b32_e64 v78, v156, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[130:131], v[76:77]
	v_max3_f32 v85, v85, v74, v75
	v_cndmask_b32_e64 v81, v156, v81, s[30:31]
	v_cndmask_b32_e64 v80, v156, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[132:133], v[78:79]
	v_max3_f32 v85, v85, v76, v77
	v_pk_mul_f32 v[80:81], v[134:135], v[80:81]
	v_max3_f32 v85, v85, v78, v79
	v_max3_f32 v85, v85, v80, v81
	ds_bpermute_b32 v86, v153, v85
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v86, v86, v86
	v_max_f32_e32 v86, v85, v86
	;;#ASMSTART
	v_add_f32 v85, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v86, v85
	v_mov_b32_e32 v85, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_69
	;;#ASMSTART
	v_add_f32 v136, v86, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v85, v120, v136
	v_exp_f32_e32 v85, v85
	v_mov_b32_e32 v120, v136
	v_mov_b32_e32 v137, v136
	v_mov_b32_e32 v138, v136
	v_mov_b32_e32 v139, v136
	v_mov_b32_e32 v140, v136
	v_mov_b32_e32 v141, v136
	v_mov_b32_e32 v142, v136
	v_mov_b32_e32 v143, v136
	v_mov_b32_e32 v144, v136
	v_mov_b32_e32 v145, v136
	v_mov_b32_e32 v146, v136
	v_mov_b32_e32 v147, v136
	v_mov_b32_e32 v148, v136
	v_mov_b32_e32 v149, v136
	v_mov_b32_e32 v150, v136
	v_mov_b32_e32 v151, v136
.LBB0_69:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, 0, v66
	v_add_f32_e32 v86, v86, v67
	;;#ASMSTART
	v_exp_f32 v68, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v69, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v70, v70
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, v86, v68
	v_add_f32_e32 v86, v86, v69
	v_add_f32_e32 v86, v86, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, v86, v71
	v_add_f32_e32 v86, v86, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, v86, v73
	v_add_f32_e32 v86, v86, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, v86, v75
	v_add_f32_e32 v86, v86, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[150:151] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v86, v86, v77
	v_add_f32_e32 v86, v86, v78
	;;#ASMSTART
	v_exp_f32 v79, v79
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v80, v80
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v85
	v_add_f32_e32 v86, v86, v79
	v_add_f32_e32 v86, v86, v80
	;;#ASMSTART
	v_exp_f32 v81, v81
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v86, v86, v81
	;;#ASMSTART
	v_fma_f32 v157, v84, v85, v86
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_71
	;;#ASMSTART
	v_mul_f32 v34, v34, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v85
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v85
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[166:167], v[66:67], v[34:49]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[66:67], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[112:113], v[66:67], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[104:105], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[168:169], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[110:111], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[114:115], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[100:101], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[92:93], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[96:97], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[88:89], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[102:103], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[94:95], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[98:99], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[90:91], v[72:73], v[18:33]
	;;#ASMSTART
	v_mov_b32 v66, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[226:229], v68 offset:8192
	v_or_b32_e32 v68, 0x1010, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[210:213], v68
	v_or_b32_e32 v68, 0x1020, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[166:169], v68
	v_or_b32_e32 v68, 0x1030, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[206:209], v66
	v_or_b32_e32 v66, 0x1040, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[214:217], v66
	v_or_b32_e32 v66, 0x1050, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[218:221], v66
	v_or_b32_e32 v66, 0x1060, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[222:225], v66
	v_or_b32_e32 v66, 0x1070, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[170:173], v66
	s_cmp_gt_i32 s72, s43
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_le_i32 s72, s43
	s_cbranch_scc1 .LBB0_88
	ds_write_b128 v152, v[158:161]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[174:175], 0
	v_mov_b32_e32 v84, s44
	s_lshl_b32 s47, s92, 14
	s_add_i32 s47, s47, s98
	v_mfma_f32_32x32x8_bf16 v[66:81], v[228:229], v[176:177], v[66:81]
	buffer_load_dword v158, v84, s[68:71], 0 offen
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s46, v158
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[178:179], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[166:167], v[182:183], v[66:81]
	global_load_dwordx4 v[210:213], v[82:83], off offset:1536
	;;#ASMSTART
	v_mov_b32 v82, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v83, 3, v82
	v_lshlrev_b32_e32 v82, 5, v82
	v_and_b32_e32 v83, 0xf8, v83
	v_and_b32_e32 v82, 0x400, v82
	v_or3_b32 v82, v83, v82, s47
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, s[84:85]
	global_load_dwordx4 v[108:111], v[82:83], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[168:169], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[206:207], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[188:189], v[66:81]
	global_load_dwordx4 v[112:115], v[82:83], off offset:512
	global_load_dwordx4 v[100:103], v[82:83], off offset:1024
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[214:215], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[216:217], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[218:219], v[194:195], v[66:81]
	global_load_dwordx4 v[104:107], v[82:83], off offset:1536
	v_add_co_u32_e32 v82, vcc, s90, v82
	s_nop 1
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	global_load_dwordx4 v[92:95], v[82:83], off
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[200:201], v[66:81]
	global_load_dwordx4 v[96:99], v[82:83], off offset:512
	global_load_dwordx4 v[88:91], v[82:83], off offset:1024
	global_load_dwordx4 v[84:87], v[82:83], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[170:171], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[172:173], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v82, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v82, 2, v82
	v_and_b32_e32 v82, 8, v82
	v_add_u32_e32 v82, s42, v82
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
	v_cndmask_b32_e64 v67, v156, v67, s[0:1]
	v_cndmask_b32_e32 v66, v156, v66, vcc
	v_cndmask_b32_e64 v69, v156, v69, s[6:7]
	v_cndmask_b32_e64 v68, v156, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
	v_cndmask_b32_e64 v71, v156, v71, s[10:11]
	v_cndmask_b32_e64 v70, v156, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[122:123], v[68:69]
	v_max_f32_e32 v82, v66, v67
	v_cndmask_b32_e64 v73, v156, v73, s[14:15]
	v_cndmask_b32_e64 v72, v156, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[124:125], v[70:71]
	v_max3_f32 v82, v82, v68, v69
	v_cndmask_b32_e64 v75, v156, v75, s[18:19]
	v_cndmask_b32_e64 v74, v156, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[126:127], v[72:73]
	v_max3_f32 v82, v82, v70, v71
	v_cndmask_b32_e64 v77, v156, v77, s[22:23]
	v_cndmask_b32_e64 v76, v156, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[128:129], v[74:75]
	v_max3_f32 v82, v82, v72, v73
	v_cndmask_b32_e64 v79, v156, v79, s[26:27]
	v_cndmask_b32_e64 v78, v156, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[130:131], v[76:77]
	v_max3_f32 v82, v82, v74, v75
	v_cndmask_b32_e64 v81, v156, v81, s[30:31]
	v_cndmask_b32_e64 v80, v156, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[132:133], v[78:79]
	v_max3_f32 v82, v82, v76, v77
	v_pk_mul_f32 v[80:81], v[134:135], v[80:81]
	v_max3_f32 v82, v82, v78, v79
	v_max3_f32 v82, v82, v80, v81
	ds_bpermute_b32 v83, v153, v82
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v83, v83, v83
	v_max_f32_e32 v82, v82, v83
	;;#ASMSTART
	v_add_f32 v83, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v82, v83
	v_mov_b32_e32 v83, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_74
	;;#ASMSTART
	v_add_f32 v136, v82, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v82, v120, v136
	v_exp_f32_e32 v83, v82
	v_mov_b32_e32 v120, v136
	v_mov_b32_e32 v137, v136
	v_mov_b32_e32 v138, v136
	v_mov_b32_e32 v139, v136
	v_mov_b32_e32 v140, v136
	v_mov_b32_e32 v141, v136
	v_mov_b32_e32 v142, v136
	v_mov_b32_e32 v143, v136
	v_mov_b32_e32 v144, v136
	v_mov_b32_e32 v145, v136
	v_mov_b32_e32 v146, v136
	v_mov_b32_e32 v147, v136
	v_mov_b32_e32 v148, v136
	v_mov_b32_e32 v149, v136
	v_mov_b32_e32 v150, v136
	v_mov_b32_e32 v151, v136
.LBB0_74:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v66, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v67, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[72:73], v[72:73], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v68
	v_add_f32_e32 v82, v82, v69
	v_add_f32_e32 v82, v82, v70
	;;#ASMSTART
	v_exp_f32 v71, v71
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v72, v72
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v71
	v_add_f32_e32 v82, v82, v72
	;;#ASMSTART
	v_exp_f32 v73, v73
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v74, v74
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v73
	v_add_f32_e32 v82, v82, v74
	;;#ASMSTART
	v_exp_f32 v75, v75
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v76, v76
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v82, v82, v75
	v_add_f32_e32 v82, v82, v76
	;;#ASMSTART
	v_exp_f32 v77, v77
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v78, v78
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[150:151] neg_lo:[0,1] neg_hi:[0,1]
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
	v_fma_f32 v82, v157, v83, v82
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_76
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
.LBB0_76:
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
	v_perm_b32 v66, v67, v66, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v67, v69, v68, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v68, v71, v70, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v69, v73, v72, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v70, v75, v74, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v71, v77, v76, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v72, v79, v78, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v73, v81, v80, s91
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
	v_lshrrev_b32_e32 v68, 2, v66
	v_lshlrev_b32_e32 v67, 7, v66
	v_and_b32_e32 v68, 8, v68
	v_lshlrev_b32_e32 v66, 4, v66
	v_and_or_b32 v67, v67, s89, v68
	v_and_b32_e32 v66, 48, v66
	v_or_b32_e32 v68, v67, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[84:87], v68
	v_or_b32_e32 v68, 16, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[88:91], v68
	v_or_b32_e32 v68, 32, v67
	v_xor_b32_e32 v68, v68, v66
	v_lshlrev_b32_e32 v68, 1, v68
	ds_read_b128 v[92:95], v68
	v_or_b32_e32 v68, 48, v67
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[96:99], v66
	v_or_b32_e32 v66, 64, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[100:103], v66
	v_or_b32_e32 v66, 0x50, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[104:107], v66
	v_or_b32_e32 v66, 0x60, v67
	v_lshrrev_b32_e32 v68, 3, v66
	v_and_b32_e32 v68, 56, v68
	v_xor_b32_e32 v66, v68, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[108:111], v66
	v_or_b32_e32 v66, 0x70, v67
	v_lshrrev_b32_e32 v67, 3, v66
	v_and_b32_e32 v67, 56, v67
	v_xor_b32_e32 v66, v67, v66
	v_lshlrev_b32_e32 v66, 1, v66
	ds_read_b128 v[112:115], v66
	ds_write_b128 v152, v[162:165] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[84:85], v[174:175], 0
	s_lshl_b32 s40, s75, 13
	v_or_b32_e32 v84, s40, v116
	v_ashrrev_i32_e32 v85, 31, v84
	v_lshl_add_u64 v[84:85], v[84:85], 2, s[80:81]
	s_addk_i32 s47, 0x1000
	v_mfma_f32_32x32x8_bf16 v[66:81], v[86:87], v[176:177], v[66:81]
	global_load_dwordx4 v[206:209], v[84:85], off
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v84, 3, v83
	v_lshlrev_b32_e32 v83, 5, v83
	v_and_b32_e32 v84, 0xf8, v84
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[88:89], v[178:179], v[66:81]
	v_and_b32_e32 v83, 0x400, v83
	v_or3_b32 v84, v84, v83, s47
	v_ashrrev_i32_e32 v85, 31, v84
	v_lshl_add_u64 v[84:85], v[84:85], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[90:91], v[180:181], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[92:93], v[182:183], v[66:81]
	global_load_dwordx4 v[242:245], v[84:85], off
	global_load_dwordx4 v[234:237], v[84:85], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[94:95], v[184:185], v[66:81]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[96:97], v[186:187], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[188:189], v[66:81]
	global_load_dwordx4 v[238:241], v[84:85], off offset:1024
	global_load_dwordx4 v[230:233], v[84:85], off offset:1536
	v_add_co_u32_e32 v84, vcc, s90, v84
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[190:191], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[192:193], v[66:81]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[104:105], v[194:195], v[66:81]
	global_load_dwordx4 v[226:229], v[84:85], off
	global_load_dwordx4 v[218:221], v[84:85], off offset:512
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[196:197], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[198:199], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[110:111], v[200:201], v[66:81]
	global_load_dwordx4 v[222:225], v[84:85], off offset:1024
	global_load_dwordx4 v[214:217], v[84:85], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[112:113], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[114:115], v[204:205], v[66:81]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v83, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v83, 2, v83
	v_and_b32_e32 v83, 8, v83
	v_add_u32_e32 v83, s42, v83
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
	v_cndmask_b32_e64 v67, v156, v67, s[0:1]
	v_cndmask_b32_e32 v66, v156, v66, vcc
	v_cndmask_b32_e64 v69, v156, v69, s[6:7]
	v_cndmask_b32_e64 v68, v156, v68, s[4:5]
	v_pk_mul_f32 v[66:67], v[118:119], v[66:67]
	v_cndmask_b32_e64 v71, v156, v71, s[10:11]
	v_cndmask_b32_e64 v70, v156, v70, s[8:9]
	v_pk_mul_f32 v[68:69], v[122:123], v[68:69]
	v_max_f32_e32 v83, v66, v67
	v_cndmask_b32_e64 v73, v156, v73, s[14:15]
	v_cndmask_b32_e64 v72, v156, v72, s[12:13]
	v_pk_mul_f32 v[70:71], v[124:125], v[70:71]
	v_max3_f32 v83, v83, v68, v69
	v_cndmask_b32_e64 v75, v156, v75, s[18:19]
	v_cndmask_b32_e64 v74, v156, v74, s[16:17]
	v_pk_mul_f32 v[72:73], v[126:127], v[72:73]
	v_max3_f32 v83, v83, v70, v71
	v_cndmask_b32_e64 v77, v156, v77, s[22:23]
	v_cndmask_b32_e64 v76, v156, v76, s[20:21]
	v_pk_mul_f32 v[74:75], v[128:129], v[74:75]
	v_max3_f32 v83, v83, v72, v73
	v_cndmask_b32_e64 v79, v156, v79, s[26:27]
	v_cndmask_b32_e64 v78, v156, v78, s[24:25]
	v_pk_mul_f32 v[76:77], v[130:131], v[76:77]
	v_max3_f32 v83, v83, v74, v75
	v_cndmask_b32_e64 v81, v156, v81, s[30:31]
	v_cndmask_b32_e64 v80, v156, v80, s[28:29]
	v_pk_mul_f32 v[78:79], v[132:133], v[78:79]
	v_max3_f32 v83, v83, v76, v77
	v_pk_mul_f32 v[80:81], v[134:135], v[80:81]
	v_max3_f32 v83, v83, v78, v79
	v_max3_f32 v83, v83, v80, v81
	ds_bpermute_b32 v84, v153, v83
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v84, v84, v84
	v_max_f32_e32 v83, v83, v84
	;;#ASMSTART
	v_add_f32 v84, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v83, v84
	v_mov_b32_e32 v84, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_78
	;;#ASMSTART
	v_add_f32 v136, v83, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v83, v120, v136
	v_exp_f32_e32 v84, v83
	v_mov_b32_e32 v120, v136
	v_mov_b32_e32 v137, v136
	v_mov_b32_e32 v138, v136
	v_mov_b32_e32 v139, v136
	v_mov_b32_e32 v140, v136
	v_mov_b32_e32 v141, v136
	v_mov_b32_e32 v142, v136
	v_mov_b32_e32 v143, v136
	v_mov_b32_e32 v144, v136
	v_mov_b32_e32 v145, v136
	v_mov_b32_e32 v146, v136
	v_mov_b32_e32 v147, v136
	v_mov_b32_e32 v148, v136
	v_mov_b32_e32 v149, v136
	v_mov_b32_e32 v150, v136
	v_mov_b32_e32 v151, v136
.LBB0_78:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[66:67], v[66:67], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v158, v66
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v159, v67
	;;#ASMEND
	v_pk_add_f32 v[70:71], v[70:71], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, 0, v158
	v_add_f32_e32 v66, v66, v159
	;;#ASMSTART
	v_exp_f32 v160, v68
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v161, v69
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v162, v70
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v163, v71
	;;#ASMEND
	v_pk_add_f32 v[72:73], v[72:73], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, v66, v160
	v_add_f32_e32 v66, v66, v161
	v_add_f32_e32 v66, v66, v162
	v_add_f32_e32 v66, v66, v163
	;;#ASMSTART
	v_exp_f32 v164, v72
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v165, v73
	;;#ASMEND
	v_pk_add_f32 v[74:75], v[74:75], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, v66, v164
	v_add_f32_e32 v66, v66, v165
	;;#ASMSTART
	v_exp_f32 v166, v74
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v167, v75
	;;#ASMEND
	v_pk_add_f32 v[76:77], v[76:77], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, v66, v166
	v_add_f32_e32 v66, v66, v167
	;;#ASMSTART
	v_exp_f32 v168, v76
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v169, v77
	;;#ASMEND
	v_pk_add_f32 v[78:79], v[78:79], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, v66, v168
	v_add_f32_e32 v66, v66, v169
	;;#ASMSTART
	v_exp_f32 v170, v78
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v171, v79
	;;#ASMEND
	v_pk_add_f32 v[80:81], v[80:81], v[150:151] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v66, v66, v170
	v_add_f32_e32 v66, v66, v171
	;;#ASMSTART
	v_exp_f32 v172, v80
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v173, v81
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v84
	v_add_f32_e32 v66, v66, v172
	v_add_f32_e32 v66, v66, v173
	;;#ASMSTART
	v_fma_f32 v157, v82, v84, v66
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
	s_cbranch_execz .LBB0_80
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
.LBB0_80:
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
	v_add_u32_e32 v9, 0x8000, v173
	v_add_u32_e32 v10, 0x8000, v172
	v_add_u32_e32 v8, 0x8000, v171
	v_add_u32_e32 v11, 0x8000, v170
	v_add_u32_e32 v7, 0x8000, v169
	v_add_u32_e32 v12, 0x8000, v168
	v_add_u32_e32 v6, 0x8000, v167
	v_add_u32_e32 v13, 0x8000, v166
	v_add_u32_e32 v5, 0x8000, v165
	v_add_u32_e32 v14, 0x8000, v164
	v_add_u32_e32 v4, 0x8000, v163
	v_add_u32_e32 v15, 0x8000, v162
	v_add_u32_e32 v3, 0x8000, v161
	v_add_u32_e32 v16, 0x8000, v160
	v_add_u32_e32 v2, 0x8000, v159
	v_add_u32_e32 v17, 0x8000, v158
	;;#ASMSTART
	v_perm_b32 v2, v2, v17, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v3, v16, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v4, v15, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v5, v14, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v13, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v7, v12, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v8, v11, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v9, v10, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[52:67], v[242:243], v[2:3], v[52:67]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[68:83], v[234:235], v[2:3], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[238:239], v[2:3], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[230:231], v[2:3], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[244:245], v[4:5], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[236:237], v[4:5], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[240:241], v[4:5], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[232:233], v[4:5], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[226:227], v[6:7], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[218:219], v[6:7], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[222:223], v[6:7], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[214:215], v[6:7], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[228:229], v[8:9], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[220:221], v[8:9], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[224:225], v[8:9], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[216:217], v[8:9], v[100:115]
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v4, 2, v2
	v_lshlrev_b32_e32 v3, 7, v2
	v_and_b32_e32 v4, 8, v4
	v_lshlrev_b32_e32 v2, 4, v2
	v_and_or_b32 v3, v3, s89, v4
	v_and_b32_e32 v2, 48, v2
	v_or_b32_e32 v4, v3, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[18:21], v4 offset:8192
	v_or_b32_e32 v4, 0x1010, v3
	v_xor_b32_e32 v4, v4, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[22:25], v4
	v_or_b32_e32 v4, 0x1020, v3
	v_xor_b32_e32 v4, v4, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[26:29], v4
	v_or_b32_e32 v4, 0x1030, v3
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[30:33], v2
	v_or_b32_e32 v2, 0x1040, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[34:37], v2
	v_or_b32_e32 v2, 0x1050, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[166:169], v2
	v_or_b32_e32 v2, 0x1060, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[170:173], v2
	v_or_b32_e32 v2, 0x1070, v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[214:217], v2
	ds_write_b128 v152, v[210:213]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[18:19], v[174:175], 0
	s_ashr_i32 s41, s40, 31
	v_lshl_add_u64 v[18:19], s[40:41], 0, v[116:117]
	v_lshl_add_u64 v[18:19], v[18:19], 2, s[80:81]
	s_add_i32 s0, s47, 0x1000
	v_mfma_f32_32x32x8_bf16 v[2:17], v[20:21], v[176:177], v[2:17]
	global_load_dwordx4 v[158:161], v[18:19], off offset:512
	;;#ASMSTART
	v_mov_b32 v20, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v21, 3, v20
	v_lshlrev_b32_e32 v20, 5, v20
	v_and_b32_e32 v21, 0xf8, v21
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[22:23], v[178:179], v[2:17]
	v_and_b32_e32 v20, 0x400, v20
	v_or3_b32 v20, v21, v20, s0
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[24:25], v[180:181], v[2:17]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[26:27], v[182:183], v[2:17]
	global_load_dwordx4 v[162:165], v[20:21], off
	global_load_dwordx4 v[42:45], v[20:21], off offset:512
	v_mfma_f32_32x32x8_bf16 v[2:17], v[28:29], v[184:185], v[2:17]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[30:31], v[186:187], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[32:33], v[188:189], v[2:17]
	global_load_dwordx4 v[46:49], v[20:21], off offset:1024
	global_load_dwordx4 v[38:41], v[20:21], off offset:1536
	v_add_co_u32_e32 v20, vcc, s90, v20
	s_nop 1
	v_addc_co_u32_e32 v21, vcc, 0, v21, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[34:35], v[190:191], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[36:37], v[192:193], v[2:17]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[166:167], v[194:195], v[2:17]
	global_load_dwordx4 v[34:37], v[20:21], off
	global_load_dwordx4 v[26:29], v[20:21], off offset:512
	v_mfma_f32_32x32x8_bf16 v[2:17], v[168:169], v[196:197], v[2:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[170:171], v[198:199], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[172:173], v[200:201], v[2:17]
	global_load_dwordx4 v[30:33], v[20:21], off offset:1024
	global_load_dwordx4 v[22:25], v[20:21], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[214:215], v[202:203], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[216:217], v[204:205], v[2:17]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v20, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v20, 2, v20
	v_and_b32_e32 v20, 8, v20
	v_add_u32_e32 v20, s42, v20
	v_subrev_u32_e32 v21, 55, v20
	v_cmp_gt_i32_e32 vcc, s37, v21
	v_subrev_u32_e32 v21, 54, v20
	v_cmp_gt_i32_e64 s[0:1], s37, v21
	v_subrev_u32_e32 v21, 53, v20
	v_cmp_gt_i32_e64 s[4:5], s37, v21
	v_subrev_u32_e32 v21, 52, v20
	v_cmp_gt_i32_e64 s[6:7], s37, v21
	v_subrev_u32_e32 v21, 51, v20
	v_cmp_gt_i32_e64 s[8:9], s37, v21
	v_subrev_u32_e32 v21, 50, v20
	v_cmp_gt_i32_e64 s[10:11], s37, v21
	v_subrev_u32_e32 v21, 49, v20
	v_cmp_gt_i32_e64 s[12:13], s37, v21
	v_subrev_u32_e32 v21, 48, v20
	v_cmp_gt_i32_e64 s[14:15], s37, v21
	v_subrev_u32_e32 v21, 39, v20
	v_cmp_gt_i32_e64 s[16:17], s37, v21
	v_subrev_u32_e32 v21, 38, v20
	v_cmp_gt_i32_e64 s[18:19], s37, v21
	v_subrev_u32_e32 v21, 37, v20
	v_cmp_gt_i32_e64 s[20:21], s37, v21
	v_subrev_u32_e32 v21, 36, v20
	v_cmp_gt_i32_e64 s[22:23], s37, v21
	v_subrev_u32_e32 v21, 35, v20
	v_cmp_gt_i32_e64 s[24:25], s37, v21
	v_subrev_u32_e32 v21, 34, v20
	v_cmp_gt_i32_e64 s[26:27], s37, v21
	v_subrev_u32_e32 v21, 33, v20
	v_subrev_u32_e32 v20, 32, v20
	v_cmp_gt_i32_e64 s[28:29], s37, v21
	v_cmp_gt_i32_e64 s[30:31], s37, v20
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
	v_cndmask_b32_e64 v3, v156, v3, s[0:1]
	v_cndmask_b32_e32 v2, v156, v2, vcc
	v_cndmask_b32_e64 v5, v156, v5, s[6:7]
	v_cndmask_b32_e64 v4, v156, v4, s[4:5]
	v_pk_mul_f32 v[2:3], v[118:119], v[2:3]
	v_cndmask_b32_e64 v7, v156, v7, s[10:11]
	v_cndmask_b32_e64 v6, v156, v6, s[8:9]
	v_pk_mul_f32 v[4:5], v[122:123], v[4:5]
	v_max_f32_e32 v20, v2, v3
	v_cndmask_b32_e64 v9, v156, v9, s[14:15]
	v_cndmask_b32_e64 v8, v156, v8, s[12:13]
	v_pk_mul_f32 v[6:7], v[124:125], v[6:7]
	v_max3_f32 v20, v20, v4, v5
	v_cndmask_b32_e64 v11, v156, v11, s[18:19]
	v_cndmask_b32_e64 v10, v156, v10, s[16:17]
	v_pk_mul_f32 v[8:9], v[126:127], v[8:9]
	v_max3_f32 v20, v20, v6, v7
	v_cndmask_b32_e64 v13, v156, v13, s[22:23]
	v_cndmask_b32_e64 v12, v156, v12, s[20:21]
	v_pk_mul_f32 v[10:11], v[128:129], v[10:11]
	v_max3_f32 v20, v20, v8, v9
	v_cndmask_b32_e64 v15, v156, v15, s[26:27]
	v_cndmask_b32_e64 v14, v156, v14, s[24:25]
	v_pk_mul_f32 v[12:13], v[130:131], v[12:13]
	v_max3_f32 v20, v20, v10, v11
	v_cndmask_b32_e64 v17, v156, v17, s[30:31]
	v_cndmask_b32_e64 v16, v156, v16, s[28:29]
	v_pk_mul_f32 v[14:15], v[132:133], v[14:15]
	v_max3_f32 v20, v20, v12, v13
	v_pk_mul_f32 v[16:17], v[134:135], v[16:17]
	v_max3_f32 v20, v20, v14, v15
	v_max3_f32 v20, v20, v16, v17
	ds_bpermute_b32 v21, v153, v20
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v21, v21, v21
	v_max_f32_e32 v20, v20, v21
	;;#ASMSTART
	v_add_f32 v21, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v20, v21
	v_mov_b32_e32 v21, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_82
	;;#ASMSTART
	v_add_f32 v136, v20, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v20, v120, v136
	v_exp_f32_e32 v21, v20
	v_mov_b32_e32 v120, v136
	v_mov_b32_e32 v137, v136
	v_mov_b32_e32 v138, v136
	v_mov_b32_e32 v139, v136
	v_mov_b32_e32 v140, v136
	v_mov_b32_e32 v141, v136
	v_mov_b32_e32 v142, v136
	v_mov_b32_e32 v143, v136
	v_mov_b32_e32 v144, v136
	v_mov_b32_e32 v145, v136
	v_mov_b32_e32 v146, v136
	v_mov_b32_e32 v147, v136
	v_mov_b32_e32 v148, v136
	v_mov_b32_e32 v149, v136
	v_mov_b32_e32 v150, v136
	v_mov_b32_e32 v151, v136
.LBB0_82:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[2:3], v[2:3], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v2, v2
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v3, v3
	;;#ASMEND
	v_pk_add_f32 v[6:7], v[6:7], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[8:9], v[8:9], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, v20, v4
	v_add_f32_e32 v20, v20, v5
	v_add_f32_e32 v20, v20, v6
	;;#ASMSTART
	v_exp_f32 v7, v7
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v8, v8
	;;#ASMEND
	v_pk_add_f32 v[10:11], v[10:11], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, v20, v7
	v_add_f32_e32 v20, v20, v8
	;;#ASMSTART
	v_exp_f32 v9, v9
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v10, v10
	;;#ASMEND
	v_pk_add_f32 v[12:13], v[12:13], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, v20, v9
	v_add_f32_e32 v20, v20, v10
	;;#ASMSTART
	v_exp_f32 v11, v11
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v12, v12
	;;#ASMEND
	v_pk_add_f32 v[14:15], v[14:15], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, v20, v11
	v_add_f32_e32 v20, v20, v12
	;;#ASMSTART
	v_exp_f32 v13, v13
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v14, v14
	;;#ASMEND
	v_pk_add_f32 v[16:17], v[16:17], v[150:151] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v20, v20, v13
	v_add_f32_e32 v20, v20, v14
	;;#ASMSTART
	v_exp_f32 v15, v15
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v16, v16
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v17, v17
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v21
	v_add_f32_e32 v20, v20, v15
	v_add_f32_e32 v20, v20, v16
	v_add_f32_e32 v20, v20, v17
	;;#ASMSTART
	v_fma_f32 v20, v157, v21, v20
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_84
	;;#ASMSTART
	v_mul_f32 v52, v52, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v98, v98, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v99, v99, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v100, v100, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v101, v101, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v102, v102, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v103, v103, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v104, v104, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v105, v105, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v106, v106, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v107, v107, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v108, v108, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v109, v109, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v110, v110, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v111, v111, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v112, v112, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v113, v113, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v114, v114, v21
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v115, v115, v21
	;;#ASMEND
.LBB0_84:
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
	v_perm_b32 v2, v3, v2, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v5, v4, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v7, v6, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v9, v8, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v11, v10, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v13, v12, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v15, v14, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v17, v16, s91
	;;#ASMEND
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[52:67], v[162:163], v[2:3], v[52:67]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[68:83], v[42:43], v[2:3], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[46:47], v[2:3], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[38:39], v[2:3], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[164:165], v[4:5], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[44:45], v[4:5], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[48:49], v[4:5], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[40:41], v[4:5], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[34:35], v[6:7], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[26:27], v[6:7], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[30:31], v[6:7], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[22:23], v[6:7], v[100:115]
	v_mfma_f32_32x32x8_bf16 v[52:67], v[36:37], v[8:9], v[52:67]
	v_mfma_f32_32x32x8_bf16 v[68:83], v[28:29], v[8:9], v[68:83]
	v_mfma_f32_32x32x8_bf16 v[84:99], v[32:33], v[8:9], v[84:99]
	v_mfma_f32_32x32x8_bf16 v[100:115], v[24:25], v[8:9], v[100:115]
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v4, 2, v2
	v_lshlrev_b32_e32 v3, 7, v2
	v_and_b32_e32 v4, 8, v4
	v_lshlrev_b32_e32 v2, 4, v2
	v_and_or_b32 v3, v3, s89, v4
	v_and_b32_e32 v2, 48, v2
	v_or_b32_e32 v4, v3, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[22:25], v4
	v_or_b32_e32 v4, 16, v3
	v_xor_b32_e32 v4, v4, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[26:29], v4
	v_or_b32_e32 v4, 32, v3
	v_xor_b32_e32 v4, v4, v2
	v_lshlrev_b32_e32 v4, 1, v4
	ds_read_b128 v[30:33], v4
	v_or_b32_e32 v4, 48, v3
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[34:37], v2
	v_or_b32_e32 v2, 64, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[170:173], v2
	v_or_b32_e32 v2, 0x50, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[210:213], v2
	v_or_b32_e32 v2, 0x60, v3
	v_lshrrev_b32_e32 v4, 3, v2
	v_and_b32_e32 v4, 56, v4
	v_xor_b32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[214:217], v2
	v_or_b32_e32 v2, 0x70, v3
	v_lshrrev_b32_e32 v3, 3, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[218:221], v2
	ds_write_b128 v152, v[206:209] offset:8192
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[22:23], v[174:175], 0
	s_addk_i32 s47, 0x2000
	v_mfma_f32_32x32x8_bf16 v[2:17], v[24:25], v[176:177], v[2:17]
	global_load_dwordx4 v[162:165], v[18:19], off offset:1024
	;;#ASMSTART
	v_mov_b32 v18, v0
	;;#ASMEND
	s_nop 0
	v_lshlrev_b32_e32 v19, 3, v18
	v_lshlrev_b32_e32 v18, 5, v18
	v_and_b32_e32 v19, 0xf8, v19
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[26:27], v[178:179], v[2:17]
	v_and_b32_e32 v18, 0x400, v18
	v_or3_b32 v18, v19, v18, s47
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[84:85]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[28:29], v[180:181], v[2:17]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[30:31], v[182:183], v[2:17]
	global_load_dwordx4 v[166:169], v[18:19], off
	global_load_dwordx4 v[42:45], v[18:19], off offset:512
	v_mfma_f32_32x32x8_bf16 v[2:17], v[32:33], v[184:185], v[2:17]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[34:35], v[186:187], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[36:37], v[188:189], v[2:17]
	global_load_dwordx4 v[46:49], v[18:19], off offset:1024
	global_load_dwordx4 v[38:41], v[18:19], off offset:1536
	v_add_co_u32_e32 v18, vcc, s90, v18
	s_nop 1
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[170:171], v[190:191], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[172:173], v[192:193], v[2:17]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[210:211], v[194:195], v[2:17]
	global_load_dwordx4 v[34:37], v[18:19], off
	global_load_dwordx4 v[26:29], v[18:19], off offset:512
	v_mfma_f32_32x32x8_bf16 v[2:17], v[212:213], v[196:197], v[2:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[214:215], v[198:199], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[216:217], v[200:201], v[2:17]
	global_load_dwordx4 v[30:33], v[18:19], off offset:1024
	global_load_dwordx4 v[22:25], v[18:19], off offset:1536
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[218:219], v[202:203], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[220:221], v[204:205], v[2:17]
	s_waitcnt vmcnt(9) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v18, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v18, 2, v18
	v_and_b32_e32 v18, 8, v18
	v_add_u32_e32 v18, s42, v18
	v_subrev_u32_e32 v19, 23, v18
	v_cmp_gt_i32_e32 vcc, s37, v19
	v_subrev_u32_e32 v19, 22, v18
	v_cmp_gt_i32_e64 s[0:1], s37, v19
	v_subrev_u32_e32 v19, 21, v18
	v_cmp_gt_i32_e64 s[4:5], s37, v19
	v_subrev_u32_e32 v19, 20, v18
	v_cmp_gt_i32_e64 s[6:7], s37, v19
	v_subrev_u32_e32 v19, 19, v18
	v_cmp_gt_i32_e64 s[8:9], s37, v19
	v_subrev_u32_e32 v19, 18, v18
	v_cmp_gt_i32_e64 s[10:11], s37, v19
	v_subrev_u32_e32 v19, 17, v18
	v_cmp_gt_i32_e64 s[12:13], s37, v19
	v_add_u32_e32 v19, -16, v18
	v_cmp_gt_i32_e64 s[14:15], s37, v19
	v_add_u32_e32 v19, -7, v18
	v_cmp_gt_i32_e64 s[16:17], s37, v19
	v_add_u32_e32 v19, -6, v18
	v_cmp_gt_i32_e64 s[18:19], s37, v19
	v_add_u32_e32 v19, -5, v18
	v_cmp_gt_i32_e64 s[20:21], s37, v19
	v_add_u32_e32 v19, -4, v18
	v_cmp_gt_i32_e64 s[22:23], s37, v19
	v_add_u32_e32 v19, -3, v18
	v_cmp_gt_i32_e64 s[24:25], s37, v19
	v_add_u32_e32 v19, -2, v18
	v_cmp_gt_i32_e64 s[26:27], s37, v19
	v_add_u32_e32 v19, -1, v18
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
	v_cndmask_b32_e64 v3, v156, v3, s[0:1]
	v_cndmask_b32_e32 v2, v156, v2, vcc
	v_cndmask_b32_e64 v5, v156, v5, s[6:7]
	v_cndmask_b32_e64 v4, v156, v4, s[4:5]
	v_pk_mul_f32 v[2:3], v[118:119], v[2:3]
	v_cndmask_b32_e64 v7, v156, v7, s[10:11]
	v_cndmask_b32_e64 v6, v156, v6, s[8:9]
	v_pk_mul_f32 v[4:5], v[122:123], v[4:5]
	v_max_f32_e32 v18, v2, v3
	v_cndmask_b32_e64 v9, v156, v9, s[14:15]
	v_cndmask_b32_e64 v8, v156, v8, s[12:13]
	v_pk_mul_f32 v[6:7], v[124:125], v[6:7]
	v_max3_f32 v18, v18, v4, v5
	v_cndmask_b32_e64 v11, v156, v11, s[18:19]
	v_cndmask_b32_e64 v10, v156, v10, s[16:17]
	v_pk_mul_f32 v[8:9], v[126:127], v[8:9]
	v_max3_f32 v18, v18, v6, v7
	v_cndmask_b32_e64 v13, v156, v13, s[22:23]
	v_cndmask_b32_e64 v12, v156, v12, s[20:21]
	v_pk_mul_f32 v[10:11], v[128:129], v[10:11]
	v_max3_f32 v18, v18, v8, v9
	v_cndmask_b32_e64 v15, v156, v15, s[26:27]
	v_cndmask_b32_e64 v14, v156, v14, s[24:25]
	v_pk_mul_f32 v[12:13], v[130:131], v[12:13]
	v_max3_f32 v18, v18, v10, v11
	v_cndmask_b32_e64 v17, v156, v17, s[30:31]
	v_cndmask_b32_e64 v16, v156, v16, s[28:29]
	v_pk_mul_f32 v[14:15], v[132:133], v[14:15]
	v_max3_f32 v18, v18, v12, v13
	v_pk_mul_f32 v[16:17], v[134:135], v[16:17]
	v_max3_f32 v18, v18, v14, v15
	v_max3_f32 v18, v18, v16, v17
	ds_bpermute_b32 v19, v153, v18
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v19, v19, v19
	v_max_f32_e32 v19, v18, v19
	;;#ASMSTART
	v_add_f32 v18, v120, v154
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v19, v18
	v_mov_b32_e32 v18, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_86
	;;#ASMSTART
	v_add_f32 v136, v19, v155
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v18, v120, v136
	v_exp_f32_e32 v18, v18
	v_mov_b32_e32 v120, v136
	v_mov_b32_e32 v137, v136
	v_mov_b32_e32 v138, v136
	v_mov_b32_e32 v139, v136
	v_mov_b32_e32 v140, v136
	v_mov_b32_e32 v141, v136
	v_mov_b32_e32 v142, v136
	v_mov_b32_e32 v143, v136
	v_mov_b32_e32 v144, v136
	v_mov_b32_e32 v145, v136
	v_mov_b32_e32 v146, v136
	v_mov_b32_e32 v147, v136
	v_mov_b32_e32 v148, v136
	v_mov_b32_e32 v149, v136
	v_mov_b32_e32 v150, v136
	v_mov_b32_e32 v151, v136
.LBB0_86:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[2:3], v[2:3], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v2, v2
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v3, v3
	;;#ASMEND
	v_pk_add_f32 v[6:7], v[6:7], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v19, 0, v2
	v_add_f32_e32 v19, v19, v3
	;;#ASMSTART
	v_exp_f32 v4, v4
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v5, v5
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v6, v6
	;;#ASMEND
	v_pk_add_f32 v[8:9], v[8:9], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v19, v19, v4
	v_add_f32_e32 v19, v19, v5
	v_add_f32_e32 v19, v19, v6
	;;#ASMSTART
	v_exp_f32 v7, v7
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v8, v8
	;;#ASMEND
	v_pk_add_f32 v[10:11], v[10:11], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v19, v19, v7
	v_add_f32_e32 v19, v19, v8
	;;#ASMSTART
	v_exp_f32 v9, v9
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v10, v10
	;;#ASMEND
	v_pk_add_f32 v[12:13], v[12:13], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v19, v19, v9
	v_add_f32_e32 v19, v19, v10
	;;#ASMSTART
	v_exp_f32 v11, v11
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v12, v12
	;;#ASMEND
	v_pk_add_f32 v[14:15], v[14:15], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v19, v19, v11
	v_add_f32_e32 v19, v19, v12
	;;#ASMSTART
	v_exp_f32 v13, v13
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v14, v14
	;;#ASMEND
	v_pk_add_f32 v[16:17], v[16:17], v[150:151] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v19, v19, v13
	v_add_f32_e32 v19, v19, v14
	;;#ASMSTART
	v_exp_f32 v15, v15
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v16, v16
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v18
	v_add_f32_e32 v19, v19, v15
	v_add_f32_e32 v19, v19, v16
	;;#ASMSTART
	v_exp_f32 v17, v17
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v19, v19, v17
	;;#ASMSTART
	v_fma_f32 v157, v20, v18, v19
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_53
	;;#ASMSTART
	v_mul_f32 v52, v52, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v98, v98, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v99, v99, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v100, v100, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v101, v101, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v102, v102, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v103, v103, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v104, v104, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v105, v105, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v106, v106, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v107, v107, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v108, v108, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v109, v109, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v110, v110, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v111, v111, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v112, v112, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v113, v113, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v114, v114, v18
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v115, v115, v18
	;;#ASMEND
	s_branch .LBB0_53
.LBB0_88:
	s_mov_b32 s46, s45
	s_branch .LBB0_54
.LBB0_89:
	ds_bpermute_b32 v66, v153, v157
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v66, v157, v66
	;;#ASMEND
	s_nop 0
	v_div_scale_f32 v67, s[0:1], v66, v66, s87
	v_rcp_f32_e32 v68, v67
	v_div_scale_f32 v69, vcc, s87, v66, s87
	v_fma_f32 v70, -v67, v68, 1.0
	v_fmac_f32_e32 v68, v70, v68
	v_mul_f32_e32 v70, v69, v68
	v_fma_f32 v71, -v67, v70, v69
	v_fmac_f32_e32 v70, v71, v68
	v_fma_f32 v67, -v67, v70, v69
	v_div_fmas_f32 v67, v67, v68, v70
	v_div_fixup_f32 v66, v67, v66, s87
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
	v_perm_b32 v28, v28, v34, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v29, v36, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v32, v38, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v33, v40, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v30, v42, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v31, v44, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v26, v46, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v27, v48, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v24, v50, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v25, v52, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v22, v54, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v21, v23, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v19, v20, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v17, v18, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v15, v16, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v12, v14, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v3, v2, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v5, v4, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v7, v6, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v9, v8, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v11, v10, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v13, v86, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v84, v85, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v82, v83, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v80, v81, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v78, v79, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v76, v77, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v74, v75, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v72, v73, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v70, v71, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v68, v69, s91
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v66, v67, s91
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v34, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v34, 0x100, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_91
	s_barrier
.LBB0_91:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v37, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v34, 3, v37
	v_bfe_u32 v36, v37, 6, 3
	v_lshlrev_b32_e32 v35, 7, v37
	v_and_b32_e32 v34, 4, v34
	v_and_or_b32 v34, v35, s89, v34
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_93
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
.LBB0_93:
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
	s_cbranch_execz .LBB0_95
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
.LBB0_95:
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
	s_cbranch_execz .LBB0_97
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
.LBB0_97:
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
	s_cbranch_execz .LBB0_99
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
.LBB0_99:
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
	s_cbranch_execz .LBB0_101
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
.LBB0_101:
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
	s_cbranch_execz .LBB0_103
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
.LBB0_103:
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
	s_add_i32 s0, s4, 0x50000
	v_add_lshl_u32 v39, s0, v38, 1
	s_mov_b32 s75, s71
	v_cmp_eq_u32_e32 vcc, 7, v36
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v39, s[72:75], 0 offen
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_107
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
.LBB0_107:
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
	s_cbranch_execz .LBB0_111
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_110
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v3, s8
	global_atomic_add v3, v117, v3, s[94:95] sc0
.LBB0_110:
	s_or_b64 exec, exec, s[6:7]
	s_lshl_b64 s[6:7], s[0:1], 2
	s_add_u32 s6, s94, s6
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v3
	s_addc_u32 s7, s95, s7
	s_nop 0
	v_add_u32_e32 v2, s8, v2
	global_store_dword v117, v2, s[6:7]
	s_waitcnt vmcnt(0)
.LBB0_111:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s94, s0
	s_addc_u32 s1, s95, s1
	s_barrier
	global_load_dword v2, v117, s[0:1]
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s4, v2
	s_add_i32 s0, s4, s33
	s_sub_i32 s33, s0, s2
	s_cmp_ge_i32 s78, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s86
	s_cselect_b64 s[6:7], -1, 0
	s_or_b64 s[0:1], s[0:1], s[6:7]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_114
	s_branch .LBB0_12
.LBB0_112:
	s_mov_b32 s78, s5
.LBB0_113:
	s_sub_i32 s33, s33, s86
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s88, 0, s2
	s_cmp_ge_i32 s78, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s86, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_114:
	s_add_i32 s2, s88, 1
	s_cmp_gt_i32 s2, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s2, 16
	s_cbranch_scc1 .LBB0_117
	s_add_i32 s5, s78, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s86
	s_cbranch_scc1 .LBB0_112
	s_ashr_i32 s79, s78, 31
	s_lshl_b64 s[6:7], s[78:79], 2
	s_add_u32 s6, s34, s6
	s_addc_u32 s7, s35, s7
	global_load_dwordx2 v[2:3], v117, s[6:7] offset:4
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
	s_branch .LBB0_112
.LBB0_117:
	s_mov_b32 s6, s86
	s_branch .LBB0_113
.LBB0_118:
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
		.amdhsa_next_free_vgpr 246
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

	.set .Lattn_kernel_0.num_vgpr, 246
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
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     246
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
