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
	s_cmp_lt_i32 s33, s48
	s_cselect_b64 s[16:17], -1, 0
	s_or_b64 s[14:15], s[14:15], s[16:17]
	s_and_b64 vcc, exec, s[14:15]
	s_mov_b32 s16, s48
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s17, s22, 1
	s_cmp_gt_i32 s17, 15
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_lt_i32 s17, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s18, s12, 1
	s_cmp_ge_i32 s18, s3
	s_mov_b32 s48, s16
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
	s_subb_u32 s48, s22, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s48, s16
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s48, s16
	s_mov_b32 s22, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s49, s[4:5], 0x0
	s_load_dword s50, s[6:7], 0x0
	s_cmp_ge_i32 s12, s3
	s_cbranch_scc1 .LBB0_138
	s_load_dwordx2 s[14:15], s[0:1], 0x0
	s_load_dwordx2 s[16:17], s[0:1], 0x10
	s_load_dwordx2 s[18:19], s[0:1], 0x20
	s_load_dwordx2 s[20:21], s[0:1], 0x50
	s_load_dwordx2 s[24:25], s[0:1], 0x60
	s_load_dwordx2 s[26:27], s[0:1], 0x70
	s_load_dwordx2 s[28:29], s[0:1], 0xa0
	s_load_dwordx2 s[30:31], s[0:1], 0xb0
	s_load_dwordx2 s[34:35], s[0:1], 0xc0
	v_lshrrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v1, 31, v0
	s_movk_i32 s51, 0xe0
	v_and_or_b32 v3, v2, s51, v1
	v_lshrrev_b32_e32 v4, 2, v0
	v_lshlrev_b32_e32 v6, 10, v0
	v_lshlrev_b32_e32 v119, 6, v3
	v_lshlrev_b32_e32 v3, 11, v1
	v_and_b32_e32 v5, 8, v4
	v_and_b32_e32 v6, 0x70000, v6
	v_or3_b32 v3, v3, v5, v6
	v_lshrrev_b32_e32 v6, 3, v0
	v_lshlrev_b32_e32 v166, 1, v3
	v_lshlrev_b32_e32 v3, 9, v0
	v_and_b32_e32 v5, 12, v4
	v_and_b32_e32 v2, 32, v2
	v_and_b32_e32 v6, 16, v6
	v_and_b32_e32 v3, 0x1e00, v3
	v_or3_b32 v2, v5, v2, v6
	v_and_b32_e32 v4, 64, v4
	v_or3_b32 v114, v2, v4, v3
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v167, v2, v3
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_mov_b32_e32 v115, 0
	v_xor_b32_e32 v168, 0x80, v2
	s_mov_b32 s7, 0x27000
	s_movk_i32 s52, 0xf80
	v_mov_b32_e32 v169, 0x40e00000
	v_mov_b32_e32 v170, 1.0
	s_mov_b32 s53, 0x7060302
	s_movk_i32 s54, 0x1000
	s_movk_i32 s55, 0x2000
	s_mov_b32 s56, 0x800000
	s_mov_b32 s57, 0xaaab
	s_movk_i32 s58, 0xff
	s_mov_b32 s59, s2
	v_mov_b32_e32 v171, 0xff800000
	v_mov_b32_e32 v172, 0x42000000
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s48, s6
.LBB0_12:
	s_cmp_ge_i32 s12, s3
	s_cbranch_scc1 .LBB0_138
.LBB0_13:
	s_ashr_i32 s13, s12, 31
	s_lshl_b32 s46, s33, 8
	s_lshl_b64 s[0:1], s[12:13], 2
	s_add_u32 s4, s8, s0
	s_addc_u32 s5, s9, s1
	global_load_dwordx2 v[2:3], v115, s[4:5]
	v_lshl_add_u32 v7, s22, 8, v166
	s_mov_b32 s43, s7
	v_lshl_add_u32 v6, s22, 2, v119
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s23, v2
	s_add_i32 s36, s23, s46
	v_readfirstlane_b32 s37, v3
	s_add_i32 s4, s36, 0x100
	s_min_i32 s4, s4, s37
	s_sub_i32 s13, s4, s36
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s20, s0
	s_addc_u32 s5, s21, s1
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s40, s36, 11
	s_ashr_i32 s41, s40, 31
	s_lshl_b32 s38, s36, 4
	global_load_dwordx2 v[4:5], v115, s[4:5]
	global_load_dword v3, v115, s[0:1]
	s_lshl_b64 s[0:1], s[40:41], 1
	s_add_u32 s4, s14, s0
	s_addc_u32 s0, s15, s1
	s_ashr_i32 s39, s38, 31
	s_lshl_b32 s6, s13, 12
	s_and_b32 s5, s0, 0xffff
	s_lshl_b64 s[0:1], s[38:39], 2
	s_add_u32 s40, s26, s0
	buffer_load_dwordx4 v[176:179], v7, s[4:7], 0 offen
	buffer_load_dwordx4 v[180:183], v7, s[4:7], 0 offen offset:32
	buffer_load_dwordx4 v[184:187], v7, s[4:7], 0 offen offset:64
	buffer_load_dwordx4 v[188:191], v7, s[4:7], 0 offen offset:96
	buffer_load_dwordx4 v[192:195], v7, s[4:7], 0 offen offset:128
	buffer_load_dwordx4 v[196:199], v7, s[4:7], 0 offen offset:160
	s_addc_u32 s0, s27, s1
	s_lshl_b32 s42, s13, 6
	s_and_b32 s41, s0, 0xffff
	buffer_load_dword v2, v6, s[40:43], 0 offen
	buffer_load_dwordx4 v[200:203], v7, s[4:7], 0 offen offset:192
	buffer_load_dwordx4 v[204:207], v7, s[4:7], 0 offen offset:224
	s_waitcnt vmcnt(10)
	v_readfirstlane_b32 s4, v4
	s_waitcnt vmcnt(9)
	v_readfirstlane_b32 s38, v3
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
	s_sub_i32 s39, s23, s37
	s_ashr_i32 s23, s22, 31
	s_lshr_b32 s0, s23, 28
	s_sub_i32 s47, s5, s4
	s_add_i32 s0, s22, s0
	s_lshl_b32 s40, s47, 7
	s_ashr_i32 s5, s0, 4
	s_and_b32 s0, s0, -16
	s_cmp_lg_u32 s22, s0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s22, 0
	s_cselect_b64 s[42:43], -1, 0
	s_and_b64 s[0:1], s[42:43], s[0:1]
	s_subb_u32 s37, s5, 0
	s_lshl_b32 s0, s37, 14
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s16, s0
	s_addc_u32 s1, s17, s1
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s4, s24, s4
	s_addc_u32 s5, s25, s5
	s_lshl_b32 s6, s47, 2
	s_and_b32 s5, s5, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[4:7], 0
	s_waitcnt vmcnt(3)
	v_mul_f32_e32 v2, s49, v2
	v_mul_f32_e32 v116, 0x3e0293ee, v2
	s_mulk_i32 s37, 0x6000
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v4
	v_readfirstlane_b32 s62, v5
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
	buffer_load_dword v2, off, s[4:7], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s60, v2
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
	buffer_load_dword v2, off, s[4:7], 0 offset:12
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s63, v2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_lshl_b32 s41, s44, 13
	v_or_b32_e32 v2, s41, v114
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 2, s[0:1]
	global_load_dwordx4 v[4:7], v[4:5], off
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_ashr_i32 s41, s41, 31
	v_mov_b32_e32 v3, s41
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[0:1]
	global_load_dwordx4 v[208:211], v[2:3], off offset:512
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	global_load_dwordx4 v[212:215], v[2:3], off offset:1024
	ds_write_b128 v167, v[4:7] offset:8192
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_add_i32 s39, s39, s40
	s_add_i32 s61, s39, s38
	s_addk_i32 s61, 0xff80
	s_add_i32 s64, s61, s46
	s_add_i32 s40, s64, 1
	s_ashr_i32 s38, s40, 31
	s_lshr_b32 s38, s38, 25
	s_add_i32 s38, s40, s38
	s_ashr_i32 s42, s38, 7
	s_and_b32 s38, s38, 0xffffff80
	s_cmp_lg_u32 s40, s38
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_lt_i32 s40, 0
	s_cselect_b64 s[40:41], -1, 0
	s_and_b64 s[38:39], s[40:41], s[38:39]
	s_subb_u32 s40, s42, 0
	s_lshr_b32 s38, s40, 31
	s_add_i32 s38, s40, s38
	s_ashr_i32 s42, s38, 1
	s_and_b32 s38, s38, -2
	s_cmp_lg_u32 s40, s38
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_lt_i32 s40, 0
	s_cselect_b64 s[40:41], -1, 0
	s_and_b64 s[38:39], s[40:41], s[38:39]
	s_subb_u32 s65, s42, 0
	s_lshl_b32 s38, s65, 1
	s_ashr_i32 s39, s38, 31
	s_cmp_lt_i32 s65, 1
	s_cbranch_scc1 .LBB0_51
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v117, v116
	v_mov_b32_e32 v120, v116
	v_mov_b32_e32 v121, v116
	v_mov_b32_e32 v122, v116
	v_mov_b32_e32 v123, v116
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
	s_mov_b64 s[40:41], 0
	s_mov_b32 s66, 20
	v_mov_b32_e32 v118, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v174
	v_mov_b32_e32 v4, v174
	v_mov_b32_e32 v5, v174
	v_mov_b32_e32 v6, v174
	v_mov_b32_e32 v7, v174
	v_mov_b32_e32 v8, v174
	v_mov_b32_e32 v9, v174
	v_mov_b32_e32 v10, v174
	v_mov_b32_e32 v11, v174
	v_mov_b32_e32 v12, v174
	v_mov_b32_e32 v13, v174
	v_mov_b32_e32 v14, v174
	v_mov_b32_e32 v15, v174
	v_mov_b32_e32 v16, v174
	v_mov_b32_e32 v17, v174
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v174
	v_mov_b32_e32 v20, v174
	v_mov_b32_e32 v21, v174
	v_mov_b32_e32 v22, v174
	v_mov_b32_e32 v23, v174
	v_mov_b32_e32 v24, v174
	v_mov_b32_e32 v25, v174
	v_mov_b32_e32 v26, v174
	v_mov_b32_e32 v27, v174
	v_mov_b32_e32 v28, v174
	v_mov_b32_e32 v29, v174
	v_mov_b32_e32 v30, v174
	v_mov_b32_e32 v31, v174
	v_mov_b32_e32 v32, v174
	v_mov_b32_e32 v33, v174
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v174
	v_mov_b32_e32 v36, v174
	v_mov_b32_e32 v37, v174
	v_mov_b32_e32 v38, v174
	v_mov_b32_e32 v39, v174
	v_mov_b32_e32 v40, v174
	v_mov_b32_e32 v41, v174
	v_mov_b32_e32 v42, v174
	v_mov_b32_e32 v43, v174
	v_mov_b32_e32 v44, v174
	v_mov_b32_e32 v45, v174
	v_mov_b32_e32 v46, v174
	v_mov_b32_e32 v47, v174
	v_mov_b32_e32 v48, v174
	v_mov_b32_e32 v49, v174
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, v174
	v_mov_b32_e32 v84, v174
	v_mov_b32_e32 v85, v174
	v_mov_b32_e32 v86, v174
	v_mov_b32_e32 v87, v174
	v_mov_b32_e32 v88, v174
	v_mov_b32_e32 v89, v174
	v_mov_b32_e32 v90, v174
	v_mov_b32_e32 v91, v174
	v_mov_b32_e32 v92, v174
	v_mov_b32_e32 v93, v174
	v_mov_b32_e32 v94, v174
	v_mov_b32_e32 v95, v174
	v_mov_b32_e32 v96, v174
	v_mov_b32_e32 v97, v174
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v174
	v_mov_b32_e32 v52, v174
	v_mov_b32_e32 v53, v174
	v_mov_b32_e32 v54, v174
	v_mov_b32_e32 v55, v174
	v_mov_b32_e32 v56, v174
	v_mov_b32_e32 v57, v174
	v_mov_b32_e32 v58, v174
	v_mov_b32_e32 v59, v174
	v_mov_b32_e32 v60, v174
	v_mov_b32_e32 v61, v174
	v_mov_b32_e32 v62, v174
	v_mov_b32_e32 v63, v174
	v_mov_b32_e32 v64, v174
	v_mov_b32_e32 v65, v174
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, v174
	v_mov_b32_e32 v68, v174
	v_mov_b32_e32 v69, v174
	v_mov_b32_e32 v70, v174
	v_mov_b32_e32 v71, v174
	v_mov_b32_e32 v72, v174
	v_mov_b32_e32 v73, v174
	v_mov_b32_e32 v74, v174
	v_mov_b32_e32 v75, v174
	v_mov_b32_e32 v76, v174
	v_mov_b32_e32 v77, v174
	v_mov_b32_e32 v78, v174
	v_mov_b32_e32 v79, v174
	v_mov_b32_e32 v80, v174
	v_mov_b32_e32 v81, v174
	s_branch .LBB0_18
.LBB0_17:
	s_or_b64 exec, exec, s[42:43]
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v100, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	;;#ASMSTART
	v_perm_b32 v98, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v101, v100, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v100, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v150, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v151, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v104, v0
	;;#ASMEND
	s_addk_i32 s44, 0x3000
	v_lshlrev_b32_e32 v105, 3, v104
	v_bfe_i32 v104, v104, 5, 1
	v_and_b32_e32 v105, 0xf8, v105
	v_and_b32_e32 v104, 0x600, v104
	v_or3_b32 v104, v105, v104, s44
	v_ashrrev_i32_e32 v105, 31, v104
	v_lshl_add_u64 v[104:105], v[104:105], 1, s[18:19]
	global_load_dwordx4 v[106:109], v[104:105], off
	global_load_dwordx4 v[110:113], v[104:105], off offset:512
	global_load_dwordx4 v[134:137], v[104:105], off offset:1024
	global_load_dwordx4 v[138:141], v[104:105], off offset:1536
	global_load_dwordx4 v[142:145], v[104:105], off offset:2048
	global_load_dwordx4 v[146:149], v[104:105], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[106:107], v[98:99], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[110:111], v[98:99], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[134:135], v[98:99], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[138:139], v[98:99], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[98:99], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[98:99], v[66:81]
	v_add_co_u32_e32 v98, vcc, s54, v104
	s_nop 1
	v_addc_co_u32_e32 v99, vcc, 0, v105, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	global_load_dwordx4 v[106:109], v[98:99], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[112:113], v[100:101], v[18:33]
	global_load_dwordx4 v[110:113], v[98:99], off offset:3584
	v_mfma_f32_32x32x8_bf16 v[34:49], v[136:137], v[100:101], v[34:49]
	global_load_dwordx4 v[134:137], v[98:99], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[82:97], v[140:141], v[100:101], v[82:97]
	global_load_dwordx4 v[140:143], v[98:99], off offset:3072
	v_add_co_u32_e32 v98, vcc, s55, v104
	s_nop 1
	v_addc_co_u32_e32 v99, vcc, 0, v105, vcc
	global_load_dwordx4 v[152:155], v[98:99], off
	v_mfma_f32_32x32x8_bf16 v[50:65], v[144:145], v[100:101], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[148:149], v[100:101], v[66:81]
	global_load_dwordx4 v[98:101], v[98:99], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[106:107], v[102:103], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[140:141], v[102:103], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[152:153], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[150:151], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[112:113], v[150:151], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[136:137], v[150:151], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[142:143], v[150:151], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[154:155], v[150:151], v[50:65]
	s_add_u32 s40, s40, 2
	s_addc_u32 s41, s41, 0
	v_mov_b64_e32 v[150:151], s[38:39]
	v_cmp_lt_i64_e32 vcc, s[40:41], v[150:151]
	s_add_i32 s66, s66, 8
	s_mov_b32 s62, s68
	s_mov_b32 s44, s67
	s_cbranch_vccz .LBB0_50
.LBB0_18:
	s_add_i32 s42, s66, -4
	v_mov_b32_e32 v98, s42
	s_lshl_b32 s42, s44, 13
	s_ashr_i32 s43, s42, 31
	buffer_load_dword v134, v98, s[4:7], 0 offen
	v_or_b32_e32 v98, s42, v114
	v_mov_b32_e32 v99, s43
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[160:163], v[98:99], off offset:1536
	ds_write_b128 v167, v[208:211]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mov_b32 s67, s60
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v135, 48, v98
	v_and_or_b32 v136, v99, s52, v100
	v_or_b32_e32 v98, v136, v135
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[138:141], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[176:177], 0
	s_mov_b32 s68, s63
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s60, v134
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[178:179], v[98:113]
	v_or_b32_e32 v134, 0x1010, v136
	v_xor_b32_e32 v134, v134, v135
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[182:183], v[98:113]
	v_or_b32_e32 v134, 0x1020, v136
	v_xor_b32_e32 v134, v134, v135
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[186:187], v[98:113]
	v_or_b32_e32 v134, 0x1030, v136
	v_xor_b32_e32 v134, v134, v135
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[190:191], v[98:113]
	v_or_b32_e32 v134, 0x1040, v136
	v_lshrrev_b32_e32 v135, 3, v134
	v_and_b32_e32 v135, 56, v135
	v_xor_b32_e32 v134, v135, v134
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[194:195], v[98:113]
	v_or_b32_e32 v134, 0x1050, v136
	v_lshrrev_b32_e32 v135, 3, v134
	v_and_b32_e32 v135, 56, v135
	v_xor_b32_e32 v134, v135, v134
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[198:199], v[98:113]
	v_or_b32_e32 v134, 0x1060, v136
	v_lshrrev_b32_e32 v135, 3, v134
	v_and_b32_e32 v135, 56, v135
	v_xor_b32_e32 v134, v135, v134
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[202:203], v[98:113]
	v_or_b32_e32 v134, 0x1070, v136
	v_lshrrev_b32_e32 v135, 3, v134
	v_and_b32_e32 v135, 56, v135
	v_xor_b32_e32 v134, v135, v134
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_max_f32_e32 v134, v98, v99
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_max3_f32 v134, v134, v100, v101
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_max3_f32 v134, v134, v102, v103
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_max3_f32 v134, v134, v104, v105
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_max3_f32 v134, v134, v106, v107
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	v_max3_f32 v134, v134, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v134, v134, v110, v111
	v_max3_f32 v134, v134, v112, v113
	ds_bpermute_b32 v135, v168, v134
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v135, v135, v135
	v_max_f32_e32 v135, v134, v135
	;;#ASMSTART
	v_add_f32 v134, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v135, v134
	v_mov_b32_e32 v134, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_20
	;;#ASMSTART
	v_add_f32 v135, v135, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v135
	v_exp_f32_e32 v134, v118
	v_mov_b32_e32 v118, v135
.LBB0_20:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[98:99], v[98:99], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v135, 0, v98
	v_add_f32_e32 v135, v135, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v135, v135, v100
	v_add_f32_e32 v135, v135, v101
	v_add_f32_e32 v135, v135, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v135, v135, v103
	v_add_f32_e32 v135, v135, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v135, v135, v105
	v_add_f32_e32 v135, v135, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v135, v135, v107
	v_add_f32_e32 v135, v135, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v135, v135, v109
	v_add_f32_e32 v135, v135, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v134
	v_add_f32_e32 v135, v135, v111
	v_add_f32_e32 v135, v135, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v135, v135, v113
	;;#ASMSTART
	v_fma_f32 v150, v174, v134, v135
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_22
	;;#ASMSTART
	v_mul_f32 v2, v2, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v134
	;;#ASMEND
.LBB0_22:
	s_or_b64 exec, exec, s[42:43]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v134, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v134, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s43, s44, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s43, s43, s37
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s43
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[134:137], v[106:107], off offset:512
	global_load_dwordx4 v[138:141], v[106:107], off offset:1024
	global_load_dwordx4 v[142:145], v[106:107], off offset:1536
	global_load_dwordx4 v[146:149], v[106:107], off offset:2048
	global_load_dwordx4 v[152:155], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[138:139], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[142:143], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[146:147], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[152:153], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[136:137], v[102:103], v[18:33]
	global_load_dwordx4 v[134:137], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[140:141], v[102:103], v[34:49]
	global_load_dwordx4 v[138:141], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[144:145], v[102:103], v[82:97]
	global_load_dwordx4 v[142:145], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[154:155], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[138:139], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[142:143], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[136:137], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[140:141], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[144:145], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	s_lshl_b32 s42, s62, 13
	v_or_b32_e32 v98, s42, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[156:159], v[98:99], off
	ds_write_b128 v167, v[212:215] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v134, v99, s52, v100
	v_and_b32_e32 v135, 48, v98
	v_or_b32_e32 v98, v134, v135
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[136:139], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[178:179], v[98:113]
	v_or_b32_e32 v136, 16, v134
	v_xor_b32_e32 v136, v136, v135
	v_lshlrev_b32_e32 v136, 1, v136
	ds_read_b128 v[136:139], v136
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[182:183], v[98:113]
	v_or_b32_e32 v136, 32, v134
	v_xor_b32_e32 v136, v136, v135
	v_lshlrev_b32_e32 v136, 1, v136
	ds_read_b128 v[136:139], v136
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[186:187], v[98:113]
	v_or_b32_e32 v136, 48, v134
	v_xor_b32_e32 v135, v136, v135
	v_lshlrev_b32_e32 v135, 1, v135
	ds_read_b128 v[136:139], v135
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[190:191], v[98:113]
	v_or_b32_e32 v135, 64, v134
	v_lshrrev_b32_e32 v136, 3, v135
	v_and_b32_e32 v136, 56, v136
	v_xor_b32_e32 v135, v136, v135
	v_lshlrev_b32_e32 v135, 1, v135
	ds_read_b128 v[136:139], v135
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[194:195], v[98:113]
	v_or_b32_e32 v135, 0x50, v134
	v_lshrrev_b32_e32 v136, 3, v135
	v_and_b32_e32 v136, 56, v136
	v_xor_b32_e32 v135, v136, v135
	v_lshlrev_b32_e32 v135, 1, v135
	ds_read_b128 v[136:139], v135
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[198:199], v[98:113]
	v_or_b32_e32 v135, 0x60, v134
	v_lshrrev_b32_e32 v136, 3, v135
	v_and_b32_e32 v136, 56, v136
	v_xor_b32_e32 v135, v136, v135
	v_lshlrev_b32_e32 v135, 1, v135
	ds_read_b128 v[136:139], v135
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[202:203], v[98:113]
	v_or_b32_e32 v134, 0x70, v134
	v_lshrrev_b32_e32 v135, 3, v134
	v_and_b32_e32 v135, 56, v135
	v_xor_b32_e32 v134, v135, v134
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_max_f32_e32 v134, v98, v99
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_max3_f32 v134, v134, v100, v101
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_max3_f32 v134, v134, v102, v103
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_max3_f32 v134, v134, v104, v105
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_max3_f32 v134, v134, v106, v107
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	v_max3_f32 v134, v134, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v134, v134, v110, v111
	v_max3_f32 v134, v134, v112, v113
	ds_bpermute_b32 v135, v168, v134
	v_mov_b32_e32 v151, 1.0
	v_mov_b32_e32 v136, v118
	v_mov_b32_e32 v137, v118
	v_mov_b32_e32 v138, v118
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v135, v135, v135
	v_max_f32_e32 v152, v134, v135
	;;#ASMSTART
	v_add_f32 v134, v118, v169
	;;#ASMEND
	v_mov_b32_e32 v135, v118
	v_cmp_gt_f32_e32 vcc, v152, v134
	v_mov_b32_e32 v134, v118
	v_mov_b32_e32 v139, v118
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
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_24
	;;#ASMSTART
	v_add_f32 v134, v152, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v151, v118
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
.LBB0_24:
	s_or_b64 exec, exec, s[44:45]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, 0, v98
	v_add_f32_e32 v152, v152, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v100
	v_add_f32_e32 v152, v152, v101
	v_add_f32_e32 v152, v152, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v103
	v_add_f32_e32 v152, v152, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v105
	v_add_f32_e32 v152, v152, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v107
	v_add_f32_e32 v152, v152, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v109
	v_add_f32_e32 v152, v152, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v151
	v_add_f32_e32 v152, v152, v111
	v_add_f32_e32 v152, v152, v112
	v_add_f32_e32 v152, v152, v113
	;;#ASMSTART
	v_fma_f32 v152, v150, v151, v152
	;;#ASMEND
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_26
	;;#ASMSTART
	v_mul_f32 v2, v2, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v151
	;;#ASMEND
.LBB0_26:
	s_or_b64 exec, exec, s[44:45]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v150, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v150, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s44, s43, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s44
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[208:211], v[106:107], off offset:512
	global_load_dwordx4 v[212:215], v[106:107], off offset:1024
	global_load_dwordx4 v[216:219], v[106:107], off offset:1536
	global_load_dwordx4 v[220:223], v[106:107], off offset:2048
	global_load_dwordx4 v[224:227], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[216:217], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[210:211], v[102:103], v[18:33]
	global_load_dwordx4 v[208:211], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[102:103], v[34:49]
	global_load_dwordx4 v[212:215], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[218:219], v[102:103], v[82:97]
	global_load_dwordx4 v[216:219], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[216:217], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[210:211], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[218:219], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	s_ashr_i32 s43, s42, 31
	v_lshl_add_u64 v[98:99], s[42:43], 0, v[114:115]
	v_lshl_add_u64 v[150:151], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[208:211], v[150:151], off offset:512
	ds_write_b128 v167, v[160:163]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v153, v99, s52, v100
	v_and_b32_e32 v154, 48, v98
	v_or_b32_e32 v98, v153, v154
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[160:163], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[178:179], v[98:113]
	v_or_b32_e32 v155, 0x1010, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[160:163], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[182:183], v[98:113]
	v_or_b32_e32 v155, 0x1020, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[160:163], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[186:187], v[98:113]
	v_or_b32_e32 v155, 0x1030, v153
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[160:163], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[190:191], v[98:113]
	v_or_b32_e32 v154, 0x1040, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[160:163], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[194:195], v[98:113]
	v_or_b32_e32 v154, 0x1050, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[160:163], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[198:199], v[98:113]
	v_or_b32_e32 v154, 0x1060, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[160:163], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[202:203], v[98:113]
	v_or_b32_e32 v153, 0x1070, v153
	v_lshrrev_b32_e32 v154, 3, v153
	v_and_b32_e32 v154, 56, v154
	v_xor_b32_e32 v153, v154, v153
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[160:163], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_max_f32_e32 v153, v98, v99
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_max3_f32 v153, v153, v100, v101
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_max3_f32 v153, v153, v102, v103
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_max3_f32 v153, v153, v104, v105
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_max3_f32 v153, v153, v106, v107
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	v_max3_f32 v153, v153, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v153, v153, v110, v111
	v_max3_f32 v153, v153, v112, v113
	ds_bpermute_b32 v154, v168, v153
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v154, v154, v154
	v_max_f32_e32 v154, v153, v154
	;;#ASMSTART
	v_add_f32 v153, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v154, v153
	v_mov_b32_e32 v153, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_28
	;;#ASMSTART
	v_add_f32 v134, v154, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v153, v118
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
.LBB0_28:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, 0, v98
	v_add_f32_e32 v154, v154, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v100
	v_add_f32_e32 v154, v154, v101
	v_add_f32_e32 v154, v154, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v103
	v_add_f32_e32 v154, v154, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v105
	v_add_f32_e32 v154, v154, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v107
	v_add_f32_e32 v154, v154, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v109
	v_add_f32_e32 v154, v154, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v153
	v_add_f32_e32 v154, v154, v111
	v_add_f32_e32 v154, v154, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v154, v154, v113
	;;#ASMSTART
	v_fma_f32 v152, v152, v153, v154
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_30
	;;#ASMSTART
	v_mul_f32 v2, v2, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v153
	;;#ASMEND
.LBB0_30:
	s_or_b64 exec, exec, s[42:43]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v153, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v153, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s42, s44, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s42
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[160:163], v[106:107], off offset:512
	global_load_dwordx4 v[212:215], v[106:107], off offset:1024
	global_load_dwordx4 v[216:219], v[106:107], off offset:1536
	global_load_dwordx4 v[220:223], v[106:107], off offset:2048
	global_load_dwordx4 v[224:227], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[160:161], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[216:217], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[162:163], v[102:103], v[18:33]
	global_load_dwordx4 v[160:163], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[102:103], v[34:49]
	global_load_dwordx4 v[212:215], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[218:219], v[102:103], v[82:97]
	global_load_dwordx4 v[216:219], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[160:161], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[216:217], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[162:163], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[218:219], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	global_load_dwordx4 v[160:163], v[150:151], off offset:1024
	ds_write_b128 v167, v[156:159] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v153, v99, s52, v100
	v_and_b32_e32 v154, 48, v98
	v_or_b32_e32 v98, v153, v154
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[156:159], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[178:179], v[98:113]
	v_or_b32_e32 v155, 16, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[156:159], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[182:183], v[98:113]
	v_or_b32_e32 v155, 32, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[156:159], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[186:187], v[98:113]
	v_or_b32_e32 v155, 48, v153
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[190:191], v[98:113]
	v_or_b32_e32 v154, 64, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[194:195], v[98:113]
	v_or_b32_e32 v154, 0x50, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[198:199], v[98:113]
	v_or_b32_e32 v154, 0x60, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[202:203], v[98:113]
	v_or_b32_e32 v153, 0x70, v153
	v_lshrrev_b32_e32 v154, 3, v153
	v_and_b32_e32 v154, 56, v154
	v_xor_b32_e32 v153, v154, v153
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[154:157], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_max_f32_e32 v153, v98, v99
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_max3_f32 v153, v153, v100, v101
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_max3_f32 v153, v153, v102, v103
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_max3_f32 v153, v153, v104, v105
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_max3_f32 v153, v153, v106, v107
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	v_max3_f32 v153, v153, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v153, v153, v110, v111
	v_max3_f32 v153, v153, v112, v113
	ds_bpermute_b32 v154, v168, v153
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v154, v154, v154
	v_max_f32_e32 v154, v153, v154
	;;#ASMSTART
	v_add_f32 v153, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v154, v153
	v_mov_b32_e32 v153, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v134, v154, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v153, v118
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
.LBB0_32:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, 0, v98
	v_add_f32_e32 v154, v154, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v100
	v_add_f32_e32 v154, v154, v101
	v_add_f32_e32 v154, v154, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v103
	v_add_f32_e32 v154, v154, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v105
	v_add_f32_e32 v154, v154, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v107
	v_add_f32_e32 v154, v154, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v109
	v_add_f32_e32 v154, v154, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v153
	v_add_f32_e32 v154, v154, v111
	v_add_f32_e32 v154, v154, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v154, v154, v113
	;;#ASMSTART
	v_fma_f32 v152, v152, v153, v154
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_34
	;;#ASMSTART
	v_mul_f32 v2, v2, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v153
	;;#ASMEND
.LBB0_34:
	s_or_b64 exec, exec, s[42:43]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_u32_e32 v153, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v153, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s44, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s44
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[154:157], v[106:107], off offset:512
	global_load_dwordx4 v[212:215], v[106:107], off offset:1024
	global_load_dwordx4 v[216:219], v[106:107], off offset:1536
	global_load_dwordx4 v[220:223], v[106:107], off offset:2048
	global_load_dwordx4 v[224:227], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[154:155], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[216:217], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[156:157], v[102:103], v[18:33]
	global_load_dwordx4 v[154:157], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[102:103], v[34:49]
	global_load_dwordx4 v[212:215], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[218:219], v[102:103], v[82:97]
	global_load_dwordx4 v[216:219], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[154:155], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[216:217], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[156:157], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[218:219], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	v_mov_b32_e32 v98, s66
	buffer_load_dword v153, v98, s[4:7], 0 offen
	global_load_dwordx4 v[154:157], v[150:151], off offset:1536
	ds_write_b128 v167, v[208:211]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s63, v153
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v150, 48, v98
	v_and_or_b32 v151, v99, s52, v100
	v_or_b32_e32 v98, v151, v150
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[208:211], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[208:209], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[210:211], v[178:179], v[98:113]
	v_or_b32_e32 v153, 0x1010, v151
	v_xor_b32_e32 v153, v153, v150
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[208:211], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[208:209], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[210:211], v[182:183], v[98:113]
	v_or_b32_e32 v153, 0x1020, v151
	v_xor_b32_e32 v153, v153, v150
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[208:211], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[208:209], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[210:211], v[186:187], v[98:113]
	v_or_b32_e32 v153, 0x1030, v151
	v_xor_b32_e32 v150, v153, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[208:211], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[208:209], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[210:211], v[190:191], v[98:113]
	v_or_b32_e32 v150, 0x1040, v151
	v_lshrrev_b32_e32 v153, 3, v150
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v150, v153, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[208:211], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[208:209], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[210:211], v[194:195], v[98:113]
	v_or_b32_e32 v150, 0x1050, v151
	v_lshrrev_b32_e32 v153, 3, v150
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v150, v153, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[208:211], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[208:209], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[210:211], v[198:199], v[98:113]
	v_or_b32_e32 v150, 0x1060, v151
	v_lshrrev_b32_e32 v153, 3, v150
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v150, v153, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[208:211], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[208:209], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[210:211], v[202:203], v[98:113]
	v_or_b32_e32 v150, 0x1070, v151
	v_lshrrev_b32_e32 v151, 3, v150
	v_and_b32_e32 v151, 56, v151
	v_xor_b32_e32 v150, v151, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[208:211], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[208:209], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[210:211], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_max_f32_e32 v150, v98, v99
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_max3_f32 v150, v150, v100, v101
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_max3_f32 v150, v150, v102, v103
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_max3_f32 v150, v150, v104, v105
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_max3_f32 v150, v150, v106, v107
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	v_max3_f32 v150, v150, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v150, v150, v110, v111
	v_max3_f32 v150, v150, v112, v113
	ds_bpermute_b32 v151, v168, v150
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v151, v151, v151
	v_max_f32_e32 v150, v150, v151
	;;#ASMSTART
	v_add_f32 v151, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v150, v151
	v_mov_b32_e32 v151, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_36
	;;#ASMSTART
	v_add_f32 v134, v150, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v151, v118
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
.LBB0_36:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, 0, v98
	v_add_f32_e32 v150, v150, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, v150, v100
	v_add_f32_e32 v150, v150, v101
	v_add_f32_e32 v150, v150, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, v150, v103
	v_add_f32_e32 v150, v150, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, v150, v105
	v_add_f32_e32 v150, v150, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, v150, v107
	v_add_f32_e32 v150, v150, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, v150, v109
	v_add_f32_e32 v150, v150, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v151
	v_add_f32_e32 v150, v150, v111
	v_add_f32_e32 v150, v150, v112
	v_add_f32_e32 v150, v150, v113
	;;#ASMSTART
	v_fma_f32 v150, v152, v151, v150
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_38
	;;#ASMSTART
	v_mul_f32 v2, v2, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v151
	;;#ASMEND
.LBB0_38:
	s_or_b64 exec, exec, s[42:43]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v151, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v151, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s43, s62, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s43, s43, s37
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s43
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[208:211], v[106:107], off offset:512
	global_load_dwordx4 v[212:215], v[106:107], off offset:1024
	global_load_dwordx4 v[216:219], v[106:107], off offset:1536
	global_load_dwordx4 v[220:223], v[106:107], off offset:2048
	global_load_dwordx4 v[224:227], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[216:217], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[210:211], v[102:103], v[18:33]
	global_load_dwordx4 v[208:211], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[102:103], v[34:49]
	global_load_dwordx4 v[212:215], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[218:219], v[102:103], v[82:97]
	global_load_dwordx4 v[216:219], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[208:209], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[216:217], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[210:211], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[218:219], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	s_lshl_b32 s42, s67, 13
	v_or_b32_e32 v98, s42, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[216:219], v[98:99], off
	ds_write_b128 v167, v[160:163] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v151, v99, s52, v100
	v_and_b32_e32 v152, 48, v98
	v_or_b32_e32 v98, v151, v152
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[158:161], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[178:179], v[98:113]
	v_or_b32_e32 v153, 16, v151
	v_xor_b32_e32 v153, v153, v152
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[158:161], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[182:183], v[98:113]
	v_or_b32_e32 v153, 32, v151
	v_xor_b32_e32 v153, v153, v152
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[158:161], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[186:187], v[98:113]
	v_or_b32_e32 v153, 48, v151
	v_xor_b32_e32 v152, v153, v152
	v_lshlrev_b32_e32 v152, 1, v152
	ds_read_b128 v[158:161], v152
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[190:191], v[98:113]
	v_or_b32_e32 v152, 64, v151
	v_lshrrev_b32_e32 v153, 3, v152
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v152, v153, v152
	v_lshlrev_b32_e32 v152, 1, v152
	ds_read_b128 v[158:161], v152
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[194:195], v[98:113]
	v_or_b32_e32 v152, 0x50, v151
	v_lshrrev_b32_e32 v153, 3, v152
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v152, v153, v152
	v_lshlrev_b32_e32 v152, 1, v152
	ds_read_b128 v[158:161], v152
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[198:199], v[98:113]
	v_or_b32_e32 v152, 0x60, v151
	v_lshrrev_b32_e32 v153, 3, v152
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v152, v153, v152
	v_lshlrev_b32_e32 v152, 1, v152
	ds_read_b128 v[158:161], v152
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[202:203], v[98:113]
	v_or_b32_e32 v151, 0x70, v151
	v_lshrrev_b32_e32 v152, 3, v151
	v_and_b32_e32 v152, 56, v152
	v_xor_b32_e32 v151, v152, v151
	v_lshlrev_b32_e32 v151, 1, v151
	ds_read_b128 v[158:161], v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_max_f32_e32 v151, v98, v99
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_max3_f32 v151, v151, v100, v101
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_max3_f32 v151, v151, v102, v103
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_max3_f32 v151, v151, v104, v105
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_max3_f32 v151, v151, v106, v107
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	v_max3_f32 v151, v151, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v151, v151, v110, v111
	v_max3_f32 v151, v151, v112, v113
	ds_bpermute_b32 v152, v168, v151
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v152, v152, v152
	v_max_f32_e32 v152, v151, v152
	;;#ASMSTART
	v_add_f32 v151, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v152, v151
	v_mov_b32_e32 v151, 1.0
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_40
	;;#ASMSTART
	v_add_f32 v134, v152, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v151, v118
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
.LBB0_40:
	s_or_b64 exec, exec, s[44:45]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, 0, v98
	v_add_f32_e32 v152, v152, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v100
	v_add_f32_e32 v152, v152, v101
	v_add_f32_e32 v152, v152, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v103
	v_add_f32_e32 v152, v152, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v105
	v_add_f32_e32 v152, v152, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v107
	v_add_f32_e32 v152, v152, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v109
	v_add_f32_e32 v152, v152, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v151
	v_add_f32_e32 v152, v152, v111
	v_add_f32_e32 v152, v152, v112
	v_add_f32_e32 v152, v152, v113
	;;#ASMSTART
	v_fma_f32 v152, v150, v151, v152
	;;#ASMEND
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_42
	;;#ASMSTART
	v_mul_f32 v2, v2, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v151
	;;#ASMEND
.LBB0_42:
	s_or_b64 exec, exec, s[44:45]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v150, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v150, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s44, s43, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s44
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[158:161], v[106:107], off offset:512
	global_load_dwordx4 v[162:165], v[106:107], off offset:1024
	global_load_dwordx4 v[208:211], v[106:107], off offset:1536
	global_load_dwordx4 v[212:215], v[106:107], off offset:2048
	global_load_dwordx4 v[220:223], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[158:159], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[162:163], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[208:209], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[212:213], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[160:161], v[102:103], v[18:33]
	global_load_dwordx4 v[158:161], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[164:165], v[102:103], v[34:49]
	global_load_dwordx4 v[162:165], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[210:211], v[102:103], v[82:97]
	global_load_dwordx4 v[208:211], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[214:215], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[158:159], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[162:163], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[208:209], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[160:161], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[164:165], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[210:211], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	s_ashr_i32 s43, s42, 31
	v_lshl_add_u64 v[98:99], s[42:43], 0, v[114:115]
	v_lshl_add_u64 v[150:151], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[208:211], v[150:151], off offset:512
	ds_write_b128 v167, v[154:157]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v153, v99, s52, v100
	v_and_b32_e32 v154, 48, v98
	v_or_b32_e32 v98, v153, v154
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[156:159], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[178:179], v[98:113]
	v_or_b32_e32 v155, 0x1010, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[156:159], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[182:183], v[98:113]
	v_or_b32_e32 v155, 0x1020, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[156:159], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[186:187], v[98:113]
	v_or_b32_e32 v155, 0x1030, v153
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[190:191], v[98:113]
	v_or_b32_e32 v154, 0x1040, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[194:195], v[98:113]
	v_or_b32_e32 v154, 0x1050, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[198:199], v[98:113]
	v_or_b32_e32 v154, 0x1060, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[202:203], v[98:113]
	v_or_b32_e32 v153, 0x1070, v153
	v_lshrrev_b32_e32 v154, 3, v153
	v_and_b32_e32 v154, 56, v154
	v_xor_b32_e32 v153, v154, v153
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[154:157], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_max_f32_e32 v153, v98, v99
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_max3_f32 v153, v153, v100, v101
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_max3_f32 v153, v153, v102, v103
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_max3_f32 v153, v153, v104, v105
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_max3_f32 v153, v153, v106, v107
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	v_max3_f32 v153, v153, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v153, v153, v110, v111
	v_max3_f32 v153, v153, v112, v113
	ds_bpermute_b32 v154, v168, v153
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v154, v154, v154
	v_max_f32_e32 v154, v153, v154
	;;#ASMSTART
	v_add_f32 v153, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v154, v153
	v_mov_b32_e32 v153, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_44
	;;#ASMSTART
	v_add_f32 v134, v154, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v153, v118
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
.LBB0_44:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, 0, v98
	v_add_f32_e32 v154, v154, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v100
	v_add_f32_e32 v154, v154, v101
	v_add_f32_e32 v154, v154, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v103
	v_add_f32_e32 v154, v154, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v105
	v_add_f32_e32 v154, v154, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v107
	v_add_f32_e32 v154, v154, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v109
	v_add_f32_e32 v154, v154, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v153
	v_add_f32_e32 v154, v154, v111
	v_add_f32_e32 v154, v154, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v154, v154, v113
	;;#ASMSTART
	v_fma_f32 v152, v152, v153, v154
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_46
	;;#ASMSTART
	v_mul_f32 v2, v2, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v153
	;;#ASMEND
.LBB0_46:
	s_or_b64 exec, exec, s[42:43]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_u32_e32 v153, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v153, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s42, s44, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s42
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[154:157], v[106:107], off offset:512
	global_load_dwordx4 v[158:161], v[106:107], off offset:1024
	global_load_dwordx4 v[162:165], v[106:107], off offset:1536
	global_load_dwordx4 v[212:215], v[106:107], off offset:2048
	global_load_dwordx4 v[220:223], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[154:155], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[158:159], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[162:163], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[212:213], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[220:221], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[156:157], v[102:103], v[18:33]
	global_load_dwordx4 v[154:157], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[160:161], v[102:103], v[34:49]
	global_load_dwordx4 v[158:161], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[164:165], v[102:103], v[82:97]
	global_load_dwordx4 v[162:165], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[214:215], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[222:223], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[154:155], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[158:159], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[162:163], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[156:157], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[160:161], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[164:165], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	global_load_dwordx4 v[212:215], v[150:151], off offset:1024
	ds_write_b128 v167, v[216:219] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v150, v99, s52, v100
	v_and_b32_e32 v151, 48, v98
	v_or_b32_e32 v98, v150, v151
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[154:157], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[178:179], v[98:113]
	v_or_b32_e32 v153, 16, v150
	v_xor_b32_e32 v153, v153, v151
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[154:157], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[182:183], v[98:113]
	v_or_b32_e32 v153, 32, v150
	v_xor_b32_e32 v153, v153, v151
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[154:157], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[186:187], v[98:113]
	v_or_b32_e32 v153, 48, v150
	v_xor_b32_e32 v151, v153, v151
	v_lshlrev_b32_e32 v151, 1, v151
	ds_read_b128 v[154:157], v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[190:191], v[98:113]
	v_or_b32_e32 v151, 64, v150
	v_lshrrev_b32_e32 v153, 3, v151
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v151, v153, v151
	v_lshlrev_b32_e32 v151, 1, v151
	ds_read_b128 v[154:157], v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[194:195], v[98:113]
	v_or_b32_e32 v151, 0x50, v150
	v_lshrrev_b32_e32 v153, 3, v151
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v151, v153, v151
	v_lshlrev_b32_e32 v151, 1, v151
	ds_read_b128 v[154:157], v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[198:199], v[98:113]
	v_or_b32_e32 v151, 0x60, v150
	v_lshrrev_b32_e32 v153, 3, v151
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v151, v153, v151
	v_lshlrev_b32_e32 v151, 1, v151
	ds_read_b128 v[154:157], v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[202:203], v[98:113]
	v_or_b32_e32 v150, 0x70, v150
	v_lshrrev_b32_e32 v151, 3, v150
	v_and_b32_e32 v151, 56, v151
	v_xor_b32_e32 v150, v151, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[154:157], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_max_f32_e32 v150, v98, v99
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_max3_f32 v150, v150, v100, v101
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_max3_f32 v150, v150, v102, v103
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_max3_f32 v150, v150, v104, v105
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_max3_f32 v150, v150, v106, v107
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	v_max3_f32 v150, v150, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v150, v150, v110, v111
	v_max3_f32 v150, v150, v112, v113
	ds_bpermute_b32 v151, v168, v150
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v151, v151, v151
	v_max_f32_e32 v151, v150, v151
	;;#ASMSTART
	v_add_f32 v150, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v151, v150
	v_mov_b32_e32 v150, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_48
	;;#ASMSTART
	v_add_f32 v134, v151, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v150, v118
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
.LBB0_48:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, 0, v98
	v_add_f32_e32 v134, v134, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, v134, v100
	v_add_f32_e32 v134, v134, v101
	v_add_f32_e32 v134, v134, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, v134, v103
	v_add_f32_e32 v134, v134, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, v134, v105
	v_add_f32_e32 v134, v134, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, v134, v107
	v_add_f32_e32 v134, v134, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, v134, v109
	v_add_f32_e32 v134, v134, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v150
	v_add_f32_e32 v134, v134, v111
	v_add_f32_e32 v134, v134, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v134, v134, v113
	;;#ASMSTART
	v_fma_f32 v174, v152, v150, v134
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_17
	;;#ASMSTART
	v_mul_f32 v2, v2, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v150
	;;#ASMEND
	s_branch .LBB0_17
.LBB0_50:
	v_mov_b32_e32 v99, v81
	v_mov_b32_e32 v98, v80
	v_mov_b32_e32 v101, v79
	v_mov_b32_e32 v100, v78
	v_mov_b32_e32 v103, v77
	v_mov_b32_e32 v102, v76
	v_mov_b32_e32 v105, v75
	v_mov_b32_e32 v104, v74
	v_mov_b32_e32 v107, v73
	v_mov_b32_e32 v106, v72
	v_mov_b32_e32 v109, v71
	v_mov_b32_e32 v108, v70
	v_mov_b32_e32 v111, v69
	v_mov_b32_e32 v110, v68
	v_mov_b32_e32 v113, v67
	v_mov_b32_e32 v112, v66
	v_mov_b32_e32 v135, v97
	v_mov_b32_e32 v134, v96
	v_mov_b32_e32 v137, v95
	v_mov_b32_e32 v136, v94
	v_mov_b32_e32 v139, v93
	v_mov_b32_e32 v138, v92
	v_mov_b32_e32 v141, v91
	v_mov_b32_e32 v140, v90
	v_mov_b32_e32 v143, v89
	v_mov_b32_e32 v142, v88
	v_mov_b32_e32 v145, v87
	v_mov_b32_e32 v144, v86
	v_mov_b32_e32 v147, v85
	v_mov_b32_e32 v146, v84
	v_mov_b32_e32 v149, v83
	v_mov_b32_e32 v148, v82
	v_mov_b32_e32 v82, v2
	v_mov_b32_e32 v83, v3
	v_mov_b32_e32 v84, v4
	v_mov_b32_e32 v85, v5
	v_mov_b32_e32 v86, v6
	v_mov_b32_e32 v87, v7
	v_mov_b32_e32 v88, v8
	v_mov_b32_e32 v89, v9
	v_mov_b32_e32 v90, v10
	v_mov_b32_e32 v91, v11
	v_mov_b32_e32 v92, v12
	v_mov_b32_e32 v93, v13
	v_mov_b32_e32 v94, v14
	v_mov_b32_e32 v95, v15
	v_mov_b32_e32 v96, v16
	v_mov_b32_e32 v97, v17
	v_mov_b32_e32 v66, v18
	v_mov_b32_e32 v67, v19
	v_mov_b32_e32 v68, v20
	v_mov_b32_e32 v69, v21
	v_mov_b32_e32 v70, v22
	v_mov_b32_e32 v71, v23
	v_mov_b32_e32 v72, v24
	v_mov_b32_e32 v73, v25
	v_mov_b32_e32 v74, v26
	v_mov_b32_e32 v75, v27
	v_mov_b32_e32 v76, v28
	v_mov_b32_e32 v77, v29
	v_mov_b32_e32 v78, v30
	v_mov_b32_e32 v79, v31
	v_mov_b32_e32 v80, v32
	v_mov_b32_e32 v81, v33
	v_mov_b32_e32 v150, v34
	v_mov_b32_e32 v151, v35
	v_mov_b32_e32 v152, v36
	v_mov_b32_e32 v153, v37
	v_mov_b32_e32 v154, v38
	v_mov_b32_e32 v155, v39
	v_mov_b32_e32 v156, v40
	v_mov_b32_e32 v157, v41
	v_mov_b32_e32 v158, v42
	v_mov_b32_e32 v159, v43
	v_mov_b32_e32 v160, v44
	v_mov_b32_e32 v161, v45
	v_mov_b32_e32 v162, v46
	v_mov_b32_e32 v163, v47
	v_mov_b32_e32 v164, v48
	v_mov_b32_e32 v165, v49
	v_mov_b32_e32 v18, v50
	v_mov_b32_e32 v19, v51
	v_mov_b32_e32 v20, v52
	v_mov_b32_e32 v21, v53
	v_mov_b32_e32 v22, v54
	v_mov_b32_e32 v23, v55
	v_mov_b32_e32 v24, v56
	v_mov_b32_e32 v25, v57
	v_mov_b32_e32 v26, v58
	v_mov_b32_e32 v27, v59
	v_mov_b32_e32 v28, v60
	v_mov_b32_e32 v29, v61
	v_mov_b32_e32 v30, v62
	v_mov_b32_e32 v31, v63
	v_mov_b32_e32 v32, v64
	v_mov_b32_e32 v33, v65
	s_mov_b32 s62, s68
	s_mov_b32 s44, s67
	s_branch .LBB0_52
.LBB0_51:
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, v82
	v_mov_b32_e32 v84, v82
	v_mov_b32_e32 v85, v82
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
	v_mov_b32_e32 v66, v82
	v_mov_b32_e32 v67, v82
	v_mov_b32_e32 v68, v82
	v_mov_b32_e32 v69, v82
	v_mov_b32_e32 v70, v82
	v_mov_b32_e32 v71, v82
	v_mov_b32_e32 v72, v82
	v_mov_b32_e32 v73, v82
	v_mov_b32_e32 v74, v82
	v_mov_b32_e32 v75, v82
	v_mov_b32_e32 v76, v82
	v_mov_b32_e32 v77, v82
	v_mov_b32_e32 v78, v82
	v_mov_b32_e32 v79, v82
	v_mov_b32_e32 v80, v82
	v_mov_b32_e32 v81, v82
	v_mov_b32_e32 v150, v82
	v_mov_b32_e32 v151, v82
	v_mov_b32_e32 v152, v82
	v_mov_b32_e32 v153, v82
	v_mov_b32_e32 v154, v82
	v_mov_b32_e32 v155, v82
	v_mov_b32_e32 v156, v82
	v_mov_b32_e32 v157, v82
	v_mov_b32_e32 v158, v82
	v_mov_b32_e32 v159, v82
	v_mov_b32_e32 v160, v82
	v_mov_b32_e32 v161, v82
	v_mov_b32_e32 v162, v82
	v_mov_b32_e32 v163, v82
	v_mov_b32_e32 v164, v82
	v_mov_b32_e32 v165, v82
	v_mov_b32_e32 v148, v82
	v_mov_b32_e32 v149, v82
	v_mov_b32_e32 v146, v82
	v_mov_b32_e32 v147, v82
	v_mov_b32_e32 v144, v82
	v_mov_b32_e32 v145, v82
	v_mov_b32_e32 v142, v82
	v_mov_b32_e32 v143, v82
	v_mov_b32_e32 v140, v82
	v_mov_b32_e32 v141, v82
	v_mov_b32_e32 v138, v82
	v_mov_b32_e32 v139, v82
	v_mov_b32_e32 v136, v82
	v_mov_b32_e32 v137, v82
	v_mov_b32_e32 v134, v82
	v_mov_b32_e32 v135, v82
	v_mov_b32_e32 v18, v82
	v_mov_b32_e32 v19, v82
	v_mov_b32_e32 v20, v82
	v_mov_b32_e32 v21, v82
	v_mov_b32_e32 v22, v82
	v_mov_b32_e32 v23, v82
	v_mov_b32_e32 v24, v82
	v_mov_b32_e32 v25, v82
	v_mov_b32_e32 v26, v82
	v_mov_b32_e32 v27, v82
	v_mov_b32_e32 v28, v82
	v_mov_b32_e32 v29, v82
	v_mov_b32_e32 v30, v82
	v_mov_b32_e32 v31, v82
	v_mov_b32_e32 v32, v82
	v_mov_b32_e32 v33, v82
	v_mov_b32_e32 v112, v82
	v_mov_b32_e32 v113, v82
	v_mov_b32_e32 v110, v82
	v_mov_b32_e32 v111, v82
	v_mov_b32_e32 v108, v82
	v_mov_b32_e32 v109, v82
	v_mov_b32_e32 v106, v82
	v_mov_b32_e32 v107, v82
	v_mov_b32_e32 v104, v82
	v_mov_b32_e32 v105, v82
	v_mov_b32_e32 v102, v82
	v_mov_b32_e32 v103, v82
	v_mov_b32_e32 v100, v82
	v_mov_b32_e32 v101, v82
	v_mov_b32_e32 v98, v82
	v_mov_b32_e32 v99, v82
	v_mov_b32_e32 v174, v82
	v_mov_b32_e32 v118, 0xff800000
.LBB0_52:
	s_add_i32 s40, s13, s64
	s_add_i32 s42, s40, 0x7f
	s_ashr_i32 s40, s42, 31
	s_lshr_b32 s40, s40, 25
	s_add_i32 s40, s42, s40
	s_ashr_i32 s45, s40, 7
	s_and_b32 s40, s40, 0xffffff80
	s_cmp_lg_u32 s42, s40
	s_cselect_b64 s[40:41], -1, 0
	s_cmp_lt_i32 s42, 0
	s_cselect_b64 s[42:43], -1, 0
	s_and_b64 s[40:41], s[42:43], s[40:41]
	s_subb_u32 s40, s45, 0
	s_min_i32 s40, s40, s47
	s_cmp_ge_i32 s38, s40
	s_cbranch_scc1 .LBB0_91
	s_lshl_b32 s42, s65, 8
	s_ashr_i32 s41, s40, 31
	v_mov_b32_e32 v117, v116
	v_mov_b32_e32 v120, v116
	v_mov_b32_e32 v121, v116
	v_mov_b32_e32 v122, v116
	v_mov_b32_e32 v123, v116
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
	v_or_b32_e32 v173, s46, v1
	s_or_b32 s64, s42, 0xf7
	s_lshl3_add_u32 s65, s65, 20
	v_mov_b32_e32 v50, v150
	v_mov_b32_e32 v51, v151
	v_mov_b32_e32 v52, v152
	v_mov_b32_e32 v53, v153
	v_mov_b32_e32 v54, v154
	v_mov_b32_e32 v55, v155
	v_mov_b32_e32 v56, v156
	v_mov_b32_e32 v57, v157
	v_mov_b32_e32 v58, v158
	v_mov_b32_e32 v59, v159
	v_mov_b32_e32 v60, v160
	v_mov_b32_e32 v61, v161
	v_mov_b32_e32 v62, v162
	v_mov_b32_e32 v63, v163
	v_mov_b32_e32 v64, v164
	v_mov_b32_e32 v65, v165
	v_mov_b32_e32 v34, v148
	v_mov_b32_e32 v35, v149
	v_mov_b32_e32 v36, v146
	v_mov_b32_e32 v37, v147
	v_mov_b32_e32 v38, v144
	v_mov_b32_e32 v39, v145
	v_mov_b32_e32 v40, v142
	v_mov_b32_e32 v41, v143
	v_mov_b32_e32 v42, v140
	v_mov_b32_e32 v43, v141
	v_mov_b32_e32 v44, v138
	v_mov_b32_e32 v45, v139
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v47, v137
	v_mov_b32_e32 v48, v134
	v_mov_b32_e32 v49, v135
	v_mov_b32_e32 v2, v112
	v_mov_b32_e32 v3, v113
	v_mov_b32_e32 v4, v110
	v_mov_b32_e32 v5, v111
	v_mov_b32_e32 v6, v108
	v_mov_b32_e32 v7, v109
	v_mov_b32_e32 v8, v106
	v_mov_b32_e32 v9, v107
	v_mov_b32_e32 v10, v104
	v_mov_b32_e32 v11, v105
	v_mov_b32_e32 v12, v102
	v_mov_b32_e32 v13, v103
	v_mov_b32_e32 v14, v100
	v_mov_b32_e32 v15, v101
	v_mov_b32_e32 v16, v98
	v_mov_b32_e32 v17, v99
	s_branch .LBB0_56
.LBB0_54:
	s_or_b64 exec, exec, s[44:45]
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v134, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v134, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s46, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s46
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[134:137], v[106:107], off offset:512
	global_load_dwordx4 v[138:141], v[106:107], off offset:1024
	global_load_dwordx4 v[142:145], v[106:107], off offset:1536
	global_load_dwordx4 v[146:149], v[106:107], off offset:2048
	global_load_dwordx4 v[150:153], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[142:143], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[146:147], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[150:151], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[102:103], v[66:81]
	global_load_dwordx4 v[134:137], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[102:103], v[50:65]
	global_load_dwordx4 v[138:141], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[102:103], v[34:49]
	global_load_dwordx4 v[142:145], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[148:149], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[152:153], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[142:143], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
.LBB0_55:
	s_and_b64 s[42:43], s[42:43], exec
	s_cselect_b32 s44, s60, s62
	s_cselect_b32 s62, s63, s60
	s_cselect_b32 s60, s66, s63
	s_add_u32 s38, s38, 2
	s_addc_u32 s39, s39, 0
	v_mov_b64_e32 v[98:99], s[40:41]
	v_cmp_lt_i64_e32 vcc, s[38:39], v[98:99]
	s_addk_i32 s64, 0x100
	s_add_i32 s65, s65, 8
	s_mov_b32 s63, s67
	s_cbranch_vccz .LBB0_90
.LBB0_56:
	s_add_i32 s42, s65, -4
	v_mov_b32_e32 v98, s42
	s_lshl_b32 s42, s44, 13
	s_ashr_i32 s43, s42, 31
	buffer_load_dword v134, v98, s[4:7], 0 offen
	v_or_b32_e32 v98, s42, v114
	v_mov_b32_e32 v99, s43
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[160:163], v[98:99], off offset:1536
	ds_write_b128 v167, v[208:211]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s66, v134
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v135, 48, v98
	v_and_or_b32 v136, v99, s52, v100
	v_or_b32_e32 v98, v136, v135
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[138:141], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[178:179], v[98:113]
	v_or_b32_e32 v134, 0x1010, v136
	v_xor_b32_e32 v134, v134, v135
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[182:183], v[98:113]
	v_or_b32_e32 v134, 0x1020, v136
	v_xor_b32_e32 v134, v134, v135
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[186:187], v[98:113]
	v_or_b32_e32 v134, 0x1030, v136
	v_xor_b32_e32 v134, v134, v135
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[190:191], v[98:113]
	v_or_b32_e32 v134, 0x1040, v136
	v_lshrrev_b32_e32 v135, 3, v134
	v_and_b32_e32 v135, 56, v135
	v_xor_b32_e32 v134, v135, v134
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[194:195], v[98:113]
	v_or_b32_e32 v134, 0x1050, v136
	v_lshrrev_b32_e32 v135, 3, v134
	v_and_b32_e32 v135, 56, v135
	v_xor_b32_e32 v134, v135, v134
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[198:199], v[98:113]
	v_or_b32_e32 v134, 0x1060, v136
	v_lshrrev_b32_e32 v135, 3, v134
	v_and_b32_e32 v135, 56, v135
	v_xor_b32_e32 v134, v135, v134
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[138:141], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[202:203], v[98:113]
	v_or_b32_e32 v134, 0x1070, v136
	v_lshrrev_b32_e32 v135, 3, v134
	v_and_b32_e32 v135, 56, v135
	v_xor_b32_e32 v134, v135, v134
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v134, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v135, 2, v134
	v_and_b32_e32 v135, 8, v135
	v_lshrrev_b32_e32 v134, 1, v134
	v_and_or_b32 v134, v134, s51, v173
	v_add_u32_e32 v143, s64, v135
	v_add_u32_e32 v142, s61, v134
	v_add_u32_e32 v134, 0xffffff09, v143
	v_cmp_lt_i32_e32 vcc, v134, v142
	s_nop 1
	v_cndmask_b32_e32 v135, v171, v99, vcc
	v_cmp_le_i32_e32 vcc, v134, v142
	v_add_u32_e32 v99, 0xffffff20, v143
	s_nop 0
	v_cndmask_b32_e32 v134, v171, v98, vcc
	v_add_u32_e32 v98, 0xffffff0b, v143
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff0c, v143
	s_nop 0
	v_cndmask_b32_e32 v136, v171, v100, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff0d, v143
	s_nop 0
	v_cndmask_b32_e32 v137, v171, v101, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff0e, v143
	s_nop 0
	v_cndmask_b32_e32 v138, v171, v102, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff0f, v143
	s_nop 0
	v_cndmask_b32_e32 v139, v171, v103, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff10, v143
	s_nop 0
	v_cndmask_b32_e32 v140, v171, v104, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff19, v143
	s_nop 0
	v_cndmask_b32_e32 v141, v171, v105, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff1a, v143
	s_nop 0
	v_cndmask_b32_e32 v104, v171, v106, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff1b, v143
	s_nop 0
	v_cndmask_b32_e32 v105, v171, v107, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff1c, v143
	v_pk_mul_f32 v[106:107], v[124:125], v[140:141]
	v_cndmask_b32_e32 v102, v171, v108, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff1d, v143
	v_pk_mul_f32 v[104:105], v[126:127], v[104:105]
	v_cndmask_b32_e32 v103, v171, v109, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff1e, v143
	v_pk_mul_f32 v[108:109], v[122:123], v[138:139]
	v_cndmask_b32_e32 v100, v171, v110, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_add_u32_e32 v98, 0xffffff1f, v143
	v_pk_mul_f32 v[102:103], v[128:129], v[102:103]
	v_cndmask_b32_e32 v101, v171, v111, vcc
	v_cmp_le_i32_e32 vcc, v98, v142
	v_pk_mul_f32 v[110:111], v[120:121], v[136:137]
	v_pk_mul_f32 v[100:101], v[130:131], v[100:101]
	v_cndmask_b32_e32 v98, v171, v112, vcc
	v_cmp_le_i32_e32 vcc, v99, v142
	s_nop 1
	v_cndmask_b32_e32 v99, v171, v113, vcc
	v_pk_mul_f32 v[112:113], v[116:117], v[134:135]
	v_pk_mul_f32 v[98:99], v[132:133], v[98:99]
	v_max_f32_e32 v134, v112, v113
	v_max3_f32 v134, v134, v110, v111
	v_max3_f32 v134, v134, v108, v109
	v_max3_f32 v134, v134, v106, v107
	v_max3_f32 v134, v134, v104, v105
	v_max3_f32 v134, v134, v102, v103
	v_max3_f32 v134, v134, v100, v101
	v_max3_f32 v134, v134, v98, v99
	ds_bpermute_b32 v135, v168, v134
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v135, v135, v135
	v_max_f32_e32 v135, v134, v135
	;;#ASMSTART
	v_add_f32 v134, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v135, v134
	v_mov_b32_e32 v134, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_58
	;;#ASMSTART
	v_add_f32 v135, v135, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v135
	v_exp_f32_e32 v134, v118
	v_mov_b32_e32 v118, v135
.LBB0_58:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[136:137], v[98:99], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[98:99], v[112:113], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[138:139], v[100:101], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	v_pk_add_f32 v[100:101], v[110:111], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v135, 0, v98
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	v_pk_add_f32 v[140:141], v[102:103], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v135, v135, v99
	v_add_f32_e32 v135, v135, v100
	v_pk_add_f32 v[102:103], v[108:109], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	v_pk_add_f32 v[142:143], v[104:105], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v135, v135, v101
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[106:107], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v135, v135, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v142
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v107, v143
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v140
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v135, v135, v103
	v_add_f32_e32 v135, v135, v104
	v_add_f32_e32 v135, v135, v105
	v_add_f32_e32 v135, v135, v106
	v_add_f32_e32 v135, v135, v107
	v_add_f32_e32 v135, v135, v108
	;;#ASMSTART
	v_exp_f32 v109, v141
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v138
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v111, v139
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v136
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v134
	v_add_f32_e32 v135, v135, v109
	v_add_f32_e32 v135, v135, v110
	v_add_f32_e32 v135, v135, v111
	v_add_f32_e32 v135, v135, v112
	;;#ASMSTART
	v_exp_f32 v113, v137
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v135, v135, v113
	;;#ASMSTART
	v_fma_f32 v150, v174, v134, v135
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_60
	;;#ASMSTART
	v_mul_f32 v82, v82, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v134
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v134
	;;#ASMEND
.LBB0_60:
	s_or_b64 exec, exec, s[42:43]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v134, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v134, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s43, s44, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s43, s43, s37
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s43
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[134:137], v[106:107], off offset:512
	global_load_dwordx4 v[138:141], v[106:107], off offset:1024
	global_load_dwordx4 v[142:145], v[106:107], off offset:1536
	global_load_dwordx4 v[146:149], v[106:107], off offset:2048
	global_load_dwordx4 v[152:155], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[142:143], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[146:147], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[152:153], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[102:103], v[66:81]
	global_load_dwordx4 v[134:137], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[102:103], v[50:65]
	global_load_dwordx4 v[138:141], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[102:103], v[34:49]
	global_load_dwordx4 v[142:145], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[148:149], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[154:155], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[142:143], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[144:145], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	s_lshl_b32 s42, s62, 13
	v_or_b32_e32 v98, s42, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[156:159], v[98:99], off
	ds_write_b128 v167, v[212:215] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v134, v99, s52, v100
	v_and_b32_e32 v135, 48, v98
	v_or_b32_e32 v98, v134, v135
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[136:139], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[178:179], v[98:113]
	v_or_b32_e32 v136, 16, v134
	v_xor_b32_e32 v136, v136, v135
	v_lshlrev_b32_e32 v136, 1, v136
	ds_read_b128 v[136:139], v136
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[182:183], v[98:113]
	v_or_b32_e32 v136, 32, v134
	v_xor_b32_e32 v136, v136, v135
	v_lshlrev_b32_e32 v136, 1, v136
	ds_read_b128 v[136:139], v136
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[186:187], v[98:113]
	v_or_b32_e32 v136, 48, v134
	v_xor_b32_e32 v135, v136, v135
	v_lshlrev_b32_e32 v135, 1, v135
	ds_read_b128 v[136:139], v135
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[190:191], v[98:113]
	v_or_b32_e32 v135, 64, v134
	v_lshrrev_b32_e32 v136, 3, v135
	v_and_b32_e32 v136, 56, v136
	v_xor_b32_e32 v135, v136, v135
	v_lshlrev_b32_e32 v135, 1, v135
	ds_read_b128 v[136:139], v135
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[194:195], v[98:113]
	v_or_b32_e32 v135, 0x50, v134
	v_lshrrev_b32_e32 v136, 3, v135
	v_and_b32_e32 v136, 56, v136
	v_xor_b32_e32 v135, v136, v135
	v_lshlrev_b32_e32 v135, 1, v135
	ds_read_b128 v[136:139], v135
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[198:199], v[98:113]
	v_or_b32_e32 v135, 0x60, v134
	v_lshrrev_b32_e32 v136, 3, v135
	v_and_b32_e32 v136, 56, v136
	v_xor_b32_e32 v135, v136, v135
	v_lshlrev_b32_e32 v135, 1, v135
	ds_read_b128 v[136:139], v135
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[202:203], v[98:113]
	v_or_b32_e32 v134, 0x70, v134
	v_lshrrev_b32_e32 v135, 3, v134
	v_and_b32_e32 v135, 56, v135
	v_xor_b32_e32 v134, v135, v134
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v134, v0
	;;#ASMEND
	v_mov_b32_e32 v151, 1.0
	v_lshrrev_b32_e32 v135, 2, v134
	v_and_b32_e32 v135, 8, v135
	v_lshrrev_b32_e32 v134, 1, v134
	v_and_or_b32 v134, v134, s51, v173
	v_add_u32_e32 v135, s64, v135
	v_add_u32_e32 v134, s61, v134
	v_add_u32_e32 v136, 0xffffff29, v135
	v_cmp_lt_i32_e32 vcc, v136, v134
	v_mov_b32_e32 v137, v118
	v_mov_b32_e32 v138, v118
	v_cndmask_b32_e32 v99, v171, v99, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff2b, v135
	v_mov_b32_e32 v139, v118
	v_cndmask_b32_e32 v98, v171, v98, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff2c, v135
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v171, v100, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff2d, v135
	v_mov_b32_e32 v140, v118
	v_cndmask_b32_e32 v101, v171, v101, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff2e, v135
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_cndmask_b32_e32 v102, v171, v102, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff2f, v135
	v_mov_b32_e32 v141, v118
	v_cndmask_b32_e32 v103, v171, v103, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff30, v135
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_cndmask_b32_e32 v104, v171, v104, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff39, v135
	v_mov_b32_e32 v142, v118
	v_cndmask_b32_e32 v105, v171, v105, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff3a, v135
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_cndmask_b32_e32 v106, v171, v106, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff3b, v135
	v_mov_b32_e32 v143, v118
	v_cndmask_b32_e32 v107, v171, v107, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff3c, v135
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_cndmask_b32_e32 v108, v171, v108, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff3d, v135
	v_mov_b32_e32 v144, v118
	v_cndmask_b32_e32 v109, v171, v109, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff3e, v135
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_cndmask_b32_e32 v110, v171, v110, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_add_u32_e32 v136, 0xffffff3f, v135
	v_add_u32_e32 v135, 0xffffff40, v135
	v_cndmask_b32_e32 v111, v171, v111, vcc
	v_cmp_le_i32_e32 vcc, v136, v134
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	v_mov_b32_e32 v136, v118
	v_cndmask_b32_e32 v112, v171, v112, vcc
	v_cmp_le_i32_e32 vcc, v135, v134
	v_max_f32_e32 v134, v98, v99
	v_max3_f32 v134, v134, v100, v101
	v_max3_f32 v134, v134, v102, v103
	v_max3_f32 v134, v134, v104, v105
	v_max3_f32 v134, v134, v106, v107
	v_cndmask_b32_e32 v113, v171, v113, vcc
	v_max3_f32 v134, v134, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v134, v134, v110, v111
	v_max3_f32 v134, v134, v112, v113
	ds_bpermute_b32 v135, v168, v134
	v_mov_b32_e32 v145, v118
	v_mov_b32_e32 v146, v118
	v_mov_b32_e32 v147, v118
	v_mov_b32_e32 v148, v118
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v135, v135, v135
	v_max_f32_e32 v152, v134, v135
	;;#ASMSTART
	v_add_f32 v134, v118, v169
	;;#ASMEND
	v_mov_b32_e32 v135, v118
	v_cmp_gt_f32_e32 vcc, v152, v134
	v_mov_b32_e32 v134, v118
	v_mov_b32_e32 v149, v118
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_62
	;;#ASMSTART
	v_add_f32 v134, v152, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v151, v118
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
.LBB0_62:
	s_or_b64 exec, exec, s[44:45]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, 0, v98
	v_add_f32_e32 v152, v152, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v100
	v_add_f32_e32 v152, v152, v101
	v_add_f32_e32 v152, v152, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v103
	v_add_f32_e32 v152, v152, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v105
	v_add_f32_e32 v152, v152, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v107
	v_add_f32_e32 v152, v152, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v109
	v_add_f32_e32 v152, v152, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v151
	v_add_f32_e32 v152, v152, v111
	v_add_f32_e32 v152, v152, v112
	v_add_f32_e32 v152, v152, v113
	;;#ASMSTART
	v_fma_f32 v152, v150, v151, v152
	;;#ASMEND
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_64
	;;#ASMSTART
	v_mul_f32 v82, v82, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v151
	;;#ASMEND
.LBB0_64:
	s_or_b64 exec, exec, s[44:45]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v150, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v150, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s44, s43, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s44
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[208:211], v[106:107], off offset:512
	global_load_dwordx4 v[212:215], v[106:107], off offset:1024
	global_load_dwordx4 v[216:219], v[106:107], off offset:1536
	global_load_dwordx4 v[220:223], v[106:107], off offset:2048
	global_load_dwordx4 v[224:227], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[212:213], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[220:221], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[224:225], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[102:103], v[66:81]
	global_load_dwordx4 v[208:211], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[214:215], v[102:103], v[50:65]
	global_load_dwordx4 v[212:215], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[102:103], v[34:49]
	global_load_dwordx4 v[216:219], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[222:223], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[226:227], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[212:213], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[214:215], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	s_ashr_i32 s43, s42, 31
	v_lshl_add_u64 v[98:99], s[42:43], 0, v[114:115]
	v_lshl_add_u64 v[150:151], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[208:211], v[150:151], off offset:512
	ds_write_b128 v167, v[160:163]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v153, v99, s52, v100
	v_and_b32_e32 v154, 48, v98
	v_or_b32_e32 v98, v153, v154
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[160:163], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[178:179], v[98:113]
	v_or_b32_e32 v155, 0x1010, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[160:163], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[182:183], v[98:113]
	v_or_b32_e32 v155, 0x1020, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[160:163], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[186:187], v[98:113]
	v_or_b32_e32 v155, 0x1030, v153
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[160:163], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[190:191], v[98:113]
	v_or_b32_e32 v154, 0x1040, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[160:163], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[194:195], v[98:113]
	v_or_b32_e32 v154, 0x1050, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[160:163], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[198:199], v[98:113]
	v_or_b32_e32 v154, 0x1060, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[160:163], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[202:203], v[98:113]
	v_or_b32_e32 v153, 0x1070, v153
	v_lshrrev_b32_e32 v154, 3, v153
	v_and_b32_e32 v154, 56, v154
	v_xor_b32_e32 v153, v154, v153
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[160:163], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v153, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v154, 2, v153
	v_and_b32_e32 v154, 8, v154
	v_lshrrev_b32_e32 v153, 1, v153
	v_and_or_b32 v153, v153, s51, v173
	v_add_u32_e32 v154, s64, v154
	v_add_u32_e32 v153, s61, v153
	v_add_u32_e32 v155, 0xffffff49, v154
	v_cmp_lt_i32_e32 vcc, v155, v153
	s_nop 1
	v_cndmask_b32_e32 v99, v171, v99, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff4b, v154
	s_nop 0
	v_cndmask_b32_e32 v98, v171, v98, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff4c, v154
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v171, v100, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff4d, v154
	s_nop 0
	v_cndmask_b32_e32 v101, v171, v101, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff4e, v154
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_cndmask_b32_e32 v102, v171, v102, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff4f, v154
	s_nop 0
	v_cndmask_b32_e32 v103, v171, v103, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff50, v154
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_cndmask_b32_e32 v104, v171, v104, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff59, v154
	s_nop 0
	v_cndmask_b32_e32 v105, v171, v105, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff5a, v154
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_cndmask_b32_e32 v106, v171, v106, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff5b, v154
	s_nop 0
	v_cndmask_b32_e32 v107, v171, v107, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff5c, v154
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_cndmask_b32_e32 v108, v171, v108, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff5d, v154
	s_nop 0
	v_cndmask_b32_e32 v109, v171, v109, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff5e, v154
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_cndmask_b32_e32 v110, v171, v110, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff5f, v154
	v_add_u32_e32 v154, 0xffffff60, v154
	v_cndmask_b32_e32 v111, v171, v111, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v171, v112, vcc
	v_cmp_le_i32_e32 vcc, v154, v153
	v_max_f32_e32 v153, v98, v99
	v_max3_f32 v153, v153, v100, v101
	v_max3_f32 v153, v153, v102, v103
	v_max3_f32 v153, v153, v104, v105
	v_max3_f32 v153, v153, v106, v107
	v_cndmask_b32_e32 v113, v171, v113, vcc
	v_max3_f32 v153, v153, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v153, v153, v110, v111
	v_max3_f32 v153, v153, v112, v113
	ds_bpermute_b32 v154, v168, v153
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v154, v154, v154
	v_max_f32_e32 v154, v153, v154
	;;#ASMSTART
	v_add_f32 v153, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v154, v153
	v_mov_b32_e32 v153, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_66
	;;#ASMSTART
	v_add_f32 v134, v154, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v153, v118
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
.LBB0_66:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, 0, v98
	v_add_f32_e32 v154, v154, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v100
	v_add_f32_e32 v154, v154, v101
	v_add_f32_e32 v154, v154, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v103
	v_add_f32_e32 v154, v154, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v105
	v_add_f32_e32 v154, v154, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v107
	v_add_f32_e32 v154, v154, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v109
	v_add_f32_e32 v154, v154, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v153
	v_add_f32_e32 v154, v154, v111
	v_add_f32_e32 v154, v154, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v154, v154, v113
	;;#ASMSTART
	v_fma_f32 v152, v152, v153, v154
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_68
	;;#ASMSTART
	v_mul_f32 v82, v82, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v153
	;;#ASMEND
.LBB0_68:
	s_or_b64 exec, exec, s[42:43]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v153, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v153, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s42, s44, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s42
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[160:163], v[106:107], off offset:512
	global_load_dwordx4 v[212:215], v[106:107], off offset:1024
	global_load_dwordx4 v[216:219], v[106:107], off offset:1536
	global_load_dwordx4 v[220:223], v[106:107], off offset:2048
	global_load_dwordx4 v[224:227], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[212:213], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[220:221], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[224:225], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[102:103], v[66:81]
	global_load_dwordx4 v[160:163], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[214:215], v[102:103], v[50:65]
	global_load_dwordx4 v[212:215], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[102:103], v[34:49]
	global_load_dwordx4 v[216:219], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[222:223], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[226:227], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[212:213], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[214:215], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	global_load_dwordx4 v[212:215], v[150:151], off offset:1024
	ds_write_b128 v167, v[156:159] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v153, v99, s52, v100
	v_and_b32_e32 v154, 48, v98
	v_or_b32_e32 v98, v153, v154
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[156:159], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[178:179], v[98:113]
	v_or_b32_e32 v155, 16, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[156:159], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[182:183], v[98:113]
	v_or_b32_e32 v155, 32, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[156:159], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[186:187], v[98:113]
	v_or_b32_e32 v155, 48, v153
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[190:191], v[98:113]
	v_or_b32_e32 v154, 64, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[194:195], v[98:113]
	v_or_b32_e32 v154, 0x50, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[198:199], v[98:113]
	v_or_b32_e32 v154, 0x60, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[202:203], v[98:113]
	v_or_b32_e32 v153, 0x70, v153
	v_lshrrev_b32_e32 v154, 3, v153
	v_and_b32_e32 v154, 56, v154
	v_xor_b32_e32 v153, v154, v153
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[154:157], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v153, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v154, 2, v153
	v_and_b32_e32 v154, 8, v154
	v_lshrrev_b32_e32 v153, 1, v153
	v_and_or_b32 v153, v153, s51, v173
	v_add_u32_e32 v154, s64, v154
	v_add_u32_e32 v153, s61, v153
	v_add_u32_e32 v155, 0xffffff69, v154
	v_cmp_lt_i32_e32 vcc, v155, v153
	s_nop 1
	v_cndmask_b32_e32 v99, v171, v99, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff6b, v154
	s_nop 0
	v_cndmask_b32_e32 v98, v171, v98, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff6c, v154
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v171, v100, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff6d, v154
	s_nop 0
	v_cndmask_b32_e32 v101, v171, v101, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff6e, v154
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_cndmask_b32_e32 v102, v171, v102, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff6f, v154
	s_nop 0
	v_cndmask_b32_e32 v103, v171, v103, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff70, v154
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_cndmask_b32_e32 v104, v171, v104, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff79, v154
	s_nop 0
	v_cndmask_b32_e32 v105, v171, v105, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff7a, v154
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_cndmask_b32_e32 v106, v171, v106, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff7b, v154
	s_nop 0
	v_cndmask_b32_e32 v107, v171, v107, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff7c, v154
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_cndmask_b32_e32 v108, v171, v108, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff7d, v154
	s_nop 0
	v_cndmask_b32_e32 v109, v171, v109, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff7e, v154
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_cndmask_b32_e32 v110, v171, v110, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_add_u32_e32 v155, 0xffffff7f, v154
	v_add_u32_e32 v154, 0xffffff80, v154
	v_cndmask_b32_e32 v111, v171, v111, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v171, v112, vcc
	v_cmp_le_i32_e32 vcc, v154, v153
	v_max_f32_e32 v153, v98, v99
	v_max3_f32 v153, v153, v100, v101
	v_max3_f32 v153, v153, v102, v103
	v_max3_f32 v153, v153, v104, v105
	v_max3_f32 v153, v153, v106, v107
	v_cndmask_b32_e32 v113, v171, v113, vcc
	v_max3_f32 v153, v153, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v153, v153, v110, v111
	v_max3_f32 v153, v153, v112, v113
	ds_bpermute_b32 v154, v168, v153
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v154, v154, v154
	v_max_f32_e32 v154, v153, v154
	;;#ASMSTART
	v_add_f32 v153, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v154, v153
	v_mov_b32_e32 v153, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_70
	;;#ASMSTART
	v_add_f32 v134, v154, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v153, v118
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
.LBB0_70:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, 0, v98
	v_add_f32_e32 v154, v154, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v100
	v_add_f32_e32 v154, v154, v101
	v_add_f32_e32 v154, v154, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v103
	v_add_f32_e32 v154, v154, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v105
	v_add_f32_e32 v154, v154, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v107
	v_add_f32_e32 v154, v154, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v109
	v_add_f32_e32 v154, v154, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v153
	v_add_f32_e32 v154, v154, v111
	v_add_f32_e32 v154, v154, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v154, v154, v113
	;;#ASMSTART
	v_fma_f32 v174, v152, v153, v154
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_72
	;;#ASMSTART
	v_mul_f32 v82, v82, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v153
	;;#ASMEND
.LBB0_72:
	s_or_b64 exec, exec, s[42:43]
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v152, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v152, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s44, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s44
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[152:155], v[106:107], off offset:512
	global_load_dwordx4 v[156:159], v[106:107], off offset:1024
	global_load_dwordx4 v[160:163], v[106:107], off offset:1536
	global_load_dwordx4 v[216:219], v[106:107], off offset:2048
	global_load_dwordx4 v[220:223], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[152:153], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[156:157], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[160:161], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[216:217], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[220:221], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[154:155], v[102:103], v[66:81]
	global_load_dwordx4 v[152:155], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[158:159], v[102:103], v[50:65]
	global_load_dwordx4 v[156:159], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[162:163], v[102:103], v[34:49]
	global_load_dwordx4 v[160:163], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[218:219], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[222:223], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[152:153], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[156:157], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[160:161], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[154:155], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[158:159], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[162:163], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	s_add_i32 s44, s38, 1
	s_cmp_gt_i32 s40, s44
	s_cselect_b64 s[42:43], -1, 0
	s_cmp_le_i32 s40, s44
	s_cbranch_scc1 .LBB0_89
	v_mov_b32_e32 v98, s65
	buffer_load_dword v152, v98, s[4:7], 0 offen
	global_load_dwordx4 v[154:157], v[150:151], off offset:1536
	ds_write_b128 v167, v[208:211]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s67, v152
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v150, 48, v98
	v_and_or_b32 v151, v99, s52, v100
	v_or_b32_e32 v98, v151, v150
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[158:161], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[178:179], v[98:113]
	v_or_b32_e32 v152, 0x1010, v151
	v_xor_b32_e32 v152, v152, v150
	v_lshlrev_b32_e32 v152, 1, v152
	ds_read_b128 v[158:161], v152
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[182:183], v[98:113]
	v_or_b32_e32 v152, 0x1020, v151
	v_xor_b32_e32 v152, v152, v150
	v_lshlrev_b32_e32 v152, 1, v152
	ds_read_b128 v[158:161], v152
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[186:187], v[98:113]
	v_or_b32_e32 v152, 0x1030, v151
	v_xor_b32_e32 v150, v152, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[158:161], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[190:191], v[98:113]
	v_or_b32_e32 v150, 0x1040, v151
	v_lshrrev_b32_e32 v152, 3, v150
	v_and_b32_e32 v152, 56, v152
	v_xor_b32_e32 v150, v152, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[158:161], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[194:195], v[98:113]
	v_or_b32_e32 v150, 0x1050, v151
	v_lshrrev_b32_e32 v152, 3, v150
	v_and_b32_e32 v152, 56, v152
	v_xor_b32_e32 v150, v152, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[158:161], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[198:199], v[98:113]
	v_or_b32_e32 v150, 0x1060, v151
	v_lshrrev_b32_e32 v152, 3, v150
	v_and_b32_e32 v152, 56, v152
	v_xor_b32_e32 v150, v152, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[158:161], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[158:159], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[160:161], v[202:203], v[98:113]
	v_or_b32_e32 v150, 0x1070, v151
	v_lshrrev_b32_e32 v151, 3, v150
	v_and_b32_e32 v151, 56, v151
	v_xor_b32_e32 v150, v151, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[150:153], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[152:153], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v150, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v151, 2, v150
	v_and_b32_e32 v151, 8, v151
	v_lshrrev_b32_e32 v150, 1, v150
	v_and_or_b32 v150, v150, s51, v173
	v_add_u32_e32 v151, s64, v151
	v_add_u32_e32 v150, s61, v150
	v_add_u32_e32 v152, 0xffffff89, v151
	v_cmp_lt_i32_e32 vcc, v152, v150
	s_nop 1
	v_cndmask_b32_e32 v99, v171, v99, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff8b, v151
	s_nop 0
	v_cndmask_b32_e32 v98, v171, v98, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff8c, v151
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v171, v100, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff8d, v151
	s_nop 0
	v_cndmask_b32_e32 v101, v171, v101, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff8e, v151
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_cndmask_b32_e32 v102, v171, v102, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff8f, v151
	s_nop 0
	v_cndmask_b32_e32 v103, v171, v103, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff90, v151
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_cndmask_b32_e32 v104, v171, v104, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff99, v151
	s_nop 0
	v_cndmask_b32_e32 v105, v171, v105, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff9a, v151
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_cndmask_b32_e32 v106, v171, v106, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff9b, v151
	s_nop 0
	v_cndmask_b32_e32 v107, v171, v107, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff9c, v151
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_cndmask_b32_e32 v108, v171, v108, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff9d, v151
	s_nop 0
	v_cndmask_b32_e32 v109, v171, v109, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff9e, v151
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_cndmask_b32_e32 v110, v171, v110, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_add_u32_e32 v152, 0xffffff9f, v151
	v_add_u32_e32 v151, 0xffffffa0, v151
	v_cndmask_b32_e32 v111, v171, v111, vcc
	v_cmp_le_i32_e32 vcc, v152, v150
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v171, v112, vcc
	v_cmp_le_i32_e32 vcc, v151, v150
	v_max_f32_e32 v150, v98, v99
	v_max3_f32 v150, v150, v100, v101
	v_max3_f32 v150, v150, v102, v103
	v_max3_f32 v150, v150, v104, v105
	v_max3_f32 v150, v150, v106, v107
	v_cndmask_b32_e32 v113, v171, v113, vcc
	v_max3_f32 v150, v150, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v150, v150, v110, v111
	v_max3_f32 v150, v150, v112, v113
	ds_bpermute_b32 v151, v168, v150
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v151, v151, v151
	v_max_f32_e32 v150, v150, v151
	;;#ASMSTART
	v_add_f32 v151, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v150, v151
	v_mov_b32_e32 v151, 1.0
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_75
	;;#ASMSTART
	v_add_f32 v134, v150, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v151, v118
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
.LBB0_75:
	s_or_b64 exec, exec, s[44:45]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, 0, v98
	v_add_f32_e32 v150, v150, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, v150, v100
	v_add_f32_e32 v150, v150, v101
	v_add_f32_e32 v150, v150, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, v150, v103
	v_add_f32_e32 v150, v150, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, v150, v105
	v_add_f32_e32 v150, v150, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, v150, v107
	v_add_f32_e32 v150, v150, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v150, v150, v109
	v_add_f32_e32 v150, v150, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v151
	v_add_f32_e32 v150, v150, v111
	v_add_f32_e32 v150, v150, v112
	v_add_f32_e32 v150, v150, v113
	;;#ASMSTART
	v_fma_f32 v150, v174, v151, v150
	;;#ASMEND
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_77
	;;#ASMSTART
	v_mul_f32 v82, v82, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v151
	;;#ASMEND
.LBB0_77:
	s_or_b64 exec, exec, s[44:45]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v151, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v151, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s45, s62, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s45, s45, s37
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s45
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[158:161], v[106:107], off offset:512
	global_load_dwordx4 v[162:165], v[106:107], off offset:1024
	global_load_dwordx4 v[208:211], v[106:107], off offset:1536
	global_load_dwordx4 v[216:219], v[106:107], off offset:2048
	global_load_dwordx4 v[220:223], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[158:159], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[162:163], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[208:209], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[216:217], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[220:221], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[102:103], v[66:81]
	global_load_dwordx4 v[158:161], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[164:165], v[102:103], v[50:65]
	global_load_dwordx4 v[162:165], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[210:211], v[102:103], v[34:49]
	global_load_dwordx4 v[208:211], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[218:219], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[222:223], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[158:159], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[162:163], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[208:209], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[160:161], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[164:165], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[210:211], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	s_lshl_b32 s44, s60, 13
	v_or_b32_e32 v98, s44, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[158:161], v[98:99], off
	ds_write_b128 v167, v[212:215] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v151, v99, s52, v100
	v_and_b32_e32 v152, 48, v98
	v_or_b32_e32 v98, v151, v152
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[162:165], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[178:179], v[98:113]
	v_or_b32_e32 v153, 16, v151
	v_xor_b32_e32 v153, v153, v152
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[162:165], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[182:183], v[98:113]
	v_or_b32_e32 v153, 32, v151
	v_xor_b32_e32 v153, v153, v152
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[162:165], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[186:187], v[98:113]
	v_or_b32_e32 v153, 48, v151
	v_xor_b32_e32 v152, v153, v152
	v_lshlrev_b32_e32 v152, 1, v152
	ds_read_b128 v[162:165], v152
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[190:191], v[98:113]
	v_or_b32_e32 v152, 64, v151
	v_lshrrev_b32_e32 v153, 3, v152
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v152, v153, v152
	v_lshlrev_b32_e32 v152, 1, v152
	ds_read_b128 v[162:165], v152
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[194:195], v[98:113]
	v_or_b32_e32 v152, 0x50, v151
	v_lshrrev_b32_e32 v153, 3, v152
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v152, v153, v152
	v_lshlrev_b32_e32 v152, 1, v152
	ds_read_b128 v[162:165], v152
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[198:199], v[98:113]
	v_or_b32_e32 v152, 0x60, v151
	v_lshrrev_b32_e32 v153, 3, v152
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v152, v153, v152
	v_lshlrev_b32_e32 v152, 1, v152
	ds_read_b128 v[162:165], v152
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[202:203], v[98:113]
	v_or_b32_e32 v151, 0x70, v151
	v_lshrrev_b32_e32 v152, 3, v151
	v_and_b32_e32 v152, 56, v152
	v_xor_b32_e32 v151, v152, v151
	v_lshlrev_b32_e32 v151, 1, v151
	ds_read_b128 v[162:165], v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v151, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v152, 2, v151
	v_and_b32_e32 v152, 8, v152
	v_lshrrev_b32_e32 v151, 1, v151
	v_and_or_b32 v151, v151, s51, v173
	v_add_u32_e32 v152, s64, v152
	v_add_u32_e32 v151, s61, v151
	v_add_u32_e32 v153, 0xffffffa9, v152
	v_cmp_lt_i32_e32 vcc, v153, v151
	s_nop 1
	v_cndmask_b32_e32 v99, v171, v99, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffab, v152
	s_nop 0
	v_cndmask_b32_e32 v98, v171, v98, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffac, v152
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v171, v100, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffad, v152
	s_nop 0
	v_cndmask_b32_e32 v101, v171, v101, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffae, v152
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_cndmask_b32_e32 v102, v171, v102, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffaf, v152
	s_nop 0
	v_cndmask_b32_e32 v103, v171, v103, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffb0, v152
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_cndmask_b32_e32 v104, v171, v104, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffb9, v152
	s_nop 0
	v_cndmask_b32_e32 v105, v171, v105, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffba, v152
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_cndmask_b32_e32 v106, v171, v106, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffbb, v152
	s_nop 0
	v_cndmask_b32_e32 v107, v171, v107, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffbc, v152
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_cndmask_b32_e32 v108, v171, v108, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffbd, v152
	s_nop 0
	v_cndmask_b32_e32 v109, v171, v109, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffbe, v152
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_cndmask_b32_e32 v110, v171, v110, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_add_u32_e32 v153, 0xffffffbf, v152
	v_subrev_u32_e32 v152, 64, v152
	v_cndmask_b32_e32 v111, v171, v111, vcc
	v_cmp_le_i32_e32 vcc, v153, v151
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v171, v112, vcc
	v_cmp_le_i32_e32 vcc, v152, v151
	v_max_f32_e32 v151, v98, v99
	v_max3_f32 v151, v151, v100, v101
	v_max3_f32 v151, v151, v102, v103
	v_max3_f32 v151, v151, v104, v105
	v_max3_f32 v151, v151, v106, v107
	v_cndmask_b32_e32 v113, v171, v113, vcc
	v_max3_f32 v151, v151, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v151, v151, v110, v111
	v_max3_f32 v151, v151, v112, v113
	ds_bpermute_b32 v152, v168, v151
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v152, v152, v152
	v_max_f32_e32 v152, v151, v152
	;;#ASMSTART
	v_add_f32 v151, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v152, v151
	v_mov_b32_e32 v151, 1.0
	s_and_saveexec_b64 s[46:47], vcc
	s_cbranch_execz .LBB0_79
	;;#ASMSTART
	v_add_f32 v134, v152, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v151, v118
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
.LBB0_79:
	s_or_b64 exec, exec, s[46:47]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, 0, v98
	v_add_f32_e32 v152, v152, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v100
	v_add_f32_e32 v152, v152, v101
	v_add_f32_e32 v152, v152, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v103
	v_add_f32_e32 v152, v152, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v105
	v_add_f32_e32 v152, v152, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v107
	v_add_f32_e32 v152, v152, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v152, v152, v109
	v_add_f32_e32 v152, v152, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v151
	v_add_f32_e32 v152, v152, v111
	v_add_f32_e32 v152, v152, v112
	v_add_f32_e32 v152, v152, v113
	;;#ASMSTART
	v_fma_f32 v152, v150, v151, v152
	;;#ASMEND
	s_and_saveexec_b64 s[46:47], vcc
	s_cbranch_execz .LBB0_81
	;;#ASMSTART
	v_mul_f32 v82, v82, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v151
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v151
	;;#ASMEND
.LBB0_81:
	s_or_b64 exec, exec, s[46:47]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v150, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v150, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s46, s45, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s46
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[162:165], v[106:107], off offset:512
	global_load_dwordx4 v[208:211], v[106:107], off offset:1024
	global_load_dwordx4 v[212:215], v[106:107], off offset:1536
	global_load_dwordx4 v[216:219], v[106:107], off offset:2048
	global_load_dwordx4 v[220:223], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[208:209], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[216:217], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[220:221], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[102:103], v[66:81]
	global_load_dwordx4 v[162:165], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[210:211], v[102:103], v[50:65]
	global_load_dwordx4 v[208:211], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[102:103], v[34:49]
	global_load_dwordx4 v[212:215], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[218:219], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[222:223], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[162:163], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[208:209], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[164:165], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[210:211], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	s_ashr_i32 s45, s44, 31
	v_lshl_add_u64 v[98:99], s[44:45], 0, v[114:115]
	v_lshl_add_u64 v[150:151], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[208:211], v[150:151], off offset:512
	ds_write_b128 v167, v[154:157]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v153, v99, s52, v100
	v_and_b32_e32 v154, 48, v98
	v_or_b32_e32 v98, v153, v154
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[162:165], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[178:179], v[98:113]
	v_or_b32_e32 v155, 0x1010, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[162:165], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[182:183], v[98:113]
	v_or_b32_e32 v155, 0x1020, v153
	v_xor_b32_e32 v155, v155, v154
	v_lshlrev_b32_e32 v155, 1, v155
	ds_read_b128 v[162:165], v155
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[162:163], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[164:165], v[186:187], v[98:113]
	v_or_b32_e32 v155, 0x1030, v153
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[190:191], v[98:113]
	v_or_b32_e32 v154, 0x1040, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[194:195], v[98:113]
	v_or_b32_e32 v154, 0x1050, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[198:199], v[98:113]
	v_or_b32_e32 v154, 0x1060, v153
	v_lshrrev_b32_e32 v155, 3, v154
	v_and_b32_e32 v155, 56, v155
	v_xor_b32_e32 v154, v155, v154
	v_lshlrev_b32_e32 v154, 1, v154
	ds_read_b128 v[154:157], v154
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[202:203], v[98:113]
	v_or_b32_e32 v153, 0x1070, v153
	v_lshrrev_b32_e32 v154, 3, v153
	v_and_b32_e32 v154, 56, v154
	v_xor_b32_e32 v153, v154, v153
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[154:157], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v153, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v154, 2, v153
	v_and_b32_e32 v154, 8, v154
	v_lshrrev_b32_e32 v153, 1, v153
	v_and_or_b32 v153, v153, s51, v173
	v_add_u32_e32 v154, s64, v154
	v_add_u32_e32 v153, s61, v153
	v_subrev_u32_e32 v155, 55, v154
	v_cmp_lt_i32_e32 vcc, v155, v153
	s_nop 1
	v_cndmask_b32_e32 v99, v171, v99, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 53, v154
	s_nop 0
	v_cndmask_b32_e32 v98, v171, v98, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 52, v154
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v171, v100, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 51, v154
	s_nop 0
	v_cndmask_b32_e32 v101, v171, v101, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 50, v154
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_cndmask_b32_e32 v102, v171, v102, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 49, v154
	s_nop 0
	v_cndmask_b32_e32 v103, v171, v103, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 48, v154
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_cndmask_b32_e32 v104, v171, v104, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 39, v154
	s_nop 0
	v_cndmask_b32_e32 v105, v171, v105, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 38, v154
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_cndmask_b32_e32 v106, v171, v106, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 37, v154
	s_nop 0
	v_cndmask_b32_e32 v107, v171, v107, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 36, v154
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_cndmask_b32_e32 v108, v171, v108, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 35, v154
	s_nop 0
	v_cndmask_b32_e32 v109, v171, v109, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 34, v154
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_cndmask_b32_e32 v110, v171, v110, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_subrev_u32_e32 v155, 33, v154
	v_subrev_u32_e32 v154, 32, v154
	v_cndmask_b32_e32 v111, v171, v111, vcc
	v_cmp_le_i32_e32 vcc, v155, v153
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v171, v112, vcc
	v_cmp_le_i32_e32 vcc, v154, v153
	v_max_f32_e32 v153, v98, v99
	v_max3_f32 v153, v153, v100, v101
	v_max3_f32 v153, v153, v102, v103
	v_max3_f32 v153, v153, v104, v105
	v_max3_f32 v153, v153, v106, v107
	v_cndmask_b32_e32 v113, v171, v113, vcc
	v_max3_f32 v153, v153, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v153, v153, v110, v111
	v_max3_f32 v153, v153, v112, v113
	ds_bpermute_b32 v154, v168, v153
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v154, v154, v154
	v_max_f32_e32 v154, v153, v154
	;;#ASMSTART
	v_add_f32 v153, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v154, v153
	v_mov_b32_e32 v153, 1.0
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_83
	;;#ASMSTART
	v_add_f32 v134, v154, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v153, v118
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
.LBB0_83:
	s_or_b64 exec, exec, s[44:45]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, 0, v98
	v_add_f32_e32 v154, v154, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v100
	v_add_f32_e32 v154, v154, v101
	v_add_f32_e32 v154, v154, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v103
	v_add_f32_e32 v154, v154, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v105
	v_add_f32_e32 v154, v154, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v107
	v_add_f32_e32 v154, v154, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v154, v154, v109
	v_add_f32_e32 v154, v154, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v153
	v_add_f32_e32 v154, v154, v111
	v_add_f32_e32 v154, v154, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v154, v154, v113
	;;#ASMSTART
	v_fma_f32 v152, v152, v153, v154
	;;#ASMEND
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_85
	;;#ASMSTART
	v_mul_f32 v82, v82, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v153
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v153
	;;#ASMEND
.LBB0_85:
	s_or_b64 exec, exec, s[44:45]
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_u32_e32 v153, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v153, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s53
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s44, s46, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s44
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[18:19]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[154:157], v[106:107], off offset:512
	global_load_dwordx4 v[162:165], v[106:107], off offset:1024
	global_load_dwordx4 v[212:215], v[106:107], off offset:1536
	global_load_dwordx4 v[216:219], v[106:107], off offset:2048
	global_load_dwordx4 v[220:223], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[154:155], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[162:163], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[216:217], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[220:221], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s54, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[156:157], v[102:103], v[66:81]
	global_load_dwordx4 v[154:157], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[164:165], v[102:103], v[50:65]
	global_load_dwordx4 v[162:165], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[102:103], v[34:49]
	global_load_dwordx4 v[212:215], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s55, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[218:219], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[222:223], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[154:155], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[162:163], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[212:213], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[156:157], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[164:165], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[214:215], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	global_load_dwordx4 v[212:215], v[150:151], off offset:1024
	ds_write_b128 v167, v[158:161] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v150, v99, s52, v100
	v_and_b32_e32 v151, 48, v98
	v_or_b32_e32 v98, v150, v151
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[154:157], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[178:179], v[98:113]
	v_or_b32_e32 v153, 16, v150
	v_xor_b32_e32 v153, v153, v151
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[154:157], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[182:183], v[98:113]
	v_or_b32_e32 v153, 32, v150
	v_xor_b32_e32 v153, v153, v151
	v_lshlrev_b32_e32 v153, 1, v153
	ds_read_b128 v[154:157], v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[186:187], v[98:113]
	v_or_b32_e32 v153, 48, v150
	v_xor_b32_e32 v151, v153, v151
	v_lshlrev_b32_e32 v151, 1, v151
	ds_read_b128 v[154:157], v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[190:191], v[98:113]
	v_or_b32_e32 v151, 64, v150
	v_lshrrev_b32_e32 v153, 3, v151
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v151, v153, v151
	v_lshlrev_b32_e32 v151, 1, v151
	ds_read_b128 v[154:157], v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[194:195], v[98:113]
	v_or_b32_e32 v151, 0x50, v150
	v_lshrrev_b32_e32 v153, 3, v151
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v151, v153, v151
	v_lshlrev_b32_e32 v151, 1, v151
	ds_read_b128 v[154:157], v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[198:199], v[98:113]
	v_or_b32_e32 v151, 0x60, v150
	v_lshrrev_b32_e32 v153, 3, v151
	v_and_b32_e32 v153, 56, v153
	v_xor_b32_e32 v151, v153, v151
	v_lshlrev_b32_e32 v151, 1, v151
	ds_read_b128 v[154:157], v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[202:203], v[98:113]
	v_or_b32_e32 v150, 0x70, v150
	v_lshrrev_b32_e32 v151, 3, v150
	v_and_b32_e32 v151, 56, v151
	v_xor_b32_e32 v150, v151, v150
	v_lshlrev_b32_e32 v150, 1, v150
	ds_read_b128 v[154:157], v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[154:155], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[156:157], v[206:207], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v150, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v151, 2, v150
	v_and_b32_e32 v151, 8, v151
	v_lshrrev_b32_e32 v150, 1, v150
	v_and_or_b32 v150, v150, s51, v173
	v_add_u32_e32 v151, s64, v151
	v_add_u32_e32 v150, s61, v150
	v_subrev_u32_e32 v153, 23, v151
	v_cmp_lt_i32_e32 vcc, v153, v150
	s_nop 1
	v_cndmask_b32_e32 v99, v171, v99, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_subrev_u32_e32 v153, 21, v151
	s_nop 0
	v_cndmask_b32_e32 v98, v171, v98, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_subrev_u32_e32 v153, 20, v151
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v171, v100, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_subrev_u32_e32 v153, 19, v151
	s_nop 0
	v_cndmask_b32_e32 v101, v171, v101, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_subrev_u32_e32 v153, 18, v151
	v_pk_mul_f32 v[100:101], v[120:121], v[100:101]
	v_cndmask_b32_e32 v102, v171, v102, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_subrev_u32_e32 v153, 17, v151
	s_nop 0
	v_cndmask_b32_e32 v103, v171, v103, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_add_u32_e32 v153, -16, v151
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_cndmask_b32_e32 v104, v171, v104, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_add_u32_e32 v153, -7, v151
	s_nop 0
	v_cndmask_b32_e32 v105, v171, v105, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_add_u32_e32 v153, -6, v151
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_cndmask_b32_e32 v106, v171, v106, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_add_u32_e32 v153, -5, v151
	s_nop 0
	v_cndmask_b32_e32 v107, v171, v107, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_add_u32_e32 v153, -4, v151
	v_pk_mul_f32 v[106:107], v[126:127], v[106:107]
	v_cndmask_b32_e32 v108, v171, v108, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_add_u32_e32 v153, -3, v151
	s_nop 0
	v_cndmask_b32_e32 v109, v171, v109, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_add_u32_e32 v153, -2, v151
	v_pk_mul_f32 v[108:109], v[128:129], v[108:109]
	v_cndmask_b32_e32 v110, v171, v110, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_add_u32_e32 v153, -1, v151
	s_nop 0
	v_cndmask_b32_e32 v111, v171, v111, vcc
	v_cmp_le_i32_e32 vcc, v153, v150
	v_pk_mul_f32 v[110:111], v[130:131], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v171, v112, vcc
	v_cmp_le_i32_e32 vcc, v151, v150
	v_max_f32_e32 v150, v98, v99
	v_max3_f32 v150, v150, v100, v101
	v_max3_f32 v150, v150, v102, v103
	v_max3_f32 v150, v150, v104, v105
	v_max3_f32 v150, v150, v106, v107
	v_cndmask_b32_e32 v113, v171, v113, vcc
	v_max3_f32 v150, v150, v108, v109
	v_pk_mul_f32 v[112:113], v[132:133], v[112:113]
	v_max3_f32 v150, v150, v110, v111
	v_max3_f32 v150, v150, v112, v113
	ds_bpermute_b32 v151, v168, v150
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v151, v151, v151
	v_max_f32_e32 v151, v150, v151
	;;#ASMSTART
	v_add_f32 v150, v118, v169
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v151, v150
	v_mov_b32_e32 v150, 1.0
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_87
	;;#ASMSTART
	v_add_f32 v134, v151, v170
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v134
	v_exp_f32_e32 v150, v118
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
.LBB0_87:
	s_or_b64 exec, exec, s[44:45]
	v_pk_add_f32 v[98:99], v[98:99], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, 0, v98
	v_add_f32_e32 v134, v134, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, v134, v100
	v_add_f32_e32 v134, v134, v101
	v_add_f32_e32 v134, v134, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, v134, v103
	v_add_f32_e32 v134, v134, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, v134, v105
	v_add_f32_e32 v134, v134, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, v134, v107
	v_add_f32_e32 v134, v134, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[148:149] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v134, v134, v109
	v_add_f32_e32 v134, v134, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v150
	v_add_f32_e32 v134, v134, v111
	v_add_f32_e32 v134, v134, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v134, v134, v113
	;;#ASMSTART
	v_fma_f32 v174, v152, v150, v134
	;;#ASMEND
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_54
	;;#ASMSTART
	v_mul_f32 v82, v82, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v150
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v150
	;;#ASMEND
	s_branch .LBB0_54
.LBB0_89:
	s_mov_b32 s67, s66
	s_branch .LBB0_55
.LBB0_90:
	v_mov_b32_e32 v150, v50
	v_mov_b32_e32 v151, v51
	v_mov_b32_e32 v152, v52
	v_mov_b32_e32 v153, v53
	v_mov_b32_e32 v154, v54
	v_mov_b32_e32 v155, v55
	v_mov_b32_e32 v156, v56
	v_mov_b32_e32 v157, v57
	v_mov_b32_e32 v158, v58
	v_mov_b32_e32 v159, v59
	v_mov_b32_e32 v160, v60
	v_mov_b32_e32 v161, v61
	v_mov_b32_e32 v162, v62
	v_mov_b32_e32 v163, v63
	v_mov_b32_e32 v164, v64
	v_mov_b32_e32 v165, v65
	v_mov_b32_e32 v148, v34
	v_mov_b32_e32 v149, v35
	v_mov_b32_e32 v146, v36
	v_mov_b32_e32 v147, v37
	v_mov_b32_e32 v144, v38
	v_mov_b32_e32 v145, v39
	v_mov_b32_e32 v142, v40
	v_mov_b32_e32 v143, v41
	v_mov_b32_e32 v140, v42
	v_mov_b32_e32 v141, v43
	v_mov_b32_e32 v138, v44
	v_mov_b32_e32 v139, v45
	v_mov_b32_e32 v136, v46
	v_mov_b32_e32 v137, v47
	v_mov_b32_e32 v134, v48
	v_mov_b32_e32 v135, v49
	v_mov_b32_e32 v112, v2
	v_mov_b32_e32 v113, v3
	v_mov_b32_e32 v110, v4
	v_mov_b32_e32 v111, v5
	v_mov_b32_e32 v108, v6
	v_mov_b32_e32 v109, v7
	v_mov_b32_e32 v106, v8
	v_mov_b32_e32 v107, v9
	v_mov_b32_e32 v104, v10
	v_mov_b32_e32 v105, v11
	v_mov_b32_e32 v102, v12
	v_mov_b32_e32 v103, v13
	v_mov_b32_e32 v100, v14
	v_mov_b32_e32 v101, v15
	v_mov_b32_e32 v98, v16
	v_mov_b32_e32 v99, v17
.LBB0_91:
	;;#ASMSTART
	v_mov_b32 v4, v0
	;;#ASMEND
	ds_bpermute_b32 v2, v168, v174
	v_lshrrev_b32_e32 v3, 1, v4
	v_and_b32_e32 v5, 31, v4
	v_and_or_b32 v3, v3, s51, v5
	v_and_b32_e32 v4, 32, v4
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_cmp_gt_i32_e64 s[0:1], s13, v3
	s_and_b64 s[4:5], vcc, s[0:1]
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v2, v174, v2
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_93
	v_cmp_gt_f32_e32 vcc, s56, v2
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[4:5], s[36:37], 6
	v_cndmask_b32_e64 v5, 0, 32, vcc
	v_ldexp_f32 v5, v2, v5
	v_log_f32_e32 v5, v5
	v_cndmask_b32_e32 v4, 0, v172, vcc
	s_add_u32 s6, s30, s4
	s_addc_u32 s37, s31, s5
	v_sub_f32_e32 v4, v5, v4
	v_add_f32_e32 v4, v118, v4
	s_lshl_b64 s[4:5], s[22:23], 2
	v_mul_f32_e32 v4, 0x3f317218, v4
	v_cmp_lt_f32_e32 vcc, 0, v2
	s_add_u32 s4, s6, s4
	s_addc_u32 s5, s37, s5
	v_cndmask_b32_e32 v4, v171, v4, vcc
	v_lshlrev_b32_e32 v3, 6, v3
	global_store_dword v3, v4, s[4:5]
.LBB0_93:
	s_or_b64 exec, exec, s[0:1]
	v_div_scale_f32 v3, s[0:1], v2, v2, s50
	v_rcp_f32_e32 v4, v3
	v_div_scale_f32 v5, vcc, s50, v2, s50
	v_fma_f32 v6, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v6, v4
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v2, v3, v2, s50
	v_pk_mul_f32 v[4:5], v[82:83], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[84:85], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[86:87], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[88:89], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[90:91], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[92:93], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[94:95], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[96:97], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[66:67], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[68:69], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[70:71], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[72:73], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[74:75], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[76:77], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[78:79], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[80:81], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[150:151], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[152:153], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[154:155], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[156:157], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[158:159], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[160:161], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[162:163], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[164:165], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[148:149], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[146:147], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[144:145], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[142:143], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[140:141], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[138:139], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[136:137], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[134:135], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[112:113], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[110:111], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[108:109], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[106:107], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[104:105], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[102:103], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[100:101], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[98:99], v[2:3] op_sel_hi:[1,0]
	v_add_u32_e32 v99, 0x8000, v32
	v_add_u32_e32 v3, 0x8000, v3
	v_add_u32_e32 v98, 0x8000, v2
	v_add_u32_e32 v2, 0x8000, v97
	v_add_u32_e32 v97, 0x8000, v33
	v_add_u32_e32 v100, 0x8000, v31
	v_add_u32_e32 v101, 0x8000, v30
	v_add_u32_e32 v102, 0x8000, v29
	v_add_u32_e32 v103, 0x8000, v28
	v_add_u32_e32 v104, 0x8000, v27
	v_add_u32_e32 v105, 0x8000, v26
	v_add_u32_e32 v106, 0x8000, v25
	v_add_u32_e32 v107, 0x8000, v24
	v_add_u32_e32 v108, 0x8000, v23
	v_add_u32_e32 v109, 0x8000, v22
	v_add_u32_e32 v110, 0x8000, v21
	v_add_u32_e32 v111, 0x8000, v20
	v_add_u32_e32 v112, 0x8000, v19
	v_add_u32_e32 v113, 0x8000, v18
	v_add_u32_e32 v19, 0x8000, v83
	v_add_u32_e32 v18, 0x8000, v81
	v_add_u32_e32 v21, 0x8000, v79
	v_add_u32_e32 v20, 0x8000, v77
	v_add_u32_e32 v23, 0x8000, v75
	v_add_u32_e32 v22, 0x8000, v73
	v_add_u32_e32 v25, 0x8000, v71
	v_add_u32_e32 v24, 0x8000, v69
	v_add_u32_e32 v27, 0x8000, v67
	v_add_u32_e32 v26, 0x8000, v65
	v_add_u32_e32 v29, 0x8000, v63
	v_add_u32_e32 v28, 0x8000, v61
	v_add_u32_e32 v31, 0x8000, v59
	v_add_u32_e32 v30, 0x8000, v57
	v_add_u32_e32 v33, 0x8000, v55
	v_add_u32_e32 v32, 0x8000, v53
	v_add_u32_e32 v50, 0x8000, v50
	v_add_u32_e32 v39, 0x8000, v39
	v_add_u32_e32 v38, 0x8000, v38
	v_add_u32_e32 v37, 0x8000, v37
	v_add_u32_e32 v36, 0x8000, v36
	v_add_u32_e32 v35, 0x8000, v35
	v_add_u32_e32 v34, 0x8000, v34
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
	v_add_u32_e32 v96, 0x8000, v96
	v_add_u32_e32 v95, 0x8000, v95
	v_add_u32_e32 v94, 0x8000, v94
	v_add_u32_e32 v93, 0x8000, v93
	v_add_u32_e32 v92, 0x8000, v92
	v_add_u32_e32 v91, 0x8000, v91
	v_add_u32_e32 v90, 0x8000, v90
	v_add_u32_e32 v89, 0x8000, v89
	v_add_u32_e32 v88, 0x8000, v88
	v_add_u32_e32 v87, 0x8000, v87
	v_add_u32_e32 v86, 0x8000, v86
	v_add_u32_e32 v85, 0x8000, v85
	v_add_u32_e32 v84, 0x8000, v84
	v_add_u32_e32 v82, 0x8000, v82
	v_add_u32_e32 v80, 0x8000, v80
	v_add_u32_e32 v78, 0x8000, v78
	v_add_u32_e32 v76, 0x8000, v76
	v_add_u32_e32 v74, 0x8000, v74
	v_add_u32_e32 v72, 0x8000, v72
	v_add_u32_e32 v70, 0x8000, v70
	v_add_u32_e32 v68, 0x8000, v68
	v_add_u32_e32 v66, 0x8000, v66
	v_add_u32_e32 v64, 0x8000, v64
	v_add_u32_e32 v62, 0x8000, v62
	v_add_u32_e32 v60, 0x8000, v60
	v_add_u32_e32 v58, 0x8000, v58
	v_add_u32_e32 v56, 0x8000, v56
	v_add_u32_e32 v54, 0x8000, v54
	v_add_u32_e32 v52, 0x8000, v52
	v_add_u32_e32 v51, 0x8000, v51
	v_add_u32_e32 v53, 0x8000, v49
	v_add_u32_e32 v55, 0x8000, v48
	v_add_u32_e32 v57, 0x8000, v47
	v_add_u32_e32 v59, 0x8000, v46
	v_add_u32_e32 v61, 0x8000, v45
	v_add_u32_e32 v63, 0x8000, v44
	v_add_u32_e32 v65, 0x8000, v43
	v_add_u32_e32 v67, 0x8000, v42
	v_add_u32_e32 v69, 0x8000, v41
	v_add_u32_e32 v71, 0x8000, v40
	;;#ASMSTART
	v_perm_b32 v48, v5, v4, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v49, v7, v6, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v46, v9, v8, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v47, v11, v10, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v44, v13, v12, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v45, v15, v14, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v42, v17, v16, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v43, v35, v34, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v40, v37, v36, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v41, v39, v38, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v38, v69, v71, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v39, v65, v67, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v36, v61, v63, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v37, v57, v59, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v34, v53, v55, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v35, v51, v50, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v32, v52, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v33, v54, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v30, v56, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v31, v58, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v28, v28, v60, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v29, v62, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v26, v64, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v27, v66, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v24, v68, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v25, v70, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v22, v72, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v23, v74, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v20, v76, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v21, v78, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v18, v80, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v19, v82, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v112, v113, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v110, v111, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v108, v109, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v106, v107, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v104, v105, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v102, v103, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v100, v101, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v97, v99, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v85, v84, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v87, v86, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v89, v88, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v91, v90, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v93, v92, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v95, v94, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v2, v96, s53
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v3, v98, s53
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v50, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v50, 0x100, v50
	v_cmp_eq_u32_e32 vcc, 0, v50
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_95
	s_barrier
.LBB0_95:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v91, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v50, 31, v91
	v_mul_u32_u24_e32 v50, 0xc0, v50
	v_lshrrev_b32_e32 v51, 3, v91
	v_and_or_b32 v58, v51, 4, v50
	v_bfe_u32 v87, v91, 6, 3
	v_lshrrev_b32_e32 v59, 3, v50
	v_lshrrev_b32_e32 v61, 2, v58
	v_add_u32_e32 v73, 0x48, v58
	v_add_u32_e32 v71, 0x50, v58
	v_add_u32_e32 v68, 0x58, v58
	v_add_u32_e32 v66, 0x60, v58
	v_add_u32_e32 v64, 0x68, v58
	v_add_u32_e32 v62, 0x70, v58
	v_add_u32_e32 v60, 0x78, v58
	v_add_u32_e32 v57, 0x88, v58
	v_add_u32_e32 v56, 0x90, v58
	v_add_u32_e32 v55, 0x98, v58
	v_add_u32_e32 v54, 0xa0, v58
	v_add_u32_e32 v53, 0xa8, v58
	v_add_u32_e32 v51, 0xb0, v58
	v_add_u32_e32 v50, 0xb8, v58
	v_cmp_eq_u32_e32 vcc, 0, v87
	v_lshlrev_b32_e32 v52, 1, v58
	v_or_b32_e32 v85, 8, v58
	v_or_b32_e32 v84, 16, v58
	v_or_b32_e32 v82, 24, v58
	v_or_b32_e32 v80, 32, v58
	v_or_b32_e32 v79, 40, v58
	v_or_b32_e32 v77, 48, v58
	v_or_b32_e32 v75, 56, v58
	v_and_b32_e32 v90, 56, v59
	v_add_u32_e32 v89, 16, v61
	v_lshrrev_b32_e32 v86, 3, v73
	v_lshrrev_b32_e32 v83, 3, v71
	v_lshrrev_b32_e32 v81, 3, v68
	v_lshrrev_b32_e32 v78, 3, v66
	v_lshrrev_b32_e32 v76, 3, v64
	v_lshrrev_b32_e32 v74, 3, v62
	v_lshrrev_b32_e32 v72, 3, v60
	v_add_u32_e32 v70, 32, v61
	v_lshrrev_b32_e32 v69, 3, v57
	v_lshrrev_b32_e32 v67, 3, v56
	v_lshrrev_b32_e32 v65, 3, v55
	v_lshrrev_b32_e32 v63, 3, v54
	v_lshrrev_b32_e32 v61, 3, v53
	v_lshrrev_b32_e32 v59, 3, v51
	v_lshrrev_b32_e32 v58, 3, v50
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_97
	v_lshl_add_u32 v88, v90, 1, v52
	ds_write_b64 v88, v[48:49]
	v_xor_b32_e32 v88, v85, v90
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[46:47]
	v_xor_b32_e32 v88, v84, v90
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[44:45]
	v_xor_b32_e32 v88, v82, v90
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[42:43]
	v_xor_b32_e32 v88, v80, v90
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[40:41]
	v_xor_b32_e32 v88, v79, v90
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[38:39]
	v_xor_b32_e32 v88, v77, v90
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[36:37]
	v_xor_b32_e32 v88, v75, v90
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[34:35]
	v_and_b32_e32 v88, 0x70, v89
	v_add_u32_e32 v88, v52, v88
	ds_write_b64 v88, v[32:33] offset:128
	v_and_b32_e32 v88, 56, v86
	v_xor_b32_e32 v88, v88, v73
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[30:31]
	v_and_b32_e32 v88, 56, v83
	v_xor_b32_e32 v88, v88, v71
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[28:29]
	v_and_b32_e32 v88, 56, v81
	v_xor_b32_e32 v88, v88, v68
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[26:27]
	v_and_b32_e32 v88, 56, v78
	v_xor_b32_e32 v88, v88, v66
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[24:25]
	v_and_b32_e32 v88, 56, v76
	v_xor_b32_e32 v88, v88, v64
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[22:23]
	v_and_b32_e32 v88, 56, v74
	v_xor_b32_e32 v88, v88, v62
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[20:21]
	v_and_b32_e32 v88, 56, v72
	v_xor_b32_e32 v88, v88, v60
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[18:19]
	v_and_b32_e32 v88, 0x70, v70
	v_add_u32_e32 v88, v52, v88
	ds_write_b64 v88, v[16:17] offset:256
	v_and_b32_e32 v88, 56, v69
	v_xor_b32_e32 v88, v88, v57
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[14:15]
	v_and_b32_e32 v88, 56, v67
	v_xor_b32_e32 v88, v88, v56
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[12:13]
	v_and_b32_e32 v88, 56, v65
	v_xor_b32_e32 v88, v88, v55
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[10:11]
	v_and_b32_e32 v88, 56, v63
	v_xor_b32_e32 v88, v88, v54
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[8:9]
	v_and_b32_e32 v88, 56, v61
	v_xor_b32_e32 v88, v88, v53
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[6:7]
	v_and_b32_e32 v88, 56, v59
	v_xor_b32_e32 v88, v88, v51
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[4:5]
	v_and_b32_e32 v88, 56, v58
	v_xor_b32_e32 v88, v88, v50
	v_lshlrev_b32_e32 v88, 1, v88
	ds_write_b64 v88, v[2:3]
.LBB0_97:
	s_or_b64 exec, exec, s[0:1]
	s_mul_i32 s0, s36, 0xc00
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_and_b32_e32 v88, 0x1ff, v91
	s_add_u32 s4, s28, s0
	s_addc_u32 s0, s29, s1
	v_lshlrev_b32_e32 v92, 3, v88
	s_mul_i32 s6, s13, 0x1800
	s_and_b32 s5, s0, 0xffff
	s_mul_i32 s13, s22, 0xc0
	v_and_b32_e32 v91, 56, v91
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_98:
	v_mul_u32_u24_sdwa v95, v94, s57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s58, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[4:7], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_98
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 1, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_101
	v_lshl_add_u32 v93, v90, 1, v52
	ds_write_b64 v93, v[48:49]
	v_xor_b32_e32 v93, v85, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[46:47]
	v_xor_b32_e32 v93, v84, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[44:45]
	v_xor_b32_e32 v93, v82, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[42:43]
	v_xor_b32_e32 v93, v80, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[40:41]
	v_xor_b32_e32 v93, v79, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[38:39]
	v_xor_b32_e32 v93, v77, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[36:37]
	v_xor_b32_e32 v93, v75, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[34:35]
	v_and_b32_e32 v93, 0x70, v89
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[32:33] offset:128
	v_and_b32_e32 v93, 56, v86
	v_xor_b32_e32 v93, v93, v73
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[30:31]
	v_and_b32_e32 v93, 56, v83
	v_xor_b32_e32 v93, v93, v71
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[28:29]
	v_and_b32_e32 v93, 56, v81
	v_xor_b32_e32 v93, v93, v68
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[26:27]
	v_and_b32_e32 v93, 56, v78
	v_xor_b32_e32 v93, v93, v66
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[24:25]
	v_and_b32_e32 v93, 56, v76
	v_xor_b32_e32 v93, v93, v64
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[22:23]
	v_and_b32_e32 v93, 56, v74
	v_xor_b32_e32 v93, v93, v62
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[20:21]
	v_and_b32_e32 v93, 56, v72
	v_xor_b32_e32 v93, v93, v60
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[18:19]
	v_and_b32_e32 v93, 0x70, v70
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[16:17] offset:256
	v_and_b32_e32 v93, 56, v69
	v_xor_b32_e32 v93, v93, v57
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[14:15]
	v_and_b32_e32 v93, 56, v67
	v_xor_b32_e32 v93, v93, v56
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[12:13]
	v_and_b32_e32 v93, 56, v65
	v_xor_b32_e32 v93, v93, v55
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[10:11]
	v_and_b32_e32 v93, 56, v63
	v_xor_b32_e32 v93, v93, v54
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[8:9]
	v_and_b32_e32 v93, 56, v61
	v_xor_b32_e32 v93, v93, v53
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[6:7]
	v_and_b32_e32 v93, 56, v59
	v_xor_b32_e32 v93, v93, v51
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[4:5]
	v_and_b32_e32 v93, 56, v58
	v_xor_b32_e32 v93, v93, v50
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[2:3]
.LBB0_101:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s13, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_102:
	v_mul_u32_u24_sdwa v95, v94, s57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s58, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[4:7], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_102
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 2, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_105
	v_lshl_add_u32 v93, v90, 1, v52
	ds_write_b64 v93, v[48:49]
	v_xor_b32_e32 v93, v85, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[46:47]
	v_xor_b32_e32 v93, v84, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[44:45]
	v_xor_b32_e32 v93, v82, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[42:43]
	v_xor_b32_e32 v93, v80, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[40:41]
	v_xor_b32_e32 v93, v79, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[38:39]
	v_xor_b32_e32 v93, v77, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[36:37]
	v_xor_b32_e32 v93, v75, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[34:35]
	v_and_b32_e32 v93, 0x70, v89
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[32:33] offset:128
	v_and_b32_e32 v93, 56, v86
	v_xor_b32_e32 v93, v93, v73
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[30:31]
	v_and_b32_e32 v93, 56, v83
	v_xor_b32_e32 v93, v93, v71
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[28:29]
	v_and_b32_e32 v93, 56, v81
	v_xor_b32_e32 v93, v93, v68
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[26:27]
	v_and_b32_e32 v93, 56, v78
	v_xor_b32_e32 v93, v93, v66
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[24:25]
	v_and_b32_e32 v93, 56, v76
	v_xor_b32_e32 v93, v93, v64
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[22:23]
	v_and_b32_e32 v93, 56, v74
	v_xor_b32_e32 v93, v93, v62
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[20:21]
	v_and_b32_e32 v93, 56, v72
	v_xor_b32_e32 v93, v93, v60
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[18:19]
	v_and_b32_e32 v93, 0x70, v70
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[16:17] offset:256
	v_and_b32_e32 v93, 56, v69
	v_xor_b32_e32 v93, v93, v57
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[14:15]
	v_and_b32_e32 v93, 56, v67
	v_xor_b32_e32 v93, v93, v56
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[12:13]
	v_and_b32_e32 v93, 56, v65
	v_xor_b32_e32 v93, v93, v55
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[10:11]
	v_and_b32_e32 v93, 56, v63
	v_xor_b32_e32 v93, v93, v54
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[8:9]
	v_and_b32_e32 v93, 56, v61
	v_xor_b32_e32 v93, v93, v53
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[6:7]
	v_and_b32_e32 v93, 56, v59
	v_xor_b32_e32 v93, v93, v51
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[4:5]
	v_and_b32_e32 v93, 56, v58
	v_xor_b32_e32 v93, v93, v50
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[2:3]
.LBB0_105:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s23, s13, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_106:
	v_mul_u32_u24_sdwa v95, v94, s57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s58, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s23, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[4:7], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_106
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 3, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_109
	v_lshl_add_u32 v93, v90, 1, v52
	ds_write_b64 v93, v[48:49]
	v_xor_b32_e32 v93, v85, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[46:47]
	v_xor_b32_e32 v93, v84, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[44:45]
	v_xor_b32_e32 v93, v82, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[42:43]
	v_xor_b32_e32 v93, v80, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[40:41]
	v_xor_b32_e32 v93, v79, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[38:39]
	v_xor_b32_e32 v93, v77, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[36:37]
	v_xor_b32_e32 v93, v75, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[34:35]
	v_and_b32_e32 v93, 0x70, v89
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[32:33] offset:128
	v_and_b32_e32 v93, 56, v86
	v_xor_b32_e32 v93, v93, v73
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[30:31]
	v_and_b32_e32 v93, 56, v83
	v_xor_b32_e32 v93, v93, v71
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[28:29]
	v_and_b32_e32 v93, 56, v81
	v_xor_b32_e32 v93, v93, v68
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[26:27]
	v_and_b32_e32 v93, 56, v78
	v_xor_b32_e32 v93, v93, v66
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[24:25]
	v_and_b32_e32 v93, 56, v76
	v_xor_b32_e32 v93, v93, v64
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[22:23]
	v_and_b32_e32 v93, 56, v74
	v_xor_b32_e32 v93, v93, v62
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[20:21]
	v_and_b32_e32 v93, 56, v72
	v_xor_b32_e32 v93, v93, v60
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[18:19]
	v_and_b32_e32 v93, 0x70, v70
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[16:17] offset:256
	v_and_b32_e32 v93, 56, v69
	v_xor_b32_e32 v93, v93, v57
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[14:15]
	v_and_b32_e32 v93, 56, v67
	v_xor_b32_e32 v93, v93, v56
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[12:13]
	v_and_b32_e32 v93, 56, v65
	v_xor_b32_e32 v93, v93, v55
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[10:11]
	v_and_b32_e32 v93, 56, v63
	v_xor_b32_e32 v93, v93, v54
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[8:9]
	v_and_b32_e32 v93, 56, v61
	v_xor_b32_e32 v93, v93, v53
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[6:7]
	v_and_b32_e32 v93, 56, v59
	v_xor_b32_e32 v93, v93, v51
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[4:5]
	v_and_b32_e32 v93, 56, v58
	v_xor_b32_e32 v93, v93, v50
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[2:3]
.LBB0_109:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s23, s13, 0x30000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_110:
	v_mul_u32_u24_sdwa v95, v94, s57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s58, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s23, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[4:7], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_110
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 4, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_113
	v_lshl_add_u32 v93, v90, 1, v52
	ds_write_b64 v93, v[48:49]
	v_xor_b32_e32 v93, v85, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[46:47]
	v_xor_b32_e32 v93, v84, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[44:45]
	v_xor_b32_e32 v93, v82, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[42:43]
	v_xor_b32_e32 v93, v80, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[40:41]
	v_xor_b32_e32 v93, v79, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[38:39]
	v_xor_b32_e32 v93, v77, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[36:37]
	v_xor_b32_e32 v93, v75, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[34:35]
	v_and_b32_e32 v93, 0x70, v89
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[32:33] offset:128
	v_and_b32_e32 v93, 56, v86
	v_xor_b32_e32 v93, v93, v73
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[30:31]
	v_and_b32_e32 v93, 56, v83
	v_xor_b32_e32 v93, v93, v71
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[28:29]
	v_and_b32_e32 v93, 56, v81
	v_xor_b32_e32 v93, v93, v68
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[26:27]
	v_and_b32_e32 v93, 56, v78
	v_xor_b32_e32 v93, v93, v66
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[24:25]
	v_and_b32_e32 v93, 56, v76
	v_xor_b32_e32 v93, v93, v64
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[22:23]
	v_and_b32_e32 v93, 56, v74
	v_xor_b32_e32 v93, v93, v62
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[20:21]
	v_and_b32_e32 v93, 56, v72
	v_xor_b32_e32 v93, v93, v60
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[18:19]
	v_and_b32_e32 v93, 0x70, v70
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[16:17] offset:256
	v_and_b32_e32 v93, 56, v69
	v_xor_b32_e32 v93, v93, v57
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[14:15]
	v_and_b32_e32 v93, 56, v67
	v_xor_b32_e32 v93, v93, v56
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[12:13]
	v_and_b32_e32 v93, 56, v65
	v_xor_b32_e32 v93, v93, v55
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[10:11]
	v_and_b32_e32 v93, 56, v63
	v_xor_b32_e32 v93, v93, v54
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[8:9]
	v_and_b32_e32 v93, 56, v61
	v_xor_b32_e32 v93, v93, v53
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[6:7]
	v_and_b32_e32 v93, 56, v59
	v_xor_b32_e32 v93, v93, v51
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[4:5]
	v_and_b32_e32 v93, 56, v58
	v_xor_b32_e32 v93, v93, v50
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[2:3]
.LBB0_113:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s23, s13, 0x48000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_114:
	v_mul_u32_u24_sdwa v95, v94, s57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s58, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s23, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[4:7], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_114
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 5, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_117
	v_lshl_add_u32 v93, v90, 1, v52
	ds_write_b64 v93, v[48:49]
	v_xor_b32_e32 v93, v85, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[46:47]
	v_xor_b32_e32 v93, v84, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[44:45]
	v_xor_b32_e32 v93, v82, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[42:43]
	v_xor_b32_e32 v93, v80, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[40:41]
	v_xor_b32_e32 v93, v79, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[38:39]
	v_xor_b32_e32 v93, v77, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[36:37]
	v_xor_b32_e32 v93, v75, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[34:35]
	v_and_b32_e32 v93, 0x70, v89
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[32:33] offset:128
	v_and_b32_e32 v93, 56, v86
	v_xor_b32_e32 v93, v93, v73
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[30:31]
	v_and_b32_e32 v93, 56, v83
	v_xor_b32_e32 v93, v93, v71
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[28:29]
	v_and_b32_e32 v93, 56, v81
	v_xor_b32_e32 v93, v93, v68
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[26:27]
	v_and_b32_e32 v93, 56, v78
	v_xor_b32_e32 v93, v93, v66
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[24:25]
	v_and_b32_e32 v93, 56, v76
	v_xor_b32_e32 v93, v93, v64
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[22:23]
	v_and_b32_e32 v93, 56, v74
	v_xor_b32_e32 v93, v93, v62
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[20:21]
	v_and_b32_e32 v93, 56, v72
	v_xor_b32_e32 v93, v93, v60
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[18:19]
	v_and_b32_e32 v93, 0x70, v70
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[16:17] offset:256
	v_and_b32_e32 v93, 56, v69
	v_xor_b32_e32 v93, v93, v57
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[14:15]
	v_and_b32_e32 v93, 56, v67
	v_xor_b32_e32 v93, v93, v56
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[12:13]
	v_and_b32_e32 v93, 56, v65
	v_xor_b32_e32 v93, v93, v55
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[10:11]
	v_and_b32_e32 v93, 56, v63
	v_xor_b32_e32 v93, v93, v54
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[8:9]
	v_and_b32_e32 v93, 56, v61
	v_xor_b32_e32 v93, v93, v53
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[6:7]
	v_and_b32_e32 v93, 56, v59
	v_xor_b32_e32 v93, v93, v51
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[4:5]
	v_and_b32_e32 v93, 56, v58
	v_xor_b32_e32 v93, v93, v50
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[2:3]
.LBB0_117:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s23, s13, 0x60000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_118:
	v_mul_u32_u24_sdwa v95, v94, s57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s58, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s23, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[4:7], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_118
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 6, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_121
	v_lshl_add_u32 v93, v90, 1, v52
	ds_write_b64 v93, v[48:49]
	v_xor_b32_e32 v93, v85, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[46:47]
	v_xor_b32_e32 v93, v84, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[44:45]
	v_xor_b32_e32 v93, v82, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[42:43]
	v_xor_b32_e32 v93, v80, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[40:41]
	v_xor_b32_e32 v93, v79, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[38:39]
	v_xor_b32_e32 v93, v77, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[36:37]
	v_xor_b32_e32 v93, v75, v90
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[34:35]
	v_and_b32_e32 v93, 0x70, v89
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[32:33] offset:128
	v_and_b32_e32 v93, 56, v86
	v_xor_b32_e32 v93, v93, v73
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[30:31]
	v_and_b32_e32 v93, 56, v83
	v_xor_b32_e32 v93, v93, v71
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[28:29]
	v_and_b32_e32 v93, 56, v81
	v_xor_b32_e32 v93, v93, v68
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[26:27]
	v_and_b32_e32 v93, 56, v78
	v_xor_b32_e32 v93, v93, v66
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[24:25]
	v_and_b32_e32 v93, 56, v76
	v_xor_b32_e32 v93, v93, v64
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[22:23]
	v_and_b32_e32 v93, 56, v74
	v_xor_b32_e32 v93, v93, v62
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[20:21]
	v_and_b32_e32 v93, 56, v72
	v_xor_b32_e32 v93, v93, v60
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[18:19]
	v_and_b32_e32 v93, 0x70, v70
	v_add_u32_e32 v93, v52, v93
	ds_write_b64 v93, v[16:17] offset:256
	v_and_b32_e32 v93, 56, v69
	v_xor_b32_e32 v93, v93, v57
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[14:15]
	v_and_b32_e32 v93, 56, v67
	v_xor_b32_e32 v93, v93, v56
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[12:13]
	v_and_b32_e32 v93, 56, v65
	v_xor_b32_e32 v93, v93, v55
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[10:11]
	v_and_b32_e32 v93, 56, v63
	v_xor_b32_e32 v93, v93, v54
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[8:9]
	v_and_b32_e32 v93, 56, v61
	v_xor_b32_e32 v93, v93, v53
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[6:7]
	v_and_b32_e32 v93, 56, v59
	v_xor_b32_e32 v93, v93, v51
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[4:5]
	v_and_b32_e32 v93, 56, v58
	v_xor_b32_e32 v93, v93, v50
	v_lshlrev_b32_e32 v93, 1, v93
	ds_write_b64 v93, v[2:3]
.LBB0_121:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s23, s13, 0x78000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_122:
	v_mul_u32_u24_sdwa v95, v94, s57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s58, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s23, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[4:7], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_122
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 7, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_125
	v_lshl_add_u32 v87, v90, 1, v52
	ds_write_b64 v87, v[48:49]
	v_xor_b32_e32 v48, v85, v90
	v_lshlrev_b32_e32 v48, 1, v48
	ds_write_b64 v48, v[46:47]
	v_xor_b32_e32 v46, v84, v90
	v_lshlrev_b32_e32 v46, 1, v46
	ds_write_b64 v46, v[44:45]
	v_xor_b32_e32 v44, v82, v90
	v_lshlrev_b32_e32 v44, 1, v44
	ds_write_b64 v44, v[42:43]
	v_xor_b32_e32 v42, v80, v90
	v_lshlrev_b32_e32 v42, 1, v42
	ds_write_b64 v42, v[40:41]
	v_xor_b32_e32 v40, v79, v90
	v_lshlrev_b32_e32 v40, 1, v40
	ds_write_b64 v40, v[38:39]
	v_xor_b32_e32 v38, v77, v90
	v_lshlrev_b32_e32 v38, 1, v38
	ds_write_b64 v38, v[36:37]
	v_xor_b32_e32 v36, v75, v90
	v_lshlrev_b32_e32 v36, 1, v36
	ds_write_b64 v36, v[34:35]
	v_and_b32_e32 v34, 0x70, v89
	v_add_u32_e32 v34, v52, v34
	ds_write_b64 v34, v[32:33] offset:128
	v_and_b32_e32 v32, 56, v86
	v_xor_b32_e32 v32, v32, v73
	v_lshlrev_b32_e32 v32, 1, v32
	ds_write_b64 v32, v[30:31]
	v_and_b32_e32 v30, 56, v83
	v_xor_b32_e32 v30, v30, v71
	v_lshlrev_b32_e32 v30, 1, v30
	ds_write_b64 v30, v[28:29]
	v_and_b32_e32 v28, 56, v81
	v_xor_b32_e32 v28, v28, v68
	v_lshlrev_b32_e32 v28, 1, v28
	ds_write_b64 v28, v[26:27]
	v_and_b32_e32 v26, 56, v78
	v_xor_b32_e32 v26, v26, v66
	v_lshlrev_b32_e32 v26, 1, v26
	ds_write_b64 v26, v[24:25]
	v_and_b32_e32 v24, 56, v76
	v_xor_b32_e32 v24, v24, v64
	v_lshlrev_b32_e32 v24, 1, v24
	ds_write_b64 v24, v[22:23]
	v_and_b32_e32 v22, 56, v74
	v_xor_b32_e32 v22, v22, v62
	v_lshlrev_b32_e32 v22, 1, v22
	ds_write_b64 v22, v[20:21]
	v_and_b32_e32 v20, 56, v72
	v_xor_b32_e32 v20, v20, v60
	v_lshlrev_b32_e32 v20, 1, v20
	ds_write_b64 v20, v[18:19]
	v_and_b32_e32 v18, 0x70, v70
	v_add_u32_e32 v18, v52, v18
	ds_write_b64 v18, v[16:17] offset:256
	v_and_b32_e32 v16, 56, v69
	v_xor_b32_e32 v16, v16, v57
	v_lshlrev_b32_e32 v16, 1, v16
	ds_write_b64 v16, v[14:15]
	v_and_b32_e32 v14, 56, v67
	v_xor_b32_e32 v14, v14, v56
	v_lshlrev_b32_e32 v14, 1, v14
	ds_write_b64 v14, v[12:13]
	v_and_b32_e32 v12, 56, v65
	v_xor_b32_e32 v12, v12, v55
	v_lshlrev_b32_e32 v12, 1, v12
	ds_write_b64 v12, v[10:11]
	v_and_b32_e32 v10, 56, v63
	v_xor_b32_e32 v10, v10, v54
	v_lshlrev_b32_e32 v10, 1, v10
	ds_write_b64 v10, v[8:9]
	v_and_b32_e32 v8, 56, v61
	v_xor_b32_e32 v8, v8, v53
	v_lshlrev_b32_e32 v8, 1, v8
	ds_write_b64 v8, v[6:7]
	v_and_b32_e32 v6, 56, v59
	v_xor_b32_e32 v6, v6, v51
	v_lshlrev_b32_e32 v6, 1, v6
	ds_write_b64 v6, v[4:5]
	v_and_b32_e32 v4, 56, v58
	v_xor_b32_e32 v4, v4, v50
	v_lshlrev_b32_e32 v4, 1, v4
	ds_write_b64 v4, v[2:3]
.LBB0_125:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s13, 0x90000
	s_mov_b64 s[0:1], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_126:
	v_mul_u32_u24_sdwa v2, v88, s57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v3, v92, v91
	v_lshrrev_b32_e32 v2, 20, v2
	v_lshlrev_b32_e32 v3, 1, v3
	v_mul_lo_u16_e32 v5, 24, v2
	ds_read_b128 v[6:9], v3
	v_sub_u16_e32 v3, v88, v5
	v_lshlrev_b16_e32 v3, 3, v3
	v_add_u32_e32 v4, 0x200, v88
	v_cmp_lt_u32_e32 vcc, s58, v88
	v_mul_u32_u24_e32 v2, 0xc00, v2
	v_add_u32_e32 v3, s13, v3
	v_add_u32_e32 v92, 0x1000, v92
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v88, v4
	v_add_lshl_u32 v2, v3, v2, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v2, s[4:7], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_126
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	;;#ASMSTART
	s_mov_b32 s0, s59
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_131
	s_mov_b64 s[38:39], exec
	v_mbcnt_lo_u32_b32 v2, s38, 0
	v_mbcnt_hi_u32_b32 v2, s39, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_130
	s_bcnt1_i32_b64 s6, s[38:39]
	v_mov_b32_e32 v3, s6
	global_atomic_add v3, v115, v3, s[34:35] sc0
.LBB0_130:
	s_or_b64 exec, exec, s[36:37]
	s_lshl_b64 s[36:37], s[0:1], 2
	s_add_u32 s36, s34, s36
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s6, v3
	s_addc_u32 s37, s35, s37
	s_nop 0
	v_add_u32_e32 v2, s6, v2
	global_store_dword v115, v2, s[36:37]
	s_waitcnt vmcnt(0)
.LBB0_131:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s34, s0
	s_addc_u32 s1, s35, s1
	s_barrier
	global_load_dword v2, v115, s[0:1]
	s_sub_i32 s0, s33, s2
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s2, v2
	s_add_i32 s33, s0, s2
	s_cmp_ge_i32 s12, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s48
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_134
	s_branch .LBB0_12
.LBB0_132:
	s_mov_b32 s12, s5
.LBB0_133:
	s_sub_i32 s33, s33, s48
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s22, 0, s4
	s_cmp_ge_i32 s12, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s48, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_134:
	s_add_i32 s4, s22, 1
	s_cmp_gt_i32 s4, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s4, 16
	s_cbranch_scc1 .LBB0_137
	s_add_i32 s5, s12, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s48
	s_cbranch_scc1 .LBB0_132
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s12, s8, s12
	s_addc_u32 s13, s9, s13
	global_load_dwordx2 v[2:3], v115, s[12:13] offset:4
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s6, v3
	v_readfirstlane_b32 s12, v2
	s_sub_i32 s6, s6, s12
	s_addk_i32 s6, 0xff
	s_ashr_i32 s12, s6, 31
	s_lshr_b32 s12, s12, 24
	s_add_i32 s12, s6, s12
	s_ashr_i32 s36, s12, 8
	s_and_b32 s12, s12, 0xffffff00
	s_cmp_lg_u32 s6, s12
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s6, 0
	s_cselect_b64 s[22:23], -1, 0
	s_and_b64 s[12:13], s[22:23], s[12:13]
	s_subb_u32 s6, s36, 0
	s_branch .LBB0_132
.LBB0_137:
	s_mov_b32 s6, s48
	s_branch .LBB0_133
.LBB0_138:
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
		.amdhsa_next_free_vgpr 228
		.amdhsa_next_free_sgpr 69
		.amdhsa_accum_offset 228
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

	.set .Lattn_kernel_0.num_vgpr, 228
	.set .Lattn_kernel_0.num_agpr, 0
	.set .Lattn_kernel_0.numbered_sgpr, 69
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
    .sgpr_count:     75
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     228
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
