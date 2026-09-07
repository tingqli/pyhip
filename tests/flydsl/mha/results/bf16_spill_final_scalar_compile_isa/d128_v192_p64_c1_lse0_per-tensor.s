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
	s_mov_b32 s43, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s18, s20
.LBB0_3:
	s_sub_i32 s33, s33, s14
	s_and_b64 s[12:13], s[12:13], exec
	s_cselect_b32 s43, 0, s15
	s_cmp_ge_i32 s18, s3
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s33, s40
	s_cselect_b64 s[14:15], -1, 0
	s_or_b64 s[12:13], s[12:13], s[14:15]
	s_and_b64 vcc, exec, s[12:13]
	s_mov_b32 s14, s40
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s15, s43, 1
	s_cmp_gt_i32 s15, 15
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s15, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s20, s18, 1
	s_cmp_ge_i32 s20, s3
	s_mov_b32 s40, s14
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
	s_subb_u32 s40, s24, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s40, s14
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s40, s14
	s_mov_b32 s43, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s41, s[6:7], 0x0
	s_load_dword s42, s[8:9], 0x0
	s_cmp_ge_i32 s18, s3
	s_cbranch_scc1 .LBB0_87
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_load_dwordx2 s[20:21], s[0:1], 0x20
	s_load_dwordx2 s[22:23], s[0:1], 0x50
	s_load_dwordx2 s[24:25], s[0:1], 0x60
	s_load_dwordx2 s[26:27], s[0:1], 0x98
	s_load_dwordx2 s[28:29], s[0:1], 0xb8
	v_lshrrev_b32_e32 v2, 2, v0
	v_lshlrev_b32_e32 v3, 10, v0
	v_lshl_or_b32 v1, v0, 11, v2
	v_and_b32_e32 v3, 0x70000, v3
	s_mov_b32 s0, 0xf808
	v_lshrrev_b32_e32 v5, 1, v0
	v_lshrrev_b32_e32 v6, 3, v0
	v_and_or_b32 v1, v1, s0, v3
	v_lshlrev_b32_e32 v3, 8, v0
	v_and_b32_e32 v4, 12, v2
	v_and_b32_e32 v5, 32, v5
	v_and_b32_e32 v6, 16, v6
	v_and_b32_e32 v3, 0xf00, v3
	v_or3_b32 v4, v4, v5, v6
	v_and_b32_e32 v2, 64, v2
	v_or3_b32 v114, v4, v2, v3
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v119, v2, v3
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v1, 1, v1
	v_and_b32_e32 v140, 31, v0
	v_mov_b32_e32 v115, 0
	v_xor_b32_e32 v141, 0x80, v2
	s_mov_b32 s15, 0x27000
	s_movk_i32 s44, 0xf80
	v_mov_b32_e32 v142, 0x40e00000
	v_mov_b32_e32 v143, 1.0
	s_mov_b32 s45, 0x7060302
	s_movk_i32 s46, 0x1000
	s_movk_i32 s47, 0x2000
	s_movk_i32 s48, 0xe0
	s_mov_b32 s49, 0xaaab
	s_movk_i32 s50, 0xff
	s_mov_b32 s51, s2
	v_mov_b32_e32 v144, 0xff800000
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s40, s14
.LBB0_12:
	s_mov_b32 s2, s12
	s_cmp_ge_i32 s18, s3
	s_cbranch_scc1 .LBB0_87
.LBB0_13:
	s_ashr_i32 s19, s18, 31
	s_lshl_b32 s58, s33, 8
	s_lshl_b64 s[0:1], s[18:19], 2
	s_add_u32 s12, s16, s0
	s_addc_u32 s13, s17, s1
	global_load_dwordx2 v[4:5], v115, s[12:13]
	global_load_dword v2, v115, s[4:5]
	v_lshl_add_u32 v3, s43, 8, v1
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s31, v4
	s_add_i32 s52, s31, s58
	v_readfirstlane_b32 s34, v5
	s_add_i32 s12, s52, 0x100
	s_min_i32 s12, s12, s34
	s_sub_i32 s19, s12, s52
	s_waitcnt lgkmcnt(0)
	s_add_u32 s12, s22, s0
	s_addc_u32 s13, s23, s1
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s36, s52, 11
	s_ashr_i32 s37, s36, 31
	global_load_dwordx2 v[4:5], v115, s[12:13]
	global_load_dword v6, v115, s[0:1]
	s_lshl_b64 s[0:1], s[36:37], 1
	s_add_u32 s12, s6, s0
	s_addc_u32 s0, s7, s1
	s_lshl_b32 s14, s19, 12
	s_and_b32 s13, s0, 0xffff
	buffer_load_dwordx4 v[148:151], v3, s[12:15], 0 offen
	buffer_load_dwordx4 v[152:155], v3, s[12:15], 0 offen offset:32
	buffer_load_dwordx4 v[156:159], v3, s[12:15], 0 offen offset:64
	buffer_load_dwordx4 v[160:163], v3, s[12:15], 0 offen offset:96
	buffer_load_dwordx4 v[164:167], v3, s[12:15], 0 offen offset:128
	buffer_load_dwordx4 v[168:171], v3, s[12:15], 0 offen offset:160
	buffer_load_dwordx4 v[172:175], v3, s[12:15], 0 offen offset:192
	buffer_load_dwordx4 v[176:179], v3, s[12:15], 0 offen offset:224
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_waitcnt vmcnt(9)
	v_readfirstlane_b32 s12, v4
	v_and_b32_e32 v3, 0x100, v3
	v_readfirstlane_b32 s13, v5
	s_waitcnt vmcnt(8)
	v_readfirstlane_b32 s30, v6
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_15
	s_barrier
.LBB0_15:
	s_or_b64 exec, exec, s[0:1]
	s_ashr_i32 s0, s43, 31
	s_lshr_b32 s0, s0, 28
	s_sub_i32 s59, s13, s12
	s_add_i32 s0, s43, s0
	s_sub_i32 s31, s31, s34
	s_lshl_b32 s36, s59, 6
	s_ashr_i32 s13, s0, 4
	s_and_b32 s0, s0, -16
	s_cmp_lg_u32 s43, s0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s43, 0
	s_cselect_b64 s[34:35], -1, 0
	s_and_b64 s[0:1], s[34:35], s[0:1]
	s_subb_u32 s53, s13, 0
	s_lshl_b32 s0, s53, 13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s8, s0
	s_addc_u32 s1, s9, s1
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s12, s24, s12
	s_addc_u32 s13, s25, s13
	s_lshl_b32 s14, s59, 2
	s_and_b32 s13, s13, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[12:15], 0
	v_mul_f32_e32 v2, s41, v2
	v_mul_f32_e32 v116, 0x3e0293ee, v2
	s_mulk_i32 s53, 0x3000
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s38, v4
	v_readfirstlane_b32 s56, v5
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
	buffer_load_dword v2, off, s[12:15], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s54, v2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_lshl_b32 s34, s38, 12
	v_or_b32_e32 v2, s34, v114
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 2, s[0:1]
	global_load_dwordx4 v[6:9], v[4:5], off
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_ashr_i32 s34, s34, 31
	v_mov_b32_e32 v3, s34
	buffer_load_dword v4, off, s[12:15], 0 offset:12
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[2:3], off offset:512
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s57, v4
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	v_lshl_or_b32 v2, s56, 12, v114
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[0:1]
	global_load_dwordx4 v[184:187], v[2:3], off
	ds_write_b128 v119, v[6:9] offset:8192
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_add_i32 s31, s31, s36
	s_add_i32 s31, s31, s30
	s_sub_i32 s55, s31, 64
	s_add_i32 s61, s55, s58
	s_add_i32 s34, s61, 1
	s_ashr_i32 s30, s34, 31
	s_lshr_b32 s30, s30, 26
	s_add_i32 s30, s34, s30
	s_ashr_i32 s36, s30, 6
	s_andn2_b32 s30, s30, 63
	s_cmp_lg_u32 s34, s30
	s_cselect_b64 s[30:31], -1, 0
	s_cmp_lt_i32 s34, 0
	s_cselect_b64 s[34:35], -1, 0
	s_and_b64 s[30:31], s[34:35], s[30:31]
	s_subb_u32 s34, s36, 0
	s_lshr_b32 s30, s34, 31
	s_add_i32 s30, s34, s30
	s_ashr_i32 s36, s30, 1
	s_and_b32 s30, s30, -2
	s_cmp_lg_u32 s34, s30
	s_cselect_b64 s[30:31], -1, 0
	s_cmp_lt_i32 s34, 0
	s_cselect_b64 s[34:35], -1, 0
	s_and_b64 s[30:31], s[34:35], s[30:31]
	s_subb_u32 s60, s36, 0
	s_lshl_b32 s30, s60, 1
	s_ashr_i32 s31, s30, 31
	s_cmp_lt_i32 s60, 1
	s_cbranch_scc1 .LBB0_27
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v120, v116
	v_mov_b32_e32 v121, v116
	s_mov_b64 s[34:35], 0
	s_mov_b32 s62, 20
	v_mov_b32_e32 v118, 0xff800000
	v_mov_b32_e32 v122, v116
	v_mov_b32_e32 v123, v116
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v138
	v_mov_b32_e32 v4, v138
	v_mov_b32_e32 v5, v138
	v_mov_b32_e32 v6, v138
	v_mov_b32_e32 v7, v138
	v_mov_b32_e32 v8, v138
	v_mov_b32_e32 v9, v138
	v_mov_b32_e32 v10, v138
	v_mov_b32_e32 v11, v138
	v_mov_b32_e32 v12, v138
	v_mov_b32_e32 v13, v138
	v_mov_b32_e32 v14, v138
	v_mov_b32_e32 v15, v138
	v_mov_b32_e32 v16, v138
	v_mov_b32_e32 v17, v138
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v138
	v_mov_b32_e32 v20, v138
	v_mov_b32_e32 v21, v138
	v_mov_b32_e32 v22, v138
	v_mov_b32_e32 v23, v138
	v_mov_b32_e32 v24, v138
	v_mov_b32_e32 v25, v138
	v_mov_b32_e32 v26, v138
	v_mov_b32_e32 v27, v138
	v_mov_b32_e32 v28, v138
	v_mov_b32_e32 v29, v138
	v_mov_b32_e32 v30, v138
	v_mov_b32_e32 v31, v138
	v_mov_b32_e32 v32, v138
	v_mov_b32_e32 v33, v138
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v138
	v_mov_b32_e32 v36, v138
	v_mov_b32_e32 v37, v138
	v_mov_b32_e32 v38, v138
	v_mov_b32_e32 v39, v138
	v_mov_b32_e32 v40, v138
	v_mov_b32_e32 v41, v138
	v_mov_b32_e32 v42, v138
	v_mov_b32_e32 v43, v138
	v_mov_b32_e32 v44, v138
	v_mov_b32_e32 v45, v138
	v_mov_b32_e32 v46, v138
	v_mov_b32_e32 v47, v138
	v_mov_b32_e32 v48, v138
	v_mov_b32_e32 v49, v138
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v138
	v_mov_b32_e32 v52, v138
	v_mov_b32_e32 v53, v138
	v_mov_b32_e32 v54, v138
	v_mov_b32_e32 v55, v138
	v_mov_b32_e32 v56, v138
	v_mov_b32_e32 v57, v138
	v_mov_b32_e32 v58, v138
	v_mov_b32_e32 v59, v138
	v_mov_b32_e32 v60, v138
	v_mov_b32_e32 v61, v138
	v_mov_b32_e32 v62, v138
	v_mov_b32_e32 v63, v138
	v_mov_b32_e32 v64, v138
	v_mov_b32_e32 v65, v138
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, v138
	v_mov_b32_e32 v68, v138
	v_mov_b32_e32 v69, v138
	v_mov_b32_e32 v70, v138
	v_mov_b32_e32 v71, v138
	v_mov_b32_e32 v72, v138
	v_mov_b32_e32 v73, v138
	v_mov_b32_e32 v74, v138
	v_mov_b32_e32 v75, v138
	v_mov_b32_e32 v76, v138
	v_mov_b32_e32 v77, v138
	v_mov_b32_e32 v78, v138
	v_mov_b32_e32 v79, v138
	v_mov_b32_e32 v80, v138
	v_mov_b32_e32 v81, v138
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, v138
	v_mov_b32_e32 v84, v138
	v_mov_b32_e32 v85, v138
	v_mov_b32_e32 v86, v138
	v_mov_b32_e32 v87, v138
	v_mov_b32_e32 v88, v138
	v_mov_b32_e32 v89, v138
	v_mov_b32_e32 v90, v138
	v_mov_b32_e32 v91, v138
	v_mov_b32_e32 v92, v138
	v_mov_b32_e32 v93, v138
	v_mov_b32_e32 v94, v138
	v_mov_b32_e32 v95, v138
	v_mov_b32_e32 v96, v138
	v_mov_b32_e32 v97, v138
	s_branch .LBB0_18
.LBB0_17:
	s_or_b64 exec, exec, s[36:37]
	v_pk_add_f32 v[98:99], v[98:99], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, 0, v98
	v_add_f32_e32 v117, v117, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v100
	v_add_f32_e32 v117, v117, v101
	v_add_f32_e32 v117, v117, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v103
	v_add_f32_e32 v117, v117, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v105
	v_add_f32_e32 v117, v117, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v107
	v_add_f32_e32 v117, v117, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v109
	v_add_f32_e32 v117, v117, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_add_u32_e32 v105, 0x8000, v105
	v_add_f32_e32 v117, v117, v111
	v_add_f32_e32 v117, v117, v112
	v_add_f32_e32 v117, v117, v113
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_fma_f32 v138, v146, v145, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v145
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s45
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s38, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s38
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[20:21]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[124:127], v[106:107], off offset:512
	global_load_dwordx4 v[128:131], v[106:107], off offset:1024
	global_load_dwordx4 v[132:135], v[106:107], off offset:1536
	global_load_dwordx4 v[188:191], v[106:107], off offset:2048
	global_load_dwordx4 v[192:195], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[188:189], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[192:193], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s46, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[126:127], v[102:103], v[18:33]
	global_load_dwordx4 v[124:127], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[102:103], v[34:49]
	global_load_dwordx4 v[128:131], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[102:103], v[50:65]
	global_load_dwordx4 v[132:135], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s47, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[190:191], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[194:195], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[126:127], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	s_add_u32 s34, s34, 2
	s_addc_u32 s35, s35, 0
	v_mov_b64_e32 v[98:99], s[30:31]
	v_cmp_lt_i64_e32 vcc, s[34:35], v[98:99]
	s_add_i32 s62, s62, 8
	s_mov_b32 s56, s64
	s_mov_b32 s38, s63
	s_cbranch_vccz .LBB0_26
.LBB0_18:
	s_add_i32 s36, s62, -4
	v_mov_b32_e32 v98, s36
	s_lshl_b32 s36, s56, 12
	s_ashr_i32 s37, s36, 31
	buffer_load_dword v117, v98, s[12:15], 0 offen
	v_or_b32_e32 v98, s36, v114
	v_mov_b32_e32 v99, s37
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[192:195], v[98:99], off offset:512
	ds_write_b128 v119, v[180:183]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mov_b32 s63, s54
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v124, 48, v98
	v_and_or_b32 v125, v99, s44, v100
	v_or_b32_e32 v98, v125, v124
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[126:129], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[148:149], 0
	s_mov_b32 s64, s57
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s54, v117
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[150:151], v[98:113]
	v_or_b32_e32 v117, 0x1010, v125
	v_xor_b32_e32 v117, v117, v124
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[152:153], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[154:155], v[98:113]
	v_or_b32_e32 v117, 0x1020, v125
	v_xor_b32_e32 v117, v117, v124
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[158:159], v[98:113]
	v_or_b32_e32 v117, 0x1030, v125
	v_xor_b32_e32 v117, v117, v124
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[162:163], v[98:113]
	v_or_b32_e32 v117, 0x1040, v125
	v_lshrrev_b32_e32 v124, 3, v117
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v117, v124, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[166:167], v[98:113]
	v_or_b32_e32 v117, 0x1050, v125
	v_lshrrev_b32_e32 v124, 3, v117
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v117, v124, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[170:171], v[98:113]
	v_or_b32_e32 v117, 0x1060, v125
	v_lshrrev_b32_e32 v124, 3, v117
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v117, v124, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[174:175], v[98:113]
	v_or_b32_e32 v117, 0x1070, v125
	v_lshrrev_b32_e32 v124, 3, v117
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v117, v124, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[178:179], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[120:121], v[98:99]
	v_pk_mul_f32 v[100:101], v[122:123], v[100:101]
	v_max_f32_e32 v117, v98, v99
	v_pk_mul_f32 v[102:103], v[122:123], v[102:103]
	v_max3_f32 v117, v117, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v117, v117, v102, v103
	v_pk_mul_f32 v[106:107], v[122:123], v[106:107]
	v_max3_f32 v117, v117, v104, v105
	v_pk_mul_f32 v[108:109], v[122:123], v[108:109]
	v_max3_f32 v117, v117, v106, v107
	v_pk_mul_f32 v[110:111], v[122:123], v[110:111]
	v_max3_f32 v117, v117, v108, v109
	v_pk_mul_f32 v[112:113], v[122:123], v[112:113]
	v_max3_f32 v117, v117, v110, v111
	v_max3_f32 v117, v117, v112, v113
	ds_bpermute_b32 v124, v141, v117
	v_mov_b32_e32 v145, 1.0
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v124, v124, v124
	v_max_f32_e32 v124, v117, v124
	;;#ASMSTART
	v_add_f32 v117, v118, v142
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v124, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_20
	;;#ASMSTART
	v_add_f32 v124, v124, v143
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v124
	v_exp_f32_e32 v117, v117
	v_mov_b32_e32 v118, v124
.LBB0_20:
	s_or_b64 exec, exec, s[36:37]
	v_pk_add_f32 v[98:99], v[98:99], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v124, 0, v98
	v_add_f32_e32 v124, v124, v99
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
	v_add_f32_e32 v124, v124, v100
	v_add_f32_e32 v124, v124, v101
	v_add_f32_e32 v124, v124, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v124, v124, v103
	v_add_f32_e32 v124, v124, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v124, v124, v105
	v_add_f32_e32 v124, v124, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v124, v124, v107
	v_add_f32_e32 v124, v124, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v124, v124, v109
	v_add_f32_e32 v124, v124, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v117
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v124, v124, v111
	v_add_f32_e32 v124, v124, v112
	v_add_f32_e32 v124, v124, v113
	;;#ASMSTART
	v_fma_f32 v146, v138, v117, v124
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v117
	;;#ASMEND
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
	v_add_u32_e32 v117, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s45
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s37, s38, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s37, s37, s53
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s37
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[20:21]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[124:127], v[106:107], off offset:512
	global_load_dwordx4 v[128:131], v[106:107], off offset:1024
	global_load_dwordx4 v[132:135], v[106:107], off offset:1536
	global_load_dwordx4 v[136:139], v[106:107], off offset:2048
	global_load_dwordx4 v[180:183], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[180:181], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s46, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[126:127], v[102:103], v[18:33]
	global_load_dwordx4 v[124:127], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[102:103], v[34:49]
	global_load_dwordx4 v[128:131], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[102:103], v[50:65]
	global_load_dwordx4 v[132:135], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s47, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[182:183], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[126:127], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	s_lshl_b32 s36, s63, 12
	v_or_b32_e32 v98, s36, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[188:191], v[98:99], off
	ds_write_b128 v119, v[184:187] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s44, v100
	v_and_b32_e32 v124, 48, v98
	v_or_b32_e32 v98, v117, v124
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[126:129], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[148:149], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[150:151], v[98:113]
	v_or_b32_e32 v125, 16, v117
	v_xor_b32_e32 v125, v125, v124
	v_lshlrev_b32_e32 v125, 1, v125
	ds_read_b128 v[126:129], v125
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[152:153], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[154:155], v[98:113]
	v_or_b32_e32 v125, 32, v117
	v_xor_b32_e32 v125, v125, v124
	v_lshlrev_b32_e32 v125, 1, v125
	ds_read_b128 v[126:129], v125
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[158:159], v[98:113]
	v_or_b32_e32 v125, 48, v117
	v_xor_b32_e32 v124, v125, v124
	v_lshlrev_b32_e32 v124, 1, v124
	ds_read_b128 v[124:127], v124
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[162:163], v[98:113]
	v_or_b32_e32 v124, 64, v117
	v_lshrrev_b32_e32 v125, 3, v124
	v_and_b32_e32 v125, 56, v125
	v_xor_b32_e32 v124, v125, v124
	v_lshlrev_b32_e32 v124, 1, v124
	ds_read_b128 v[124:127], v124
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[166:167], v[98:113]
	v_or_b32_e32 v124, 0x50, v117
	v_lshrrev_b32_e32 v125, 3, v124
	v_and_b32_e32 v125, 56, v125
	v_xor_b32_e32 v124, v125, v124
	v_lshlrev_b32_e32 v124, 1, v124
	ds_read_b128 v[124:127], v124
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[170:171], v[98:113]
	v_or_b32_e32 v124, 0x60, v117
	v_lshrrev_b32_e32 v125, 3, v124
	v_and_b32_e32 v125, 56, v125
	v_xor_b32_e32 v124, v125, v124
	v_lshlrev_b32_e32 v124, 1, v124
	ds_read_b128 v[124:127], v124
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[174:175], v[98:113]
	v_or_b32_e32 v117, 0x70, v117
	v_lshrrev_b32_e32 v124, 3, v117
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v117, v124, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[178:179], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v117, v116
	s_nop 6
	v_pk_mul_f32 v[98:99], v[120:121], v[98:99]
	v_pk_mul_f32 v[100:101], v[116:117], v[100:101]
	v_max_f32_e32 v124, v98, v99
	v_pk_mul_f32 v[102:103], v[116:117], v[102:103]
	v_max3_f32 v124, v124, v100, v101
	v_pk_mul_f32 v[104:105], v[116:117], v[104:105]
	v_max3_f32 v124, v124, v102, v103
	v_pk_mul_f32 v[106:107], v[116:117], v[106:107]
	v_max3_f32 v124, v124, v104, v105
	v_pk_mul_f32 v[108:109], v[116:117], v[108:109]
	v_max3_f32 v124, v124, v106, v107
	v_pk_mul_f32 v[110:111], v[116:117], v[110:111]
	v_max3_f32 v124, v124, v108, v109
	v_pk_mul_f32 v[112:113], v[116:117], v[112:113]
	v_max3_f32 v124, v124, v110, v111
	v_max3_f32 v124, v124, v112, v113
	ds_bpermute_b32 v125, v141, v124
	v_mov_b32_e32 v126, v118
	v_mov_b32_e32 v127, v118
	v_mov_b32_e32 v128, v118
	v_mov_b32_e32 v129, v118
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v125, v125, v125
	v_max_f32_e32 v147, v124, v125
	;;#ASMSTART
	v_add_f32 v124, v118, v142
	;;#ASMEND
	v_mov_b32_e32 v125, v118
	v_cmp_gt_f32_e32 vcc, v147, v124
	v_mov_b32_e32 v124, v118
	v_mov_b32_e32 v130, v118
	v_mov_b32_e32 v131, v118
	v_mov_b32_e32 v132, v118
	v_mov_b32_e32 v133, v118
	v_mov_b32_e32 v134, v118
	v_mov_b32_e32 v135, v118
	v_mov_b32_e32 v136, v118
	v_mov_b32_e32 v137, v118
	v_mov_b32_e32 v138, v118
	v_mov_b32_e32 v139, v118
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_22
	;;#ASMSTART
	v_add_f32 v124, v147, v143
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v124
	v_exp_f32_e32 v145, v118
	v_mov_b32_e32 v118, v124
	v_mov_b32_e32 v125, v124
	v_mov_b32_e32 v126, v124
	v_mov_b32_e32 v127, v124
	v_mov_b32_e32 v128, v124
	v_mov_b32_e32 v129, v124
	v_mov_b32_e32 v130, v124
	v_mov_b32_e32 v131, v124
	v_mov_b32_e32 v132, v124
	v_mov_b32_e32 v133, v124
	v_mov_b32_e32 v134, v124
	v_mov_b32_e32 v135, v124
	v_mov_b32_e32 v136, v124
	v_mov_b32_e32 v137, v124
	v_mov_b32_e32 v138, v124
	v_mov_b32_e32 v139, v124
.LBB0_22:
	s_or_b64 exec, exec, s[38:39]
	v_pk_add_f32 v[98:99], v[98:99], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, 0, v98
	v_add_f32_e32 v147, v147, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v100
	v_add_f32_e32 v147, v147, v101
	v_add_f32_e32 v147, v147, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v103
	v_add_f32_e32 v147, v147, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v105
	v_add_f32_e32 v147, v147, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v107
	v_add_f32_e32 v147, v147, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v109
	v_add_f32_e32 v147, v147, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v145
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v147, v147, v111
	v_add_f32_e32 v147, v147, v112
	v_add_f32_e32 v147, v147, v113
	;;#ASMSTART
	v_fma_f32 v146, v146, v145, v147
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v145
	;;#ASMEND
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
	v_add_u32_e32 v145, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v145, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s45
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s37, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s37
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[20:21]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[180:183], v[106:107], off offset:512
	global_load_dwordx4 v[184:187], v[106:107], off offset:1024
	global_load_dwordx4 v[196:199], v[106:107], off offset:1536
	global_load_dwordx4 v[200:203], v[106:107], off offset:2048
	global_load_dwordx4 v[204:207], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[180:181], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[196:197], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[204:205], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s46, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[102:103], v[18:33]
	global_load_dwordx4 v[180:183], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[102:103], v[34:49]
	global_load_dwordx4 v[184:187], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[102:103], v[50:65]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s47, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[206:207], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[180:181], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[184:185], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[196:197], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[182:183], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[186:187], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	v_mov_b32_e32 v98, s62
	s_ashr_i32 s37, s36, 31
	buffer_load_dword v145, v98, s[12:15], 0 offen
	v_lshl_add_u64 v[98:99], s[36:37], 0, v[114:115]
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[180:183], v[98:99], off offset:512
	ds_write_b128 v119, v[192:195]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s57, v145
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v147, 48, v98
	v_and_or_b32 v184, v99, s44, v100
	v_or_b32_e32 v98, v184, v147
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[192:195], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[148:149], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[150:151], v[98:113]
	v_or_b32_e32 v145, 0x1010, v184
	v_xor_b32_e32 v145, v145, v147
	v_lshlrev_b32_e32 v145, 1, v145
	ds_read_b128 v[192:195], v145
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[152:153], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[154:155], v[98:113]
	v_or_b32_e32 v145, 0x1020, v184
	v_xor_b32_e32 v145, v145, v147
	v_lshlrev_b32_e32 v145, 1, v145
	ds_read_b128 v[192:195], v145
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[158:159], v[98:113]
	v_or_b32_e32 v145, 0x1030, v184
	v_xor_b32_e32 v145, v145, v147
	v_lshlrev_b32_e32 v145, 1, v145
	ds_read_b128 v[192:195], v145
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[162:163], v[98:113]
	v_or_b32_e32 v145, 0x1040, v184
	v_lshrrev_b32_e32 v147, 3, v145
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v145, v147, v145
	v_lshlrev_b32_e32 v145, 1, v145
	ds_read_b128 v[192:195], v145
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[166:167], v[98:113]
	v_or_b32_e32 v145, 0x1050, v184
	v_lshrrev_b32_e32 v147, 3, v145
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v145, v147, v145
	v_lshlrev_b32_e32 v145, 1, v145
	ds_read_b128 v[192:195], v145
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[170:171], v[98:113]
	v_or_b32_e32 v145, 0x1060, v184
	v_lshrrev_b32_e32 v147, 3, v145
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v145, v147, v145
	v_lshlrev_b32_e32 v145, 1, v145
	ds_read_b128 v[192:195], v145
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[174:175], v[98:113]
	v_or_b32_e32 v145, 0x1070, v184
	v_lshrrev_b32_e32 v147, 3, v145
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v145, v147, v145
	v_lshlrev_b32_e32 v145, 1, v145
	ds_read_b128 v[184:187], v145
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[184:185], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[186:187], v[178:179], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[120:121], v[98:99]
	v_pk_mul_f32 v[112:113], v[116:117], v[112:113]
	v_pk_mul_f32 v[110:111], v[116:117], v[110:111]
	v_pk_mul_f32 v[108:109], v[116:117], v[108:109]
	v_pk_mul_f32 v[106:107], v[116:117], v[106:107]
	v_pk_mul_f32 v[104:105], v[116:117], v[104:105]
	v_pk_mul_f32 v[102:103], v[116:117], v[102:103]
	v_pk_mul_f32 v[100:101], v[116:117], v[100:101]
	v_max_f32_e32 v117, v98, v99
	v_max3_f32 v117, v117, v100, v101
	v_max3_f32 v117, v117, v102, v103
	v_max3_f32 v117, v117, v104, v105
	v_max3_f32 v117, v117, v106, v107
	v_max3_f32 v117, v117, v108, v109
	v_max3_f32 v117, v117, v110, v111
	v_max3_f32 v117, v117, v112, v113
	ds_bpermute_b32 v145, v141, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v145, v145, v145
	v_max_f32_e32 v147, v117, v145
	;;#ASMSTART
	v_add_f32 v117, v118, v142
	;;#ASMEND
	v_mov_b32_e32 v145, 1.0
	v_cmp_gt_f32_e32 vcc, v147, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_24
	;;#ASMSTART
	v_add_f32 v124, v147, v143
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v124
	v_exp_f32_e32 v117, v117
	v_mov_b32_e32 v118, v124
	v_mov_b32_e32 v125, v124
	v_mov_b32_e32 v126, v124
	v_mov_b32_e32 v127, v124
	v_mov_b32_e32 v128, v124
	v_mov_b32_e32 v129, v124
	v_mov_b32_e32 v130, v124
	v_mov_b32_e32 v131, v124
	v_mov_b32_e32 v132, v124
	v_mov_b32_e32 v133, v124
	v_mov_b32_e32 v134, v124
	v_mov_b32_e32 v135, v124
	v_mov_b32_e32 v136, v124
	v_mov_b32_e32 v137, v124
	v_mov_b32_e32 v138, v124
	v_mov_b32_e32 v139, v124
.LBB0_24:
	s_or_b64 exec, exec, s[36:37]
	v_pk_add_f32 v[98:99], v[98:99], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, 0, v98
	v_add_f32_e32 v147, v147, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v100
	v_add_f32_e32 v147, v147, v101
	v_add_f32_e32 v147, v147, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v103
	v_add_f32_e32 v147, v147, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v105
	v_add_f32_e32 v147, v147, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v107
	v_add_f32_e32 v147, v147, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v109
	v_add_f32_e32 v147, v147, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v117
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v147, v147, v111
	v_add_f32_e32 v147, v147, v112
	v_add_f32_e32 v147, v147, v113
	;;#ASMSTART
	v_fma_f32 v146, v146, v117, v147
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v117
	;;#ASMEND
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
	v_add_u32_e32 v117, 0x8000, v100
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s45
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s38, s56, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s38, s38, s53
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s38
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[20:21]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[184:187], v[106:107], off offset:512
	global_load_dwordx4 v[192:195], v[106:107], off offset:1024
	global_load_dwordx4 v[196:199], v[106:107], off offset:1536
	global_load_dwordx4 v[200:203], v[106:107], off offset:2048
	global_load_dwordx4 v[204:207], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[196:197], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[204:205], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s46, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[186:187], v[102:103], v[18:33]
	global_load_dwordx4 v[184:187], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[102:103], v[34:49]
	global_load_dwordx4 v[192:195], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[102:103], v[50:65]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s47, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[206:207], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[184:185], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[196:197], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[186:187], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	v_lshl_or_b32 v98, s64, 12, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	global_load_dwordx4 v[184:187], v[98:99], off
	ds_write_b128 v119, v[188:191] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s44, v100
	v_and_b32_e32 v147, 48, v98
	v_or_b32_e32 v98, v117, v147
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[188:191], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[148:149], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[150:151], v[98:113]
	v_or_b32_e32 v188, 16, v117
	v_xor_b32_e32 v188, v188, v147
	v_lshlrev_b32_e32 v188, 1, v188
	ds_read_b128 v[188:191], v188
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[152:153], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[154:155], v[98:113]
	v_or_b32_e32 v188, 32, v117
	v_xor_b32_e32 v188, v188, v147
	v_lshlrev_b32_e32 v188, 1, v188
	ds_read_b128 v[188:191], v188
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[158:159], v[98:113]
	v_or_b32_e32 v188, 48, v117
	v_xor_b32_e32 v147, v188, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[188:191], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[162:163], v[98:113]
	v_or_b32_e32 v147, 64, v117
	v_lshrrev_b32_e32 v188, 3, v147
	v_and_b32_e32 v188, 56, v188
	v_xor_b32_e32 v147, v188, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[188:191], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[166:167], v[98:113]
	v_or_b32_e32 v147, 0x50, v117
	v_lshrrev_b32_e32 v188, 3, v147
	v_and_b32_e32 v188, 56, v188
	v_xor_b32_e32 v147, v188, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[188:191], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[170:171], v[98:113]
	v_or_b32_e32 v147, 0x60, v117
	v_lshrrev_b32_e32 v188, 3, v147
	v_and_b32_e32 v188, 56, v188
	v_xor_b32_e32 v147, v188, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[188:191], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[174:175], v[98:113]
	v_or_b32_e32 v117, 0x70, v117
	v_lshrrev_b32_e32 v147, 3, v117
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v117, v147, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[188:191], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[178:179], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v117, v116
	s_nop 6
	v_pk_mul_f32 v[98:99], v[120:121], v[98:99]
	v_pk_mul_f32 v[112:113], v[116:117], v[112:113]
	v_pk_mul_f32 v[110:111], v[116:117], v[110:111]
	v_pk_mul_f32 v[108:109], v[116:117], v[108:109]
	v_pk_mul_f32 v[106:107], v[116:117], v[106:107]
	v_pk_mul_f32 v[104:105], v[116:117], v[104:105]
	v_pk_mul_f32 v[102:103], v[116:117], v[102:103]
	v_pk_mul_f32 v[100:101], v[116:117], v[100:101]
	v_max_f32_e32 v117, v98, v99
	v_max3_f32 v117, v117, v100, v101
	v_max3_f32 v117, v117, v102, v103
	v_max3_f32 v117, v117, v104, v105
	v_max3_f32 v117, v117, v106, v107
	v_max3_f32 v117, v117, v108, v109
	v_max3_f32 v117, v117, v110, v111
	v_max3_f32 v117, v117, v112, v113
	ds_bpermute_b32 v147, v141, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v147, v147, v147
	v_max_f32_e32 v117, v117, v147
	;;#ASMSTART
	v_add_f32 v147, v118, v142
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v117, v147
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_17
	;;#ASMSTART
	v_add_f32 v124, v117, v143
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v124
	v_exp_f32_e32 v145, v117
	v_mov_b32_e32 v118, v124
	v_mov_b32_e32 v125, v124
	v_mov_b32_e32 v126, v124
	v_mov_b32_e32 v127, v124
	v_mov_b32_e32 v128, v124
	v_mov_b32_e32 v129, v124
	v_mov_b32_e32 v130, v124
	v_mov_b32_e32 v131, v124
	v_mov_b32_e32 v132, v124
	v_mov_b32_e32 v133, v124
	v_mov_b32_e32 v134, v124
	v_mov_b32_e32 v135, v124
	v_mov_b32_e32 v136, v124
	v_mov_b32_e32 v137, v124
	v_mov_b32_e32 v138, v124
	v_mov_b32_e32 v139, v124
	s_branch .LBB0_17
.LBB0_26:
	s_mov_b32 s56, s64
	s_mov_b32 s38, s63
	s_branch .LBB0_28
.LBB0_27:
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	v_mov_b32_e32 v66, v2
	v_mov_b32_e32 v67, v2
	v_mov_b32_e32 v68, v2
	v_mov_b32_e32 v69, v2
	v_mov_b32_e32 v70, v2
	v_mov_b32_e32 v71, v2
	v_mov_b32_e32 v72, v2
	v_mov_b32_e32 v73, v2
	v_mov_b32_e32 v74, v2
	v_mov_b32_e32 v75, v2
	v_mov_b32_e32 v76, v2
	v_mov_b32_e32 v77, v2
	v_mov_b32_e32 v78, v2
	v_mov_b32_e32 v79, v2
	v_mov_b32_e32 v80, v2
	v_mov_b32_e32 v81, v2
	v_mov_b32_e32 v82, v2
	v_mov_b32_e32 v83, v2
	v_mov_b32_e32 v84, v2
	v_mov_b32_e32 v85, v2
	v_mov_b32_e32 v86, v2
	v_mov_b32_e32 v87, v2
	v_mov_b32_e32 v88, v2
	v_mov_b32_e32 v89, v2
	v_mov_b32_e32 v90, v2
	v_mov_b32_e32 v91, v2
	v_mov_b32_e32 v92, v2
	v_mov_b32_e32 v93, v2
	v_mov_b32_e32 v94, v2
	v_mov_b32_e32 v95, v2
	v_mov_b32_e32 v96, v2
	v_mov_b32_e32 v97, v2
	v_mov_b32_e32 v138, v2
	v_mov_b32_e32 v118, 0xff800000
.LBB0_28:
	s_add_i32 s34, s19, s61
	s_add_i32 s36, s34, 63
	s_ashr_i32 s34, s36, 31
	s_lshr_b32 s34, s34, 26
	s_add_i32 s34, s36, s34
	s_ashr_i32 s39, s34, 6
	s_andn2_b32 s34, s34, 63
	s_cmp_lg_u32 s36, s34
	s_cselect_b64 s[34:35], -1, 0
	s_cmp_lt_i32 s36, 0
	s_cselect_b64 s[36:37], -1, 0
	s_and_b64 s[34:35], s[36:37], s[34:35]
	s_subb_u32 s34, s39, 0
	s_min_i32 s34, s34, s59
	s_cmp_ge_i32 s30, s34
	s_cbranch_scc1 .LBB0_42
	s_lshl_b32 s36, s60, 7
	s_ashr_i32 s35, s34, 31
	v_mov_b32_e32 v120, v116
	v_mov_b32_e32 v121, v116
	v_or_b32_e32 v139, s58, v140
	s_or_b32 s58, s36, 0x77
	s_lshl3_add_u32 s59, s60, 20
	s_branch .LBB0_32
.LBB0_30:
	s_or_b64 exec, exec, s[38:39]
	v_pk_add_f32 v[98:99], v[98:99], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, 0, v98
	v_add_f32_e32 v117, v117, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v100
	v_add_f32_e32 v117, v117, v101
	v_add_f32_e32 v117, v117, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v103
	v_add_f32_e32 v117, v117, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v105
	v_add_f32_e32 v117, v117, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v107
	v_add_f32_e32 v117, v117, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v109
	v_add_f32_e32 v117, v117, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_add_u32_e32 v105, 0x8000, v105
	v_add_f32_e32 v117, v117, v111
	v_add_f32_e32 v117, v117, v112
	v_add_f32_e32 v117, v117, v113
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_fma_f32 v138, v138, v145, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v145
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s45
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s62, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s62
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[20:21]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[122:125], v[106:107], off offset:512
	global_load_dwordx4 v[126:129], v[106:107], off offset:1024
	global_load_dwordx4 v[130:133], v[106:107], off offset:1536
	global_load_dwordx4 v[134:137], v[106:107], off offset:2048
	global_load_dwordx4 v[188:191], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[188:189], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s46, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[102:103], v[18:33]
	global_load_dwordx4 v[122:125], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[102:103], v[34:49]
	global_load_dwordx4 v[126:129], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[102:103], v[50:65]
	global_load_dwordx4 v[130:133], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s47, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[190:191], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
.LBB0_31:
	s_and_b64 s[36:37], s[36:37], exec
	s_cselect_b32 s38, s54, s56
	s_cselect_b32 s56, s57, s54
	s_cselect_b32 s54, s60, s57
	s_add_u32 s30, s30, 2
	s_addc_u32 s31, s31, 0
	v_mov_b64_e32 v[98:99], s[34:35]
	v_cmp_lt_i64_e32 vcc, s[30:31], v[98:99]
	s_addk_i32 s58, 0x80
	s_add_i32 s59, s59, 8
	s_mov_b32 s57, s61
	s_cbranch_vccz .LBB0_42
.LBB0_32:
	s_add_i32 s36, s59, -4
	v_mov_b32_e32 v98, s36
	s_lshl_b32 s36, s56, 12
	s_ashr_i32 s37, s36, 31
	buffer_load_dword v117, v98, s[12:15], 0 offen
	v_or_b32_e32 v98, s36, v114
	v_mov_b32_e32 v99, s37
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	ds_write_b128 v119, v[180:183]
	global_load_dwordx4 v[180:183], v[98:99], off offset:512
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s60, v117
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v122, 48, v98
	v_and_or_b32 v123, v99, s44, v100
	v_or_b32_e32 v98, v123, v122
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[124:127], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[148:149], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[150:151], v[98:113]
	v_or_b32_e32 v117, 0x1010, v123
	v_xor_b32_e32 v117, v117, v122
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[152:153], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[154:155], v[98:113]
	v_or_b32_e32 v117, 0x1020, v123
	v_xor_b32_e32 v117, v117, v122
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[158:159], v[98:113]
	v_or_b32_e32 v117, 0x1030, v123
	v_xor_b32_e32 v117, v117, v122
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[162:163], v[98:113]
	v_or_b32_e32 v117, 0x1040, v123
	v_lshrrev_b32_e32 v122, 3, v117
	v_and_b32_e32 v122, 56, v122
	v_xor_b32_e32 v117, v122, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[166:167], v[98:113]
	v_or_b32_e32 v117, 0x1050, v123
	v_lshrrev_b32_e32 v122, 3, v117
	v_and_b32_e32 v122, 56, v122
	v_xor_b32_e32 v117, v122, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[170:171], v[98:113]
	v_or_b32_e32 v117, 0x1060, v123
	v_lshrrev_b32_e32 v122, 3, v117
	v_and_b32_e32 v122, 56, v122
	v_xor_b32_e32 v117, v122, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[174:175], v[98:113]
	v_or_b32_e32 v117, 0x1070, v123
	v_lshrrev_b32_e32 v122, 3, v117
	v_and_b32_e32 v122, 56, v122
	v_xor_b32_e32 v117, v122, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[122:125], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[178:179], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	v_mov_b32_e32 v145, 1.0
	v_lshrrev_b32_e32 v122, 2, v117
	v_and_b32_e32 v122, 8, v122
	v_lshrrev_b32_e32 v117, 1, v117
	v_and_or_b32 v117, v117, s48, v139
	v_add_u32_e32 v124, s58, v122
	v_add_u32_e32 v117, s55, v117
	v_add_u32_e32 v122, 0xffffff89, v124
	v_cmp_lt_i32_e32 vcc, v122, v117
	s_nop 1
	v_cndmask_b32_e32 v123, v144, v99, vcc
	v_cmp_le_i32_e32 vcc, v122, v117
	v_add_u32_e32 v99, 0xffffff8c, v124
	s_nop 0
	v_cndmask_b32_e32 v122, v144, v98, vcc
	v_add_u32_e32 v98, 0xffffff8b, v124
	v_cmp_le_i32_e32 vcc, v98, v117
	s_nop 1
	v_cndmask_b32_e32 v98, v144, v100, vcc
	v_cmp_le_i32_e32 vcc, v99, v117
	v_add_u32_e32 v100, 0xffffff8d, v124
	s_nop 0
	v_cndmask_b32_e32 v99, v144, v101, vcc
	v_cmp_le_i32_e32 vcc, v100, v117
	v_add_u32_e32 v101, 0xffffff8e, v124
	s_nop 0
	v_cndmask_b32_e32 v100, v144, v102, vcc
	v_cmp_le_i32_e32 vcc, v101, v117
	v_add_u32_e32 v102, 0xffffff8f, v124
	s_nop 0
	v_cndmask_b32_e32 v101, v144, v103, vcc
	v_cmp_le_i32_e32 vcc, v102, v117
	v_add_u32_e32 v103, 0xffffff90, v124
	s_nop 0
	v_cndmask_b32_e32 v102, v144, v104, vcc
	v_cmp_le_i32_e32 vcc, v103, v117
	v_add_u32_e32 v104, 0xffffff99, v124
	s_nop 0
	v_cndmask_b32_e32 v103, v144, v105, vcc
	v_cmp_le_i32_e32 vcc, v104, v117
	v_add_u32_e32 v105, 0xffffff9a, v124
	s_nop 0
	v_cndmask_b32_e32 v104, v144, v106, vcc
	v_cmp_le_i32_e32 vcc, v105, v117
	v_add_u32_e32 v106, 0xffffff9b, v124
	s_nop 0
	v_cndmask_b32_e32 v105, v144, v107, vcc
	v_cmp_le_i32_e32 vcc, v106, v117
	v_add_u32_e32 v107, 0xffffff9c, v124
	s_nop 0
	v_cndmask_b32_e32 v106, v144, v108, vcc
	v_cmp_le_i32_e32 vcc, v107, v117
	v_add_u32_e32 v108, 0xffffff9d, v124
	s_nop 0
	v_cndmask_b32_e32 v107, v144, v109, vcc
	v_cmp_le_i32_e32 vcc, v108, v117
	v_add_u32_e32 v109, 0xffffff9e, v124
	s_nop 0
	v_cndmask_b32_e32 v108, v144, v110, vcc
	v_cmp_le_i32_e32 vcc, v109, v117
	v_add_u32_e32 v110, 0xffffff9f, v124
	s_nop 0
	v_cndmask_b32_e32 v109, v144, v111, vcc
	v_cmp_le_i32_e32 vcc, v110, v117
	v_add_u32_e32 v111, 0xffffffa0, v124
	s_nop 0
	v_cndmask_b32_e32 v110, v144, v112, vcc
	v_cmp_le_i32_e32 vcc, v111, v117
	v_mov_b32_e32 v117, v116
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v111, v144, v113, vcc
	v_pk_mul_f32 v[112:113], v[116:117], v[110:111]
	v_pk_mul_f32 v[110:111], v[116:117], v[108:109]
	v_pk_mul_f32 v[108:109], v[116:117], v[106:107]
	v_pk_mul_f32 v[106:107], v[116:117], v[104:105]
	v_pk_mul_f32 v[104:105], v[116:117], v[102:103]
	v_pk_mul_f32 v[102:103], v[116:117], v[100:101]
	v_pk_mul_f32 v[100:101], v[120:121], v[122:123]
	s_nop 0
	v_max_f32_e32 v122, v100, v101
	v_max3_f32 v122, v122, v98, v99
	v_max3_f32 v122, v122, v102, v103
	v_max3_f32 v122, v122, v104, v105
	v_max3_f32 v122, v122, v106, v107
	v_max3_f32 v122, v122, v108, v109
	v_max3_f32 v122, v122, v110, v111
	v_max3_f32 v122, v122, v112, v113
	ds_bpermute_b32 v123, v141, v122
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v123, v123, v123
	v_max_f32_e32 v123, v122, v123
	;;#ASMSTART
	v_add_f32 v122, v118, v142
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v123, v122
	v_mov_b32_e32 v122, 1.0
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_34
	;;#ASMSTART
	v_add_f32 v123, v123, v143
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v123
	v_exp_f32_e32 v122, v118
	v_mov_b32_e32 v118, v123
.LBB0_34:
	s_or_b64 exec, exec, s[36:37]
	v_pk_add_f32 v[100:101], v[100:101], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[98:99], v[98:99], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v123, 0, v100
	v_add_f32_e32 v123, v123, v101
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v123, v123, v98
	v_add_f32_e32 v123, v123, v99
	v_add_f32_e32 v123, v123, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v123, v123, v103
	v_add_f32_e32 v123, v123, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v123, v123, v105
	v_add_f32_e32 v123, v123, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v123, v123, v107
	v_add_f32_e32 v123, v123, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v123, v123, v109
	v_add_f32_e32 v123, v123, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_add_u32_e32 v110, 0x8000, v110
	v_add_f32_e32 v123, v123, v111
	v_add_f32_e32 v123, v123, v112
	v_add_f32_e32 v123, v123, v113
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v100, 0x8000, v100
	;;#ASMSTART
	v_fma_f32 v138, v138, v122, v123
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v122
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v122
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v100, v101, v100, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v99, v98, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s45
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s39, s38, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s39, s39, s53
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s39
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[20:21]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[122:125], v[106:107], off offset:512
	global_load_dwordx4 v[126:129], v[106:107], off offset:1024
	global_load_dwordx4 v[130:133], v[106:107], off offset:1536
	global_load_dwordx4 v[134:137], v[106:107], off offset:2048
	global_load_dwordx4 v[188:191], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[188:189], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s46, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[102:103], v[18:33]
	global_load_dwordx4 v[122:125], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[102:103], v[34:49]
	global_load_dwordx4 v[126:129], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[102:103], v[50:65]
	global_load_dwordx4 v[130:133], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s47, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[190:191], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	s_lshl_b32 s38, s54, 12
	v_or_b32_e32 v98, s38, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	ds_write_b128 v119, v[184:187] offset:8192
	global_load_dwordx4 v[184:187], v[98:99], off
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v122, v99, s44, v100
	v_and_b32_e32 v123, 48, v98
	v_or_b32_e32 v98, v122, v123
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[124:127], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[148:149], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[150:151], v[98:113]
	v_or_b32_e32 v124, 16, v122
	v_xor_b32_e32 v124, v124, v123
	v_lshlrev_b32_e32 v124, 1, v124
	ds_read_b128 v[124:127], v124
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[152:153], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[154:155], v[98:113]
	v_or_b32_e32 v124, 32, v122
	v_xor_b32_e32 v124, v124, v123
	v_lshlrev_b32_e32 v124, 1, v124
	ds_read_b128 v[124:127], v124
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[158:159], v[98:113]
	v_or_b32_e32 v124, 48, v122
	v_xor_b32_e32 v123, v124, v123
	v_lshlrev_b32_e32 v123, 1, v123
	ds_read_b128 v[124:127], v123
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[162:163], v[98:113]
	v_or_b32_e32 v123, 64, v122
	v_lshrrev_b32_e32 v124, 3, v123
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v123, v124, v123
	v_lshlrev_b32_e32 v123, 1, v123
	ds_read_b128 v[124:127], v123
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[166:167], v[98:113]
	v_or_b32_e32 v123, 0x50, v122
	v_lshrrev_b32_e32 v124, 3, v123
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v123, v124, v123
	v_lshlrev_b32_e32 v123, 1, v123
	ds_read_b128 v[124:127], v123
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[170:171], v[98:113]
	v_or_b32_e32 v123, 0x60, v122
	v_lshrrev_b32_e32 v124, 3, v123
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v123, v124, v123
	v_lshlrev_b32_e32 v123, 1, v123
	ds_read_b128 v[124:127], v123
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[174:175], v[98:113]
	v_or_b32_e32 v122, 0x70, v122
	v_lshrrev_b32_e32 v123, 3, v122
	v_and_b32_e32 v123, 56, v123
	v_xor_b32_e32 v122, v123, v122
	v_lshlrev_b32_e32 v122, 1, v122
	ds_read_b128 v[122:125], v122
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[178:179], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v122, v0
	;;#ASMEND
	v_mov_b32_e32 v125, v118
	v_lshrrev_b32_e32 v123, 2, v122
	v_and_b32_e32 v123, 8, v123
	v_lshrrev_b32_e32 v122, 1, v122
	v_and_or_b32 v122, v122, s48, v139
	v_add_u32_e32 v123, s58, v123
	v_add_u32_e32 v122, s55, v122
	v_add_u32_e32 v124, 0xffffffa9, v123
	v_cmp_lt_i32_e32 vcc, v124, v122
	v_mov_b32_e32 v126, v118
	v_mov_b32_e32 v127, v118
	v_cndmask_b32_e32 v99, v144, v99, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffab, v123
	v_mov_b32_e32 v128, v118
	v_cndmask_b32_e32 v98, v144, v98, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffac, v123
	v_pk_mul_f32 v[98:99], v[120:121], v[98:99]
	v_cndmask_b32_e32 v100, v144, v100, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffad, v123
	v_mov_b32_e32 v129, v118
	v_cndmask_b32_e32 v101, v144, v101, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffae, v123
	v_pk_mul_f32 v[100:101], v[116:117], v[100:101]
	v_cndmask_b32_e32 v102, v144, v102, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffaf, v123
	v_mov_b32_e32 v130, v118
	v_cndmask_b32_e32 v103, v144, v103, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffb0, v123
	v_pk_mul_f32 v[102:103], v[116:117], v[102:103]
	v_cndmask_b32_e32 v104, v144, v104, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffb9, v123
	v_mov_b32_e32 v131, v118
	v_cndmask_b32_e32 v105, v144, v105, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffba, v123
	v_pk_mul_f32 v[104:105], v[116:117], v[104:105]
	v_cndmask_b32_e32 v106, v144, v106, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffbb, v123
	v_mov_b32_e32 v132, v118
	v_cndmask_b32_e32 v107, v144, v107, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffbc, v123
	v_pk_mul_f32 v[106:107], v[116:117], v[106:107]
	v_cndmask_b32_e32 v108, v144, v108, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffbd, v123
	v_mov_b32_e32 v133, v118
	v_cndmask_b32_e32 v109, v144, v109, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffbe, v123
	v_pk_mul_f32 v[108:109], v[116:117], v[108:109]
	v_cndmask_b32_e32 v110, v144, v110, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_add_u32_e32 v124, 0xffffffbf, v123
	v_subrev_u32_e32 v123, 64, v123
	v_cndmask_b32_e32 v111, v144, v111, vcc
	v_cmp_le_i32_e32 vcc, v124, v122
	v_pk_mul_f32 v[110:111], v[116:117], v[110:111]
	v_mov_b32_e32 v124, v118
	v_cndmask_b32_e32 v112, v144, v112, vcc
	v_cmp_le_i32_e32 vcc, v123, v122
	v_mov_b32_e32 v123, v118
	v_mov_b32_e32 v134, v118
	v_cndmask_b32_e32 v113, v144, v113, vcc
	v_pk_mul_f32 v[112:113], v[116:117], v[112:113]
	v_max_f32_e32 v117, v98, v99
	v_max3_f32 v117, v117, v100, v101
	v_max3_f32 v117, v117, v102, v103
	v_max3_f32 v117, v117, v104, v105
	v_max3_f32 v117, v117, v106, v107
	v_max3_f32 v117, v117, v108, v109
	v_max3_f32 v117, v117, v110, v111
	v_max3_f32 v117, v117, v112, v113
	ds_bpermute_b32 v122, v141, v117
	v_mov_b32_e32 v135, v118
	v_mov_b32_e32 v136, v118
	v_mov_b32_e32 v137, v118
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v122, v122, v122
	v_max_f32_e32 v117, v117, v122
	;;#ASMSTART
	v_add_f32 v122, v118, v142
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v117, v122
	v_mov_b32_e32 v122, v118
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_36
	;;#ASMSTART
	v_add_f32 v122, v117, v143
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v122
	v_exp_f32_e32 v145, v117
	v_mov_b32_e32 v118, v122
	v_mov_b32_e32 v123, v122
	v_mov_b32_e32 v124, v122
	v_mov_b32_e32 v125, v122
	v_mov_b32_e32 v126, v122
	v_mov_b32_e32 v127, v122
	v_mov_b32_e32 v128, v122
	v_mov_b32_e32 v129, v122
	v_mov_b32_e32 v130, v122
	v_mov_b32_e32 v131, v122
	v_mov_b32_e32 v132, v122
	v_mov_b32_e32 v133, v122
	v_mov_b32_e32 v134, v122
	v_mov_b32_e32 v135, v122
	v_mov_b32_e32 v136, v122
	v_mov_b32_e32 v137, v122
.LBB0_36:
	s_or_b64 exec, exec, s[36:37]
	v_pk_add_f32 v[98:99], v[98:99], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, 0, v98
	v_add_f32_e32 v117, v117, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v100
	v_add_f32_e32 v117, v117, v101
	v_add_f32_e32 v117, v117, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v103
	v_add_f32_e32 v117, v117, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v105
	v_add_f32_e32 v117, v117, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v107
	v_add_f32_e32 v117, v117, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v117, v117, v109
	v_add_f32_e32 v117, v117, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_add_u32_e32 v105, 0x8000, v105
	v_add_f32_e32 v117, v117, v111
	v_add_f32_e32 v117, v117, v112
	v_add_f32_e32 v117, v117, v113
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_fma_f32 v138, v138, v145, v117
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v145
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v145
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s45
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s39, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s39
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[20:21]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[188:191], v[106:107], off offset:512
	global_load_dwordx4 v[192:195], v[106:107], off offset:1024
	global_load_dwordx4 v[196:199], v[106:107], off offset:1536
	global_load_dwordx4 v[200:203], v[106:107], off offset:2048
	global_load_dwordx4 v[204:207], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[188:189], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[196:197], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[204:205], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s46, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[102:103], v[18:33]
	global_load_dwordx4 v[188:191], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[102:103], v[34:49]
	global_load_dwordx4 v[192:195], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[102:103], v[50:65]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s47, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[206:207], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[188:189], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[196:197], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	s_add_i32 s39, s30, 1
	s_cmp_gt_i32 s34, s39
	s_cselect_b64 s[36:37], -1, 0
	s_cmp_le_i32 s34, s39
	s_cbranch_scc1 .LBB0_41
	v_mov_b32_e32 v98, s59
	s_ashr_i32 s39, s38, 31
	buffer_load_dword v117, v98, s[12:15], 0 offen
	v_lshl_add_u64 v[98:99], s[38:39], 0, v[114:115]
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	ds_write_b128 v119, v[180:183]
	global_load_dwordx4 v[180:183], v[98:99], off offset:512
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s61, v117
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v145, 48, v98
	v_and_or_b32 v146, v99, s44, v100
	v_or_b32_e32 v98, v146, v145
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[188:191], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[148:149], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[150:151], v[98:113]
	v_or_b32_e32 v117, 0x1010, v146
	v_xor_b32_e32 v117, v117, v145
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[188:191], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[152:153], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[154:155], v[98:113]
	v_or_b32_e32 v117, 0x1020, v146
	v_xor_b32_e32 v117, v117, v145
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[188:191], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[158:159], v[98:113]
	v_or_b32_e32 v117, 0x1030, v146
	v_xor_b32_e32 v117, v117, v145
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[188:191], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[162:163], v[98:113]
	v_or_b32_e32 v117, 0x1040, v146
	v_lshrrev_b32_e32 v145, 3, v117
	v_and_b32_e32 v145, 56, v145
	v_xor_b32_e32 v117, v145, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[188:191], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[166:167], v[98:113]
	v_or_b32_e32 v117, 0x1050, v146
	v_lshrrev_b32_e32 v145, 3, v117
	v_and_b32_e32 v145, 56, v145
	v_xor_b32_e32 v117, v145, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[188:191], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[170:171], v[98:113]
	v_or_b32_e32 v117, 0x1060, v146
	v_lshrrev_b32_e32 v145, 3, v117
	v_and_b32_e32 v145, 56, v145
	v_xor_b32_e32 v117, v145, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[188:191], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[174:175], v[98:113]
	v_or_b32_e32 v117, 0x1070, v146
	v_lshrrev_b32_e32 v145, 3, v117
	v_and_b32_e32 v145, 56, v145
	v_xor_b32_e32 v117, v145, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[188:191], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[178:179], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v145, 2, v117
	v_and_b32_e32 v145, 8, v145
	v_lshrrev_b32_e32 v117, 1, v117
	v_and_or_b32 v117, v117, s48, v139
	v_add_u32_e32 v145, s58, v145
	v_add_u32_e32 v117, s55, v117
	v_subrev_u32_e32 v146, 55, v145
	v_cmp_lt_i32_e32 vcc, v146, v117
	s_nop 1
	v_cndmask_b32_e32 v99, v144, v99, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 53, v145
	s_nop 0
	v_cndmask_b32_e32 v98, v144, v98, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 52, v145
	v_pk_mul_f32 v[98:99], v[120:121], v[98:99]
	v_cndmask_b32_e32 v100, v144, v100, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 51, v145
	s_nop 0
	v_cndmask_b32_e32 v101, v144, v101, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 50, v145
	s_nop 0
	v_cndmask_b32_e32 v102, v144, v102, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 49, v145
	s_nop 0
	v_cndmask_b32_e32 v103, v144, v103, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 48, v145
	s_nop 0
	v_cndmask_b32_e32 v104, v144, v104, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 39, v145
	s_nop 0
	v_cndmask_b32_e32 v105, v144, v105, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 38, v145
	s_nop 0
	v_cndmask_b32_e32 v106, v144, v106, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 37, v145
	s_nop 0
	v_cndmask_b32_e32 v107, v144, v107, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 36, v145
	s_nop 0
	v_cndmask_b32_e32 v108, v144, v108, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 35, v145
	s_nop 0
	v_cndmask_b32_e32 v109, v144, v109, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 34, v145
	s_nop 0
	v_cndmask_b32_e32 v110, v144, v110, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	v_subrev_u32_e32 v146, 33, v145
	v_subrev_u32_e32 v145, 32, v145
	v_cndmask_b32_e32 v111, v144, v111, vcc
	v_cmp_le_i32_e32 vcc, v146, v117
	s_nop 1
	v_cndmask_b32_e32 v112, v144, v112, vcc
	v_cmp_le_i32_e32 vcc, v145, v117
	v_mov_b32_e32 v117, v116
	v_pk_mul_f32 v[100:101], v[116:117], v[100:101]
	v_max_f32_e32 v145, v98, v99
	v_pk_mul_f32 v[102:103], v[116:117], v[102:103]
	v_max3_f32 v145, v145, v100, v101
	v_pk_mul_f32 v[104:105], v[116:117], v[104:105]
	v_max3_f32 v145, v145, v102, v103
	v_pk_mul_f32 v[106:107], v[116:117], v[106:107]
	v_max3_f32 v145, v145, v104, v105
	v_pk_mul_f32 v[108:109], v[116:117], v[108:109]
	v_max3_f32 v145, v145, v106, v107
	v_cndmask_b32_e32 v113, v144, v113, vcc
	v_pk_mul_f32 v[110:111], v[116:117], v[110:111]
	v_max3_f32 v145, v145, v108, v109
	v_pk_mul_f32 v[112:113], v[116:117], v[112:113]
	v_max3_f32 v145, v145, v110, v111
	v_max3_f32 v145, v145, v112, v113
	ds_bpermute_b32 v146, v141, v145
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v146, v146, v146
	v_max_f32_e32 v147, v145, v146
	;;#ASMSTART
	v_add_f32 v145, v118, v142
	;;#ASMEND
	v_mov_b32_e32 v146, 1.0
	v_cmp_gt_f32_e32 vcc, v147, v145
	v_mov_b32_e32 v145, 1.0
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_39
	;;#ASMSTART
	v_add_f32 v122, v147, v143
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v118, v118, v122
	v_exp_f32_e32 v146, v118
	v_mov_b32_e32 v118, v122
	v_mov_b32_e32 v123, v122
	v_mov_b32_e32 v124, v122
	v_mov_b32_e32 v125, v122
	v_mov_b32_e32 v126, v122
	v_mov_b32_e32 v127, v122
	v_mov_b32_e32 v128, v122
	v_mov_b32_e32 v129, v122
	v_mov_b32_e32 v130, v122
	v_mov_b32_e32 v131, v122
	v_mov_b32_e32 v132, v122
	v_mov_b32_e32 v133, v122
	v_mov_b32_e32 v134, v122
	v_mov_b32_e32 v135, v122
	v_mov_b32_e32 v136, v122
	v_mov_b32_e32 v137, v122
.LBB0_39:
	s_or_b64 exec, exec, s[38:39]
	v_pk_add_f32 v[98:99], v[98:99], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, 0, v98
	v_add_f32_e32 v147, v147, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v100
	v_add_f32_e32 v147, v147, v101
	v_add_f32_e32 v147, v147, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v103
	v_add_f32_e32 v147, v147, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v105
	v_add_f32_e32 v147, v147, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v107
	v_add_f32_e32 v147, v147, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v109
	v_add_f32_e32 v147, v147, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_add_u32_e32 v110, 0x8000, v110
	v_add_f32_e32 v147, v147, v111
	v_add_f32_e32 v147, v147, v112
	v_add_f32_e32 v147, v147, v113
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
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
	;;#ASMSTART
	v_fma_f32 v138, v138, v146, v147
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v146
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v146
	;;#ASMEND
	v_add_u32_e32 v146, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v146, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s45
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s62, s56, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s62, s62, s53
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s62
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[20:21]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[188:191], v[106:107], off offset:512
	global_load_dwordx4 v[192:195], v[106:107], off offset:1024
	global_load_dwordx4 v[196:199], v[106:107], off offset:1536
	global_load_dwordx4 v[200:203], v[106:107], off offset:2048
	global_load_dwordx4 v[204:207], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[188:189], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[196:197], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[200:201], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[204:205], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s46, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[102:103], v[18:33]
	global_load_dwordx4 v[188:191], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[102:103], v[34:49]
	global_load_dwordx4 v[192:195], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[102:103], v[50:65]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s47, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[202:203], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[206:207], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[188:189], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[196:197], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[190:191], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[198:199], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	v_lshl_or_b32 v98, s57, 12, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[0:1]
	ds_write_b128 v119, v[184:187] offset:8192
	global_load_dwordx4 v[184:187], v[98:99], off
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v146, v99, s44, v100
	v_and_b32_e32 v147, 48, v98
	v_or_b32_e32 v98, v146, v147
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[188:191], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[148:149], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[150:151], v[98:113]
	v_or_b32_e32 v188, 16, v146
	v_xor_b32_e32 v188, v188, v147
	v_lshlrev_b32_e32 v188, 1, v188
	ds_read_b128 v[188:191], v188
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[152:153], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[154:155], v[98:113]
	v_or_b32_e32 v188, 32, v146
	v_xor_b32_e32 v188, v188, v147
	v_lshlrev_b32_e32 v188, 1, v188
	ds_read_b128 v[188:191], v188
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[158:159], v[98:113]
	v_or_b32_e32 v188, 48, v146
	v_xor_b32_e32 v147, v188, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[188:191], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[162:163], v[98:113]
	v_or_b32_e32 v147, 64, v146
	v_lshrrev_b32_e32 v188, 3, v147
	v_and_b32_e32 v188, 56, v188
	v_xor_b32_e32 v147, v188, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[188:191], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[166:167], v[98:113]
	v_or_b32_e32 v147, 0x50, v146
	v_lshrrev_b32_e32 v188, 3, v147
	v_and_b32_e32 v188, 56, v188
	v_xor_b32_e32 v147, v188, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[188:191], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[170:171], v[98:113]
	v_or_b32_e32 v147, 0x60, v146
	v_lshrrev_b32_e32 v188, 3, v147
	v_and_b32_e32 v188, 56, v188
	v_xor_b32_e32 v147, v188, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[188:191], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[174:175], v[98:113]
	v_or_b32_e32 v146, 0x70, v146
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[188:191], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[188:189], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[190:191], v[178:179], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v146, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v147, 2, v146
	v_and_b32_e32 v147, 8, v147
	v_lshrrev_b32_e32 v146, 1, v146
	v_and_or_b32 v146, v146, s48, v139
	v_add_u32_e32 v147, s58, v147
	v_add_u32_e32 v146, s55, v146
	v_subrev_u32_e32 v188, 23, v147
	v_cmp_lt_i32_e32 vcc, v188, v146
	s_nop 1
	v_cndmask_b32_e32 v99, v144, v99, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_subrev_u32_e32 v188, 21, v147
	s_nop 0
	v_cndmask_b32_e32 v98, v144, v98, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_subrev_u32_e32 v188, 20, v147
	v_pk_mul_f32 v[98:99], v[120:121], v[98:99]
	v_cndmask_b32_e32 v100, v144, v100, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_subrev_u32_e32 v188, 19, v147
	s_nop 0
	v_cndmask_b32_e32 v101, v144, v101, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_subrev_u32_e32 v188, 18, v147
	v_pk_mul_f32 v[100:101], v[116:117], v[100:101]
	v_cndmask_b32_e32 v102, v144, v102, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_subrev_u32_e32 v188, 17, v147
	s_nop 0
	v_cndmask_b32_e32 v103, v144, v103, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_add_u32_e32 v188, -16, v147
	v_pk_mul_f32 v[102:103], v[116:117], v[102:103]
	v_cndmask_b32_e32 v104, v144, v104, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_add_u32_e32 v188, -7, v147
	s_nop 0
	v_cndmask_b32_e32 v105, v144, v105, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_add_u32_e32 v188, -6, v147
	v_pk_mul_f32 v[104:105], v[116:117], v[104:105]
	v_cndmask_b32_e32 v106, v144, v106, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_add_u32_e32 v188, -5, v147
	s_nop 0
	v_cndmask_b32_e32 v107, v144, v107, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_add_u32_e32 v188, -4, v147
	v_pk_mul_f32 v[106:107], v[116:117], v[106:107]
	v_cndmask_b32_e32 v108, v144, v108, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_add_u32_e32 v188, -3, v147
	s_nop 0
	v_cndmask_b32_e32 v109, v144, v109, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_add_u32_e32 v188, -2, v147
	v_pk_mul_f32 v[108:109], v[116:117], v[108:109]
	v_cndmask_b32_e32 v110, v144, v110, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_add_u32_e32 v188, -1, v147
	s_nop 0
	v_cndmask_b32_e32 v111, v144, v111, vcc
	v_cmp_le_i32_e32 vcc, v188, v146
	v_pk_mul_f32 v[110:111], v[116:117], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v144, v112, vcc
	v_cmp_le_i32_e32 vcc, v147, v146
	s_nop 1
	v_cndmask_b32_e32 v113, v144, v113, vcc
	v_pk_mul_f32 v[112:113], v[116:117], v[112:113]
	v_max_f32_e32 v117, v98, v99
	v_max3_f32 v117, v117, v100, v101
	v_max3_f32 v117, v117, v102, v103
	v_max3_f32 v117, v117, v104, v105
	v_max3_f32 v117, v117, v106, v107
	v_max3_f32 v117, v117, v108, v109
	v_max3_f32 v117, v117, v110, v111
	v_max3_f32 v117, v117, v112, v113
	ds_bpermute_b32 v146, v141, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v146, v146, v146
	v_max_f32_e32 v117, v117, v146
	;;#ASMSTART
	v_add_f32 v146, v118, v142
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v117, v146
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_30
	;;#ASMSTART
	v_add_f32 v122, v117, v143
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v122
	v_exp_f32_e32 v145, v117
	v_mov_b32_e32 v118, v122
	v_mov_b32_e32 v123, v122
	v_mov_b32_e32 v124, v122
	v_mov_b32_e32 v125, v122
	v_mov_b32_e32 v126, v122
	v_mov_b32_e32 v127, v122
	v_mov_b32_e32 v128, v122
	v_mov_b32_e32 v129, v122
	v_mov_b32_e32 v130, v122
	v_mov_b32_e32 v131, v122
	v_mov_b32_e32 v132, v122
	v_mov_b32_e32 v133, v122
	v_mov_b32_e32 v134, v122
	v_mov_b32_e32 v135, v122
	v_mov_b32_e32 v136, v122
	v_mov_b32_e32 v137, v122
	s_branch .LBB0_30
.LBB0_41:
	s_mov_b32 s61, s60
	s_branch .LBB0_31
.LBB0_42:
	ds_bpermute_b32 v98, v141, v138
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v98, v138, v98
	;;#ASMEND
	s_nop 0
	v_div_scale_f32 v99, s[0:1], v98, v98, s42
	v_rcp_f32_e32 v100, v99
	v_div_scale_f32 v101, vcc, s42, v98, s42
	v_fma_f32 v102, -v99, v100, 1.0
	v_fmac_f32_e32 v100, v102, v100
	v_mul_f32_e32 v102, v101, v100
	v_fma_f32 v103, -v99, v102, v101
	v_fmac_f32_e32 v102, v103, v100
	v_fma_f32 v99, -v99, v102, v101
	v_div_fmas_f32 v99, v99, v100, v102
	v_div_fixup_f32 v98, v99, v98, s42
	v_pk_mul_f32 v[2:3], v[2:3], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[98:99] op_sel_hi:[1,0]
	v_add_u32_e32 v50, 0x8000, v50
	v_add_u32_e32 v33, 0x8000, v33
	v_add_u32_e32 v32, 0x8000, v32
	v_add_u32_e32 v31, 0x8000, v31
	v_add_u32_e32 v30, 0x8000, v30
	v_add_u32_e32 v29, 0x8000, v29
	v_add_u32_e32 v28, 0x8000, v28
	v_add_u32_e32 v27, 0x8000, v27
	v_add_u32_e32 v26, 0x8000, v26
	v_add_u32_e32 v25, 0x8000, v25
	v_add_u32_e32 v24, 0x8000, v24
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
	v_add_u32_e32 v97, 0x8000, v97
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
	v_add_u32_e32 v83, 0x8000, v83
	v_add_u32_e32 v82, 0x8000, v82
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
	v_add_u32_e32 v98, 0x8000, v49
	v_add_u32_e32 v99, 0x8000, v48
	v_add_u32_e32 v100, 0x8000, v47
	v_add_u32_e32 v101, 0x8000, v46
	v_add_u32_e32 v102, 0x8000, v45
	v_add_u32_e32 v103, 0x8000, v44
	v_add_u32_e32 v104, 0x8000, v43
	v_add_u32_e32 v105, 0x8000, v42
	v_add_u32_e32 v106, 0x8000, v41
	v_add_u32_e32 v107, 0x8000, v40
	v_add_u32_e32 v108, 0x8000, v39
	v_add_u32_e32 v109, 0x8000, v38
	v_add_u32_e32 v110, 0x8000, v37
	v_add_u32_e32 v111, 0x8000, v36
	v_add_u32_e32 v112, 0x8000, v35
	v_add_u32_e32 v113, 0x8000, v34
	;;#ASMSTART
	v_perm_b32 v48, v3, v2, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v49, v5, v4, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v46, v7, v6, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v47, v9, v8, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v44, v11, v10, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v45, v13, v12, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v42, v15, v14, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v43, v17, v16, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v40, v19, v18, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v41, v21, v20, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v38, v23, v22, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v39, v25, v24, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v36, v27, v26, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v37, v29, v28, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v34, v31, v30, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v35, v33, v32, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v112, v113, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v110, v111, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v108, v109, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v106, v107, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v28, v104, v105, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v102, v103, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v100, v101, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v98, v99, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v51, v50, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v53, v52, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v55, v54, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v57, v56, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v59, v58, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v61, v60, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v63, v62, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v65, v64, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v67, v66, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v69, v68, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v71, v70, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v73, v72, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v75, v74, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v77, v76, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v79, v78, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v81, v80, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v83, v82, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v85, v84, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v87, v86, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v89, v88, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v91, v90, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v93, v92, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v95, v94, s45
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v97, v96, s45
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v50, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v50, 0x100, v50
	v_cmp_eq_u32_e32 vcc, 0, v50
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_44
	s_barrier
.LBB0_44:
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
	s_cbranch_execz .LBB0_46
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
.LBB0_46:
	s_or_b64 exec, exec, s[0:1]
	s_mul_i32 s0, s52, 0xc00
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_and_b32_e32 v88, 0x1ff, v91
	s_add_u32 s12, s26, s0
	s_addc_u32 s0, s27, s1
	v_lshlrev_b32_e32 v92, 3, v88
	s_mul_i32 s14, s19, 0x1800
	s_and_b32 s13, s0, 0xffff
	s_mul_i32 s19, s43, 0xc0
	v_and_b32_e32 v91, 56, v91
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_47:
	v_mul_u32_u24_sdwa v95, v94, s49 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s50, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s19, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[12:15], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_47
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 1, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_50
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
.LBB0_50:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s19, s19, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_51:
	v_mul_u32_u24_sdwa v95, v94, s49 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s50, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s19, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[12:15], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_51
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 2, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_54
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
.LBB0_54:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s30, s19, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_55:
	v_mul_u32_u24_sdwa v95, v94, s49 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s50, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s30, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[12:15], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_55
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 3, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_58
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
.LBB0_58:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s30, s19, 0x30000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_59:
	v_mul_u32_u24_sdwa v95, v94, s49 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s50, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s30, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[12:15], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_59
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 4, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_62
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
.LBB0_62:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s30, s19, 0x48000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_63:
	v_mul_u32_u24_sdwa v95, v94, s49 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s50, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s30, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[12:15], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_63
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 5, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_66
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
.LBB0_66:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s30, s19, 0x60000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_67:
	v_mul_u32_u24_sdwa v95, v94, s49 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s50, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s30, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[12:15], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_67
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 6, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_70
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
.LBB0_70:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s30, s19, 0x78000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_71:
	v_mul_u32_u24_sdwa v95, v94, s49 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s50, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s30, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[12:15], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_71
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 7, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_74
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
.LBB0_74:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s19, s19, 0x90000
	s_mov_b64 s[0:1], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_75:
	v_mul_u32_u24_sdwa v2, v88, s49 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v3, v92, v91
	v_lshrrev_b32_e32 v2, 20, v2
	v_lshlrev_b32_e32 v3, 1, v3
	v_mul_lo_u16_e32 v5, 24, v2
	ds_read_b128 v[6:9], v3
	v_sub_u16_e32 v3, v88, v5
	v_lshlrev_b16_e32 v3, 3, v3
	v_add_u32_e32 v4, 0x200, v88
	v_cmp_lt_u32_e32 vcc, s50, v88
	v_mul_u32_u24_e32 v2, 0xc00, v2
	v_add_u32_e32 v3, s19, v3
	v_add_u32_e32 v92, 0x1000, v92
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v88, v4
	v_add_lshl_u32 v2, v3, v2, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v2, s[12:15], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_75
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	;;#ASMSTART
	s_mov_b32 s0, s51
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_80
	s_mov_b64 s[34:35], exec
	v_mbcnt_lo_u32_b32 v2, s34, 0
	v_mbcnt_hi_u32_b32 v2, s35, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB0_79
	s_bcnt1_i32_b64 s14, s[34:35]
	v_mov_b32_e32 v3, s14
	global_atomic_add v3, v115, v3, s[28:29] sc0
.LBB0_79:
	s_or_b64 exec, exec, s[30:31]
	s_lshl_b64 s[30:31], s[0:1], 2
	s_add_u32 s30, s28, s30
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s14, v3
	s_addc_u32 s31, s29, s31
	s_nop 0
	v_add_u32_e32 v2, s14, v2
	global_store_dword v115, v2, s[30:31]
	s_waitcnt vmcnt(0)
.LBB0_80:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s28, s0
	s_addc_u32 s1, s29, s1
	s_barrier
	global_load_dword v2, v115, s[0:1]
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s12, v2
	s_add_i32 s0, s12, s33
	s_sub_i32 s33, s0, s2
	s_cmp_ge_i32 s18, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s40
	s_cselect_b64 s[30:31], -1, 0
	s_or_b64 s[0:1], s[0:1], s[30:31]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_83
	s_branch .LBB0_12
.LBB0_81:
	s_mov_b32 s18, s13
.LBB0_82:
	s_sub_i32 s33, s33, s40
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s43, 0, s2
	s_cmp_ge_i32 s18, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s14
	s_cselect_b64 s[30:31], -1, 0
	s_or_b64 s[0:1], s[0:1], s[30:31]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s40, s14
	s_cbranch_vccnz .LBB0_11
.LBB0_83:
	s_add_i32 s2, s43, 1
	s_cmp_gt_i32 s2, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s2, 16
	s_cbranch_scc1 .LBB0_86
	s_add_i32 s13, s18, 1
	s_cmp_ge_i32 s13, s3
	s_mov_b32 s14, s40
	s_cbranch_scc1 .LBB0_81
	s_ashr_i32 s19, s18, 31
	s_lshl_b64 s[18:19], s[18:19], 2
	s_add_u32 s18, s16, s18
	s_addc_u32 s19, s17, s19
	global_load_dwordx2 v[2:3], v115, s[18:19] offset:4
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s14, v3
	v_readfirstlane_b32 s18, v2
	s_sub_i32 s14, s14, s18
	s_addk_i32 s14, 0xff
	s_ashr_i32 s18, s14, 31
	s_lshr_b32 s18, s18, 24
	s_add_i32 s18, s14, s18
	s_ashr_i32 s34, s18, 8
	s_and_b32 s18, s18, 0xffffff00
	s_cmp_lg_u32 s14, s18
	s_cselect_b64 s[18:19], -1, 0
	s_cmp_lt_i32 s14, 0
	s_cselect_b64 s[30:31], -1, 0
	s_and_b64 s[18:19], s[30:31], s[18:19]
	s_subb_u32 s14, s34, 0
	s_branch .LBB0_81
.LBB0_86:
	s_mov_b32 s14, s40
	s_branch .LBB0_82
.LBB0_87:
	s_endpgm
.Lfunc_end0:
	.size	attn_kernel_0, .Lfunc_end0-attn_kernel_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_kernel_0
		.amdhsa_group_segment_fixed_size 16384
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
		.amdhsa_next_free_vgpr 208
		.amdhsa_next_free_sgpr 65
		.amdhsa_accum_offset 208
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

	.set .Lattn_kernel_0.num_vgpr, 208
	.set .Lattn_kernel_0.num_agpr, 0
	.set .Lattn_kernel_0.numbered_sgpr, 65
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
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 196
    .max_flat_workgroup_size: 512
    .name:           attn_kernel_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 512
      - 1
      - 1
    .sgpr_count:     71
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     208
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
