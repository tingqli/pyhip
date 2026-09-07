	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	attn_kernel_0
	.p2align	8
	.type	attn_kernel_0,@function
attn_kernel_0:
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dword s3, s[0:1], 0x38
	s_load_dwordx2 s[44:45], s[0:1], 0x90
	s_load_dwordx4 s[4:7], s[0:1], 0x80
	s_mov_b32 s46, 0
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
	s_mov_b32 s77, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s46, s12
.LBB0_3:
	s_sub_i32 s33, s33, s10
	s_and_b64 s[8:9], s[8:9], exec
	s_cselect_b32 s77, 0, s11
	s_cmp_ge_i32 s46, s3
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s33, s74
	s_cselect_b64 s[10:11], -1, 0
	s_or_b64 s[8:9], s[8:9], s[10:11]
	s_and_b64 vcc, exec, s[8:9]
	s_mov_b32 s10, s74
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s11, s77, 1
	s_cmp_gt_i32 s11, 15
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s11, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s12, s46, 1
	s_cmp_ge_i32 s12, s3
	s_mov_b32 s74, s10
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s47, s46, 31
	s_lshl_b64 s[14:15], s[46:47], 2
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
	s_subb_u32 s74, s18, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s74, s10
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s74, s10
	s_mov_b32 s77, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s75, s[4:5], 0x0
	s_load_dword s76, s[6:7], 0x0
	s_cmp_ge_i32 s46, s3
	s_cbranch_scc1 .LBB0_126
	v_lshrrev_b32_e32 v3, 6, v0
	s_load_dwordx2 s[48:49], s[0:1], 0x0
	s_load_dwordx2 s[50:51], s[0:1], 0x10
	s_load_dwordx2 s[52:53], s[0:1], 0x20
	s_load_dwordx2 s[54:55], s[0:1], 0x50
	s_load_dwordx2 s[56:57], s[0:1], 0x60
	s_load_dwordx2 s[58:59], s[0:1], 0x70
	s_load_dwordx2 s[60:61], s[0:1], 0xa0
	s_load_dwordx2 s[62:63], s[0:1], 0xc0
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v2, 31, v0
	s_movk_i32 s0, 0xe0
	v_lshrrev_b32_e32 v4, 2, v0
	v_mul_u32_u24_e32 v3, 0x18000, v3
	v_and_or_b32 v1, v1, s0, v2
	v_and_or_b32 v3, v4, 8, v3
	s_movk_i32 s0, 0xc00
	v_mad_u32_u24 v117, v2, s0, v3
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
	v_lshl_or_b32 v3, v3, 8, v5
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_u32_e32 v3, v3, v4
	v_and_or_b32 v148, v2, 12, v3
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v150, v2, v3
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v1, 6, v1
	v_or_b32_e32 v149, 0x80, v148
	v_xor_b32_e32 v151, 0x80, v2
	v_mov_b32_e32 v152, 0
	s_mov_b32 s39, 0x27000
	s_movk_i32 s78, 0x180
	v_mov_b32_e32 v153, 0x40e00000
	v_mov_b32_e32 v154, 1.0
	s_mov_b32 s79, 0x7060302
	s_movk_i32 s80, 0x1000
	s_movk_i32 s81, 0x2000
	s_mov_b32 s82, 0xaaab
	s_movk_i32 s83, 0xff
	s_mov_b32 s84, s2
	v_mov_b32_e32 v155, 0xff800000
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s74, s6
.LBB0_12:
	s_cmp_ge_i32 s46, s3
	s_cbranch_scc1 .LBB0_126
.LBB0_13:
	s_ashr_i32 s47, s46, 31
	s_lshl_b32 s6, s33, 8
	s_lshl_b64 s[0:1], s[46:47], 2
	s_add_u32 s4, s34, s0
	s_addc_u32 s5, s35, s1
	global_load_dwordx2 v[2:3], v152, s[4:5]
	s_mul_i32 s47, s77, 0xc0
	s_mov_b32 s43, s39
	v_add_lshl_u32 v7, s47, v117, 1
	v_lshl_add_u32 v6, s77, 2, v1
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s4, v2
	s_add_i32 s8, s4, s6
	v_readfirstlane_b32 s5, v3
	s_add_i32 s6, s8, 0x100
	s_min_i32 s5, s6, s5
	s_sub_i32 s10, s5, s8
	s_waitcnt lgkmcnt(0)
	s_add_u32 s6, s54, s0
	s_addc_u32 s7, s55, s1
	s_mul_i32 s4, s8, 0xc00
	s_add_u32 s0, s44, s0
	s_addc_u32 s1, s45, s1
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s8, s8, 4
	s_lshl_b64 s[64:65], s[4:5], 1
	s_add_u32 s40, s48, s64
	global_load_dwordx2 v[4:5], v152, s[6:7]
	global_load_dword v3, v152, s[0:1]
	s_addc_u32 s0, s49, s65
	s_mul_i32 s42, s10, 0x1800
	s_and_b32 s41, s0, 0xffff
	buffer_load_dwordx4 v[176:179], v7, s[40:43], 0 offen
	buffer_load_dwordx4 v[180:183], v7, s[40:43], 0 offen offset:32
	buffer_load_dwordx4 v[184:187], v7, s[40:43], 0 offen offset:64
	buffer_load_dwordx4 v[188:191], v7, s[40:43], 0 offen offset:96
	buffer_load_dwordx4 v[192:195], v7, s[40:43], 0 offen offset:128
	buffer_load_dwordx4 v[196:199], v7, s[40:43], 0 offen offset:160
	buffer_load_dwordx4 v[200:203], v7, s[40:43], 0 offen offset:192
	buffer_load_dwordx4 v[204:207], v7, s[40:43], 0 offen offset:224
	buffer_load_dwordx4 v[208:211], v7, s[40:43], 0 offen offset:256
	buffer_load_dwordx4 v[212:215], v7, s[40:43], 0 offen offset:288
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[0:1], s[8:9], 2
	s_add_u32 s36, s58, s0
	s_addc_u32 s0, s59, s1
	s_lshl_b32 s38, s10, 6
	s_and_b32 s37, s0, 0xffff
	buffer_load_dword v2, v6, s[36:39], 0 offen
	buffer_load_dwordx4 v[216:219], v7, s[40:43], 0 offen offset:320
	buffer_load_dwordx4 v[220:223], v7, s[40:43], 0 offen offset:352
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
	s_sub_i32 s40, s1, s0
	s_ashr_i32 s1, s77, 31
	s_lshr_b32 s1, s1, 28
	s_add_i32 s1, s77, s1
	s_ashr_i32 s7, s1, 4
	s_and_b32 s1, s1, -16
	s_cmp_lg_u32 s77, s1
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s77, 0
	s_cselect_b64 s[8:9], -1, 0
	s_and_b64 s[4:5], s[8:9], s[4:5]
	s_subb_u32 s1, s7, 0
	s_mul_i32 s66, s1, 0x3000
	s_ashr_i32 s67, s66, 31
	s_lshl_b64 s[4:5], s[66:67], 1
	s_add_u32 s68, s50, s4
	s_addc_u32 s69, s51, s5
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s36, s56, s0
	s_addc_u32 s0, s57, s1
	s_lshl_b32 s38, s40, 2
	s_and_b32 s37, s0, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[36:39], 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s72, v4
	v_readfirstlane_b32 s67, v5
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
	buffer_load_dword v3, off, s[36:39], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_mul_i32 s4, s72, 0x1800
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s78, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_17
	v_add_u32_e32 v4, s4, v148
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[68:69]
	global_load_dwordx4 v[160:163], v[4:5], off
	global_load_dwordx4 v[164:167], v[4:5], off offset:256
.LBB0_17:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	buffer_load_dword v3, off, s[36:39], 0 offset:12
	;;#ASMSTART
	v_mov_b32 v4, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s85, v3
	v_and_b32_e32 v4, 0x180, v4
	v_cmp_ne_u32_e32 vcc, s78, v4
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_19
	v_add_u32_e32 v4, s4, v149
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[68:69]
	global_load_dwordx4 v[168:171], v[4:5], off
	global_load_dwordx4 v[172:175], v[4:5], off offset:256
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
	v_cmp_ne_u32_e32 vcc, s78, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_21
	ds_write_b128 v150, v[160:163] offset:12288
	ds_write_b128 v150, v[164:167] offset:18432
.LBB0_21:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s78, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_23
	s_mul_i32 s4, s67, 0x1800
	v_add_u32_e32 v4, s4, v148
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[68:69]
	global_load_dwordx4 v[160:163], v[4:5], off
	global_load_dwordx4 v[164:167], v[4:5], off offset:256
.LBB0_23:
	s_or_b64 exec, exec, s[0:1]
	v_mul_f32_e32 v2, s75, v2
	s_add_i32 s7, s40, -1
	v_mul_f32_e32 v114, 0x3dd53b95, v2
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_add_i32 s0, s40, -2
	s_bitcmp0_b32 s40, 0
	s_cselect_b32 s70, s0, s7
	s_ashr_i32 s71, s70, 31
	s_cmp_lt_i32 s70, 1
	s_cbranch_scc1 .LBB0_51
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v115, v114
	v_mov_b32_e32 v118, v114
	v_mov_b32_e32 v119, v114
	v_mov_b32_e32 v120, v114
	v_mov_b32_e32 v121, v114
	v_mov_b32_e32 v122, v114
	v_mov_b32_e32 v123, v114
	v_mov_b32_e32 v124, v114
	v_mov_b32_e32 v125, v114
	v_mov_b32_e32 v126, v114
	v_mov_b32_e32 v127, v114
	v_mov_b32_e32 v128, v114
	v_mov_b32_e32 v129, v114
	v_mov_b32_e32 v130, v114
	v_mov_b32_e32 v131, v114
	s_mov_b64 s[0:1], 0
	s_mov_b32 s8, 20
	v_mov_b32_e32 v116, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v156
	v_mov_b32_e32 v4, v156
	v_mov_b32_e32 v5, v156
	v_mov_b32_e32 v6, v156
	v_mov_b32_e32 v7, v156
	v_mov_b32_e32 v8, v156
	v_mov_b32_e32 v9, v156
	v_mov_b32_e32 v10, v156
	v_mov_b32_e32 v11, v156
	v_mov_b32_e32 v12, v156
	v_mov_b32_e32 v13, v156
	v_mov_b32_e32 v14, v156
	v_mov_b32_e32 v15, v156
	v_mov_b32_e32 v16, v156
	v_mov_b32_e32 v17, v156
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v156
	v_mov_b32_e32 v20, v156
	v_mov_b32_e32 v21, v156
	v_mov_b32_e32 v22, v156
	v_mov_b32_e32 v23, v156
	v_mov_b32_e32 v24, v156
	v_mov_b32_e32 v25, v156
	v_mov_b32_e32 v26, v156
	v_mov_b32_e32 v27, v156
	v_mov_b32_e32 v28, v156
	v_mov_b32_e32 v29, v156
	v_mov_b32_e32 v30, v156
	v_mov_b32_e32 v31, v156
	v_mov_b32_e32 v32, v156
	v_mov_b32_e32 v33, v156
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v156
	v_mov_b32_e32 v36, v156
	v_mov_b32_e32 v37, v156
	v_mov_b32_e32 v38, v156
	v_mov_b32_e32 v39, v156
	v_mov_b32_e32 v40, v156
	v_mov_b32_e32 v41, v156
	v_mov_b32_e32 v42, v156
	v_mov_b32_e32 v43, v156
	v_mov_b32_e32 v44, v156
	v_mov_b32_e32 v45, v156
	v_mov_b32_e32 v46, v156
	v_mov_b32_e32 v47, v156
	v_mov_b32_e32 v48, v156
	v_mov_b32_e32 v49, v156
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v156
	v_mov_b32_e32 v52, v156
	v_mov_b32_e32 v53, v156
	v_mov_b32_e32 v54, v156
	v_mov_b32_e32 v55, v156
	v_mov_b32_e32 v56, v156
	v_mov_b32_e32 v57, v156
	v_mov_b32_e32 v58, v156
	v_mov_b32_e32 v59, v156
	v_mov_b32_e32 v60, v156
	v_mov_b32_e32 v61, v156
	v_mov_b32_e32 v62, v156
	v_mov_b32_e32 v63, v156
	v_mov_b32_e32 v64, v156
	v_mov_b32_e32 v65, v156
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, v156
	v_mov_b32_e32 v68, v156
	v_mov_b32_e32 v69, v156
	v_mov_b32_e32 v70, v156
	v_mov_b32_e32 v71, v156
	v_mov_b32_e32 v72, v156
	v_mov_b32_e32 v73, v156
	v_mov_b32_e32 v74, v156
	v_mov_b32_e32 v75, v156
	v_mov_b32_e32 v76, v156
	v_mov_b32_e32 v77, v156
	v_mov_b32_e32 v78, v156
	v_mov_b32_e32 v79, v156
	v_mov_b32_e32 v80, v156
	v_mov_b32_e32 v81, v156
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, v156
	v_mov_b32_e32 v84, v156
	v_mov_b32_e32 v85, v156
	v_mov_b32_e32 v86, v156
	v_mov_b32_e32 v87, v156
	v_mov_b32_e32 v88, v156
	v_mov_b32_e32 v89, v156
	v_mov_b32_e32 v90, v156
	v_mov_b32_e32 v91, v156
	v_mov_b32_e32 v92, v156
	v_mov_b32_e32 v93, v156
	v_mov_b32_e32 v94, v156
	v_mov_b32_e32 v95, v156
	v_mov_b32_e32 v96, v156
	v_mov_b32_e32 v97, v156
	s_branch .LBB0_26
.LBB0_25:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, 0, v98
	v_add_f32_e32 v132, v132, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, v132, v100
	v_add_f32_e32 v132, v132, v101
	v_add_f32_e32 v132, v132, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, v132, v103
	v_add_f32_e32 v132, v132, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, v132, v105
	v_add_f32_e32 v132, v132, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, v132, v107
	v_add_f32_e32 v132, v132, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, v132, v109
	v_add_f32_e32 v132, v132, v110
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
	v_add_f32_e32 v132, v132, v111
	v_add_f32_e32 v132, v132, v112
	v_add_f32_e32 v132, v132, v113
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_fma_f32 v156, v156, v157, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v157
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v132, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v132, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s11, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s11
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[52:53]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[132:135], v[106:107], off offset:512
	global_load_dwordx4 v[136:139], v[106:107], off offset:1024
	global_load_dwordx4 v[140:143], v[106:107], off offset:1536
	global_load_dwordx4 v[144:147], v[106:107], off offset:2048
	global_load_dwordx4 v[224:227], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[136:137], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[224:225], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s80, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[102:103], v[18:33]
	global_load_dwordx4 v[132:135], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[138:139], v[102:103], v[34:49]
	global_load_dwordx4 v[136:139], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[102:103], v[50:65]
	global_load_dwordx4 v[140:143], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s81, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[226:227], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[136:137], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[138:139], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	s_add_u32 s0, s0, 2
	s_addc_u32 s1, s1, 0
	v_mov_b64_e32 v[98:99], s[70:71]
	v_cmp_lt_i64_e32 vcc, s[0:1], v[98:99]
	s_add_i32 s8, s8, 8
	s_mov_b32 s67, s10
	s_mov_b32 s72, s9
	s_cbranch_vccz .LBB0_50
.LBB0_26:
	s_add_i32 s4, s8, -4
	v_mov_b32_e32 v98, s4
	buffer_load_dword v98, v98, s[36:39], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_mov_b32 s9, s43
	v_and_b32_e32 v99, 0x180, v99
	s_mov_b32 s10, s85
	v_cmp_ne_u32_e32 vcc, s78, v99
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v98
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_28
	ds_write_b128 v150, v[168:171]
	ds_write_b128 v150, v[172:175] offset:6144
.LBB0_28:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_30
	s_mul_i32 s11, s67, 0x1800
	v_add_u32_e32 v98, s11, v149
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[68:69]
	global_load_dwordx4 v[168:171], v[98:99], off
	global_load_dwordx4 v[172:175], v[98:99], off offset:256
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v132, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v132
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[134:137], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[178:179], v[98:113]
	v_add_u32_e32 v133, 0x1810, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[182:183], v[98:113]
	v_add_u32_e32 v133, 0x1820, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[186:187], v[98:113]
	v_add_u32_e32 v133, 0x1830, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[190:191], v[98:113]
	v_add_u32_e32 v133, 0x1840, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[194:195], v[98:113]
	v_add_u32_e32 v133, 0x1850, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[198:199], v[98:113]
	v_add_u32_e32 v133, 0x1860, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[202:203], v[98:113]
	v_add_u32_e32 v133, 0x1870, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	v_add_u32_e32 v133, 0x1880, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[210:211], v[98:113]
	v_add_u32_e32 v133, 0x1890, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[212:213], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[214:215], v[98:113]
	v_add_u32_e32 v133, 0x18a0, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[216:217], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[218:219], v[98:113]
	v_add_u32_e32 v132, 0x18b0, v132
	v_lshrrev_b32_e32 v133, 3, v132
	v_and_b32_e32 v133, 56, v133
	v_xor_b32_e32 v132, v133, v132
	v_lshlrev_b32_e32 v132, 1, v132
	ds_read_b128 v[132:135], v132
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[132:133], v[220:221], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[222:223], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[114:115], v[98:99]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v132, v98, v99
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v132, v132, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v132, v132, v102, v103
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v132, v132, v104, v105
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v132, v132, v106, v107
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v132, v132, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v132, v132, v110, v111
	v_max3_f32 v132, v132, v112, v113
	ds_bpermute_b32 v133, v151, v132
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v133, v133, v133
	v_max_f32_e32 v133, v132, v133
	;;#ASMSTART
	v_add_f32 v132, v116, v153
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v133, v132
	v_mov_b32_e32 v132, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v133, v133, v154
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v116, v116, v133
	v_exp_f32_e32 v132, v116
	v_mov_b32_e32 v116, v133
.LBB0_32:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[98:99], v[98:99], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, 0, v98
	v_add_f32_e32 v133, v133, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v100
	v_add_f32_e32 v133, v133, v101
	v_add_f32_e32 v133, v133, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v103
	v_add_f32_e32 v133, v133, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v105
	v_add_f32_e32 v133, v133, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v107
	v_add_f32_e32 v133, v133, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v109
	v_add_f32_e32 v133, v133, v110
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
	v_add_f32_e32 v133, v133, v111
	v_add_f32_e32 v133, v133, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v133, v133, v113
	;;#ASMSTART
	v_fma_f32 v156, v156, v132, v133
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v132
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v132, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v132, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s12, s72, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s12, s12, s66
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s12
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[52:53]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[132:135], v[106:107], off offset:512
	global_load_dwordx4 v[136:139], v[106:107], off offset:1024
	global_load_dwordx4 v[140:143], v[106:107], off offset:1536
	global_load_dwordx4 v[144:147], v[106:107], off offset:2048
	global_load_dwordx4 v[224:227], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[136:137], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[224:225], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s80, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[102:103], v[18:33]
	global_load_dwordx4 v[132:135], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[138:139], v[102:103], v[34:49]
	global_load_dwordx4 v[136:139], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[102:103], v[50:65]
	global_load_dwordx4 v[140:143], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s81, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[226:227], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[136:137], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[138:139], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_34
	ds_write_b128 v150, v[160:163] offset:12288
	ds_write_b128 v150, v[164:167] offset:18432
.LBB0_34:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mul_i32 s11, s9, 0x1800
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_36
	v_add_u32_e32 v98, s11, v148
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[68:69]
	global_load_dwordx4 v[160:163], v[98:99], off
	global_load_dwordx4 v[164:167], v[98:99], off offset:256
.LBB0_36:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v132, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v133, 56, v98
	v_xor_b32_e32 v98, v133, v132
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[134:137], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[178:179], v[98:113]
	v_or_b32_e32 v134, 16, v132
	v_xor_b32_e32 v134, v134, v133
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[182:183], v[98:113]
	v_or_b32_e32 v134, 32, v132
	v_xor_b32_e32 v134, v134, v133
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[186:187], v[98:113]
	v_or_b32_e32 v134, 48, v132
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[190:191], v[98:113]
	v_add_u32_e32 v133, 64, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[194:195], v[98:113]
	v_add_u32_e32 v133, 0x50, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[198:199], v[98:113]
	v_add_u32_e32 v133, 0x60, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[202:203], v[98:113]
	v_add_u32_e32 v133, 0x70, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	v_add_u32_e32 v133, 0x80, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[210:211], v[98:113]
	v_add_u32_e32 v133, 0x90, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[212:213], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[214:215], v[98:113]
	v_add_u32_e32 v133, 0xa0, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[216:217], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[218:219], v[98:113]
	v_add_u32_e32 v132, 0xb0, v132
	v_lshrrev_b32_e32 v133, 3, v132
	v_and_b32_e32 v133, 56, v133
	v_xor_b32_e32 v132, v133, v132
	v_lshlrev_b32_e32 v132, 1, v132
	ds_read_b128 v[132:135], v132
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[132:133], v[220:221], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[222:223], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[114:115], v[98:99]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v132, v98, v99
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v132, v132, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v132, v132, v102, v103
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v132, v132, v104, v105
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v132, v132, v106, v107
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v132, v132, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v132, v132, v110, v111
	v_max3_f32 v132, v132, v112, v113
	ds_bpermute_b32 v133, v151, v132
	v_mov_b32_e32 v157, 1.0
	v_mov_b32_e32 v134, v116
	v_mov_b32_e32 v135, v116
	v_mov_b32_e32 v136, v116
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v133, v133, v133
	v_max_f32_e32 v158, v132, v133
	;;#ASMSTART
	v_add_f32 v132, v116, v153
	;;#ASMEND
	v_mov_b32_e32 v133, v116
	v_cmp_gt_f32_e32 vcc, v158, v132
	v_mov_b32_e32 v132, v116
	v_mov_b32_e32 v137, v116
	v_mov_b32_e32 v138, v116
	v_mov_b32_e32 v139, v116
	v_mov_b32_e32 v140, v116
	v_mov_b32_e32 v141, v116
	v_mov_b32_e32 v142, v116
	v_mov_b32_e32 v143, v116
	v_mov_b32_e32 v144, v116
	v_mov_b32_e32 v145, v116
	v_mov_b32_e32 v146, v116
	v_mov_b32_e32 v147, v116
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_38
	;;#ASMSTART
	v_add_f32 v132, v158, v154
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v116, v116, v132
	v_exp_f32_e32 v157, v116
	v_mov_b32_e32 v116, v132
	v_mov_b32_e32 v133, v132
	v_mov_b32_e32 v134, v132
	v_mov_b32_e32 v135, v132
	v_mov_b32_e32 v136, v132
	v_mov_b32_e32 v137, v132
	v_mov_b32_e32 v138, v132
	v_mov_b32_e32 v139, v132
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v132
	v_mov_b32_e32 v142, v132
	v_mov_b32_e32 v143, v132
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v132
	v_mov_b32_e32 v146, v132
	v_mov_b32_e32 v147, v132
.LBB0_38:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, 0, v98
	v_add_f32_e32 v158, v158, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v100
	v_add_f32_e32 v158, v158, v101
	v_add_f32_e32 v158, v158, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v103
	v_add_f32_e32 v158, v158, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v105
	v_add_f32_e32 v158, v158, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v107
	v_add_f32_e32 v158, v158, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v109
	v_add_f32_e32 v158, v158, v110
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
	v_add_f32_e32 v158, v158, v111
	v_add_f32_e32 v158, v158, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v158, v158, v113
	;;#ASMSTART
	v_fma_f32 v156, v156, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v157
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v157, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v157, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s12, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s12
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[52:53]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[224:227], v[106:107], off offset:512
	global_load_dwordx4 v[228:231], v[106:107], off offset:1024
	global_load_dwordx4 v[232:235], v[106:107], off offset:1536
	global_load_dwordx4 v[236:239], v[106:107], off offset:2048
	global_load_dwordx4 v[240:243], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[224:225], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[228:229], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[232:233], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[240:241], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s80, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[102:103], v[18:33]
	global_load_dwordx4 v[224:227], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[102:103], v[34:49]
	global_load_dwordx4 v[228:231], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[234:235], v[102:103], v[50:65]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s81, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[242:243], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[224:225], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[228:229], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[232:233], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[234:235], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	v_mov_b32_e32 v98, s8
	buffer_load_dword v98, v98, s[36:39], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s85, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s78, v99
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_40
	ds_write_b128 v150, v[168:171]
	ds_write_b128 v150, v[172:175] offset:6144
.LBB0_40:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_42
	v_add_u32_e32 v98, s11, v149
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[68:69]
	global_load_dwordx4 v[168:171], v[98:99], off
	global_load_dwordx4 v[172:175], v[98:99], off offset:256
.LBB0_42:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v157, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v157
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[224:227], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[178:179], v[98:113]
	v_add_u32_e32 v158, 0x1810, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[182:183], v[98:113]
	v_add_u32_e32 v158, 0x1820, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[186:187], v[98:113]
	v_add_u32_e32 v158, 0x1830, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[190:191], v[98:113]
	v_add_u32_e32 v158, 0x1840, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[194:195], v[98:113]
	v_add_u32_e32 v158, 0x1850, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[198:199], v[98:113]
	v_add_u32_e32 v158, 0x1860, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[202:203], v[98:113]
	v_add_u32_e32 v158, 0x1870, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[206:207], v[98:113]
	v_add_u32_e32 v158, 0x1880, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[210:211], v[98:113]
	v_add_u32_e32 v158, 0x1890, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[212:213], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[214:215], v[98:113]
	v_add_u32_e32 v158, 0x18a0, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[216:217], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[218:219], v[98:113]
	v_add_u32_e32 v157, 0x18b0, v157
	v_lshrrev_b32_e32 v158, 3, v157
	v_and_b32_e32 v158, 56, v158
	v_xor_b32_e32 v157, v158, v157
	v_lshlrev_b32_e32 v157, 1, v157
	ds_read_b128 v[224:227], v157
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[220:221], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[222:223], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[114:115], v[98:99]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v157, v98, v99
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v157, v157, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v157, v157, v102, v103
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v157, v157, v104, v105
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v157, v157, v106, v107
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v157, v157, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v157, v157, v110, v111
	v_max3_f32 v157, v157, v112, v113
	ds_bpermute_b32 v158, v151, v157
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v158, v158, v158
	v_max_f32_e32 v158, v157, v158
	;;#ASMSTART
	v_add_f32 v157, v116, v153
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v158, v157
	v_mov_b32_e32 v157, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_44
	;;#ASMSTART
	v_add_f32 v132, v158, v154
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v116, v116, v132
	v_exp_f32_e32 v157, v116
	v_mov_b32_e32 v116, v132
	v_mov_b32_e32 v133, v132
	v_mov_b32_e32 v134, v132
	v_mov_b32_e32 v135, v132
	v_mov_b32_e32 v136, v132
	v_mov_b32_e32 v137, v132
	v_mov_b32_e32 v138, v132
	v_mov_b32_e32 v139, v132
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v132
	v_mov_b32_e32 v142, v132
	v_mov_b32_e32 v143, v132
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v132
	v_mov_b32_e32 v146, v132
	v_mov_b32_e32 v147, v132
.LBB0_44:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, 0, v98
	v_add_f32_e32 v158, v158, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v100
	v_add_f32_e32 v158, v158, v101
	v_add_f32_e32 v158, v158, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v103
	v_add_f32_e32 v158, v158, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v105
	v_add_f32_e32 v158, v158, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v107
	v_add_f32_e32 v158, v158, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v109
	v_add_f32_e32 v158, v158, v110
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
	v_add_f32_e32 v158, v158, v111
	v_add_f32_e32 v158, v158, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v158, v158, v113
	;;#ASMSTART
	v_fma_f32 v156, v156, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v157
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v157, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v157, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s11, s67, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s11, s11, s66
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s11
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[52:53]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[224:227], v[106:107], off offset:512
	global_load_dwordx4 v[228:231], v[106:107], off offset:1024
	global_load_dwordx4 v[232:235], v[106:107], off offset:1536
	global_load_dwordx4 v[236:239], v[106:107], off offset:2048
	global_load_dwordx4 v[240:243], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[224:225], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[228:229], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[232:233], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[240:241], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s80, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[102:103], v[18:33]
	global_load_dwordx4 v[224:227], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[102:103], v[34:49]
	global_load_dwordx4 v[228:231], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[234:235], v[102:103], v[50:65]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s81, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[242:243], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[224:225], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[228:229], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[232:233], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[234:235], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_46
	ds_write_b128 v150, v[160:163] offset:12288
	ds_write_b128 v150, v[164:167] offset:18432
.LBB0_46:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_48
	s_mul_i32 s12, s10, 0x1800
	v_add_u32_e32 v98, s12, v148
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[68:69]
	global_load_dwordx4 v[160:163], v[98:99], off
	global_load_dwordx4 v[164:167], v[98:99], off offset:256
.LBB0_48:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v157, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v158, 56, v98
	v_xor_b32_e32 v98, v158, v157
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[224:227], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[178:179], v[98:113]
	v_or_b32_e32 v159, 16, v157
	v_xor_b32_e32 v159, v159, v158
	v_lshlrev_b32_e32 v159, 1, v159
	ds_read_b128 v[224:227], v159
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[182:183], v[98:113]
	v_or_b32_e32 v159, 32, v157
	v_xor_b32_e32 v159, v159, v158
	v_lshlrev_b32_e32 v159, 1, v159
	ds_read_b128 v[224:227], v159
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[186:187], v[98:113]
	v_or_b32_e32 v159, 48, v157
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[190:191], v[98:113]
	v_add_u32_e32 v158, 64, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[194:195], v[98:113]
	v_add_u32_e32 v158, 0x50, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[198:199], v[98:113]
	v_add_u32_e32 v158, 0x60, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[202:203], v[98:113]
	v_add_u32_e32 v158, 0x70, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[206:207], v[98:113]
	v_add_u32_e32 v158, 0x80, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[210:211], v[98:113]
	v_add_u32_e32 v158, 0x90, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[212:213], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[214:215], v[98:113]
	v_add_u32_e32 v158, 0xa0, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[216:217], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[218:219], v[98:113]
	v_add_u32_e32 v157, 0xb0, v157
	v_lshrrev_b32_e32 v158, 3, v157
	v_and_b32_e32 v158, 56, v158
	v_xor_b32_e32 v157, v158, v157
	v_lshlrev_b32_e32 v157, 1, v157
	ds_read_b128 v[224:227], v157
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[220:221], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[222:223], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[114:115], v[98:99]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v157, v98, v99
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v157, v157, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v157, v157, v102, v103
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v157, v157, v104, v105
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v157, v157, v106, v107
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v157, v157, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v157, v157, v110, v111
	v_max3_f32 v157, v157, v112, v113
	ds_bpermute_b32 v158, v151, v157
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v158, v158, v158
	v_max_f32_e32 v158, v157, v158
	;;#ASMSTART
	v_add_f32 v157, v116, v153
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v158, v157
	v_mov_b32_e32 v157, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_25
	;;#ASMSTART
	v_add_f32 v132, v158, v154
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v116, v116, v132
	v_exp_f32_e32 v157, v116
	v_mov_b32_e32 v116, v132
	v_mov_b32_e32 v133, v132
	v_mov_b32_e32 v134, v132
	v_mov_b32_e32 v135, v132
	v_mov_b32_e32 v136, v132
	v_mov_b32_e32 v137, v132
	v_mov_b32_e32 v138, v132
	v_mov_b32_e32 v139, v132
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v132
	v_mov_b32_e32 v142, v132
	v_mov_b32_e32 v143, v132
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v132
	v_mov_b32_e32 v146, v132
	v_mov_b32_e32 v147, v132
	s_branch .LBB0_25
.LBB0_50:
	s_mov_b32 s67, s10
	s_mov_b32 s72, s9
	s_cmp_ge_i32 s70, s40
	s_cbranch_scc0 .LBB0_52
	s_branch .LBB0_81
.LBB0_51:
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v116, 0xff800000
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
	v_mov_b32_e32 v156, v2
	s_cmp_ge_i32 s70, s40
	s_cbranch_scc1 .LBB0_81
.LBB0_52:
	s_lshl_b32 s86, s7, 6
	s_lshl_b32 s0, s70, 6
	s_ashr_i32 s41, s40, 31
	s_add_i32 s86, s86, s6
	v_mov_b32_e32 v115, v114
	v_mov_b32_e32 v118, v114
	v_mov_b32_e32 v119, v114
	v_mov_b32_e32 v120, v114
	v_mov_b32_e32 v121, v114
	v_mov_b32_e32 v122, v114
	v_mov_b32_e32 v123, v114
	v_mov_b32_e32 v124, v114
	v_mov_b32_e32 v125, v114
	v_mov_b32_e32 v126, v114
	v_mov_b32_e32 v127, v114
	v_mov_b32_e32 v128, v114
	v_mov_b32_e32 v129, v114
	v_mov_b32_e32 v130, v114
	v_mov_b32_e32 v131, v114
	s_add_i32 s87, s0, 0x77
	s_add_i32 s88, s70, 1
	s_lshl2_add_u32 s89, s70, 20
	s_branch .LBB0_55
.LBB0_53:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, 0, v98
	v_add_f32_e32 v132, v132, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, v132, v100
	v_add_f32_e32 v132, v132, v101
	v_add_f32_e32 v132, v132, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, v132, v103
	v_add_f32_e32 v132, v132, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, v132, v105
	v_add_f32_e32 v132, v132, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, v132, v107
	v_add_f32_e32 v132, v132, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v132, v132, v109
	v_add_f32_e32 v132, v132, v110
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
	v_add_f32_e32 v132, v132, v111
	v_add_f32_e32 v132, v132, v112
	v_add_f32_e32 v132, v132, v113
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	;;#ASMSTART
	v_fma_f32 v156, v156, v157, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v157
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v132, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v132, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s92, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s92
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[52:53]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[132:135], v[106:107], off offset:512
	global_load_dwordx4 v[136:139], v[106:107], off offset:1024
	global_load_dwordx4 v[140:143], v[106:107], off offset:1536
	global_load_dwordx4 v[144:147], v[106:107], off offset:2048
	global_load_dwordx4 v[224:227], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[136:137], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[224:225], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s80, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[102:103], v[18:33]
	global_load_dwordx4 v[132:135], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[138:139], v[102:103], v[34:49]
	global_load_dwordx4 v[136:139], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[102:103], v[50:65]
	global_load_dwordx4 v[140:143], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s81, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[226:227], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[136:137], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[138:139], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
.LBB0_54:
	s_and_b64 s[0:1], s[72:73], exec
	s_cselect_b32 s72, s43, s67
	s_cselect_b32 s67, s85, s43
	s_cselect_b32 s43, s90, s85
	s_add_u32 s70, s70, 2
	s_addc_u32 s71, s71, 0
	v_mov_b64_e32 v[98:99], s[40:41]
	v_cmp_lt_i64_e32 vcc, s[70:71], v[98:99]
	s_addk_i32 s87, 0x80
	s_add_i32 s88, s88, 2
	s_add_i32 s89, s89, 8
	s_mov_b32 s85, s91
	s_cbranch_vccz .LBB0_81
.LBB0_55:
	s_add_i32 s0, s89, -4
	v_mov_b32_e32 v98, s0
	buffer_load_dword v98, v98, s[36:39], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s90, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s78, v99
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_57
	ds_write_b128 v150, v[168:171]
	ds_write_b128 v150, v[172:175] offset:6144
.LBB0_57:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_59
	s_mul_i32 s4, s67, 0x1800
	v_add_u32_e32 v98, s4, v149
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[68:69]
	global_load_dwordx4 v[168:171], v[98:99], off
	global_load_dwordx4 v[172:175], v[98:99], off offset:256
.LBB0_59:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v132, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v132
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[134:137], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[178:179], v[98:113]
	v_add_u32_e32 v133, 0x1810, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[182:183], v[98:113]
	v_add_u32_e32 v133, 0x1820, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[186:187], v[98:113]
	v_add_u32_e32 v133, 0x1830, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[190:191], v[98:113]
	v_add_u32_e32 v133, 0x1840, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[194:195], v[98:113]
	v_add_u32_e32 v133, 0x1850, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[198:199], v[98:113]
	v_add_u32_e32 v133, 0x1860, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[202:203], v[98:113]
	v_add_u32_e32 v133, 0x1870, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	v_add_u32_e32 v133, 0x1880, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[210:211], v[98:113]
	v_add_u32_e32 v133, 0x1890, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[212:213], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[214:215], v[98:113]
	v_add_u32_e32 v133, 0x18a0, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[216:217], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[218:219], v[98:113]
	v_add_u32_e32 v132, 0x18b0, v132
	v_lshrrev_b32_e32 v133, 3, v132
	v_and_b32_e32 v133, 56, v133
	v_xor_b32_e32 v132, v133, v132
	v_lshlrev_b32_e32 v132, 1, v132
	ds_read_b128 v[132:135], v132
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[132:133], v[220:221], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[222:223], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v132, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v132, 2, v132
	v_and_b32_e32 v132, 8, v132
	v_add_u32_e32 v132, s87, v132
	v_add_u32_e32 v133, 0xffffff89, v132
	v_cmp_gt_i32_e32 vcc, s86, v133
	v_add_u32_e32 v133, 0xffffff8a, v132
	v_cmp_gt_i32_e64 s[0:1], s86, v133
	v_add_u32_e32 v133, 0xffffff8b, v132
	v_cmp_gt_i32_e64 s[4:5], s86, v133
	v_add_u32_e32 v133, 0xffffff8c, v132
	v_cmp_gt_i32_e64 s[6:7], s86, v133
	v_add_u32_e32 v133, 0xffffff8d, v132
	v_cmp_gt_i32_e64 s[8:9], s86, v133
	v_add_u32_e32 v133, 0xffffff8e, v132
	v_cmp_gt_i32_e64 s[10:11], s86, v133
	v_add_u32_e32 v133, 0xffffff8f, v132
	v_cmp_gt_i32_e64 s[12:13], s86, v133
	v_add_u32_e32 v133, 0xffffff90, v132
	v_cmp_gt_i32_e64 s[14:15], s86, v133
	v_add_u32_e32 v133, 0xffffff99, v132
	v_cmp_gt_i32_e64 s[16:17], s86, v133
	v_add_u32_e32 v133, 0xffffff9a, v132
	v_cmp_gt_i32_e64 s[18:19], s86, v133
	v_add_u32_e32 v133, 0xffffff9b, v132
	v_cmp_gt_i32_e64 s[20:21], s86, v133
	v_add_u32_e32 v133, 0xffffff9c, v132
	v_cmp_gt_i32_e64 s[22:23], s86, v133
	v_add_u32_e32 v133, 0xffffff9d, v132
	v_cmp_gt_i32_e64 s[24:25], s86, v133
	v_add_u32_e32 v133, 0xffffff9e, v132
	v_cmp_gt_i32_e64 s[26:27], s86, v133
	v_add_u32_e32 v133, 0xffffff9f, v132
	v_add_u32_e32 v132, 0xffffffa0, v132
	v_cmp_gt_i32_e64 s[28:29], s86, v133
	v_cmp_gt_i32_e64 s[30:31], s86, v132
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
	v_cndmask_b32_e64 v113, v155, v113, s[30:31]
	v_cndmask_b32_e64 v112, v155, v112, s[28:29]
	v_cndmask_b32_e64 v139, v155, v99, s[0:1]
	v_cndmask_b32_e32 v138, v155, v98, vcc
	v_cndmask_b32_e64 v111, v155, v111, s[26:27]
	v_cndmask_b32_e64 v110, v155, v110, s[24:25]
	v_cndmask_b32_e64 v107, v155, v107, s[18:19]
	v_cndmask_b32_e64 v106, v155, v106, s[16:17]
	v_cndmask_b32_e64 v133, v155, v105, s[14:15]
	v_cndmask_b32_e64 v132, v155, v104, s[12:13]
	v_cndmask_b32_e64 v137, v155, v101, s[6:7]
	v_cndmask_b32_e64 v136, v155, v100, s[4:5]
	v_pk_mul_f32 v[98:99], v[130:131], v[112:113]
	v_pk_mul_f32 v[112:113], v[114:115], v[138:139]
	v_cndmask_b32_e64 v109, v155, v109, s[22:23]
	v_cndmask_b32_e64 v108, v155, v108, s[20:21]
	v_cndmask_b32_e64 v135, v155, v103, s[10:11]
	v_cndmask_b32_e64 v134, v155, v102, s[8:9]
	v_pk_mul_f32 v[100:101], v[128:129], v[110:111]
	v_pk_mul_f32 v[104:105], v[124:125], v[106:107]
	v_pk_mul_f32 v[106:107], v[122:123], v[132:133]
	v_pk_mul_f32 v[110:111], v[118:119], v[136:137]
	v_max_f32_e32 v132, v112, v113
	v_pk_mul_f32 v[102:103], v[126:127], v[108:109]
	v_pk_mul_f32 v[108:109], v[120:121], v[134:135]
	v_max3_f32 v132, v132, v110, v111
	v_max3_f32 v132, v132, v108, v109
	v_max3_f32 v132, v132, v106, v107
	v_max3_f32 v132, v132, v104, v105
	v_max3_f32 v132, v132, v102, v103
	v_max3_f32 v132, v132, v100, v101
	v_max3_f32 v132, v132, v98, v99
	ds_bpermute_b32 v133, v151, v132
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v133, v133, v133
	v_max_f32_e32 v133, v132, v133
	;;#ASMSTART
	v_add_f32 v132, v116, v153
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v133, v132
	v_mov_b32_e32 v132, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_61
	;;#ASMSTART
	v_add_f32 v133, v133, v154
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v116, v116, v133
	v_exp_f32_e32 v132, v116
	v_mov_b32_e32 v116, v133
.LBB0_61:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[112:113], v[112:113], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[110:111], v[110:111], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, 0, v112
	v_add_f32_e32 v133, v133, v113
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v110
	v_add_f32_e32 v133, v133, v111
	v_add_f32_e32 v133, v133, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v109
	v_add_f32_e32 v133, v133, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v107
	v_add_f32_e32 v133, v133, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[100:101], v[100:101], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v105
	v_add_f32_e32 v133, v133, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	v_pk_add_f32 v[98:99], v[98:99], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v103
	v_add_f32_e32 v133, v133, v100
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v132
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v133, v133, v101
	v_add_f32_e32 v133, v133, v98
	v_add_f32_e32 v133, v133, v99
	;;#ASMSTART
	v_fma_f32 v156, v156, v132, v133
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v132
	;;#ASMEND
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v132, 0x8000, v98
	v_add_u32_e32 v98, 0x8000, v101
	v_add_u32_e32 v133, 0x8000, v100
	v_add_u32_e32 v134, 0x8000, v103
	v_add_u32_e32 v135, 0x8000, v102
	v_add_u32_e32 v105, 0x8000, v105
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v107
	v_add_u32_e32 v102, 0x8000, v109
	v_add_u32_e32 v101, 0x8000, v111
	v_add_u32_e32 v100, 0x8000, v113
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v107, 0x8000, v108
	v_add_u32_e32 v108, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v112
	;;#ASMSTART
	v_perm_b32 v100, v100, v109, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v108, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v102, v107, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v103, v106, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v105, v104, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v134, v135, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v98, v133, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v99, v132, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mulk_i32 s72, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s72, s72, s66
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s72
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[52:53]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[132:135], v[106:107], off offset:512
	global_load_dwordx4 v[136:139], v[106:107], off offset:1024
	global_load_dwordx4 v[140:143], v[106:107], off offset:1536
	global_load_dwordx4 v[144:147], v[106:107], off offset:2048
	global_load_dwordx4 v[224:227], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[136:137], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[224:225], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s80, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[102:103], v[18:33]
	global_load_dwordx4 v[132:135], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[138:139], v[102:103], v[34:49]
	global_load_dwordx4 v[136:139], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[102:103], v[50:65]
	global_load_dwordx4 v[140:143], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s81, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[226:227], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[132:133], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[136:137], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[140:141], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[138:139], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_63
	ds_write_b128 v150, v[160:163] offset:12288
	ds_write_b128 v150, v[164:167] offset:18432
.LBB0_63:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mul_i32 s92, s43, 0x1800
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_65
	v_add_u32_e32 v98, s92, v148
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[68:69]
	global_load_dwordx4 v[160:163], v[98:99], off
	global_load_dwordx4 v[164:167], v[98:99], off offset:256
.LBB0_65:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v132, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v133, 56, v98
	v_xor_b32_e32 v98, v133, v132
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[134:137], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[178:179], v[98:113]
	v_or_b32_e32 v134, 16, v132
	v_xor_b32_e32 v134, v134, v133
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[182:183], v[98:113]
	v_or_b32_e32 v134, 32, v132
	v_xor_b32_e32 v134, v134, v133
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[186:187], v[98:113]
	v_or_b32_e32 v134, 48, v132
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[190:191], v[98:113]
	v_add_u32_e32 v133, 64, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[194:195], v[98:113]
	v_add_u32_e32 v133, 0x50, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[198:199], v[98:113]
	v_add_u32_e32 v133, 0x60, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[202:203], v[98:113]
	v_add_u32_e32 v133, 0x70, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	v_add_u32_e32 v133, 0x80, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[210:211], v[98:113]
	v_add_u32_e32 v133, 0x90, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[212:213], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[214:215], v[98:113]
	v_add_u32_e32 v133, 0xa0, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[216:217], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[218:219], v[98:113]
	v_add_u32_e32 v132, 0xb0, v132
	v_lshrrev_b32_e32 v133, 3, v132
	v_and_b32_e32 v133, 56, v133
	v_xor_b32_e32 v132, v133, v132
	v_lshlrev_b32_e32 v132, 1, v132
	ds_read_b128 v[132:135], v132
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[132:133], v[220:221], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[222:223], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v132, v0
	;;#ASMEND
	v_mov_b32_e32 v157, 1.0
	v_lshrrev_b32_e32 v132, 2, v132
	v_and_b32_e32 v132, 8, v132
	v_add_u32_e32 v132, s87, v132
	v_add_u32_e32 v133, 0xffffffa9, v132
	v_cmp_gt_i32_e32 vcc, s86, v133
	v_add_u32_e32 v133, 0xffffffaa, v132
	v_cmp_gt_i32_e64 s[0:1], s86, v133
	v_add_u32_e32 v133, 0xffffffab, v132
	v_cmp_gt_i32_e64 s[4:5], s86, v133
	v_add_u32_e32 v133, 0xffffffac, v132
	v_cmp_gt_i32_e64 s[6:7], s86, v133
	v_add_u32_e32 v133, 0xffffffad, v132
	v_cmp_gt_i32_e64 s[8:9], s86, v133
	v_add_u32_e32 v133, 0xffffffae, v132
	v_cmp_gt_i32_e64 s[10:11], s86, v133
	v_add_u32_e32 v133, 0xffffffaf, v132
	v_cmp_gt_i32_e64 s[12:13], s86, v133
	v_add_u32_e32 v133, 0xffffffb0, v132
	v_cmp_gt_i32_e64 s[14:15], s86, v133
	v_add_u32_e32 v133, 0xffffffb9, v132
	v_cmp_gt_i32_e64 s[16:17], s86, v133
	v_add_u32_e32 v133, 0xffffffba, v132
	v_cmp_gt_i32_e64 s[18:19], s86, v133
	v_add_u32_e32 v133, 0xffffffbb, v132
	v_cmp_gt_i32_e64 s[20:21], s86, v133
	v_add_u32_e32 v133, 0xffffffbc, v132
	v_cmp_gt_i32_e64 s[22:23], s86, v133
	v_add_u32_e32 v133, 0xffffffbd, v132
	v_cmp_gt_i32_e64 s[24:25], s86, v133
	v_add_u32_e32 v133, 0xffffffbe, v132
	v_cmp_gt_i32_e64 s[26:27], s86, v133
	v_add_u32_e32 v133, 0xffffffbf, v132
	v_subrev_u32_e32 v132, 64, v132
	v_cmp_gt_i32_e64 s[28:29], s86, v133
	v_cmp_gt_i32_e64 s[30:31], s86, v132
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
	v_cndmask_b32_e64 v99, v155, v99, s[0:1]
	v_cndmask_b32_e32 v98, v155, v98, vcc
	v_cndmask_b32_e64 v101, v155, v101, s[6:7]
	v_cndmask_b32_e64 v100, v155, v100, s[4:5]
	v_pk_mul_f32 v[98:99], v[114:115], v[98:99]
	v_cndmask_b32_e64 v103, v155, v103, s[10:11]
	v_cndmask_b32_e64 v102, v155, v102, s[8:9]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v132, v98, v99
	v_cndmask_b32_e64 v105, v155, v105, s[14:15]
	v_cndmask_b32_e64 v104, v155, v104, s[12:13]
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v132, v132, v100, v101
	v_cndmask_b32_e64 v107, v155, v107, s[18:19]
	v_cndmask_b32_e64 v106, v155, v106, s[16:17]
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v132, v132, v102, v103
	v_cndmask_b32_e64 v109, v155, v109, s[22:23]
	v_cndmask_b32_e64 v108, v155, v108, s[20:21]
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v132, v132, v104, v105
	v_cndmask_b32_e64 v111, v155, v111, s[26:27]
	v_cndmask_b32_e64 v110, v155, v110, s[24:25]
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v132, v132, v106, v107
	v_cndmask_b32_e64 v113, v155, v113, s[30:31]
	v_cndmask_b32_e64 v112, v155, v112, s[28:29]
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v132, v132, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v132, v132, v110, v111
	v_max3_f32 v132, v132, v112, v113
	ds_bpermute_b32 v133, v151, v132
	v_mov_b32_e32 v134, v116
	v_mov_b32_e32 v135, v116
	v_mov_b32_e32 v136, v116
	v_mov_b32_e32 v137, v116
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v133, v133, v133
	v_max_f32_e32 v158, v132, v133
	;;#ASMSTART
	v_add_f32 v132, v116, v153
	;;#ASMEND
	v_mov_b32_e32 v133, v116
	v_cmp_gt_f32_e32 vcc, v158, v132
	v_mov_b32_e32 v132, v116
	v_mov_b32_e32 v138, v116
	v_mov_b32_e32 v139, v116
	v_mov_b32_e32 v140, v116
	v_mov_b32_e32 v141, v116
	v_mov_b32_e32 v142, v116
	v_mov_b32_e32 v143, v116
	v_mov_b32_e32 v144, v116
	v_mov_b32_e32 v145, v116
	v_mov_b32_e32 v146, v116
	v_mov_b32_e32 v147, v116
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_67
	;;#ASMSTART
	v_add_f32 v132, v158, v154
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v116, v116, v132
	v_exp_f32_e32 v157, v116
	v_mov_b32_e32 v116, v132
	v_mov_b32_e32 v133, v132
	v_mov_b32_e32 v134, v132
	v_mov_b32_e32 v135, v132
	v_mov_b32_e32 v136, v132
	v_mov_b32_e32 v137, v132
	v_mov_b32_e32 v138, v132
	v_mov_b32_e32 v139, v132
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v132
	v_mov_b32_e32 v142, v132
	v_mov_b32_e32 v143, v132
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v132
	v_mov_b32_e32 v146, v132
	v_mov_b32_e32 v147, v132
.LBB0_67:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, 0, v98
	v_add_f32_e32 v158, v158, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v100
	v_add_f32_e32 v158, v158, v101
	v_add_f32_e32 v158, v158, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v103
	v_add_f32_e32 v158, v158, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v105
	v_add_f32_e32 v158, v158, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v107
	v_add_f32_e32 v158, v158, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v109
	v_add_f32_e32 v158, v158, v110
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
	v_add_f32_e32 v158, v158, v111
	v_add_f32_e32 v158, v158, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v158, v158, v113
	;;#ASMSTART
	v_fma_f32 v156, v156, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v157
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v157, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v157, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s72, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s72
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[52:53]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[224:227], v[106:107], off offset:512
	global_load_dwordx4 v[228:231], v[106:107], off offset:1024
	global_load_dwordx4 v[232:235], v[106:107], off offset:1536
	global_load_dwordx4 v[236:239], v[106:107], off offset:2048
	global_load_dwordx4 v[240:243], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[224:225], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[228:229], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[232:233], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[240:241], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s80, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[102:103], v[18:33]
	global_load_dwordx4 v[224:227], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[102:103], v[34:49]
	global_load_dwordx4 v[228:231], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[234:235], v[102:103], v[50:65]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s81, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[242:243], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[224:225], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[228:229], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[232:233], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[234:235], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	s_cmp_gt_i32 s40, s88
	s_cselect_b64 s[72:73], -1, 0
	s_cmp_le_i32 s40, s88
	s_cbranch_scc1 .LBB0_80
	v_mov_b32_e32 v98, s89
	buffer_load_dword v98, v98, s[36:39], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s91, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s78, v99
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_70
	ds_write_b128 v150, v[168:171]
	ds_write_b128 v150, v[172:175] offset:6144
.LBB0_70:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_72
	v_add_u32_e32 v98, s92, v149
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[68:69]
	global_load_dwordx4 v[168:171], v[98:99], off
	global_load_dwordx4 v[172:175], v[98:99], off offset:256
.LBB0_72:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v157, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v157
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[224:227], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[178:179], v[98:113]
	v_add_u32_e32 v158, 0x1810, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[182:183], v[98:113]
	v_add_u32_e32 v158, 0x1820, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[186:187], v[98:113]
	v_add_u32_e32 v158, 0x1830, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[190:191], v[98:113]
	v_add_u32_e32 v158, 0x1840, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[194:195], v[98:113]
	v_add_u32_e32 v158, 0x1850, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[198:199], v[98:113]
	v_add_u32_e32 v158, 0x1860, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[202:203], v[98:113]
	v_add_u32_e32 v158, 0x1870, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[206:207], v[98:113]
	v_add_u32_e32 v158, 0x1880, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[210:211], v[98:113]
	v_add_u32_e32 v158, 0x1890, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[212:213], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[214:215], v[98:113]
	v_add_u32_e32 v158, 0x18a0, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[216:217], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[218:219], v[98:113]
	v_add_u32_e32 v157, 0x18b0, v157
	v_lshrrev_b32_e32 v158, 3, v157
	v_and_b32_e32 v158, 56, v158
	v_xor_b32_e32 v157, v158, v157
	v_lshlrev_b32_e32 v157, 1, v157
	ds_read_b128 v[224:227], v157
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[220:221], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[222:223], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v157, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v157, 2, v157
	v_and_b32_e32 v157, 8, v157
	v_add_u32_e32 v157, s87, v157
	v_subrev_u32_e32 v158, 55, v157
	v_cmp_gt_i32_e32 vcc, s86, v158
	v_subrev_u32_e32 v158, 54, v157
	v_cmp_gt_i32_e64 s[0:1], s86, v158
	v_subrev_u32_e32 v158, 53, v157
	v_cmp_gt_i32_e64 s[4:5], s86, v158
	v_subrev_u32_e32 v158, 52, v157
	v_cmp_gt_i32_e64 s[6:7], s86, v158
	v_subrev_u32_e32 v158, 51, v157
	v_cmp_gt_i32_e64 s[8:9], s86, v158
	v_subrev_u32_e32 v158, 50, v157
	v_cmp_gt_i32_e64 s[10:11], s86, v158
	v_subrev_u32_e32 v158, 49, v157
	v_cmp_gt_i32_e64 s[12:13], s86, v158
	v_subrev_u32_e32 v158, 48, v157
	v_cmp_gt_i32_e64 s[14:15], s86, v158
	v_subrev_u32_e32 v158, 39, v157
	v_cmp_gt_i32_e64 s[16:17], s86, v158
	v_subrev_u32_e32 v158, 38, v157
	v_cmp_gt_i32_e64 s[18:19], s86, v158
	v_subrev_u32_e32 v158, 37, v157
	v_cmp_gt_i32_e64 s[20:21], s86, v158
	v_subrev_u32_e32 v158, 36, v157
	v_cmp_gt_i32_e64 s[22:23], s86, v158
	v_subrev_u32_e32 v158, 35, v157
	v_cmp_gt_i32_e64 s[24:25], s86, v158
	v_subrev_u32_e32 v158, 34, v157
	v_cmp_gt_i32_e64 s[26:27], s86, v158
	v_subrev_u32_e32 v158, 33, v157
	v_subrev_u32_e32 v157, 32, v157
	v_cmp_gt_i32_e64 s[28:29], s86, v158
	v_cmp_gt_i32_e64 s[30:31], s86, v157
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
	v_cndmask_b32_e64 v99, v155, v99, s[0:1]
	v_cndmask_b32_e32 v98, v155, v98, vcc
	v_cndmask_b32_e64 v101, v155, v101, s[6:7]
	v_cndmask_b32_e64 v100, v155, v100, s[4:5]
	v_pk_mul_f32 v[98:99], v[114:115], v[98:99]
	v_cndmask_b32_e64 v103, v155, v103, s[10:11]
	v_cndmask_b32_e64 v102, v155, v102, s[8:9]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v157, v98, v99
	v_cndmask_b32_e64 v105, v155, v105, s[14:15]
	v_cndmask_b32_e64 v104, v155, v104, s[12:13]
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v157, v157, v100, v101
	v_cndmask_b32_e64 v107, v155, v107, s[18:19]
	v_cndmask_b32_e64 v106, v155, v106, s[16:17]
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v157, v157, v102, v103
	v_cndmask_b32_e64 v109, v155, v109, s[22:23]
	v_cndmask_b32_e64 v108, v155, v108, s[20:21]
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v157, v157, v104, v105
	v_cndmask_b32_e64 v111, v155, v111, s[26:27]
	v_cndmask_b32_e64 v110, v155, v110, s[24:25]
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v157, v157, v106, v107
	v_cndmask_b32_e64 v113, v155, v113, s[30:31]
	v_cndmask_b32_e64 v112, v155, v112, s[28:29]
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v157, v157, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v157, v157, v110, v111
	v_max3_f32 v157, v157, v112, v113
	ds_bpermute_b32 v158, v151, v157
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v158, v158, v158
	v_max_f32_e32 v158, v157, v158
	;;#ASMSTART
	v_add_f32 v157, v116, v153
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v158, v157
	v_mov_b32_e32 v157, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_74
	;;#ASMSTART
	v_add_f32 v132, v158, v154
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v116, v116, v132
	v_exp_f32_e32 v157, v116
	v_mov_b32_e32 v116, v132
	v_mov_b32_e32 v133, v132
	v_mov_b32_e32 v134, v132
	v_mov_b32_e32 v135, v132
	v_mov_b32_e32 v136, v132
	v_mov_b32_e32 v137, v132
	v_mov_b32_e32 v138, v132
	v_mov_b32_e32 v139, v132
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v132
	v_mov_b32_e32 v142, v132
	v_mov_b32_e32 v143, v132
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v132
	v_mov_b32_e32 v146, v132
	v_mov_b32_e32 v147, v132
.LBB0_74:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, 0, v98
	v_add_f32_e32 v158, v158, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v100
	v_add_f32_e32 v158, v158, v101
	v_add_f32_e32 v158, v158, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v103
	v_add_f32_e32 v158, v158, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v105
	v_add_f32_e32 v158, v158, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v107
	v_add_f32_e32 v158, v158, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v158, v158, v109
	v_add_f32_e32 v158, v158, v110
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
	v_add_f32_e32 v158, v158, v111
	v_add_f32_e32 v158, v158, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v158, v158, v113
	;;#ASMSTART
	v_fma_f32 v156, v156, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v157
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v157
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v157, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v157, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s79
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s92, s67, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s92, s92, s66
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s92
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[52:53]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[224:227], v[106:107], off offset:512
	global_load_dwordx4 v[228:231], v[106:107], off offset:1024
	global_load_dwordx4 v[232:235], v[106:107], off offset:1536
	global_load_dwordx4 v[236:239], v[106:107], off offset:2048
	global_load_dwordx4 v[240:243], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[224:225], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[228:229], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[232:233], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[236:237], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[240:241], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s80, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[102:103], v[18:33]
	global_load_dwordx4 v[224:227], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[102:103], v[34:49]
	global_load_dwordx4 v[228:231], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[234:235], v[102:103], v[50:65]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s81, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[238:239], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[242:243], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[224:225], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[228:229], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[232:233], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[226:227], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[230:231], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[234:235], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_76
	ds_write_b128 v150, v[160:163] offset:12288
	ds_write_b128 v150, v[164:167] offset:18432
.LBB0_76:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s78, v98
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_78
	s_mul_i32 s4, s85, 0x1800
	v_add_u32_e32 v98, s4, v148
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[68:69]
	global_load_dwordx4 v[160:163], v[98:99], off
	global_load_dwordx4 v[164:167], v[98:99], off offset:256
.LBB0_78:
	s_or_b64 exec, exec, s[0:1]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v157, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v158, 56, v98
	v_xor_b32_e32 v98, v158, v157
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[224:227], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[176:177], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[178:179], v[98:113]
	v_or_b32_e32 v159, 16, v157
	v_xor_b32_e32 v159, v159, v158
	v_lshlrev_b32_e32 v159, 1, v159
	ds_read_b128 v[224:227], v159
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[182:183], v[98:113]
	v_or_b32_e32 v159, 32, v157
	v_xor_b32_e32 v159, v159, v158
	v_lshlrev_b32_e32 v159, 1, v159
	ds_read_b128 v[224:227], v159
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[186:187], v[98:113]
	v_or_b32_e32 v159, 48, v157
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[190:191], v[98:113]
	v_add_u32_e32 v158, 64, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[194:195], v[98:113]
	v_add_u32_e32 v158, 0x50, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[198:199], v[98:113]
	v_add_u32_e32 v158, 0x60, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[202:203], v[98:113]
	v_add_u32_e32 v158, 0x70, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[206:207], v[98:113]
	v_add_u32_e32 v158, 0x80, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[210:211], v[98:113]
	v_add_u32_e32 v158, 0x90, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[212:213], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[214:215], v[98:113]
	v_add_u32_e32 v158, 0xa0, v157
	v_lshrrev_b32_e32 v159, 3, v158
	v_and_b32_e32 v159, 56, v159
	v_xor_b32_e32 v158, v159, v158
	v_lshlrev_b32_e32 v158, 1, v158
	ds_read_b128 v[224:227], v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[216:217], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[218:219], v[98:113]
	v_add_u32_e32 v157, 0xb0, v157
	v_lshrrev_b32_e32 v158, 3, v157
	v_and_b32_e32 v158, 56, v158
	v_xor_b32_e32 v157, v158, v157
	v_lshlrev_b32_e32 v157, 1, v157
	ds_read_b128 v[224:227], v157
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[224:225], v[220:221], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[226:227], v[222:223], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v157, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v157, 2, v157
	v_and_b32_e32 v157, 8, v157
	v_add_u32_e32 v157, s87, v157
	v_subrev_u32_e32 v158, 23, v157
	v_cmp_gt_i32_e32 vcc, s86, v158
	v_subrev_u32_e32 v158, 22, v157
	v_cmp_gt_i32_e64 s[0:1], s86, v158
	v_subrev_u32_e32 v158, 21, v157
	v_cmp_gt_i32_e64 s[4:5], s86, v158
	v_subrev_u32_e32 v158, 20, v157
	v_cmp_gt_i32_e64 s[6:7], s86, v158
	v_subrev_u32_e32 v158, 19, v157
	v_cmp_gt_i32_e64 s[8:9], s86, v158
	v_subrev_u32_e32 v158, 18, v157
	v_cmp_gt_i32_e64 s[10:11], s86, v158
	v_subrev_u32_e32 v158, 17, v157
	v_cmp_gt_i32_e64 s[12:13], s86, v158
	v_add_u32_e32 v158, -16, v157
	v_cmp_gt_i32_e64 s[14:15], s86, v158
	v_add_u32_e32 v158, -7, v157
	v_cmp_gt_i32_e64 s[16:17], s86, v158
	v_add_u32_e32 v158, -6, v157
	v_cmp_gt_i32_e64 s[18:19], s86, v158
	v_add_u32_e32 v158, -5, v157
	v_cmp_gt_i32_e64 s[20:21], s86, v158
	v_add_u32_e32 v158, -4, v157
	v_cmp_gt_i32_e64 s[22:23], s86, v158
	v_add_u32_e32 v158, -3, v157
	v_cmp_gt_i32_e64 s[24:25], s86, v158
	v_add_u32_e32 v158, -2, v157
	v_cmp_gt_i32_e64 s[26:27], s86, v158
	v_add_u32_e32 v158, -1, v157
	v_cmp_gt_i32_e64 s[28:29], s86, v158
	v_cmp_gt_i32_e64 s[30:31], s86, v157
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
	v_cndmask_b32_e64 v99, v155, v99, s[0:1]
	v_cndmask_b32_e32 v98, v155, v98, vcc
	v_cndmask_b32_e64 v101, v155, v101, s[6:7]
	v_cndmask_b32_e64 v100, v155, v100, s[4:5]
	v_pk_mul_f32 v[98:99], v[114:115], v[98:99]
	v_cndmask_b32_e64 v103, v155, v103, s[10:11]
	v_cndmask_b32_e64 v102, v155, v102, s[8:9]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v157, v98, v99
	v_cndmask_b32_e64 v105, v155, v105, s[14:15]
	v_cndmask_b32_e64 v104, v155, v104, s[12:13]
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v157, v157, v100, v101
	v_cndmask_b32_e64 v107, v155, v107, s[18:19]
	v_cndmask_b32_e64 v106, v155, v106, s[16:17]
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v157, v157, v102, v103
	v_cndmask_b32_e64 v109, v155, v109, s[22:23]
	v_cndmask_b32_e64 v108, v155, v108, s[20:21]
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v157, v157, v104, v105
	v_cndmask_b32_e64 v111, v155, v111, s[26:27]
	v_cndmask_b32_e64 v110, v155, v110, s[24:25]
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v157, v157, v106, v107
	v_cndmask_b32_e64 v113, v155, v113, s[30:31]
	v_cndmask_b32_e64 v112, v155, v112, s[28:29]
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v157, v157, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v157, v157, v110, v111
	v_max3_f32 v157, v157, v112, v113
	ds_bpermute_b32 v158, v151, v157
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v158, v158, v158
	v_max_f32_e32 v158, v157, v158
	;;#ASMSTART
	v_add_f32 v157, v116, v153
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v158, v157
	v_mov_b32_e32 v157, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_53
	;;#ASMSTART
	v_add_f32 v132, v158, v154
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v116, v116, v132
	v_exp_f32_e32 v157, v116
	v_mov_b32_e32 v116, v132
	v_mov_b32_e32 v133, v132
	v_mov_b32_e32 v134, v132
	v_mov_b32_e32 v135, v132
	v_mov_b32_e32 v136, v132
	v_mov_b32_e32 v137, v132
	v_mov_b32_e32 v138, v132
	v_mov_b32_e32 v139, v132
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v132
	v_mov_b32_e32 v142, v132
	v_mov_b32_e32 v143, v132
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v132
	v_mov_b32_e32 v146, v132
	v_mov_b32_e32 v147, v132
	s_branch .LBB0_53
.LBB0_80:
	s_mov_b32 s91, s90
	s_branch .LBB0_54
.LBB0_81:
	ds_bpermute_b32 v98, v151, v156
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v98, v156, v98
	;;#ASMEND
	s_nop 0
	v_div_scale_f32 v99, s[0:1], v98, v98, s76
	v_rcp_f32_e32 v100, v99
	v_div_scale_f32 v101, vcc, s76, v98, s76
	v_fma_f32 v102, -v99, v100, 1.0
	v_fmac_f32_e32 v100, v102, v100
	v_mul_f32_e32 v102, v101, v100
	v_fma_f32 v103, -v99, v102, v101
	v_fmac_f32_e32 v102, v103, v100
	v_fma_f32 v99, -v99, v102, v101
	v_div_fmas_f32 v99, v99, v100, v102
	v_div_fixup_f32 v98, v99, v98, s76
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
	v_perm_b32 v48, v3, v2, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v49, v5, v4, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v46, v7, v6, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v47, v9, v8, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v44, v11, v10, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v45, v13, v12, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v42, v15, v14, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v43, v17, v16, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v40, v19, v18, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v41, v21, v20, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v38, v23, v22, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v39, v25, v24, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v36, v27, v26, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v37, v29, v28, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v34, v31, v30, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v35, v33, v32, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v112, v113, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v110, v111, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v108, v109, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v106, v107, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v28, v104, v105, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v102, v103, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v100, v101, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v98, v99, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v51, v50, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v53, v52, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v55, v54, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v57, v56, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v59, v58, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v61, v60, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v63, v62, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v65, v64, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v67, v66, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v69, v68, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v71, v70, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v73, v72, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v75, v74, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v77, v76, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v79, v78, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v81, v80, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v83, v82, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v85, v84, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v87, v86, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v89, v88, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v91, v90, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v93, v92, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v95, v94, s79
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v97, v96, s79
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v50, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v50, 0x100, v50
	v_cmp_eq_u32_e32 vcc, 0, v50
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_83
	s_barrier
.LBB0_83:
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
	s_cbranch_execz .LBB0_85
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
.LBB0_85:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v88, 0x1ff, v91
	s_add_u32 s40, s60, s64
	s_addc_u32 s0, s61, s65
	v_lshlrev_b32_e32 v92, 3, v88
	s_and_b32 s41, s0, 0xffff
	s_mov_b32 s43, s39
	v_and_b32_e32 v91, 56, v91
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_86:
	v_mul_u32_u24_sdwa v95, v94, s82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s83, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s47, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[40:43], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_86
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 1, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_89
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
.LBB0_89:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s4, s47, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_90:
	v_mul_u32_u24_sdwa v95, v94, s82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s83, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s4, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[40:43], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_90
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 2, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_93
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
.LBB0_93:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_94:
	v_mul_u32_u24_sdwa v95, v94, s82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s83, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[40:43], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_94
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 3, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_97
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
.LBB0_97:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x30000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_98:
	v_mul_u32_u24_sdwa v95, v94, s82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s83, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[40:43], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_98
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 4, v87
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
	s_add_i32 s5, s4, 0x48000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_102:
	v_mul_u32_u24_sdwa v95, v94, s82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s83, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[40:43], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_102
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 5, v87
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
	s_add_i32 s5, s4, 0x60000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_106:
	v_mul_u32_u24_sdwa v95, v94, s82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s83, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[40:43], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_106
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 6, v87
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
	s_add_i32 s5, s4, 0x78000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_110:
	v_mul_u32_u24_sdwa v95, v94, s82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s83, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[40:43], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_110
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 7, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_113
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
.LBB0_113:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s4, s4, 0x90000
	s_mov_b64 s[0:1], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_114:
	v_mul_u32_u24_sdwa v2, v88, s82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v3, v92, v91
	v_lshrrev_b32_e32 v2, 20, v2
	v_lshlrev_b32_e32 v3, 1, v3
	v_mul_lo_u16_e32 v5, 24, v2
	ds_read_b128 v[6:9], v3
	v_sub_u16_e32 v3, v88, v5
	v_lshlrev_b16_e32 v3, 3, v3
	v_add_u32_e32 v4, 0x200, v88
	v_cmp_lt_u32_e32 vcc, s83, v88
	v_mul_u32_u24_e32 v2, 0xc00, v2
	v_add_u32_e32 v3, s4, v3
	v_add_u32_e32 v92, 0x1000, v92
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v88, v4
	v_add_lshl_u32 v2, v3, v2, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v2, s[40:43], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_114
	s_or_b64 exec, exec, s[0:1]
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
	s_cbranch_execz .LBB0_119
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_118
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v3, s8
	global_atomic_add v3, v152, v3, s[62:63] sc0
.LBB0_118:
	s_or_b64 exec, exec, s[6:7]
	s_lshl_b64 s[6:7], s[0:1], 2
	s_add_u32 s6, s62, s6
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v3
	s_addc_u32 s7, s63, s7
	s_nop 0
	v_add_u32_e32 v2, s8, v2
	global_store_dword v152, v2, s[6:7]
	s_waitcnt vmcnt(0)
.LBB0_119:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s62, s0
	s_addc_u32 s1, s63, s1
	s_barrier
	global_load_dword v2, v152, s[0:1]
	s_sub_i32 s0, s33, s2
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s2, v2
	s_add_i32 s33, s0, s2
	s_cmp_ge_i32 s46, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s74
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_122
	s_branch .LBB0_12
.LBB0_120:
	s_mov_b32 s46, s5
.LBB0_121:
	s_sub_i32 s33, s33, s74
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s77, 0, s4
	s_cmp_ge_i32 s46, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s74, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_122:
	s_add_i32 s4, s77, 1
	s_cmp_gt_i32 s4, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s4, 16
	s_cbranch_scc1 .LBB0_125
	s_add_i32 s5, s46, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s74
	s_cbranch_scc1 .LBB0_120
	s_ashr_i32 s47, s46, 31
	s_lshl_b64 s[6:7], s[46:47], 2
	s_add_u32 s6, s34, s6
	s_addc_u32 s7, s35, s7
	global_load_dwordx2 v[2:3], v152, s[6:7] offset:4
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
	s_mov_b32 s6, s74
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
	.set .Lattn_kernel_0.numbered_sgpr, 93
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
    .sgpr_count:     99
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
