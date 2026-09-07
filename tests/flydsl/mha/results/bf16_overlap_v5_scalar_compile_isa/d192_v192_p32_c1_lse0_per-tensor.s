	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	attn_kernel_0
	.p2align	8
	.type	attn_kernel_0,@function
attn_kernel_0:
	s_load_dwordx2 s[20:21], s[0:1], 0x30
	s_load_dword s3, s[0:1], 0x38
	s_load_dwordx8 s[4:11], s[0:1], 0x70
	s_mov_b32 s22, 0
	s_waitcnt lgkmcnt(0)
	s_load_dwordx2 s[12:13], s[20:21], 0x0
	s_add_i32 s3, s3, -1
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s12, s13, s12
	s_add_i32 s14, s12, 0xff
	s_ashr_i32 s12, s14, 31
	s_lshr_b32 s12, s12, 24
	s_add_i32 s12, s14, s12
	s_ashr_i32 s16, s12, 8
	s_and_b32 s12, s12, 0xffffff00
	s_cmp_lg_u32 s14, s12
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s14, 0
	s_cselect_b64 s[14:15], -1, 0
	s_and_b64 s[12:13], s[14:15], s[12:13]
	s_subb_u32 s14, s16, 0
	s_cmp_lt_i32 s3, 1
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s2, s14
	s_cselect_b64 s[16:17], -1, 0
	s_or_b64 s[12:13], s[12:13], s[16:17]
	s_and_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_8
	s_mov_b32 s47, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s22, s16
.LBB0_3:
	s_sub_i32 s33, s33, s14
	s_and_b64 s[12:13], s[12:13], exec
	s_cselect_b32 s47, 0, s15
	s_cmp_ge_i32 s22, s3
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s33, s44
	s_cselect_b64 s[14:15], -1, 0
	s_or_b64 s[12:13], s[12:13], s[14:15]
	s_and_b64 vcc, exec, s[12:13]
	s_mov_b32 s14, s44
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s15, s47, 1
	s_cmp_gt_i32 s15, 15
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s15, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s16, s22, 1
	s_cmp_ge_i32 s16, s3
	s_mov_b32 s44, s14
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s23, s22, 31
	s_lshl_b64 s[18:19], s[22:23], 2
	s_add_u32 s18, s20, s18
	s_addc_u32 s19, s21, s19
	s_load_dwordx2 s[22:23], s[18:19], 0x4
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s17, s23, s22
	s_addk_i32 s17, 0xff
	s_ashr_i32 s18, s17, 31
	s_lshr_b32 s18, s18, 24
	s_add_i32 s18, s17, s18
	s_ashr_i32 s24, s18, 8
	s_and_b32 s18, s18, 0xffffff00
	s_cmp_lg_u32 s17, s18
	s_cselect_b64 s[18:19], -1, 0
	s_cmp_lt_i32 s17, 0
	s_cselect_b64 s[22:23], -1, 0
	s_and_b64 s[18:19], s[22:23], s[18:19]
	s_subb_u32 s44, s24, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s44, s14
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s44, s14
	s_mov_b32 s47, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s45, s[6:7], 0x0
	s_load_dword s46, s[8:9], 0x0
	s_cmp_ge_i32 s22, s3
	s_cbranch_scc1 .LBB0_111
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_load_dwordx2 s[24:25], s[0:1], 0x20
	s_load_dwordx2 s[26:27], s[0:1], 0x50
	s_load_dwordx2 s[28:29], s[0:1], 0x60
	s_load_dwordx2 s[30:31], s[0:1], 0x98
	s_load_dwordx2 s[34:35], s[0:1], 0xb8
	v_mul_u32_u24_e32 v1, 0xaab, v0
	v_mov_b32_e32 v2, 24
	v_mul_lo_u16_sdwa v2, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_mul_u32_u24_e32 v3, 0x2ab, v0
	v_mov_b32_e32 v4, 2
	v_mov_b32_e32 v5, 5
	v_sub_u16_e32 v2, v0, v2
	v_lshlrev_b16_sdwa v1, v4, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_u32_u24_e32 v4, 0x156, v0
	v_lshlrev_b16_sdwa v3, v5, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mov_b32_e32 v5, 4
	v_lshlrev_b16_e32 v2, 7, v2
	v_lshlrev_b16_sdwa v4, v5, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v3, 32, v3
	v_or_b32_e32 v2, v2, v4
	v_and_b32_e32 v1, 12, v1
	v_add_u16_e32 v2, v2, v3
	v_lshlrev_b32_e32 v3, 1, v0
	v_or_b32_e32 v1, v2, v1
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v117, v2, v3
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_and_b32_e32 v136, 31, v0
	v_xor_b32_e32 v137, 0x80, v2
	v_mov_b32_e32 v138, 0
	s_movk_i32 s48, 0xc00
	s_mov_b32 s15, 0x27000
	s_movk_i32 s49, 0x180
	v_mov_b32_e32 v139, 0x40e00000
	v_mov_b32_e32 v140, 1.0
	s_mov_b32 s50, 0x7060302
	s_movk_i32 s51, 0x1000
	s_movk_i32 s52, 0x2000
	s_movk_i32 s53, 0xe0
	s_mov_b32 s54, 0xaaab
	s_movk_i32 s55, 0xff
	s_mov_b32 s56, s2
	v_mov_b32_e32 v141, 0xff800000
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s44, s14
.LBB0_12:
	s_cmp_ge_i32 s22, s3
	s_cbranch_scc1 .LBB0_111
.LBB0_13:
	s_ashr_i32 s23, s22, 31
	s_lshl_b32 s42, s33, 8
	s_lshl_b64 s[0:1], s[22:23], 2
	s_add_u32 s12, s20, s0
	s_addc_u32 s13, s21, s1
	global_load_dwordx2 v[4:5], v138, s[12:13]
	global_load_dword v2, v138, s[4:5]
	s_mul_i32 s23, s47, 0xc0
	s_mov_b32 s19, s15
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s38, v4
	s_add_i32 s13, s38, s42
	v_readfirstlane_b32 s39, v5
	s_add_i32 s14, s13, 0x100
	s_min_i32 s14, s14, s39
	s_sub_i32 s43, s14, s13
	s_waitcnt lgkmcnt(0)
	s_add_u32 s16, s26, s0
	s_addc_u32 s17, s27, s1
	s_mul_i32 s12, s13, 0xc00
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_ashr_i32 s13, s12, 31
	global_load_dwordx2 v[4:5], v138, s[16:17]
	global_load_dword v3, v138, s[0:1]
	;;#ASMSTART
	v_mov_b32 v6, v0
	;;#ASMEND
	s_lshl_b64 s[0:1], s[12:13], 1
	v_and_b32_e32 v7, 31, v6
	v_bfe_u32 v8, v6, 6, 3
	v_lshrrev_b32_e32 v6, 2, v6
	s_add_u32 s16, s6, s0
	v_and_or_b32 v6, v6, 8, s23
	s_addc_u32 s12, s7, s1
	v_mul_u32_u24_e32 v8, 0x18000, v8
	v_mad_u32_u24 v6, v7, s48, v6
	s_mul_i32 s18, s43, 0x1800
	s_and_b32 s17, s12, 0xffff
	v_add_lshl_u32 v6, v6, v8, 1
	buffer_load_dwordx4 v[162:165], v6, s[16:19], 0 offen
	buffer_load_dwordx4 v[166:169], v6, s[16:19], 0 offen offset:32
	buffer_load_dwordx4 v[170:173], v6, s[16:19], 0 offen offset:64
	buffer_load_dwordx4 v[174:177], v6, s[16:19], 0 offen offset:96
	buffer_load_dwordx4 v[178:181], v6, s[16:19], 0 offen offset:128
	buffer_load_dwordx4 v[182:185], v6, s[16:19], 0 offen offset:160
	buffer_load_dwordx4 v[186:189], v6, s[16:19], 0 offen offset:192
	buffer_load_dwordx4 v[190:193], v6, s[16:19], 0 offen offset:224
	buffer_load_dwordx4 v[194:197], v6, s[16:19], 0 offen offset:256
	buffer_load_dwordx4 v[198:201], v6, s[16:19], 0 offen offset:288
	buffer_load_dwordx4 v[202:205], v6, s[16:19], 0 offen offset:320
	buffer_load_dwordx4 v[206:209], v6, s[16:19], 0 offen offset:352
	s_waitcnt vmcnt(13)
	v_readfirstlane_b32 s12, v4
	s_waitcnt vmcnt(12)
	v_readfirstlane_b32 s40, v3
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	v_readfirstlane_b32 s13, v5
	v_and_b32_e32 v3, 0x100, v3
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_15
	s_barrier
.LBB0_15:
	s_or_b64 exec, exec, s[16:17]
	s_sub_i32 s61, s13, s12
	s_ashr_i32 s13, s47, 31
	s_lshr_b32 s13, s13, 28
	s_add_i32 s13, s47, s13
	s_ashr_i32 s14, s13, 4
	s_and_b32 s13, s13, -16
	s_cmp_lg_u32 s47, s13
	s_cselect_b64 s[16:17], -1, 0
	s_cmp_lt_i32 s47, 0
	s_cselect_b64 s[36:37], -1, 0
	s_and_b64 s[16:17], s[36:37], s[16:17]
	s_subb_u32 s19, s14, 0
	s_mul_i32 s16, s19, 0x1800
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[16:17], s[16:17], 1
	s_add_u32 s16, s8, s16
	s_addc_u32 s17, s9, s17
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s12, s28, s12
	s_addc_u32 s13, s29, s13
	s_lshl_b32 s14, s61, 2
	s_and_b32 s13, s13, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[12:15], 0
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s64, v4
	v_and_b32_e32 v3, 0x180, v3
	v_readfirstlane_b32 s59, v5
	v_cmp_ne_u32_e32 vcc, s49, v3
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_17
	s_mul_i32 s41, s64, 0xc00
	v_add_u32_sdwa v4, s41, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[16:17]
	global_load_dwordx4 v[146:149], v[4:5], off
	global_load_dwordx4 v[150:153], v[4:5], off offset:256
.LBB0_17:
	s_or_b64 exec, exec, s[36:37]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	buffer_load_dword v3, off, s[12:15], 0 offset:8
	;;#ASMSTART
	v_mov_b32 v4, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s57, v3
	v_and_b32_e32 v4, 0x180, v4
	v_cmp_ne_u32_e32 vcc, s49, v4
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_19
	s_mul_i32 s41, s59, 0xc00
	v_add_u32_sdwa v4, s41, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[16:17]
	global_load_dwordx4 v[154:157], v[4:5], off
	global_load_dwordx4 v[158:161], v[4:5], off offset:256
.LBB0_19:
	s_or_b64 exec, exec, s[36:37]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	buffer_load_dword v3, off, s[12:15], 0 offset:12
	;;#ASMSTART
	v_mov_b32 v4, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s60, v3
	v_and_b32_e32 v4, 0x180, v4
	v_cmp_ne_u32_e32 vcc, s49, v4
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_21
	ds_write_b128 v117, v[146:149]
	ds_write_b128 v117, v[150:153] offset:6144
.LBB0_21:
	s_or_b64 exec, exec, s[36:37]
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s49, v3
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_23
	s_mul_i32 s41, s57, 0xc00
	v_add_u32_sdwa v4, s41, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[16:17]
	global_load_dwordx4 v[146:149], v[4:5], off
	global_load_dwordx4 v[150:153], v[4:5], off offset:256
.LBB0_23:
	s_or_b64 exec, exec, s[36:37]
	v_mul_f32_e32 v2, s45, v2
	s_sub_i32 s36, s38, s39
	s_lshl_b32 s37, s61, 5
	v_mul_f32_e32 v114, 0x3dd53b95, v2
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_add_i32 s36, s36, s37
	s_add_i32 s36, s36, s40
	s_sub_i32 s58, s36, 32
	s_add_i32 s63, s58, s42
	s_add_i32 s38, s63, 1
	s_ashr_i32 s36, s38, 31
	s_lshr_b32 s36, s36, 27
	s_add_i32 s36, s38, s36
	s_ashr_i32 s40, s36, 5
	s_andn2_b32 s36, s36, 31
	s_cmp_lg_u32 s38, s36
	s_cselect_b64 s[36:37], -1, 0
	s_cmp_lt_i32 s38, 0
	s_cselect_b64 s[38:39], -1, 0
	s_and_b64 s[36:37], s[38:39], s[36:37]
	s_subb_u32 s38, s40, 0
	s_lshr_b32 s36, s38, 31
	s_add_i32 s36, s38, s36
	s_ashr_i32 s40, s36, 1
	s_and_b32 s36, s36, -2
	s_cmp_lg_u32 s38, s36
	s_cselect_b64 s[36:37], -1, 0
	s_cmp_lt_i32 s38, 0
	s_cselect_b64 s[38:39], -1, 0
	s_and_b64 s[36:37], s[38:39], s[36:37]
	s_subb_u32 s62, s40, 0
	s_lshl_b32 s36, s62, 1
	s_ashr_i32 s37, s36, 31
	s_cmp_lt_i32 s62, 1
	s_cbranch_scc1 .LBB0_43
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v118, v114
	v_mov_b32_e32 v119, v114
	s_mov_b64 s[38:39], 0
	s_mov_b32 s65, 20
	v_mov_b32_e32 v120, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v142
	v_mov_b32_e32 v4, v142
	v_mov_b32_e32 v5, v142
	v_mov_b32_e32 v6, v142
	v_mov_b32_e32 v7, v142
	v_mov_b32_e32 v8, v142
	v_mov_b32_e32 v9, v142
	v_mov_b32_e32 v10, v142
	v_mov_b32_e32 v11, v142
	v_mov_b32_e32 v12, v142
	v_mov_b32_e32 v13, v142
	v_mov_b32_e32 v14, v142
	v_mov_b32_e32 v15, v142
	v_mov_b32_e32 v16, v142
	v_mov_b32_e32 v17, v142
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v142
	v_mov_b32_e32 v20, v142
	v_mov_b32_e32 v21, v142
	v_mov_b32_e32 v22, v142
	v_mov_b32_e32 v23, v142
	v_mov_b32_e32 v24, v142
	v_mov_b32_e32 v25, v142
	v_mov_b32_e32 v26, v142
	v_mov_b32_e32 v27, v142
	v_mov_b32_e32 v28, v142
	v_mov_b32_e32 v29, v142
	v_mov_b32_e32 v30, v142
	v_mov_b32_e32 v31, v142
	v_mov_b32_e32 v32, v142
	v_mov_b32_e32 v33, v142
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v142
	v_mov_b32_e32 v36, v142
	v_mov_b32_e32 v37, v142
	v_mov_b32_e32 v38, v142
	v_mov_b32_e32 v39, v142
	v_mov_b32_e32 v40, v142
	v_mov_b32_e32 v41, v142
	v_mov_b32_e32 v42, v142
	v_mov_b32_e32 v43, v142
	v_mov_b32_e32 v44, v142
	v_mov_b32_e32 v45, v142
	v_mov_b32_e32 v46, v142
	v_mov_b32_e32 v47, v142
	v_mov_b32_e32 v48, v142
	v_mov_b32_e32 v49, v142
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v142
	v_mov_b32_e32 v52, v142
	v_mov_b32_e32 v53, v142
	v_mov_b32_e32 v54, v142
	v_mov_b32_e32 v55, v142
	v_mov_b32_e32 v56, v142
	v_mov_b32_e32 v57, v142
	v_mov_b32_e32 v58, v142
	v_mov_b32_e32 v59, v142
	v_mov_b32_e32 v60, v142
	v_mov_b32_e32 v61, v142
	v_mov_b32_e32 v62, v142
	v_mov_b32_e32 v63, v142
	v_mov_b32_e32 v64, v142
	v_mov_b32_e32 v65, v142
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, v142
	v_mov_b32_e32 v68, v142
	v_mov_b32_e32 v69, v142
	v_mov_b32_e32 v70, v142
	v_mov_b32_e32 v71, v142
	v_mov_b32_e32 v72, v142
	v_mov_b32_e32 v73, v142
	v_mov_b32_e32 v74, v142
	v_mov_b32_e32 v75, v142
	v_mov_b32_e32 v76, v142
	v_mov_b32_e32 v77, v142
	v_mov_b32_e32 v78, v142
	v_mov_b32_e32 v79, v142
	v_mov_b32_e32 v80, v142
	v_mov_b32_e32 v81, v142
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, v142
	v_mov_b32_e32 v84, v142
	v_mov_b32_e32 v85, v142
	v_mov_b32_e32 v86, v142
	v_mov_b32_e32 v87, v142
	v_mov_b32_e32 v88, v142
	v_mov_b32_e32 v89, v142
	v_mov_b32_e32 v90, v142
	v_mov_b32_e32 v91, v142
	v_mov_b32_e32 v92, v142
	v_mov_b32_e32 v93, v142
	v_mov_b32_e32 v94, v142
	v_mov_b32_e32 v95, v142
	v_mov_b32_e32 v96, v142
	v_mov_b32_e32 v97, v142
	s_branch .LBB0_26
.LBB0_25:
	s_or_b64 exec, exec, s[40:41]
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
	v_add_u32_e32 v115, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v115, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s50
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s40, s59, s19
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_mulk_i32 s40, 0x1800
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s40
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[120:123], v[106:107], off offset:512
	global_load_dwordx4 v[124:127], v[106:107], off offset:1024
	global_load_dwordx4 v[128:131], v[106:107], off offset:1536
	global_load_dwordx4 v[132:135], v[106:107], off offset:2048
	global_load_dwordx4 v[210:213], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[124:125], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[210:211], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s51, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[102:103], v[18:33]
	global_load_dwordx4 v[120:123], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[102:103], v[34:49]
	global_load_dwordx4 v[124:127], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[102:103], v[50:65]
	global_load_dwordx4 v[128:131], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s52, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[212:213], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[124:125], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	s_add_u32 s38, s38, 2
	s_addc_u32 s39, s39, 0
	v_mov_b64_e32 v[98:99], s[36:37]
	v_cmp_lt_i64_e32 vcc, s[38:39], v[98:99]
	s_add_i32 s65, s65, 8
	s_mov_b32 s59, s67
	s_mov_b32 s64, s66
	v_mov_b32_e32 v120, v116
	s_cbranch_vccz .LBB0_42
.LBB0_26:
	s_add_i32 s40, s65, -4
	v_mov_b32_e32 v98, s40
	buffer_load_dword v98, v98, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_mov_b32 s66, s57
	v_and_b32_e32 v99, 0x180, v99
	s_mov_b32 s67, s60
	v_cmp_ne_u32_e32 vcc, s49, v99
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s57, v98
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_28
	ds_write_b128 v117, v[154:157] offset:12288
	ds_write_b128 v117, v[158:161] offset:18432
.LBB0_28:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s49, v98
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_30
	s_mul_i32 s60, s67, 0xc00
	v_add_u32_sdwa v98, s60, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[154:157], v[98:99], off
	global_load_dwordx4 v[158:161], v[98:99], off offset:256
.LBB0_30:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v115, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v116, 56, v98
	v_xor_b32_e32 v98, v116, v115
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[122:125], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[162:163], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[164:165], v[98:113]
	v_or_b32_e32 v121, 16, v115
	v_xor_b32_e32 v121, v121, v116
	v_lshlrev_b32_e32 v121, 1, v121
	ds_read_b128 v[122:125], v121
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[166:167], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[168:169], v[98:113]
	v_or_b32_e32 v121, 32, v115
	v_xor_b32_e32 v121, v121, v116
	v_lshlrev_b32_e32 v121, 1, v121
	ds_read_b128 v[122:125], v121
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[170:171], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[172:173], v[98:113]
	v_or_b32_e32 v121, 48, v115
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[174:175], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[176:177], v[98:113]
	v_add_u32_e32 v116, 64, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[178:179], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[180:181], v[98:113]
	v_add_u32_e32 v116, 0x50, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[182:183], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[184:185], v[98:113]
	v_add_u32_e32 v116, 0x60, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[186:187], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[188:189], v[98:113]
	v_add_u32_e32 v116, 0x70, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[190:191], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[192:193], v[98:113]
	v_add_u32_e32 v116, 0x80, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[194:195], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[196:197], v[98:113]
	v_add_u32_e32 v116, 0x90, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[198:199], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[200:201], v[98:113]
	v_add_u32_e32 v116, 0xa0, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[202:203], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[204:205], v[98:113]
	v_add_u32_e32 v115, 0xb0, v115
	v_lshrrev_b32_e32 v116, 3, v115
	v_and_b32_e32 v116, 56, v116
	v_xor_b32_e32 v115, v116, v115
	v_lshlrev_b32_e32 v115, 1, v115
	ds_read_b128 v[122:125], v115
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[206:207], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[208:209], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v115, v114
	s_nop 6
	v_pk_mul_f32 v[98:99], v[118:119], v[98:99]
	v_pk_mul_f32 v[112:113], v[114:115], v[112:113]
	v_pk_mul_f32 v[110:111], v[114:115], v[110:111]
	v_pk_mul_f32 v[108:109], v[114:115], v[108:109]
	v_pk_mul_f32 v[106:107], v[114:115], v[106:107]
	v_pk_mul_f32 v[104:105], v[114:115], v[104:105]
	v_pk_mul_f32 v[102:103], v[114:115], v[102:103]
	v_pk_mul_f32 v[100:101], v[114:115], v[100:101]
	v_max_f32_e32 v115, v98, v99
	v_max3_f32 v115, v115, v100, v101
	v_max3_f32 v115, v115, v102, v103
	v_max3_f32 v115, v115, v104, v105
	v_max3_f32 v115, v115, v106, v107
	v_max3_f32 v115, v115, v108, v109
	v_max3_f32 v115, v115, v110, v111
	v_max3_f32 v115, v115, v112, v113
	ds_bpermute_b32 v116, v137, v115
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v116, v116, v116
	v_max_f32_e32 v116, v115, v116
	;;#ASMSTART
	v_add_f32 v115, v120, v139
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v116, v115
	v_mov_b32_e32 v115, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v116, v116, v140
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v115, v120, v116
	v_exp_f32_e32 v115, v115
	v_mov_b32_e32 v120, v116
.LBB0_32:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[98:99], v[98:99], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, 0, v98
	v_add_f32_e32 v116, v116, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v100
	v_add_f32_e32 v116, v116, v101
	v_add_f32_e32 v116, v116, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v103
	v_add_f32_e32 v116, v116, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v105
	v_add_f32_e32 v116, v116, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v107
	v_add_f32_e32 v116, v116, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[120:121] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v116, v116, v109
	v_add_f32_e32 v116, v116, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v115
	v_add_f32_e32 v116, v116, v111
	v_add_f32_e32 v116, v116, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v116, v116, v113
	;;#ASMSTART
	v_fma_f32 v142, v142, v115, v116
	;;#ASMEND
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_34
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
	v_mul_f32 v66, v66, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v115
	;;#ASMEND
.LBB0_34:
	s_or_b64 exec, exec, s[40:41]
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
	v_add_u32_e32 v115, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v115, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s50
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s40, s64, s19
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_mulk_i32 s40, 0x1800
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s40
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[122:125], v[106:107], off offset:512
	global_load_dwordx4 v[126:129], v[106:107], off offset:1024
	global_load_dwordx4 v[130:133], v[106:107], off offset:1536
	global_load_dwordx4 v[210:213], v[106:107], off offset:2048
	global_load_dwordx4 v[214:217], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[214:215], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s51, v106
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
	v_add_co_u32_e32 v100, vcc, s52, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[212:213], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[216:217], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	v_mov_b32_e32 v98, s65
	buffer_load_dword v98, v98, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s60, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s49, v99
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_36
	ds_write_b128 v117, v[146:149]
	ds_write_b128 v117, v[150:153] offset:6144
.LBB0_36:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s49, v98
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_38
	s_mul_i32 s64, s57, 0xc00
	v_add_u32_sdwa v98, s64, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[146:149], v[98:99], off
	global_load_dwordx4 v[150:153], v[98:99], off offset:256
.LBB0_38:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v115, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v115
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[122:125], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[162:163], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[164:165], v[98:113]
	v_add_u32_e32 v116, 0x1810, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[166:167], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[168:169], v[98:113]
	v_add_u32_e32 v116, 0x1820, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[170:171], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[172:173], v[98:113]
	v_add_u32_e32 v116, 0x1830, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[174:175], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[176:177], v[98:113]
	v_add_u32_e32 v116, 0x1840, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[178:179], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[180:181], v[98:113]
	v_add_u32_e32 v116, 0x1850, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[182:183], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[184:185], v[98:113]
	v_add_u32_e32 v116, 0x1860, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[186:187], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[188:189], v[98:113]
	v_add_u32_e32 v116, 0x1870, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[190:191], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[192:193], v[98:113]
	v_add_u32_e32 v116, 0x1880, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[194:195], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[196:197], v[98:113]
	v_add_u32_e32 v116, 0x1890, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[198:199], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[200:201], v[98:113]
	v_add_u32_e32 v116, 0x18a0, v115
	v_lshrrev_b32_e32 v121, 3, v116
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v116, v121, v116
	v_lshlrev_b32_e32 v116, 1, v116
	ds_read_b128 v[122:125], v116
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[202:203], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[204:205], v[98:113]
	v_add_u32_e32 v115, 0x18b0, v115
	v_lshrrev_b32_e32 v116, 3, v115
	v_and_b32_e32 v116, 56, v116
	v_xor_b32_e32 v115, v116, v115
	v_lshlrev_b32_e32 v115, 1, v115
	ds_read_b128 v[122:125], v115
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[206:207], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[208:209], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v115, v114
	s_nop 6
	v_pk_mul_f32 v[98:99], v[118:119], v[98:99]
	v_pk_mul_f32 v[112:113], v[114:115], v[112:113]
	v_pk_mul_f32 v[110:111], v[114:115], v[110:111]
	v_pk_mul_f32 v[108:109], v[114:115], v[108:109]
	v_pk_mul_f32 v[106:107], v[114:115], v[106:107]
	v_pk_mul_f32 v[104:105], v[114:115], v[104:105]
	v_pk_mul_f32 v[102:103], v[114:115], v[102:103]
	v_pk_mul_f32 v[100:101], v[114:115], v[100:101]
	v_max_f32_e32 v115, v98, v99
	v_max3_f32 v115, v115, v100, v101
	v_max3_f32 v115, v115, v102, v103
	v_max3_f32 v115, v115, v104, v105
	v_max3_f32 v115, v115, v106, v107
	v_max3_f32 v115, v115, v108, v109
	v_max3_f32 v115, v115, v110, v111
	v_max3_f32 v115, v115, v112, v113
	ds_bpermute_b32 v116, v137, v115
	v_mov_b32_e32 v121, v120
	v_mov_b32_e32 v122, v120
	v_mov_b32_e32 v123, v120
	v_mov_b32_e32 v124, v120
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v116, v116, v116
	v_max_f32_e32 v143, v115, v116
	;;#ASMSTART
	v_add_f32 v115, v120, v139
	;;#ASMEND
	v_mov_b32_e32 v116, v120
	v_cmp_gt_f32_e32 vcc, v143, v115
	v_mov_b32_e32 v115, 1.0
	v_mov_b32_e32 v125, v120
	v_mov_b32_e32 v126, v120
	v_mov_b32_e32 v127, v120
	v_mov_b32_e32 v128, v120
	v_mov_b32_e32 v129, v120
	v_mov_b32_e32 v130, v120
	v_mov_b32_e32 v131, v120
	v_mov_b32_e32 v132, v120
	v_mov_b32_e32 v133, v120
	v_mov_b32_e32 v134, v120
	v_mov_b32_e32 v135, v120
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_40
	;;#ASMSTART
	v_add_f32 v116, v143, v140
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v115, v120, v116
	v_exp_f32_e32 v115, v115
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
	v_mov_b32_e32 v134, v116
	v_mov_b32_e32 v135, v116
.LBB0_40:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[98:99], v[98:99], v[120:121] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, 0, v98
	v_add_f32_e32 v120, v120, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v100
	v_add_f32_e32 v120, v120, v101
	v_add_f32_e32 v120, v120, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v103
	v_add_f32_e32 v120, v120, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v105
	v_add_f32_e32 v120, v120, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v107
	v_add_f32_e32 v120, v120, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v109
	v_add_f32_e32 v120, v120, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v115
	v_add_f32_e32 v120, v120, v111
	v_add_f32_e32 v120, v120, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v120, v120, v113
	;;#ASMSTART
	v_fma_f32 v142, v142, v115, v120
	;;#ASMEND
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_25
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
	v_mul_f32 v66, v66, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v115
	;;#ASMEND
	s_branch .LBB0_25
.LBB0_42:
	s_mov_b32 s59, s67
	s_mov_b32 s64, s66
	s_branch .LBB0_44
.LBB0_43:
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
	v_mov_b32_e32 v142, v2
.LBB0_44:
	s_add_i32 s38, s43, s63
	s_add_i32 s40, s38, 31
	s_ashr_i32 s38, s40, 31
	s_lshr_b32 s38, s38, 27
	s_add_i32 s38, s40, s38
	s_ashr_i32 s43, s38, 5
	s_andn2_b32 s38, s38, 31
	s_cmp_lg_u32 s40, s38
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_lt_i32 s40, 0
	s_cselect_b64 s[40:41], -1, 0
	s_and_b64 s[38:39], s[40:41], s[38:39]
	s_subb_u32 s38, s43, 0
	s_min_i32 s38, s38, s61
	s_cmp_ge_i32 s36, s38
	s_cbranch_scc1 .LBB0_66
	s_lshl_b32 s40, s62, 6
	s_ashr_i32 s39, s38, 31
	v_mov_b32_e32 v118, v114
	v_mov_b32_e32 v119, v114
	v_or_b32_e32 v143, s42, v136
	s_or_b32 s61, s40, 55
	s_lshl3_add_u32 s62, s62, 20
	s_branch .LBB0_48
.LBB0_46:
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
	v_add_u32_e32 v115, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v115, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s50
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s42, s59, s19
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_mulk_i32 s42, 0x1800
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s42
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[120:123], v[106:107], off offset:512
	global_load_dwordx4 v[124:127], v[106:107], off offset:1024
	global_load_dwordx4 v[128:131], v[106:107], off offset:1536
	global_load_dwordx4 v[132:135], v[106:107], off offset:2048
	global_load_dwordx4 v[210:213], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[124:125], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[210:211], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s51, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[102:103], v[18:33]
	global_load_dwordx4 v[120:123], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[102:103], v[34:49]
	global_load_dwordx4 v[124:127], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[102:103], v[50:65]
	global_load_dwordx4 v[128:131], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s52, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[212:213], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[124:125], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
.LBB0_47:
	s_and_b64 s[40:41], s[40:41], exec
	s_cselect_b32 s64, s57, s59
	s_cselect_b32 s59, s60, s57
	s_cselect_b32 s57, s63, s60
	s_add_u32 s36, s36, 2
	s_addc_u32 s37, s37, 0
	v_mov_b64_e32 v[98:99], s[38:39]
	v_cmp_lt_i64_e32 vcc, s[36:37], v[98:99]
	s_add_i32 s61, s61, 64
	s_add_i32 s62, s62, 8
	s_mov_b32 s60, s65
	s_cbranch_vccz .LBB0_66
.LBB0_48:
	s_add_i32 s40, s62, -4
	v_mov_b32_e32 v98, s40
	buffer_load_dword v98, v98, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s63, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s49, v99
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_50
	ds_write_b128 v117, v[154:157] offset:12288
	ds_write_b128 v117, v[158:161] offset:18432
.LBB0_50:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s49, v98
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_52
	s_mul_i32 s42, s60, 0xc00
	v_add_u32_sdwa v98, s42, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[154:157], v[98:99], off
	global_load_dwordx4 v[158:161], v[98:99], off offset:256
.LBB0_52:
	s_or_b64 exec, exec, s[40:41]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v115, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v120, 56, v98
	v_xor_b32_e32 v98, v120, v115
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[122:125], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[162:163], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[164:165], v[98:113]
	v_or_b32_e32 v121, 16, v115
	v_xor_b32_e32 v121, v121, v120
	v_lshlrev_b32_e32 v121, 1, v121
	ds_read_b128 v[122:125], v121
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[166:167], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[168:169], v[98:113]
	v_or_b32_e32 v121, 32, v115
	v_xor_b32_e32 v121, v121, v120
	v_lshlrev_b32_e32 v121, 1, v121
	ds_read_b128 v[122:125], v121
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[170:171], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[172:173], v[98:113]
	v_or_b32_e32 v121, 48, v115
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[174:175], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[176:177], v[98:113]
	v_add_u32_e32 v120, 64, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[178:179], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[180:181], v[98:113]
	v_add_u32_e32 v120, 0x50, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[182:183], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[184:185], v[98:113]
	v_add_u32_e32 v120, 0x60, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[186:187], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[188:189], v[98:113]
	v_add_u32_e32 v120, 0x70, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[190:191], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[192:193], v[98:113]
	v_add_u32_e32 v120, 0x80, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[194:195], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[196:197], v[98:113]
	v_add_u32_e32 v120, 0x90, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[198:199], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[200:201], v[98:113]
	v_add_u32_e32 v120, 0xa0, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[202:203], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[204:205], v[98:113]
	v_add_u32_e32 v115, 0xb0, v115
	v_lshrrev_b32_e32 v120, 3, v115
	v_and_b32_e32 v120, 56, v120
	v_xor_b32_e32 v115, v120, v115
	v_lshlrev_b32_e32 v115, 1, v115
	ds_read_b128 v[120:123], v115
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[206:207], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[208:209], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v115, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v120, 2, v115
	v_and_b32_e32 v120, 8, v120
	v_lshrrev_b32_e32 v115, 1, v115
	v_and_or_b32 v115, v115, s53, v143
	v_add_u32_e32 v122, s61, v120
	v_add_u32_e32 v115, s58, v115
	v_subrev_u32_e32 v120, 55, v122
	v_cmp_lt_i32_e32 vcc, v120, v115
	s_nop 1
	v_cndmask_b32_e32 v121, v141, v99, vcc
	v_cmp_le_i32_e32 vcc, v120, v115
	v_subrev_u32_e32 v99, 52, v122
	s_nop 0
	v_cndmask_b32_e32 v120, v141, v98, vcc
	v_subrev_u32_e32 v98, 53, v122
	v_cmp_le_i32_e32 vcc, v98, v115
	s_nop 1
	v_cndmask_b32_e32 v98, v141, v100, vcc
	v_cmp_le_i32_e32 vcc, v99, v115
	v_subrev_u32_e32 v100, 51, v122
	s_nop 0
	v_cndmask_b32_e32 v99, v141, v101, vcc
	v_cmp_le_i32_e32 vcc, v100, v115
	v_subrev_u32_e32 v101, 50, v122
	s_nop 0
	v_cndmask_b32_e32 v100, v141, v102, vcc
	v_cmp_le_i32_e32 vcc, v101, v115
	v_subrev_u32_e32 v102, 49, v122
	s_nop 0
	v_cndmask_b32_e32 v101, v141, v103, vcc
	v_cmp_le_i32_e32 vcc, v102, v115
	v_subrev_u32_e32 v103, 48, v122
	s_nop 0
	v_cndmask_b32_e32 v102, v141, v104, vcc
	v_cmp_le_i32_e32 vcc, v103, v115
	v_subrev_u32_e32 v104, 39, v122
	s_nop 0
	v_cndmask_b32_e32 v103, v141, v105, vcc
	v_cmp_le_i32_e32 vcc, v104, v115
	v_subrev_u32_e32 v105, 38, v122
	s_nop 0
	v_cndmask_b32_e32 v104, v141, v106, vcc
	v_cmp_le_i32_e32 vcc, v105, v115
	v_subrev_u32_e32 v106, 37, v122
	s_nop 0
	v_cndmask_b32_e32 v105, v141, v107, vcc
	v_cmp_le_i32_e32 vcc, v106, v115
	v_subrev_u32_e32 v107, 36, v122
	s_nop 0
	v_cndmask_b32_e32 v106, v141, v108, vcc
	v_cmp_le_i32_e32 vcc, v107, v115
	v_subrev_u32_e32 v108, 35, v122
	s_nop 0
	v_cndmask_b32_e32 v107, v141, v109, vcc
	v_cmp_le_i32_e32 vcc, v108, v115
	v_subrev_u32_e32 v109, 34, v122
	s_nop 0
	v_cndmask_b32_e32 v108, v141, v110, vcc
	v_cmp_le_i32_e32 vcc, v109, v115
	v_subrev_u32_e32 v110, 33, v122
	s_nop 0
	v_cndmask_b32_e32 v109, v141, v111, vcc
	v_cmp_le_i32_e32 vcc, v110, v115
	v_subrev_u32_e32 v111, 32, v122
	s_nop 0
	v_cndmask_b32_e32 v110, v141, v112, vcc
	v_cmp_le_i32_e32 vcc, v111, v115
	v_mov_b32_e32 v115, v114
	v_pk_mul_f32 v[98:99], v[114:115], v[98:99]
	v_cndmask_b32_e32 v111, v141, v113, vcc
	v_pk_mul_f32 v[112:113], v[114:115], v[110:111]
	v_pk_mul_f32 v[110:111], v[114:115], v[108:109]
	v_pk_mul_f32 v[108:109], v[114:115], v[106:107]
	v_pk_mul_f32 v[106:107], v[114:115], v[104:105]
	v_pk_mul_f32 v[104:105], v[114:115], v[102:103]
	v_pk_mul_f32 v[102:103], v[114:115], v[100:101]
	v_pk_mul_f32 v[100:101], v[118:119], v[120:121]
	s_nop 0
	v_max_f32_e32 v115, v100, v101
	v_max3_f32 v115, v115, v98, v99
	v_max3_f32 v115, v115, v102, v103
	v_max3_f32 v115, v115, v104, v105
	v_max3_f32 v115, v115, v106, v107
	v_max3_f32 v115, v115, v108, v109
	v_max3_f32 v115, v115, v110, v111
	v_max3_f32 v115, v115, v112, v113
	ds_bpermute_b32 v120, v137, v115
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v120, v120, v120
	v_max_f32_e32 v120, v115, v120
	;;#ASMSTART
	v_add_f32 v115, v116, v139
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v120, v115
	v_mov_b32_e32 v115, 1.0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_54
	;;#ASMSTART
	v_add_f32 v120, v120, v140
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v115, v116, v120
	v_exp_f32_e32 v115, v115
	v_mov_b32_e32 v116, v120
.LBB0_54:
	s_or_b64 exec, exec, s[40:41]
	v_pk_add_f32 v[120:121], v[98:99], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[98:99], v[100:101], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v100, v120
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v121
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, 0, v98
	v_add_f32_e32 v120, v120, v99
	v_add_f32_e32 v120, v120, v100
	v_add_f32_e32 v120, v120, v101
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v102
	v_add_f32_e32 v120, v120, v103
	v_add_f32_e32 v120, v120, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v105
	v_add_f32_e32 v120, v120, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v107
	v_add_f32_e32 v120, v120, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[116:117] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v109
	v_add_f32_e32 v120, v120, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v115
	v_add_f32_e32 v120, v120, v111
	v_add_f32_e32 v120, v120, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v120, v120, v113
	;;#ASMSTART
	v_fma_f32 v142, v142, v115, v120
	;;#ASMEND
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_56
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
	v_mul_f32 v66, v66, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v115
	;;#ASMEND
.LBB0_56:
	s_or_b64 exec, exec, s[40:41]
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
	v_add_u32_e32 v115, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v115, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s50
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s40, s64, s19
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_mulk_i32 s40, 0x1800
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s40
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[120:123], v[106:107], off offset:512
	global_load_dwordx4 v[124:127], v[106:107], off offset:1024
	global_load_dwordx4 v[128:131], v[106:107], off offset:1536
	global_load_dwordx4 v[132:135], v[106:107], off offset:2048
	global_load_dwordx4 v[210:213], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[124:125], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[132:133], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[210:211], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s51, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[102:103], v[18:33]
	global_load_dwordx4 v[120:123], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[102:103], v[34:49]
	global_load_dwordx4 v[124:127], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[102:103], v[50:65]
	global_load_dwordx4 v[128:131], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s52, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[212:213], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[120:121], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[124:125], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[122:123], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[130:131], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	s_add_i32 s42, s36, 1
	s_cmp_gt_i32 s38, s42
	s_cselect_b64 s[40:41], -1, 0
	s_cmp_le_i32 s38, s42
	s_cbranch_scc1 .LBB0_65
	v_mov_b32_e32 v98, s62
	buffer_load_dword v98, v98, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s65, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s49, v99
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_59
	ds_write_b128 v117, v[146:149]
	ds_write_b128 v117, v[150:153] offset:6144
.LBB0_59:
	s_or_b64 exec, exec, s[42:43]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s49, v98
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_61
	s_mul_i32 s64, s63, 0xc00
	v_add_u32_sdwa v98, s64, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[146:149], v[98:99], off
	global_load_dwordx4 v[150:153], v[98:99], off offset:256
.LBB0_61:
	s_or_b64 exec, exec, s[42:43]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v115, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v115
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[120:123], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[162:163], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[164:165], v[98:113]
	v_add_u32_e32 v120, 0x1810, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[166:167], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[168:169], v[98:113]
	v_add_u32_e32 v120, 0x1820, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[170:171], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[172:173], v[98:113]
	v_add_u32_e32 v120, 0x1830, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[174:175], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[176:177], v[98:113]
	v_add_u32_e32 v120, 0x1840, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[178:179], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[180:181], v[98:113]
	v_add_u32_e32 v120, 0x1850, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[182:183], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[184:185], v[98:113]
	v_add_u32_e32 v120, 0x1860, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[186:187], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[188:189], v[98:113]
	v_add_u32_e32 v120, 0x1870, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[190:191], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[192:193], v[98:113]
	v_add_u32_e32 v120, 0x1880, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[194:195], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[196:197], v[98:113]
	v_add_u32_e32 v120, 0x1890, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[198:199], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[200:201], v[98:113]
	v_add_u32_e32 v120, 0x18a0, v115
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[202:203], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[204:205], v[98:113]
	v_add_u32_e32 v115, 0x18b0, v115
	v_lshrrev_b32_e32 v120, 3, v115
	v_and_b32_e32 v120, 56, v120
	v_xor_b32_e32 v115, v120, v115
	v_lshlrev_b32_e32 v115, 1, v115
	ds_read_b128 v[120:123], v115
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[206:207], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[208:209], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v115, v0
	;;#ASMEND
	v_mov_b32_e32 v122, v116
	v_lshrrev_b32_e32 v120, 2, v115
	v_and_b32_e32 v120, 8, v120
	v_lshrrev_b32_e32 v115, 1, v115
	v_and_or_b32 v115, v115, s53, v143
	v_add_u32_e32 v120, s61, v120
	v_add_u32_e32 v115, s58, v115
	v_subrev_u32_e32 v121, 23, v120
	v_cmp_lt_i32_e32 vcc, v121, v115
	v_mov_b32_e32 v123, v116
	v_mov_b32_e32 v124, v116
	v_cndmask_b32_e32 v99, v141, v99, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_subrev_u32_e32 v121, 21, v120
	v_mov_b32_e32 v125, v116
	v_cndmask_b32_e32 v98, v141, v98, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_subrev_u32_e32 v121, 20, v120
	v_pk_mul_f32 v[98:99], v[118:119], v[98:99]
	v_cndmask_b32_e32 v100, v141, v100, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_subrev_u32_e32 v121, 19, v120
	v_mov_b32_e32 v126, v116
	v_cndmask_b32_e32 v101, v141, v101, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_subrev_u32_e32 v121, 18, v120
	v_mov_b32_e32 v127, v116
	v_cndmask_b32_e32 v102, v141, v102, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_subrev_u32_e32 v121, 17, v120
	v_mov_b32_e32 v128, v116
	v_cndmask_b32_e32 v103, v141, v103, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_add_u32_e32 v121, -16, v120
	v_mov_b32_e32 v129, v116
	v_cndmask_b32_e32 v104, v141, v104, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_add_u32_e32 v121, -7, v120
	v_mov_b32_e32 v130, v116
	v_cndmask_b32_e32 v105, v141, v105, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_add_u32_e32 v121, -6, v120
	v_mov_b32_e32 v131, v116
	v_cndmask_b32_e32 v106, v141, v106, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_add_u32_e32 v121, -5, v120
	v_mov_b32_e32 v132, v116
	v_cndmask_b32_e32 v107, v141, v107, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_add_u32_e32 v121, -4, v120
	v_mov_b32_e32 v133, v116
	v_cndmask_b32_e32 v108, v141, v108, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_add_u32_e32 v121, -3, v120
	v_mov_b32_e32 v134, v116
	v_cndmask_b32_e32 v109, v141, v109, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_add_u32_e32 v121, -2, v120
	v_mov_b32_e32 v135, v116
	v_cndmask_b32_e32 v110, v141, v110, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_add_u32_e32 v121, -1, v120
	s_nop 0
	v_cndmask_b32_e32 v111, v141, v111, vcc
	v_cmp_le_i32_e32 vcc, v121, v115
	v_mov_b32_e32 v121, v116
	s_nop 0
	v_cndmask_b32_e32 v112, v141, v112, vcc
	v_cmp_le_i32_e32 vcc, v120, v115
	v_mov_b32_e32 v115, v114
	v_pk_mul_f32 v[110:111], v[114:115], v[110:111]
	v_cndmask_b32_e32 v113, v141, v113, vcc
	v_pk_mul_f32 v[112:113], v[114:115], v[112:113]
	v_pk_mul_f32 v[108:109], v[114:115], v[108:109]
	v_pk_mul_f32 v[106:107], v[114:115], v[106:107]
	v_pk_mul_f32 v[104:105], v[114:115], v[104:105]
	v_pk_mul_f32 v[102:103], v[114:115], v[102:103]
	v_pk_mul_f32 v[100:101], v[114:115], v[100:101]
	v_max_f32_e32 v115, v98, v99
	v_max3_f32 v115, v115, v100, v101
	v_max3_f32 v115, v115, v102, v103
	v_max3_f32 v115, v115, v104, v105
	v_max3_f32 v115, v115, v106, v107
	v_max3_f32 v115, v115, v108, v109
	v_max3_f32 v115, v115, v110, v111
	v_max3_f32 v115, v115, v112, v113
	ds_bpermute_b32 v120, v137, v115
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v120, v120, v120
	v_max_f32_e32 v144, v115, v120
	;;#ASMSTART
	v_add_f32 v115, v116, v139
	;;#ASMEND
	v_mov_b32_e32 v120, v116
	v_cmp_gt_f32_e32 vcc, v144, v115
	v_mov_b32_e32 v115, 1.0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_63
	;;#ASMSTART
	v_add_f32 v120, v144, v140
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v115, v116, v120
	v_exp_f32_e32 v115, v115
	v_mov_b32_e32 v116, v120
	v_mov_b32_e32 v121, v120
	v_mov_b32_e32 v122, v120
	v_mov_b32_e32 v123, v120
	v_mov_b32_e32 v124, v120
	v_mov_b32_e32 v125, v120
	v_mov_b32_e32 v126, v120
	v_mov_b32_e32 v127, v120
	v_mov_b32_e32 v128, v120
	v_mov_b32_e32 v129, v120
	v_mov_b32_e32 v130, v120
	v_mov_b32_e32 v131, v120
	v_mov_b32_e32 v132, v120
	v_mov_b32_e32 v133, v120
	v_mov_b32_e32 v134, v120
	v_mov_b32_e32 v135, v120
.LBB0_63:
	s_or_b64 exec, exec, s[42:43]
	v_pk_add_f32 v[98:99], v[98:99], v[120:121] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, 0, v98
	v_add_f32_e32 v120, v120, v99
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v100
	v_add_f32_e32 v120, v120, v101
	v_add_f32_e32 v120, v120, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v103
	v_add_f32_e32 v120, v120, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v105
	v_add_f32_e32 v120, v120, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v107
	v_add_f32_e32 v120, v120, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v109
	v_add_f32_e32 v120, v120, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v115
	v_add_f32_e32 v120, v120, v111
	v_add_f32_e32 v120, v120, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v120, v120, v113
	;;#ASMSTART
	v_fma_f32 v142, v142, v115, v120
	;;#ASMEND
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_46
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
	v_mul_f32 v66, v66, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v115
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v115
	;;#ASMEND
	s_branch .LBB0_46
.LBB0_65:
	s_mov_b32 s65, s63
	s_branch .LBB0_47
.LBB0_66:
	ds_bpermute_b32 v98, v137, v142
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v98, v142, v98
	;;#ASMEND
	s_nop 0
	v_div_scale_f32 v99, s[12:13], v98, v98, s46
	v_rcp_f32_e32 v100, v99
	v_div_scale_f32 v101, vcc, s46, v98, s46
	v_fma_f32 v102, -v99, v100, 1.0
	v_fmac_f32_e32 v100, v102, v100
	v_mul_f32_e32 v102, v101, v100
	v_fma_f32 v103, -v99, v102, v101
	v_fmac_f32_e32 v102, v103, v100
	v_fma_f32 v99, -v99, v102, v101
	v_div_fmas_f32 v99, v99, v100, v102
	v_div_fixup_f32 v98, v99, v98, s46
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
	v_perm_b32 v48, v3, v2, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v49, v5, v4, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v46, v7, v6, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v47, v9, v8, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v44, v11, v10, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v45, v13, v12, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v42, v15, v14, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v43, v17, v16, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v40, v19, v18, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v41, v21, v20, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v38, v23, v22, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v39, v25, v24, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v36, v27, v26, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v37, v29, v28, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v34, v31, v30, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v35, v33, v32, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v112, v113, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v110, v111, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v108, v109, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v106, v107, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v28, v104, v105, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v102, v103, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v100, v101, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v98, v99, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v51, v50, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v53, v52, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v55, v54, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v57, v56, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v59, v58, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v61, v60, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v63, v62, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v65, v64, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v67, v66, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v69, v68, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v71, v70, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v73, v72, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v75, v74, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v77, v76, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v79, v78, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v81, v80, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v83, v82, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v85, v84, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v87, v86, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v89, v88, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v91, v90, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v93, v92, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v95, v94, s50
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v97, v96, s50
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v50, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v50, 0x100, v50
	v_cmp_eq_u32_e32 vcc, 0, v50
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_68
	s_barrier
.LBB0_68:
	s_or_b64 exec, exec, s[12:13]
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
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_70
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
.LBB0_70:
	s_or_b64 exec, exec, s[12:13]
	v_and_b32_e32 v88, 0x1ff, v91
	s_add_u32 s16, s30, s0
	s_addc_u32 s0, s31, s1
	v_lshlrev_b32_e32 v92, 3, v88
	s_and_b32 s17, s0, 0xffff
	s_mov_b32 s19, s15
	v_and_b32_e32 v91, 56, v91
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_71:
	v_mul_u32_u24_sdwa v95, v94, s54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s55, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s23, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_71
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 1, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_74
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
.LBB0_74:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s12, s23, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_75:
	v_mul_u32_u24_sdwa v95, v94, s54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s55, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s12, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_75
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 2, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_78
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
.LBB0_78:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s12, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_79:
	v_mul_u32_u24_sdwa v95, v94, s54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s55, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_79
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 3, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_82
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
.LBB0_82:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s12, 0x30000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_83:
	v_mul_u32_u24_sdwa v95, v94, s54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s55, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_83
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 4, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_86
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
.LBB0_86:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s12, 0x48000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_87:
	v_mul_u32_u24_sdwa v95, v94, s54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s55, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_87
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 5, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_90
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
.LBB0_90:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s12, 0x60000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_91:
	v_mul_u32_u24_sdwa v95, v94, s54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s55, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_91
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 6, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_94
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
.LBB0_94:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s12, 0x78000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_95:
	v_mul_u32_u24_sdwa v95, v94, s54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s55, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_95
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 7, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_98
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
.LBB0_98:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s12, s12, 0x90000
	s_mov_b64 s[0:1], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_99:
	v_mul_u32_u24_sdwa v2, v88, s54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v3, v92, v91
	v_lshrrev_b32_e32 v2, 20, v2
	v_lshlrev_b32_e32 v3, 1, v3
	v_mul_lo_u16_e32 v5, 24, v2
	ds_read_b128 v[6:9], v3
	v_sub_u16_e32 v3, v88, v5
	v_lshlrev_b16_e32 v3, 3, v3
	v_add_u32_e32 v4, 0x200, v88
	v_cmp_lt_u32_e32 vcc, s55, v88
	v_mul_u32_u24_e32 v2, 0xc00, v2
	v_add_u32_e32 v3, s12, v3
	v_add_u32_e32 v92, 0x1000, v92
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v88, v4
	v_add_lshl_u32 v2, v3, v2, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v2, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_99
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	;;#ASMSTART
	s_mov_b32 s0, s56
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_104
	s_mov_b64 s[18:19], exec
	v_mbcnt_lo_u32_b32 v2, s18, 0
	v_mbcnt_hi_u32_b32 v2, s19, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_103
	s_bcnt1_i32_b64 s14, s[18:19]
	v_mov_b32_e32 v3, s14
	global_atomic_add v3, v138, v3, s[34:35] sc0
.LBB0_103:
	s_or_b64 exec, exec, s[16:17]
	s_lshl_b64 s[16:17], s[0:1], 2
	s_add_u32 s16, s34, s16
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s14, v3
	s_addc_u32 s17, s35, s17
	s_nop 0
	v_add_u32_e32 v2, s14, v2
	global_store_dword v138, v2, s[16:17]
	s_waitcnt vmcnt(0)
.LBB0_104:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s34, s0
	s_addc_u32 s1, s35, s1
	s_barrier
	global_load_dword v2, v138, s[0:1]
	s_sub_i32 s0, s33, s2
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s2, v2
	s_add_i32 s33, s0, s2
	s_cmp_ge_i32 s22, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s44
	s_cselect_b64 s[12:13], -1, 0
	s_or_b64 s[0:1], s[0:1], s[12:13]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_107
	s_branch .LBB0_12
.LBB0_105:
	s_mov_b32 s22, s13
.LBB0_106:
	s_sub_i32 s33, s33, s44
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s47, 0, s12
	s_cmp_ge_i32 s22, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s14
	s_cselect_b64 s[12:13], -1, 0
	s_or_b64 s[0:1], s[0:1], s[12:13]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s44, s14
	s_cbranch_vccnz .LBB0_11
.LBB0_107:
	s_add_i32 s12, s47, 1
	s_cmp_gt_i32 s12, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s12, 16
	s_cbranch_scc1 .LBB0_110
	s_add_i32 s13, s22, 1
	s_cmp_ge_i32 s13, s3
	s_mov_b32 s14, s44
	s_cbranch_scc1 .LBB0_105
	s_ashr_i32 s23, s22, 31
	s_lshl_b64 s[16:17], s[22:23], 2
	s_add_u32 s16, s20, s16
	s_addc_u32 s17, s21, s17
	global_load_dwordx2 v[2:3], v138, s[16:17] offset:4
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s14, v3
	v_readfirstlane_b32 s16, v2
	s_sub_i32 s14, s14, s16
	s_addk_i32 s14, 0xff
	s_ashr_i32 s16, s14, 31
	s_lshr_b32 s16, s16, 24
	s_add_i32 s16, s14, s16
	s_ashr_i32 s22, s16, 8
	s_and_b32 s16, s16, 0xffffff00
	s_cmp_lg_u32 s14, s16
	s_cselect_b64 s[16:17], -1, 0
	s_cmp_lt_i32 s14, 0
	s_cselect_b64 s[18:19], -1, 0
	s_and_b64 s[16:17], s[18:19], s[16:17]
	s_subb_u32 s14, s22, 0
	s_branch .LBB0_105
.LBB0_110:
	s_mov_b32 s14, s44
	s_branch .LBB0_106
.LBB0_111:
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
		.amdhsa_next_free_vgpr 218
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 220
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

	.set .Lattn_kernel_0.num_vgpr, 218
	.set .Lattn_kernel_0.num_agpr, 0
	.set .Lattn_kernel_0.numbered_sgpr, 68
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
    .sgpr_count:     74
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     218
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
