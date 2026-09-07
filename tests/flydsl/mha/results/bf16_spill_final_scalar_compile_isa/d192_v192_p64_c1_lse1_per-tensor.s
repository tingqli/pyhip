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
	s_mov_b32 s28, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s22, s16
.LBB0_3:
	s_sub_i32 s33, s33, s14
	s_and_b64 s[12:13], s[12:13], exec
	s_cselect_b32 s28, 0, s15
	s_cmp_ge_i32 s22, s3
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s33, s52
	s_cselect_b64 s[14:15], -1, 0
	s_or_b64 s[12:13], s[12:13], s[14:15]
	s_and_b64 vcc, exec, s[12:13]
	s_mov_b32 s14, s52
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s15, s28, 1
	s_cmp_gt_i32 s15, 15
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lt_i32 s15, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s16, s22, 1
	s_cmp_ge_i32 s16, s3
	s_mov_b32 s52, s14
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
	s_subb_u32 s52, s24, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s52, s14
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s52, s14
	s_mov_b32 s28, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s53, s[6:7], 0x0
	s_load_dword s54, s[8:9], 0x0
	s_cmp_ge_i32 s22, s3
	s_cbranch_scc1 .LBB0_129
	v_lshrrev_b32_e32 v2, 6, v0
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_load_dwordx2 s[24:25], s[0:1], 0x20
	s_load_dwordx2 s[26:27], s[0:1], 0x50
	s_load_dwordx2 s[30:31], s[0:1], 0x60
	s_load_dwordx2 s[34:35], s[0:1], 0x98
	s_load_dwordx2 s[36:37], s[0:1], 0xa8
	s_load_dwordx2 s[38:39], s[0:1], 0xb8
	v_lshrrev_b32_e32 v3, 2, v0
	v_mul_u32_u24_e32 v2, 0x18000, v2
	v_and_b32_e32 v1, 31, v0
	v_and_or_b32 v2, v3, 8, v2
	s_movk_i32 s0, 0xc00
	v_mad_u32_u24 v115, v1, s0, v2
	v_mul_u32_u24_e32 v2, 0xaab, v0
	v_mov_b32_e32 v3, 24
	v_mul_u32_u24_e32 v4, 0x2ab, v0
	v_mov_b32_e32 v5, 2
	v_mov_b32_e32 v6, 5
	v_mul_lo_u16_sdwa v3, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshlrev_b16_sdwa v2, v5, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_u32_u24_e32 v5, 0x156, v0
	v_lshlrev_b16_sdwa v4, v6, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mov_b32_e32 v6, 4
	v_sub_u16_sdwa v3, v0, v3 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_sdwa v5, v6, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v4, 32, v4
	v_or_b32_e32 v3, v3, v5
	v_and_b32_e32 v2, 12, v2
	v_add_u16_e32 v3, v3, v4
	v_or_b32_e32 v136, v3, v2
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v138, v2, v3
	v_and_b32_e32 v2, 63, v0
	s_movk_i32 s0, 0x80
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_u32_sdwa v137, v136, s0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v139, 0x80, v2
	v_mov_b32_e32 v140, 0
	s_mov_b32 s15, 0x27000
	s_movk_i32 s55, 0x180
	v_mov_b32_e32 v141, 0x40e00000
	v_mov_b32_e32 v142, 1.0
	s_mov_b32 s56, 0x7060302
	s_movk_i32 s57, 0x1000
	s_movk_i32 s58, 0x2000
	s_movk_i32 s59, 0xe0
	s_mov_b32 s60, 0x800000
	s_mov_b32 s61, 0xaaab
	s_movk_i32 s62, 0xff
	s_mov_b32 s63, s2
	v_mov_b32_e32 v143, 0xff800000
	v_mov_b32_e32 v144, 0x42000000
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s52, s14
.LBB0_12:
	s_mov_b32 s2, s12
	s_cmp_ge_i32 s22, s3
	s_cbranch_scc1 .LBB0_129
.LBB0_13:
	s_ashr_i32 s23, s22, 31
	s_lshl_b32 s50, s33, 8
	s_lshl_b64 s[0:1], s[22:23], 2
	s_add_u32 s12, s20, s0
	s_addc_u32 s13, s21, s1
	global_load_dwordx2 v[4:5], v140, s[12:13]
	global_load_dword v2, v140, s[4:5]
	s_mul_i32 s23, s28, 0xc0
	s_mov_b32 s19, s15
	v_add_lshl_u32 v3, s23, v115, 1
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s46, v4
	s_add_i32 s42, s46, s50
	v_readfirstlane_b32 s47, v5
	s_add_i32 s13, s42, 0x100
	s_min_i32 s13, s13, s47
	s_sub_i32 s43, s13, s42
	s_waitcnt lgkmcnt(0)
	s_add_u32 s16, s26, s0
	s_addc_u32 s17, s27, s1
	s_mul_i32 s12, s42, 0xc00
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[40:41], s[12:13], 1
	global_load_dwordx2 v[4:5], v140, s[16:17]
	global_load_dword v6, v140, s[0:1]
	s_add_u32 s16, s6, s40
	s_addc_u32 s0, s7, s41
	s_mul_i32 s18, s43, 0x1800
	s_and_b32 s17, s0, 0xffff
	buffer_load_dwordx4 v[164:167], v3, s[16:19], 0 offen
	buffer_load_dwordx4 v[168:171], v3, s[16:19], 0 offen offset:32
	buffer_load_dwordx4 v[172:175], v3, s[16:19], 0 offen offset:64
	buffer_load_dwordx4 v[176:179], v3, s[16:19], 0 offen offset:96
	buffer_load_dwordx4 v[180:183], v3, s[16:19], 0 offen offset:128
	buffer_load_dwordx4 v[184:187], v3, s[16:19], 0 offen offset:160
	buffer_load_dwordx4 v[188:191], v3, s[16:19], 0 offen offset:192
	buffer_load_dwordx4 v[192:195], v3, s[16:19], 0 offen offset:224
	buffer_load_dwordx4 v[196:199], v3, s[16:19], 0 offen offset:256
	buffer_load_dwordx4 v[200:203], v3, s[16:19], 0 offen offset:288
	buffer_load_dwordx4 v[204:207], v3, s[16:19], 0 offen offset:320
	buffer_load_dwordx4 v[208:211], v3, s[16:19], 0 offen offset:352
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_waitcnt vmcnt(13)
	v_readfirstlane_b32 s12, v4
	v_and_b32_e32 v3, 0x100, v3
	v_readfirstlane_b32 s13, v5
	s_waitcnt vmcnt(12)
	v_readfirstlane_b32 s19, v6
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_15
	s_barrier
.LBB0_15:
	s_or_b64 exec, exec, s[0:1]
	s_ashr_i32 s29, s28, 31
	s_lshr_b32 s0, s29, 28
	s_add_i32 s0, s28, s0
	s_sub_i32 s51, s13, s12
	s_ashr_i32 s13, s0, 4
	s_and_b32 s0, s0, -16
	s_cmp_lg_u32 s28, s0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s28, 0
	s_cselect_b64 s[16:17], -1, 0
	s_and_b64 s[0:1], s[16:17], s[0:1]
	s_subb_u32 s0, s13, 0
	s_mulk_i32 s0, 0x3000
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[16:17], s[0:1], 1
	s_add_u32 s16, s8, s16
	s_addc_u32 s17, s9, s17
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s12, s30, s12
	s_addc_u32 s1, s31, s13
	s_lshl_b32 s14, s51, 2
	s_and_b32 s13, s1, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[12:15], 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s71, v4
	v_readfirstlane_b32 s64, v5
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
	buffer_load_dword v3, off, s[12:15], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s1, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_mul_i32 s48, s71, 0x1800
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s55, v3
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_17
	v_add_u32_sdwa v4, s48, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[16:17]
	global_load_dwordx4 v[148:151], v[4:5], off
	global_load_dwordx4 v[152:155], v[4:5], off offset:256
.LBB0_17:
	s_or_b64 exec, exec, s[44:45]
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
	v_readfirstlane_b32 s65, v3
	v_and_b32_e32 v4, 0x180, v4
	v_cmp_ne_u32_e32 vcc, s55, v4
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_19
	v_add_u32_e32 v4, s48, v137
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[16:17]
	global_load_dwordx4 v[156:159], v[4:5], off
	global_load_dwordx4 v[160:163], v[4:5], off offset:256
.LBB0_19:
	s_or_b64 exec, exec, s[44:45]
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
	v_cmp_ne_u32_e32 vcc, s55, v3
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_21
	ds_write_b128 v138, v[148:151] offset:12288
	ds_write_b128 v138, v[152:155] offset:18432
.LBB0_21:
	s_or_b64 exec, exec, s[44:45]
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s55, v3
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_23
	s_mul_i32 s48, s64, 0x1800
	v_add_u32_sdwa v4, s48, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[16:17]
	global_load_dwordx4 v[148:151], v[4:5], off
	global_load_dwordx4 v[152:155], v[4:5], off offset:256
.LBB0_23:
	s_or_b64 exec, exec, s[44:45]
	v_mul_f32_e32 v2, s53, v2
	s_sub_i32 s44, s46, s47
	s_lshl_b32 s45, s51, 6
	v_mul_f32_e32 v116, 0x3dd53b95, v2
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_add_i32 s44, s44, s45
	s_add_i32 s44, s44, s19
	s_sub_i32 s19, s44, 64
	s_add_i32 s66, s19, s50
	s_add_i32 s46, s66, 1
	s_ashr_i32 s44, s46, 31
	s_lshr_b32 s44, s44, 26
	s_add_i32 s44, s46, s44
	s_ashr_i32 s48, s44, 6
	s_andn2_b32 s44, s44, 63
	s_cmp_lg_u32 s46, s44
	s_cselect_b64 s[44:45], -1, 0
	s_cmp_lt_i32 s46, 0
	s_cselect_b64 s[46:47], -1, 0
	s_and_b64 s[44:45], s[46:47], s[44:45]
	s_subb_u32 s46, s48, 0
	s_lshr_b32 s44, s46, 31
	s_add_i32 s44, s46, s44
	s_ashr_i32 s48, s44, 1
	s_and_b32 s44, s44, -2
	s_cmp_lg_u32 s46, s44
	s_cselect_b64 s[44:45], -1, 0
	s_cmp_lt_i32 s46, 0
	s_cselect_b64 s[46:47], -1, 0
	s_and_b64 s[44:45], s[46:47], s[44:45]
	s_subb_u32 s67, s48, 0
	s_lshl_b32 s44, s67, 1
	s_ashr_i32 s45, s44, 31
	s_cmp_lt_i32 s67, 1
	s_cbranch_scc1 .LBB0_51
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v118, v116
	v_mov_b32_e32 v119, v116
	s_mov_b64 s[46:47], 0
	s_mov_b32 s68, 20
	v_mov_b32_e32 v114, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v145
	v_mov_b32_e32 v4, v145
	v_mov_b32_e32 v5, v145
	v_mov_b32_e32 v6, v145
	v_mov_b32_e32 v7, v145
	v_mov_b32_e32 v8, v145
	v_mov_b32_e32 v9, v145
	v_mov_b32_e32 v10, v145
	v_mov_b32_e32 v11, v145
	v_mov_b32_e32 v12, v145
	v_mov_b32_e32 v13, v145
	v_mov_b32_e32 v14, v145
	v_mov_b32_e32 v15, v145
	v_mov_b32_e32 v16, v145
	v_mov_b32_e32 v17, v145
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v145
	v_mov_b32_e32 v20, v145
	v_mov_b32_e32 v21, v145
	v_mov_b32_e32 v22, v145
	v_mov_b32_e32 v23, v145
	v_mov_b32_e32 v24, v145
	v_mov_b32_e32 v25, v145
	v_mov_b32_e32 v26, v145
	v_mov_b32_e32 v27, v145
	v_mov_b32_e32 v28, v145
	v_mov_b32_e32 v29, v145
	v_mov_b32_e32 v30, v145
	v_mov_b32_e32 v31, v145
	v_mov_b32_e32 v32, v145
	v_mov_b32_e32 v33, v145
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v145
	v_mov_b32_e32 v36, v145
	v_mov_b32_e32 v37, v145
	v_mov_b32_e32 v38, v145
	v_mov_b32_e32 v39, v145
	v_mov_b32_e32 v40, v145
	v_mov_b32_e32 v41, v145
	v_mov_b32_e32 v42, v145
	v_mov_b32_e32 v43, v145
	v_mov_b32_e32 v44, v145
	v_mov_b32_e32 v45, v145
	v_mov_b32_e32 v46, v145
	v_mov_b32_e32 v47, v145
	v_mov_b32_e32 v48, v145
	v_mov_b32_e32 v49, v145
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v145
	v_mov_b32_e32 v52, v145
	v_mov_b32_e32 v53, v145
	v_mov_b32_e32 v54, v145
	v_mov_b32_e32 v55, v145
	v_mov_b32_e32 v56, v145
	v_mov_b32_e32 v57, v145
	v_mov_b32_e32 v58, v145
	v_mov_b32_e32 v59, v145
	v_mov_b32_e32 v60, v145
	v_mov_b32_e32 v61, v145
	v_mov_b32_e32 v62, v145
	v_mov_b32_e32 v63, v145
	v_mov_b32_e32 v64, v145
	v_mov_b32_e32 v65, v145
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, v145
	v_mov_b32_e32 v68, v145
	v_mov_b32_e32 v69, v145
	v_mov_b32_e32 v70, v145
	v_mov_b32_e32 v71, v145
	v_mov_b32_e32 v72, v145
	v_mov_b32_e32 v73, v145
	v_mov_b32_e32 v74, v145
	v_mov_b32_e32 v75, v145
	v_mov_b32_e32 v76, v145
	v_mov_b32_e32 v77, v145
	v_mov_b32_e32 v78, v145
	v_mov_b32_e32 v79, v145
	v_mov_b32_e32 v80, v145
	v_mov_b32_e32 v81, v145
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, v145
	v_mov_b32_e32 v84, v145
	v_mov_b32_e32 v85, v145
	v_mov_b32_e32 v86, v145
	v_mov_b32_e32 v87, v145
	v_mov_b32_e32 v88, v145
	v_mov_b32_e32 v89, v145
	v_mov_b32_e32 v90, v145
	v_mov_b32_e32 v91, v145
	v_mov_b32_e32 v92, v145
	v_mov_b32_e32 v93, v145
	v_mov_b32_e32 v94, v145
	v_mov_b32_e32 v95, v145
	v_mov_b32_e32 v96, v145
	v_mov_b32_e32 v97, v145
	s_branch .LBB0_26
.LBB0_25:
	s_or_b64 exec, exec, s[48:49]
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
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_add_u32_e32 v105, 0x8000, v105
	v_add_f32_e32 v120, v120, v111
	v_add_f32_e32 v120, v120, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v120, v120, v113
	;;#ASMSTART
	v_fma_f32 v145, v145, v117, v120
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
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s56
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s64, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s64
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[120:123], v[106:107], off offset:512
	global_load_dwordx4 v[124:127], v[106:107], off offset:1024
	global_load_dwordx4 v[128:131], v[106:107], off offset:1536
	global_load_dwordx4 v[132:135], v[106:107], off offset:2048
	global_load_dwordx4 v[212:215], v[106:107], off offset:2560
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
	v_mfma_f32_32x32x8_bf16 v[82:97], v[212:213], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s57, v106
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
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[214:215], v[102:103], v[82:97]
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
	s_add_u32 s46, s46, 2
	s_addc_u32 s47, s47, 0
	v_mov_b64_e32 v[98:99], s[44:45]
	v_cmp_lt_i64_e32 vcc, s[46:47], v[98:99]
	s_add_i32 s68, s68, 8
	s_mov_b32 s64, s70
	s_mov_b32 s71, s69
	s_cbranch_vccz .LBB0_50
.LBB0_26:
	s_add_i32 s48, s68, -4
	v_mov_b32_e32 v98, s48
	buffer_load_dword v98, v98, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_mov_b32 s69, s1
	v_and_b32_e32 v99, 0x180, v99
	s_mov_b32 s70, s65
	v_cmp_ne_u32_e32 vcc, s55, v99
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s1, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_28
	ds_write_b128 v138, v[156:159]
	ds_write_b128 v138, v[160:163] offset:6144
.LBB0_28:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_30
	s_mul_i32 s65, s64, 0x1800
	v_add_u32_e32 v98, s65, v137
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[156:159], v[98:99], off
	global_load_dwordx4 v[160:163], v[98:99], off offset:256
.LBB0_30:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v117, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v117
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[120:123], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[166:167], v[98:113]
	v_add_u32_e32 v120, 0x1810, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[170:171], v[98:113]
	v_add_u32_e32 v120, 0x1820, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[174:175], v[98:113]
	v_add_u32_e32 v120, 0x1830, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[178:179], v[98:113]
	v_add_u32_e32 v120, 0x1840, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[182:183], v[98:113]
	v_add_u32_e32 v120, 0x1850, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[186:187], v[98:113]
	v_add_u32_e32 v120, 0x1860, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[190:191], v[98:113]
	v_add_u32_e32 v120, 0x1870, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[194:195], v[98:113]
	v_add_u32_e32 v120, 0x1880, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[198:199], v[98:113]
	v_add_u32_e32 v120, 0x1890, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[202:203], v[98:113]
	v_add_u32_e32 v120, 0x18a0, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[206:207], v[98:113]
	v_add_u32_e32 v117, 0x18b0, v117
	v_lshrrev_b32_e32 v120, 3, v117
	v_and_b32_e32 v120, 56, v120
	v_xor_b32_e32 v117, v120, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[120:123], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v117, v116
	s_nop 6
	v_pk_mul_f32 v[98:99], v[118:119], v[98:99]
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
	ds_bpermute_b32 v120, v139, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v120, v120, v120
	v_max_f32_e32 v120, v117, v120
	;;#ASMSTART
	v_add_f32 v117, v114, v141
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v120, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v120, v120, v142
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v120
	v_exp_f32_e32 v117, v114
	v_mov_b32_e32 v114, v120
.LBB0_32:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[104:105], v[104:105], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v100
	v_add_f32_e32 v120, v120, v101
	v_add_f32_e32 v120, v120, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v103
	v_add_f32_e32 v120, v120, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v105
	v_add_f32_e32 v120, v120, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v107
	v_add_f32_e32 v120, v120, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v109
	v_add_f32_e32 v120, v120, v110
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
	v_add_f32_e32 v120, v120, v111
	v_add_f32_e32 v120, v120, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v120, v120, v113
	;;#ASMSTART
	v_fma_f32 v145, v145, v117, v120
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
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s56
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s65, s71, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s65, s65, s0
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s65
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[120:123], v[106:107], off offset:512
	global_load_dwordx4 v[124:127], v[106:107], off offset:1024
	global_load_dwordx4 v[128:131], v[106:107], off offset:1536
	global_load_dwordx4 v[132:135], v[106:107], off offset:2048
	global_load_dwordx4 v[212:215], v[106:107], off offset:2560
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
	v_mfma_f32_32x32x8_bf16 v[82:97], v[212:213], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s57, v106
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
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[214:215], v[102:103], v[82:97]
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
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_34
	ds_write_b128 v138, v[148:151] offset:12288
	ds_write_b128 v138, v[152:155] offset:18432
.LBB0_34:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mul_i32 s71, s69, 0x1800
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_36
	v_add_u32_sdwa v98, s71, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[148:151], v[98:99], off
	global_load_dwordx4 v[152:155], v[98:99], off offset:256
.LBB0_36:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v117, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v120, 56, v98
	v_xor_b32_e32 v98, v120, v117
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[122:125], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[166:167], v[98:113]
	v_or_b32_e32 v121, 16, v117
	v_xor_b32_e32 v121, v121, v120
	v_lshlrev_b32_e32 v121, 1, v121
	ds_read_b128 v[122:125], v121
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[170:171], v[98:113]
	v_or_b32_e32 v121, 32, v117
	v_xor_b32_e32 v121, v121, v120
	v_lshlrev_b32_e32 v121, 1, v121
	ds_read_b128 v[122:125], v121
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[174:175], v[98:113]
	v_or_b32_e32 v121, 48, v117
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[178:179], v[98:113]
	v_add_u32_e32 v120, 64, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[182:183], v[98:113]
	v_add_u32_e32 v120, 0x50, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[186:187], v[98:113]
	v_add_u32_e32 v120, 0x60, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[190:191], v[98:113]
	v_add_u32_e32 v120, 0x70, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[194:195], v[98:113]
	v_add_u32_e32 v120, 0x80, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[198:199], v[98:113]
	v_add_u32_e32 v120, 0x90, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[202:203], v[98:113]
	v_add_u32_e32 v120, 0xa0, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[206:207], v[98:113]
	v_add_u32_e32 v117, 0xb0, v117
	v_lshrrev_b32_e32 v120, 3, v117
	v_and_b32_e32 v120, 56, v120
	v_xor_b32_e32 v117, v120, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[120:123], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v117, v116
	s_nop 6
	v_pk_mul_f32 v[98:99], v[118:119], v[98:99]
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
	ds_bpermute_b32 v120, v139, v117
	v_mov_b32_e32 v121, v114
	v_mov_b32_e32 v122, v114
	v_mov_b32_e32 v123, v114
	v_mov_b32_e32 v124, v114
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v120, v120, v120
	v_max_f32_e32 v146, v117, v120
	;;#ASMSTART
	v_add_f32 v117, v114, v141
	;;#ASMEND
	v_mov_b32_e32 v120, v114
	v_cmp_gt_f32_e32 vcc, v146, v117
	v_mov_b32_e32 v117, 1.0
	v_mov_b32_e32 v125, v114
	v_mov_b32_e32 v126, v114
	v_mov_b32_e32 v127, v114
	v_mov_b32_e32 v128, v114
	v_mov_b32_e32 v129, v114
	v_mov_b32_e32 v130, v114
	v_mov_b32_e32 v131, v114
	v_mov_b32_e32 v132, v114
	v_mov_b32_e32 v133, v114
	v_mov_b32_e32 v134, v114
	v_mov_b32_e32 v135, v114
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_38
	;;#ASMSTART
	v_add_f32 v120, v146, v142
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v120
	v_exp_f32_e32 v117, v114
	v_mov_b32_e32 v114, v120
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
.LBB0_38:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[120:121] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v146, 0, v98
	v_add_f32_e32 v146, v146, v99
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
	v_add_f32_e32 v146, v146, v100
	v_add_f32_e32 v146, v146, v101
	v_add_f32_e32 v146, v146, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v146, v146, v103
	v_add_f32_e32 v146, v146, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v146, v146, v105
	v_add_f32_e32 v146, v146, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v146, v146, v107
	v_add_f32_e32 v146, v146, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v146, v146, v109
	v_add_f32_e32 v146, v146, v110
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
	v_add_f32_e32 v146, v146, v111
	v_add_f32_e32 v146, v146, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v146, v146, v113
	;;#ASMSTART
	v_fma_f32 v145, v145, v117, v146
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
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s56
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s65, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s65
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[212:215], v[106:107], off offset:512
	global_load_dwordx4 v[216:219], v[106:107], off offset:1024
	global_load_dwordx4 v[220:223], v[106:107], off offset:1536
	global_load_dwordx4 v[224:227], v[106:107], off offset:2048
	global_load_dwordx4 v[228:231], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[212:213], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[228:229], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s57, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[214:215], v[102:103], v[18:33]
	global_load_dwordx4 v[212:215], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[102:103], v[34:49]
	global_load_dwordx4 v[216:219], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[102:103], v[50:65]
	global_load_dwordx4 v[220:223], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[230:231], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[212:213], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[214:215], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	v_mov_b32_e32 v98, s68
	buffer_load_dword v98, v98, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s65, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s55, v99
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_40
	ds_write_b128 v138, v[156:159]
	ds_write_b128 v138, v[160:163] offset:6144
.LBB0_40:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_42
	v_add_u32_e32 v98, s71, v137
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[156:159], v[98:99], off
	global_load_dwordx4 v[160:163], v[98:99], off offset:256
.LBB0_42:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v117, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v117
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[212:215], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[166:167], v[98:113]
	v_add_u32_e32 v146, 0x1810, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[170:171], v[98:113]
	v_add_u32_e32 v146, 0x1820, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[174:175], v[98:113]
	v_add_u32_e32 v146, 0x1830, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[178:179], v[98:113]
	v_add_u32_e32 v146, 0x1840, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[182:183], v[98:113]
	v_add_u32_e32 v146, 0x1850, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[186:187], v[98:113]
	v_add_u32_e32 v146, 0x1860, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[190:191], v[98:113]
	v_add_u32_e32 v146, 0x1870, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[194:195], v[98:113]
	v_add_u32_e32 v146, 0x1880, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[198:199], v[98:113]
	v_add_u32_e32 v146, 0x1890, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[202:203], v[98:113]
	v_add_u32_e32 v146, 0x18a0, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[206:207], v[98:113]
	v_add_u32_e32 v117, 0x18b0, v117
	v_lshrrev_b32_e32 v146, 3, v117
	v_and_b32_e32 v146, 56, v146
	v_xor_b32_e32 v117, v146, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[212:215], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v117, v116
	s_nop 6
	v_pk_mul_f32 v[98:99], v[118:119], v[98:99]
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
	ds_bpermute_b32 v146, v139, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v146, v146, v146
	v_max_f32_e32 v146, v117, v146
	;;#ASMSTART
	v_add_f32 v117, v114, v141
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v146, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_44
	;;#ASMSTART
	v_add_f32 v120, v146, v142
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v120
	v_exp_f32_e32 v117, v114
	v_mov_b32_e32 v114, v120
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
.LBB0_44:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[120:121] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v146, 0, v98
	v_add_f32_e32 v146, v146, v99
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
	v_add_f32_e32 v146, v146, v100
	v_add_f32_e32 v146, v146, v101
	v_add_f32_e32 v146, v146, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v146, v146, v103
	v_add_f32_e32 v146, v146, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v146, v146, v105
	v_add_f32_e32 v146, v146, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v146, v146, v107
	v_add_f32_e32 v146, v146, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v146, v146, v109
	v_add_f32_e32 v146, v146, v110
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
	v_add_f32_e32 v146, v146, v111
	v_add_f32_e32 v146, v146, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v146, v146, v113
	;;#ASMSTART
	v_fma_f32 v145, v145, v117, v146
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
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s56
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mulk_i32 s64, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s64, s64, s0
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s64
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[212:215], v[106:107], off offset:512
	global_load_dwordx4 v[216:219], v[106:107], off offset:1024
	global_load_dwordx4 v[220:223], v[106:107], off offset:1536
	global_load_dwordx4 v[224:227], v[106:107], off offset:2048
	global_load_dwordx4 v[228:231], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[212:213], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[228:229], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s57, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[214:215], v[102:103], v[18:33]
	global_load_dwordx4 v[212:215], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[102:103], v[34:49]
	global_load_dwordx4 v[216:219], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[102:103], v[50:65]
	global_load_dwordx4 v[220:223], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[230:231], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[212:213], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[214:215], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_46
	ds_write_b128 v138, v[148:151] offset:12288
	ds_write_b128 v138, v[152:155] offset:18432
.LBB0_46:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_48
	s_mul_i32 s71, s70, 0x1800
	v_add_u32_sdwa v98, s71, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[148:151], v[98:99], off
	global_load_dwordx4 v[152:155], v[98:99], off offset:256
.LBB0_48:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v117, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v146, 56, v98
	v_xor_b32_e32 v98, v146, v117
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[212:215], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[166:167], v[98:113]
	v_or_b32_e32 v147, 16, v117
	v_xor_b32_e32 v147, v147, v146
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[170:171], v[98:113]
	v_or_b32_e32 v147, 32, v117
	v_xor_b32_e32 v147, v147, v146
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[174:175], v[98:113]
	v_or_b32_e32 v147, 48, v117
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[178:179], v[98:113]
	v_add_u32_e32 v146, 64, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[182:183], v[98:113]
	v_add_u32_e32 v146, 0x50, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[186:187], v[98:113]
	v_add_u32_e32 v146, 0x60, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[190:191], v[98:113]
	v_add_u32_e32 v146, 0x70, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[194:195], v[98:113]
	v_add_u32_e32 v146, 0x80, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[198:199], v[98:113]
	v_add_u32_e32 v146, 0x90, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[202:203], v[98:113]
	v_add_u32_e32 v146, 0xa0, v117
	v_lshrrev_b32_e32 v147, 3, v146
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v146, v147, v146
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[212:215], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[206:207], v[98:113]
	v_add_u32_e32 v117, 0xb0, v117
	v_lshrrev_b32_e32 v146, 3, v117
	v_and_b32_e32 v146, 56, v146
	v_xor_b32_e32 v117, v146, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[212:215], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	v_mov_b32_e32 v117, v116
	s_nop 6
	v_pk_mul_f32 v[98:99], v[118:119], v[98:99]
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
	ds_bpermute_b32 v146, v139, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v146, v146, v146
	v_max_f32_e32 v146, v117, v146
	;;#ASMSTART
	v_add_f32 v117, v114, v141
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v146, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_25
	;;#ASMSTART
	v_add_f32 v120, v146, v142
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v120
	v_exp_f32_e32 v117, v114
	v_mov_b32_e32 v114, v120
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
	s_branch .LBB0_25
.LBB0_50:
	s_mov_b32 s64, s70
	s_mov_b32 s71, s69
	s_branch .LBB0_52
.LBB0_51:
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v114, 0xff800000
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
	v_mov_b32_e32 v145, v2
.LBB0_52:
	s_add_i32 s46, s43, s66
	s_add_i32 s48, s46, 63
	s_ashr_i32 s46, s48, 31
	s_lshr_b32 s46, s46, 26
	s_add_i32 s46, s48, s46
	s_ashr_i32 s66, s46, 6
	s_andn2_b32 s46, s46, 63
	s_cmp_lg_u32 s48, s46
	s_cselect_b64 s[46:47], -1, 0
	s_cmp_lt_i32 s48, 0
	s_cselect_b64 s[48:49], -1, 0
	s_and_b64 s[46:47], s[48:49], s[46:47]
	s_subb_u32 s46, s66, 0
	s_min_i32 s46, s46, s51
	s_cmp_ge_i32 s44, s46
	s_cbranch_scc1 .LBB0_82
	s_lshl_b32 s48, s67, 7
	s_ashr_i32 s47, s46, 31
	v_mov_b32_e32 v118, v116
	v_mov_b32_e32 v119, v116
	v_or_b32_e32 v146, s50, v1
	s_or_b32 s66, s48, 0x77
	s_lshl3_add_u32 s67, s67, 20
	s_branch .LBB0_56
.LBB0_54:
	s_or_b64 exec, exec, s[50:51]
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
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_add_u32_e32 v105, 0x8000, v105
	v_add_f32_e32 v120, v120, v111
	v_add_f32_e32 v120, v120, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v120, v120, v113
	;;#ASMSTART
	v_fma_f32 v145, v145, v117, v120
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
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s56
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s70, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s70
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[120:123], v[106:107], off offset:512
	global_load_dwordx4 v[124:127], v[106:107], off offset:1024
	global_load_dwordx4 v[128:131], v[106:107], off offset:1536
	global_load_dwordx4 v[132:135], v[106:107], off offset:2048
	global_load_dwordx4 v[212:215], v[106:107], off offset:2560
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
	v_mfma_f32_32x32x8_bf16 v[82:97], v[212:213], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s57, v106
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
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[214:215], v[102:103], v[82:97]
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
.LBB0_55:
	s_and_b64 s[48:49], s[48:49], exec
	s_cselect_b32 s71, s1, s64
	s_cselect_b32 s64, s65, s1
	s_cselect_b32 s1, s68, s65
	s_add_u32 s44, s44, 2
	s_addc_u32 s45, s45, 0
	v_mov_b64_e32 v[98:99], s[46:47]
	v_cmp_lt_i64_e32 vcc, s[44:45], v[98:99]
	s_addk_i32 s66, 0x80
	s_add_i32 s67, s67, 8
	s_mov_b32 s65, s69
	s_cbranch_vccz .LBB0_82
.LBB0_56:
	s_add_i32 s48, s67, -4
	v_mov_b32_e32 v98, s48
	buffer_load_dword v98, v98, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s68, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s55, v99
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_58
	ds_write_b128 v138, v[156:159]
	ds_write_b128 v138, v[160:163] offset:6144
.LBB0_58:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_60
	s_mul_i32 s50, s64, 0x1800
	v_add_u32_e32 v98, s50, v137
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[156:159], v[98:99], off
	global_load_dwordx4 v[160:163], v[98:99], off offset:256
.LBB0_60:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v117, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v117
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[120:123], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[166:167], v[98:113]
	v_add_u32_e32 v120, 0x1810, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[170:171], v[98:113]
	v_add_u32_e32 v120, 0x1820, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[174:175], v[98:113]
	v_add_u32_e32 v120, 0x1830, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[178:179], v[98:113]
	v_add_u32_e32 v120, 0x1840, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[182:183], v[98:113]
	v_add_u32_e32 v120, 0x1850, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[186:187], v[98:113]
	v_add_u32_e32 v120, 0x1860, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[190:191], v[98:113]
	v_add_u32_e32 v120, 0x1870, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[194:195], v[98:113]
	v_add_u32_e32 v120, 0x1880, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[198:199], v[98:113]
	v_add_u32_e32 v120, 0x1890, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[202:203], v[98:113]
	v_add_u32_e32 v120, 0x18a0, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[206:207], v[98:113]
	v_add_u32_e32 v117, 0x18b0, v117
	v_lshrrev_b32_e32 v120, 3, v117
	v_and_b32_e32 v120, 56, v120
	v_xor_b32_e32 v117, v120, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[120:123], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v120, 2, v117
	v_and_b32_e32 v120, 8, v120
	v_lshrrev_b32_e32 v117, 1, v117
	v_and_or_b32 v117, v117, s59, v146
	v_add_u32_e32 v122, s66, v120
	v_add_u32_e32 v117, s19, v117
	v_add_u32_e32 v120, 0xffffff89, v122
	v_cmp_lt_i32_e32 vcc, v120, v117
	s_nop 1
	v_cndmask_b32_e32 v121, v143, v99, vcc
	v_cmp_le_i32_e32 vcc, v120, v117
	v_add_u32_e32 v99, 0xffffff8c, v122
	s_nop 0
	v_cndmask_b32_e32 v120, v143, v98, vcc
	v_add_u32_e32 v98, 0xffffff8b, v122
	v_cmp_le_i32_e32 vcc, v98, v117
	s_nop 1
	v_cndmask_b32_e32 v98, v143, v100, vcc
	v_cmp_le_i32_e32 vcc, v99, v117
	v_add_u32_e32 v100, 0xffffff8d, v122
	s_nop 0
	v_cndmask_b32_e32 v99, v143, v101, vcc
	v_cmp_le_i32_e32 vcc, v100, v117
	v_add_u32_e32 v101, 0xffffff8e, v122
	s_nop 0
	v_cndmask_b32_e32 v100, v143, v102, vcc
	v_cmp_le_i32_e32 vcc, v101, v117
	v_add_u32_e32 v102, 0xffffff8f, v122
	s_nop 0
	v_cndmask_b32_e32 v101, v143, v103, vcc
	v_cmp_le_i32_e32 vcc, v102, v117
	v_add_u32_e32 v103, 0xffffff90, v122
	s_nop 0
	v_cndmask_b32_e32 v102, v143, v104, vcc
	v_cmp_le_i32_e32 vcc, v103, v117
	v_add_u32_e32 v104, 0xffffff99, v122
	s_nop 0
	v_cndmask_b32_e32 v103, v143, v105, vcc
	v_cmp_le_i32_e32 vcc, v104, v117
	v_add_u32_e32 v105, 0xffffff9a, v122
	s_nop 0
	v_cndmask_b32_e32 v104, v143, v106, vcc
	v_cmp_le_i32_e32 vcc, v105, v117
	v_add_u32_e32 v106, 0xffffff9b, v122
	s_nop 0
	v_cndmask_b32_e32 v105, v143, v107, vcc
	v_cmp_le_i32_e32 vcc, v106, v117
	v_add_u32_e32 v107, 0xffffff9c, v122
	s_nop 0
	v_cndmask_b32_e32 v106, v143, v108, vcc
	v_cmp_le_i32_e32 vcc, v107, v117
	v_add_u32_e32 v108, 0xffffff9d, v122
	s_nop 0
	v_cndmask_b32_e32 v107, v143, v109, vcc
	v_cmp_le_i32_e32 vcc, v108, v117
	v_add_u32_e32 v109, 0xffffff9e, v122
	s_nop 0
	v_cndmask_b32_e32 v108, v143, v110, vcc
	v_cmp_le_i32_e32 vcc, v109, v117
	v_add_u32_e32 v110, 0xffffff9f, v122
	s_nop 0
	v_cndmask_b32_e32 v109, v143, v111, vcc
	v_cmp_le_i32_e32 vcc, v110, v117
	v_add_u32_e32 v111, 0xffffffa0, v122
	s_nop 0
	v_cndmask_b32_e32 v110, v143, v112, vcc
	v_cmp_le_i32_e32 vcc, v111, v117
	v_mov_b32_e32 v117, v116
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v111, v143, v113, vcc
	v_pk_mul_f32 v[112:113], v[116:117], v[110:111]
	v_pk_mul_f32 v[110:111], v[116:117], v[108:109]
	v_pk_mul_f32 v[108:109], v[116:117], v[106:107]
	v_pk_mul_f32 v[106:107], v[116:117], v[104:105]
	v_pk_mul_f32 v[104:105], v[116:117], v[102:103]
	v_pk_mul_f32 v[102:103], v[116:117], v[100:101]
	v_pk_mul_f32 v[100:101], v[118:119], v[120:121]
	s_nop 0
	v_max_f32_e32 v117, v100, v101
	v_max3_f32 v117, v117, v98, v99
	v_max3_f32 v117, v117, v102, v103
	v_max3_f32 v117, v117, v104, v105
	v_max3_f32 v117, v117, v106, v107
	v_max3_f32 v117, v117, v108, v109
	v_max3_f32 v117, v117, v110, v111
	v_max3_f32 v117, v117, v112, v113
	ds_bpermute_b32 v120, v139, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v120, v120, v120
	v_max_f32_e32 v120, v117, v120
	;;#ASMSTART
	v_add_f32 v117, v114, v141
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v120, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_62
	;;#ASMSTART
	v_add_f32 v120, v120, v142
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v120
	v_exp_f32_e32 v117, v114
	v_mov_b32_e32 v114, v120
.LBB0_62:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[100:101], v[100:101], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[98:99], v[98:99], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v101
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, 0, v100
	v_add_f32_e32 v120, v120, v101
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v98
	v_add_f32_e32 v120, v120, v99
	v_add_f32_e32 v120, v120, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v103
	v_add_f32_e32 v120, v120, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v105
	v_add_f32_e32 v120, v120, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v107
	v_add_f32_e32 v120, v120, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v120, v120, v109
	v_add_f32_e32 v120, v120, v110
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
	v_add_f32_e32 v120, v120, v111
	v_add_f32_e32 v120, v120, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v100, 0x8000, v100
	v_add_f32_e32 v120, v120, v113
	;;#ASMSTART
	v_fma_f32 v145, v145, v117, v120
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
	;;#ASMSTART
	v_perm_b32 v100, v101, v100, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v99, v98, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s56
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s50, s71, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s50, s50, s0
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s50
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[120:123], v[106:107], off offset:512
	global_load_dwordx4 v[124:127], v[106:107], off offset:1024
	global_load_dwordx4 v[128:131], v[106:107], off offset:1536
	global_load_dwordx4 v[132:135], v[106:107], off offset:2048
	global_load_dwordx4 v[212:215], v[106:107], off offset:2560
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
	v_mfma_f32_32x32x8_bf16 v[82:97], v[212:213], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s57, v106
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
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[134:135], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[214:215], v[102:103], v[82:97]
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
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_64
	ds_write_b128 v138, v[148:151] offset:12288
	ds_write_b128 v138, v[152:155] offset:18432
.LBB0_64:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mul_i32 s70, s1, 0x1800
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_66
	v_add_u32_sdwa v98, s70, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[148:151], v[98:99], off
	global_load_dwordx4 v[152:155], v[98:99], off offset:256
.LBB0_66:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v117, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v120, 56, v98
	v_xor_b32_e32 v98, v120, v117
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[122:125], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[166:167], v[98:113]
	v_or_b32_e32 v121, 16, v117
	v_xor_b32_e32 v121, v121, v120
	v_lshlrev_b32_e32 v121, 1, v121
	ds_read_b128 v[122:125], v121
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[170:171], v[98:113]
	v_or_b32_e32 v121, 32, v117
	v_xor_b32_e32 v121, v121, v120
	v_lshlrev_b32_e32 v121, 1, v121
	ds_read_b128 v[122:125], v121
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[174:175], v[98:113]
	v_or_b32_e32 v121, 48, v117
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[178:179], v[98:113]
	v_add_u32_e32 v120, 64, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[182:183], v[98:113]
	v_add_u32_e32 v120, 0x50, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[186:187], v[98:113]
	v_add_u32_e32 v120, 0x60, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[190:191], v[98:113]
	v_add_u32_e32 v120, 0x70, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[194:195], v[98:113]
	v_add_u32_e32 v120, 0x80, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[198:199], v[98:113]
	v_add_u32_e32 v120, 0x90, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[202:203], v[98:113]
	v_add_u32_e32 v120, 0xa0, v117
	v_lshrrev_b32_e32 v121, 3, v120
	v_and_b32_e32 v121, 56, v121
	v_xor_b32_e32 v120, v121, v120
	v_lshlrev_b32_e32 v120, 1, v120
	ds_read_b128 v[120:123], v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[206:207], v[98:113]
	v_add_u32_e32 v117, 0xb0, v117
	v_lshrrev_b32_e32 v120, 3, v117
	v_and_b32_e32 v120, 56, v120
	v_xor_b32_e32 v117, v120, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[120:123], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[120:121], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	v_mov_b32_e32 v122, v114
	v_lshrrev_b32_e32 v120, 2, v117
	v_and_b32_e32 v120, 8, v120
	v_lshrrev_b32_e32 v117, 1, v117
	v_and_or_b32 v117, v117, s59, v146
	v_add_u32_e32 v120, s66, v120
	v_add_u32_e32 v117, s19, v117
	v_add_u32_e32 v121, 0xffffffa9, v120
	v_cmp_lt_i32_e32 vcc, v121, v117
	v_mov_b32_e32 v123, v114
	v_mov_b32_e32 v124, v114
	v_cndmask_b32_e32 v99, v143, v99, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffab, v120
	v_mov_b32_e32 v125, v114
	v_cndmask_b32_e32 v98, v143, v98, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffac, v120
	v_pk_mul_f32 v[98:99], v[118:119], v[98:99]
	v_cndmask_b32_e32 v100, v143, v100, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffad, v120
	v_mov_b32_e32 v126, v114
	v_cndmask_b32_e32 v101, v143, v101, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffae, v120
	v_mov_b32_e32 v127, v114
	v_cndmask_b32_e32 v102, v143, v102, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffaf, v120
	v_mov_b32_e32 v128, v114
	v_cndmask_b32_e32 v103, v143, v103, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffb0, v120
	v_mov_b32_e32 v129, v114
	v_cndmask_b32_e32 v104, v143, v104, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffb9, v120
	v_mov_b32_e32 v130, v114
	v_cndmask_b32_e32 v105, v143, v105, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffba, v120
	v_mov_b32_e32 v131, v114
	v_cndmask_b32_e32 v106, v143, v106, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffbb, v120
	v_mov_b32_e32 v132, v114
	v_cndmask_b32_e32 v107, v143, v107, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffbc, v120
	v_mov_b32_e32 v133, v114
	v_cndmask_b32_e32 v108, v143, v108, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffbd, v120
	v_mov_b32_e32 v134, v114
	v_cndmask_b32_e32 v109, v143, v109, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffbe, v120
	v_mov_b32_e32 v135, v114
	v_cndmask_b32_e32 v110, v143, v110, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_add_u32_e32 v121, 0xffffffbf, v120
	v_subrev_u32_e32 v120, 64, v120
	v_cndmask_b32_e32 v111, v143, v111, vcc
	v_cmp_le_i32_e32 vcc, v121, v117
	v_mov_b32_e32 v121, v114
	s_nop 0
	v_cndmask_b32_e32 v112, v143, v112, vcc
	v_cmp_le_i32_e32 vcc, v120, v117
	v_mov_b32_e32 v117, v116
	v_pk_mul_f32 v[110:111], v[116:117], v[110:111]
	v_cndmask_b32_e32 v113, v143, v113, vcc
	v_pk_mul_f32 v[112:113], v[116:117], v[112:113]
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
	ds_bpermute_b32 v120, v139, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v120, v120, v120
	v_max_f32_e32 v147, v117, v120
	;;#ASMSTART
	v_add_f32 v117, v114, v141
	;;#ASMEND
	v_mov_b32_e32 v120, v114
	v_cmp_gt_f32_e32 vcc, v147, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_68
	;;#ASMSTART
	v_add_f32 v120, v147, v142
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v120
	v_exp_f32_e32 v117, v114
	v_mov_b32_e32 v114, v120
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
.LBB0_68:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[120:121] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[104:105], v[104:105], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v100
	v_add_f32_e32 v147, v147, v101
	v_add_f32_e32 v147, v147, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v103
	v_add_f32_e32 v147, v147, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v105
	v_add_f32_e32 v147, v147, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v107
	v_add_f32_e32 v147, v147, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
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
	v_add_u32_e32 v105, 0x8000, v105
	v_add_f32_e32 v147, v147, v111
	v_add_f32_e32 v147, v147, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v147, v147, v113
	;;#ASMSTART
	v_fma_f32 v145, v145, v117, v147
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
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s56
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s50, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s50
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[212:215], v[106:107], off offset:512
	global_load_dwordx4 v[216:219], v[106:107], off offset:1024
	global_load_dwordx4 v[220:223], v[106:107], off offset:1536
	global_load_dwordx4 v[224:227], v[106:107], off offset:2048
	global_load_dwordx4 v[228:231], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[212:213], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[228:229], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s57, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[214:215], v[102:103], v[18:33]
	global_load_dwordx4 v[212:215], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[102:103], v[34:49]
	global_load_dwordx4 v[216:219], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[102:103], v[50:65]
	global_load_dwordx4 v[220:223], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[230:231], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[212:213], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[214:215], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	s_add_i32 s50, s44, 1
	s_cmp_gt_i32 s46, s50
	s_cselect_b64 s[48:49], -1, 0
	s_cmp_le_i32 s46, s50
	s_cbranch_scc1 .LBB0_81
	v_mov_b32_e32 v98, s67
	buffer_load_dword v98, v98, s[12:15], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s69, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s55, v99
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_71
	ds_write_b128 v138, v[156:159]
	ds_write_b128 v138, v[160:163] offset:6144
.LBB0_71:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_73
	v_add_u32_e32 v98, s70, v137
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[156:159], v[98:99], off
	global_load_dwordx4 v[160:163], v[98:99], off offset:256
.LBB0_73:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v117, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v117
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[212:215], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[166:167], v[98:113]
	v_add_u32_e32 v147, 0x1810, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[170:171], v[98:113]
	v_add_u32_e32 v147, 0x1820, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[174:175], v[98:113]
	v_add_u32_e32 v147, 0x1830, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[178:179], v[98:113]
	v_add_u32_e32 v147, 0x1840, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[182:183], v[98:113]
	v_add_u32_e32 v147, 0x1850, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[186:187], v[98:113]
	v_add_u32_e32 v147, 0x1860, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[190:191], v[98:113]
	v_add_u32_e32 v147, 0x1870, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[194:195], v[98:113]
	v_add_u32_e32 v147, 0x1880, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[198:199], v[98:113]
	v_add_u32_e32 v147, 0x1890, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[202:203], v[98:113]
	v_add_u32_e32 v147, 0x18a0, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[206:207], v[98:113]
	v_add_u32_e32 v117, 0x18b0, v117
	v_lshrrev_b32_e32 v147, 3, v117
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v117, v147, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[212:215], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v147, 2, v117
	v_and_b32_e32 v147, 8, v147
	v_lshrrev_b32_e32 v117, 1, v117
	v_and_or_b32 v117, v117, s59, v146
	v_add_u32_e32 v147, s66, v147
	v_add_u32_e32 v117, s19, v117
	v_subrev_u32_e32 v212, 55, v147
	v_cmp_lt_i32_e32 vcc, v212, v117
	s_nop 1
	v_cndmask_b32_e32 v99, v143, v99, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 53, v147
	s_nop 0
	v_cndmask_b32_e32 v98, v143, v98, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 52, v147
	v_pk_mul_f32 v[98:99], v[118:119], v[98:99]
	v_cndmask_b32_e32 v100, v143, v100, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 51, v147
	s_nop 0
	v_cndmask_b32_e32 v101, v143, v101, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 50, v147
	s_nop 0
	v_cndmask_b32_e32 v102, v143, v102, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 49, v147
	s_nop 0
	v_cndmask_b32_e32 v103, v143, v103, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 48, v147
	s_nop 0
	v_cndmask_b32_e32 v104, v143, v104, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 39, v147
	s_nop 0
	v_cndmask_b32_e32 v105, v143, v105, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 38, v147
	s_nop 0
	v_cndmask_b32_e32 v106, v143, v106, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 37, v147
	s_nop 0
	v_cndmask_b32_e32 v107, v143, v107, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 36, v147
	s_nop 0
	v_cndmask_b32_e32 v108, v143, v108, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 35, v147
	s_nop 0
	v_cndmask_b32_e32 v109, v143, v109, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 34, v147
	s_nop 0
	v_cndmask_b32_e32 v110, v143, v110, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 33, v147
	v_subrev_u32_e32 v147, 32, v147
	v_cndmask_b32_e32 v111, v143, v111, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	s_nop 1
	v_cndmask_b32_e32 v112, v143, v112, vcc
	v_cmp_le_i32_e32 vcc, v147, v117
	v_mov_b32_e32 v117, v116
	v_pk_mul_f32 v[110:111], v[116:117], v[110:111]
	v_cndmask_b32_e32 v113, v143, v113, vcc
	v_pk_mul_f32 v[112:113], v[116:117], v[112:113]
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
	ds_bpermute_b32 v147, v139, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v147, v147, v147
	v_max_f32_e32 v147, v117, v147
	;;#ASMSTART
	v_add_f32 v117, v114, v141
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v147, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_75
	;;#ASMSTART
	v_add_f32 v120, v147, v142
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v120
	v_exp_f32_e32 v117, v114
	v_mov_b32_e32 v114, v120
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
.LBB0_75:
	s_or_b64 exec, exec, s[50:51]
	v_pk_add_f32 v[98:99], v[98:99], v[120:121] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[104:105], v[104:105], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v100
	v_add_f32_e32 v147, v147, v101
	v_add_f32_e32 v147, v147, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v103
	v_add_f32_e32 v147, v147, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v105
	v_add_f32_e32 v147, v147, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v147, v147, v107
	v_add_f32_e32 v147, v147, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
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
	v_add_u32_e32 v105, 0x8000, v105
	v_add_f32_e32 v147, v147, v111
	v_add_f32_e32 v147, v147, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v147, v147, v113
	;;#ASMSTART
	v_fma_f32 v145, v145, v117, v147
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
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s56
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s70, s64, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s70, s70, s0
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s70
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[24:25]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[212:215], v[106:107], off offset:512
	global_load_dwordx4 v[216:219], v[106:107], off offset:1024
	global_load_dwordx4 v[220:223], v[106:107], off offset:1536
	global_load_dwordx4 v[224:227], v[106:107], off offset:2048
	global_load_dwordx4 v[228:231], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[212:213], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[224:225], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[228:229], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s57, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[214:215], v[102:103], v[18:33]
	global_load_dwordx4 v[212:215], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[102:103], v[34:49]
	global_load_dwordx4 v[216:219], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[102:103], v[50:65]
	global_load_dwordx4 v[220:223], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[226:227], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[230:231], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[212:213], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[216:217], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[220:221], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[214:215], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[218:219], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[222:223], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_77
	ds_write_b128 v138, v[148:151] offset:12288
	ds_write_b128 v138, v[152:155] offset:18432
.LBB0_77:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s55, v98
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_79
	s_mul_i32 s71, s65, 0x1800
	v_add_u32_sdwa v98, s71, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[16:17]
	global_load_dwordx4 v[148:151], v[98:99], off
	global_load_dwordx4 v[152:155], v[98:99], off offset:256
.LBB0_79:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v117, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v147, 56, v98
	v_xor_b32_e32 v98, v147, v117
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[212:215], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[166:167], v[98:113]
	v_or_b32_e32 v212, 16, v117
	v_xor_b32_e32 v212, v212, v147
	v_lshlrev_b32_e32 v212, 1, v212
	ds_read_b128 v[212:215], v212
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[170:171], v[98:113]
	v_or_b32_e32 v212, 32, v117
	v_xor_b32_e32 v212, v212, v147
	v_lshlrev_b32_e32 v212, 1, v212
	ds_read_b128 v[212:215], v212
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[174:175], v[98:113]
	v_or_b32_e32 v212, 48, v117
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[178:179], v[98:113]
	v_add_u32_e32 v147, 64, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[182:183], v[98:113]
	v_add_u32_e32 v147, 0x50, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[186:187], v[98:113]
	v_add_u32_e32 v147, 0x60, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[190:191], v[98:113]
	v_add_u32_e32 v147, 0x70, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[194:195], v[98:113]
	v_add_u32_e32 v147, 0x80, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[198:199], v[98:113]
	v_add_u32_e32 v147, 0x90, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[202:203], v[98:113]
	v_add_u32_e32 v147, 0xa0, v117
	v_lshrrev_b32_e32 v212, 3, v147
	v_and_b32_e32 v212, 56, v212
	v_xor_b32_e32 v147, v212, v147
	v_lshlrev_b32_e32 v147, 1, v147
	ds_read_b128 v[212:215], v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[206:207], v[98:113]
	v_add_u32_e32 v117, 0xb0, v117
	v_lshrrev_b32_e32 v147, 3, v117
	v_and_b32_e32 v147, 56, v147
	v_xor_b32_e32 v117, v147, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[212:215], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[212:213], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[214:215], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v147, 2, v117
	v_and_b32_e32 v147, 8, v147
	v_lshrrev_b32_e32 v117, 1, v117
	v_and_or_b32 v117, v117, s59, v146
	v_add_u32_e32 v147, s66, v147
	v_add_u32_e32 v117, s19, v117
	v_subrev_u32_e32 v212, 23, v147
	v_cmp_lt_i32_e32 vcc, v212, v117
	s_nop 1
	v_cndmask_b32_e32 v99, v143, v99, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 21, v147
	s_nop 0
	v_cndmask_b32_e32 v98, v143, v98, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 20, v147
	v_pk_mul_f32 v[98:99], v[118:119], v[98:99]
	v_cndmask_b32_e32 v100, v143, v100, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 19, v147
	s_nop 0
	v_cndmask_b32_e32 v101, v143, v101, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 18, v147
	s_nop 0
	v_cndmask_b32_e32 v102, v143, v102, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_subrev_u32_e32 v212, 17, v147
	s_nop 0
	v_cndmask_b32_e32 v103, v143, v103, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_add_u32_e32 v212, -16, v147
	s_nop 0
	v_cndmask_b32_e32 v104, v143, v104, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_add_u32_e32 v212, -7, v147
	s_nop 0
	v_cndmask_b32_e32 v105, v143, v105, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_add_u32_e32 v212, -6, v147
	s_nop 0
	v_cndmask_b32_e32 v106, v143, v106, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_add_u32_e32 v212, -5, v147
	s_nop 0
	v_cndmask_b32_e32 v107, v143, v107, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_add_u32_e32 v212, -4, v147
	s_nop 0
	v_cndmask_b32_e32 v108, v143, v108, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_add_u32_e32 v212, -3, v147
	s_nop 0
	v_cndmask_b32_e32 v109, v143, v109, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_add_u32_e32 v212, -2, v147
	s_nop 0
	v_cndmask_b32_e32 v110, v143, v110, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	v_add_u32_e32 v212, -1, v147
	s_nop 0
	v_cndmask_b32_e32 v111, v143, v111, vcc
	v_cmp_le_i32_e32 vcc, v212, v117
	s_nop 1
	v_cndmask_b32_e32 v112, v143, v112, vcc
	v_cmp_le_i32_e32 vcc, v147, v117
	v_mov_b32_e32 v117, v116
	v_pk_mul_f32 v[110:111], v[116:117], v[110:111]
	v_cndmask_b32_e32 v113, v143, v113, vcc
	v_pk_mul_f32 v[112:113], v[116:117], v[112:113]
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
	ds_bpermute_b32 v147, v139, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v147, v147, v147
	v_max_f32_e32 v147, v117, v147
	;;#ASMSTART
	v_add_f32 v117, v114, v141
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v147, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_54
	;;#ASMSTART
	v_add_f32 v120, v147, v142
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v120
	v_exp_f32_e32 v117, v114
	v_mov_b32_e32 v114, v120
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
	s_branch .LBB0_54
.LBB0_81:
	s_mov_b32 s69, s68
	s_branch .LBB0_55
.LBB0_82:
	;;#ASMSTART
	v_mov_b32 v100, v0
	;;#ASMEND
	ds_bpermute_b32 v98, v139, v145
	v_lshrrev_b32_e32 v99, 1, v100
	v_and_b32_e32 v101, 31, v100
	v_and_or_b32 v99, v99, s59, v101
	v_and_b32_e32 v100, 32, v100
	v_cmp_eq_u32_e32 vcc, 0, v100
	v_cmp_gt_i32_e64 s[0:1], s43, v99
	s_and_b64 s[12:13], vcc, s[0:1]
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v98, v145, v98
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[12:13]
	s_cbranch_execz .LBB0_84
	v_cmp_gt_f32_e32 vcc, s60, v98
	s_ashr_i32 s43, s42, 31
	s_lshl_b64 s[12:13], s[42:43], 6
	v_cndmask_b32_e64 v101, 0, 32, vcc
	v_ldexp_f32 v101, v98, v101
	v_log_f32_e32 v101, v101
	v_cndmask_b32_e32 v100, 0, v144, vcc
	s_add_u32 s14, s36, s12
	s_addc_u32 s16, s37, s13
	v_sub_f32_e32 v100, v101, v100
	v_add_f32_e32 v100, v114, v100
	s_lshl_b64 s[12:13], s[28:29], 2
	v_mul_f32_e32 v100, 0x3f317218, v100
	v_cmp_lt_f32_e32 vcc, 0, v98
	s_add_u32 s12, s14, s12
	s_addc_u32 s13, s16, s13
	v_cndmask_b32_e32 v100, v143, v100, vcc
	v_lshlrev_b32_e32 v99, 6, v99
	global_store_dword v99, v100, s[12:13]
.LBB0_84:
	s_or_b64 exec, exec, s[0:1]
	v_div_scale_f32 v99, s[0:1], v98, v98, s54
	v_rcp_f32_e32 v100, v99
	v_div_scale_f32 v101, vcc, s54, v98, s54
	v_fma_f32 v102, -v99, v100, 1.0
	v_fmac_f32_e32 v100, v102, v100
	v_mul_f32_e32 v102, v101, v100
	v_fma_f32 v103, -v99, v102, v101
	v_fmac_f32_e32 v102, v103, v100
	v_fma_f32 v99, -v99, v102, v101
	v_div_fmas_f32 v99, v99, v100, v102
	v_div_fixup_f32 v98, v99, v98, s54
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
	v_perm_b32 v48, v3, v2, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v49, v5, v4, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v46, v7, v6, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v47, v9, v8, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v44, v11, v10, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v45, v13, v12, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v42, v15, v14, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v43, v17, v16, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v40, v19, v18, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v41, v21, v20, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v38, v23, v22, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v39, v25, v24, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v36, v27, v26, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v37, v29, v28, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v34, v31, v30, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v35, v33, v32, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v112, v113, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v110, v111, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v108, v109, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v106, v107, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v28, v104, v105, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v102, v103, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v100, v101, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v98, v99, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v51, v50, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v53, v52, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v55, v54, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v57, v56, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v59, v58, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v61, v60, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v63, v62, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v65, v64, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v67, v66, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v69, v68, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v71, v70, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v73, v72, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v75, v74, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v77, v76, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v79, v78, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v81, v80, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v83, v82, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v85, v84, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v87, v86, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v89, v88, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v91, v90, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v93, v92, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v95, v94, s56
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v97, v96, s56
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v50, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v50, 0x100, v50
	v_cmp_eq_u32_e32 vcc, 0, v50
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_86
	s_barrier
.LBB0_86:
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
	s_cbranch_execz .LBB0_88
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
.LBB0_88:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v88, 0x1ff, v91
	s_add_u32 s16, s34, s40
	s_addc_u32 s0, s35, s41
	v_lshlrev_b32_e32 v92, 3, v88
	s_and_b32 s17, s0, 0xffff
	s_mov_b32 s19, s15
	v_and_b32_e32 v91, 56, v91
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_89:
	v_mul_u32_u24_sdwa v95, v94, s61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s62, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s23, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_89
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 1, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_92
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
.LBB0_92:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s12, s23, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_93:
	v_mul_u32_u24_sdwa v95, v94, s61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s62, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s12, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_93
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 2, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_96
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
.LBB0_96:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s12, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_97:
	v_mul_u32_u24_sdwa v95, v94, s61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s62, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_97
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 3, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_100
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
.LBB0_100:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s12, 0x30000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_101:
	v_mul_u32_u24_sdwa v95, v94, s61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s62, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_101
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 4, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_104
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
.LBB0_104:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s12, 0x48000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_105:
	v_mul_u32_u24_sdwa v95, v94, s61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s62, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_105
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 5, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_108
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
.LBB0_108:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s12, 0x60000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_109:
	v_mul_u32_u24_sdwa v95, v94, s61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s62, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_109
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 6, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_112
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
.LBB0_112:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s13, s12, 0x78000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_113:
	v_mul_u32_u24_sdwa v95, v94, s61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s62, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s13, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_113
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 7, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_116
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
.LBB0_116:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s12, s12, 0x90000
	s_mov_b64 s[0:1], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_117:
	v_mul_u32_u24_sdwa v2, v88, s61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v3, v92, v91
	v_lshrrev_b32_e32 v2, 20, v2
	v_lshlrev_b32_e32 v3, 1, v3
	v_mul_lo_u16_e32 v5, 24, v2
	ds_read_b128 v[6:9], v3
	v_sub_u16_e32 v3, v88, v5
	v_lshlrev_b16_e32 v3, 3, v3
	v_add_u32_e32 v4, 0x200, v88
	v_cmp_lt_u32_e32 vcc, s62, v88
	v_mul_u32_u24_e32 v2, 0xc00, v2
	v_add_u32_e32 v3, s12, v3
	v_add_u32_e32 v92, 0x1000, v92
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v88, v4
	v_add_lshl_u32 v2, v3, v2, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v2, s[16:19], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_117
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	;;#ASMSTART
	s_mov_b32 s0, s63
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_122
	s_mov_b64 s[18:19], exec
	v_mbcnt_lo_u32_b32 v2, s18, 0
	v_mbcnt_hi_u32_b32 v2, s19, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_121
	s_bcnt1_i32_b64 s14, s[18:19]
	v_mov_b32_e32 v3, s14
	global_atomic_add v3, v140, v3, s[38:39] sc0
.LBB0_121:
	s_or_b64 exec, exec, s[16:17]
	s_lshl_b64 s[16:17], s[0:1], 2
	s_add_u32 s16, s38, s16
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s14, v3
	s_addc_u32 s17, s39, s17
	s_nop 0
	v_add_u32_e32 v2, s14, v2
	global_store_dword v140, v2, s[16:17]
	s_waitcnt vmcnt(0)
.LBB0_122:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s38, s0
	s_addc_u32 s1, s39, s1
	s_barrier
	global_load_dword v2, v140, s[0:1]
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s12, v2
	s_add_i32 s0, s12, s33
	s_sub_i32 s33, s0, s2
	s_cmp_ge_i32 s22, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s52
	s_cselect_b64 s[16:17], -1, 0
	s_or_b64 s[0:1], s[0:1], s[16:17]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_125
	s_branch .LBB0_12
.LBB0_123:
	s_mov_b32 s22, s13
.LBB0_124:
	s_sub_i32 s33, s33, s52
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s28, 0, s2
	s_cmp_ge_i32 s22, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s14
	s_cselect_b64 s[16:17], -1, 0
	s_or_b64 s[0:1], s[0:1], s[16:17]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s52, s14
	s_cbranch_vccnz .LBB0_11
.LBB0_125:
	s_add_i32 s2, s28, 1
	s_cmp_gt_i32 s2, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s2, 16
	s_cbranch_scc1 .LBB0_128
	s_add_i32 s13, s22, 1
	s_cmp_ge_i32 s13, s3
	s_mov_b32 s14, s52
	s_cbranch_scc1 .LBB0_123
	s_ashr_i32 s23, s22, 31
	s_lshl_b64 s[16:17], s[22:23], 2
	s_add_u32 s16, s20, s16
	s_addc_u32 s17, s21, s17
	global_load_dwordx2 v[2:3], v140, s[16:17] offset:4
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
	s_branch .LBB0_123
.LBB0_128:
	s_mov_b32 s14, s52
	s_branch .LBB0_124
.LBB0_129:
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
		.amdhsa_next_free_vgpr 232
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 232
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

	.set .Lattn_kernel_0.num_vgpr, 232
	.set .Lattn_kernel_0.num_agpr, 0
	.set .Lattn_kernel_0.numbered_sgpr, 72
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
    .sgpr_count:     78
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     232
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
