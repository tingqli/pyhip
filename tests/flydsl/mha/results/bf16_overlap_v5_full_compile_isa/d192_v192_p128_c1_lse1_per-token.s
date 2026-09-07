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
	s_cmp_lt_i32 s33, s52
	s_cselect_b64 s[10:11], -1, 0
	s_or_b64 s[8:9], s[8:9], s[10:11]
	s_and_b64 vcc, exec, s[8:9]
	s_mov_b32 s10, s52
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s11, s26, 1
	s_cmp_gt_i32 s11, 15
	s_cselect_b64 s[8:9], -1, 0
	s_cmp_lt_i32 s11, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s18, s16, 1
	s_cmp_ge_i32 s18, s3
	s_mov_b32 s52, s10
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
	s_subb_u32 s52, s22, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s52, s10
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s52, s10
	s_mov_b32 s26, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s53, s[4:5], 0x0
	s_load_dword s54, s[6:7], 0x0
	s_cmp_ge_i32 s16, s3
	s_cbranch_scc1 .LBB0_177
	v_lshrrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v1, 31, v0
	s_movk_i32 s55, 0xe0
	v_and_or_b32 v2, v2, s55, v1
	v_lshlrev_b32_e32 v115, 6, v2
	v_lshrrev_b32_e32 v2, 6, v0
	v_lshrrev_b32_e32 v3, 2, v0
	v_mul_u32_u24_e32 v2, 0x18000, v2
	s_load_dwordx2 s[18:19], s[0:1], 0x0
	s_load_dwordx2 s[20:21], s[0:1], 0x10
	s_load_dwordx2 s[22:23], s[0:1], 0x20
	s_load_dwordx2 s[24:25], s[0:1], 0x50
	s_load_dwordx2 s[28:29], s[0:1], 0x60
	s_load_dwordx2 s[30:31], s[0:1], 0x70
	s_load_dwordx2 s[34:35], s[0:1], 0xa0
	s_load_dwordx2 s[36:37], s[0:1], 0xb0
	s_load_dwordx2 s[38:39], s[0:1], 0xc0
	v_and_or_b32 v2, v3, 8, v2
	s_movk_i32 s0, 0xc00
	v_mad_u32_u24 v148, v1, s0, v2
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
	v_and_or_b32 v149, v2, 12, v3
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0x70, v3
	v_xor_b32_e32 v151, v2, v3
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_or_b32_e32 v150, 0x80, v149
	v_or_b32_e32 v152, 0x100, v149
	s_movk_i32 s56, 0x180
	v_or_b32_e32 v153, 0x180, v149
	v_xor_b32_e32 v154, 0x80, v2
	v_mov_b32_e32 v155, 0
	s_mov_b32 s7, 0x27000
	v_mov_b32_e32 v156, 0x40e00000
	v_mov_b32_e32 v157, 1.0
	s_mov_b32 s57, 0x7060302
	s_movk_i32 s58, 0x1000
	s_movk_i32 s59, 0x2000
	s_mov_b32 s60, 0x800000
	s_mov_b32 s61, 0xaaab
	s_movk_i32 s62, 0xff
	s_mov_b32 s63, s2
	v_mov_b32_e32 v158, 0xff800000
	v_mov_b32_e32 v159, 0x42000000
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s52, s6
.LBB0_12:
	s_cmp_ge_i32 s16, s3
	s_cbranch_scc1 .LBB0_177
.LBB0_13:
	s_ashr_i32 s17, s16, 31
	s_lshl_b32 s50, s33, 8
	s_lshl_b64 s[0:1], s[16:17], 2
	s_add_u32 s4, s12, s0
	s_addc_u32 s5, s13, s1
	global_load_dwordx2 v[2:3], v155, s[4:5]
	s_mul_i32 s17, s26, 0xc0
	s_mov_b32 s11, s7
	v_add_lshl_u32 v7, s17, v148, 1
	v_lshl_add_u32 v6, s26, 2, v115
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s46, v2
	s_add_i32 s42, s46, s50
	v_readfirstlane_b32 s47, v3
	s_add_i32 s5, s42, 0x100
	s_min_i32 s5, s5, s47
	s_sub_i32 s43, s5, s42
	s_waitcnt lgkmcnt(0)
	s_add_u32 s8, s24, s0
	s_addc_u32 s9, s25, s1
	s_mul_i32 s4, s42, 0xc00
	s_add_u32 s0, s14, s0
	s_addc_u32 s1, s15, s1
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s44, s42, 4
	s_lshl_b64 s[40:41], s[4:5], 1
	global_load_dwordx2 v[4:5], v155, s[8:9]
	global_load_dword v3, v155, s[0:1]
	s_add_u32 s8, s18, s40
	s_addc_u32 s0, s19, s41
	s_mul_i32 s10, s43, 0x1800
	s_and_b32 s9, s0, 0xffff
	buffer_load_dwordx4 v[164:167], v7, s[8:11], 0 offen
	buffer_load_dwordx4 v[168:171], v7, s[8:11], 0 offen offset:32
	buffer_load_dwordx4 v[172:175], v7, s[8:11], 0 offen offset:64
	buffer_load_dwordx4 v[176:179], v7, s[8:11], 0 offen offset:96
	buffer_load_dwordx4 v[180:183], v7, s[8:11], 0 offen offset:128
	buffer_load_dwordx4 v[184:187], v7, s[8:11], 0 offen offset:160
	buffer_load_dwordx4 v[188:191], v7, s[8:11], 0 offen offset:192
	buffer_load_dwordx4 v[192:195], v7, s[8:11], 0 offen offset:224
	buffer_load_dwordx4 v[196:199], v7, s[8:11], 0 offen offset:256
	buffer_load_dwordx4 v[200:203], v7, s[8:11], 0 offen offset:288
	s_ashr_i32 s45, s44, 31
	s_lshl_b64 s[0:1], s[44:45], 2
	s_add_u32 s4, s30, s0
	s_addc_u32 s0, s31, s1
	s_lshl_b32 s6, s43, 6
	s_and_b32 s5, s0, 0xffff
	buffer_load_dword v2, v6, s[4:7], 0 offen
	buffer_load_dwordx4 v[204:207], v7, s[8:11], 0 offen offset:320
	buffer_load_dwordx4 v[208:211], v7, s[8:11], 0 offen offset:352
	s_waitcnt vmcnt(14)
	v_readfirstlane_b32 s4, v4
	s_waitcnt vmcnt(13)
	v_readfirstlane_b32 s11, v3
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
	s_ashr_i32 s27, s26, 31
	s_lshr_b32 s0, s27, 28
	s_add_i32 s0, s26, s0
	s_sub_i32 s51, s5, s4
	s_ashr_i32 s5, s0, 4
	s_and_b32 s0, s0, -16
	s_cmp_lg_u32 s26, s0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s26, 0
	s_cselect_b64 s[8:9], -1, 0
	s_and_b64 s[0:1], s[8:9], s[0:1]
	s_subb_u32 s0, s5, 0
	s_mulk_i32 s0, 0x6000
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[8:9], s[0:1], 1
	s_add_u32 s8, s20, s8
	s_addc_u32 s9, s21, s9
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s4, s28, s4
	s_addc_u32 s1, s29, s5
	s_lshl_b32 s6, s51, 2
	s_and_b32 s5, s1, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[4:7], 0
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
	v_readfirstlane_b32 s1, v3
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
	v_readfirstlane_b32 s65, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_mul_i32 s48, s71, 0x3000
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s56, v3
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_17
	v_add_u32_e32 v4, s48, v149
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[8:9]
	global_load_dwordx4 v[212:215], v[4:5], off
	global_load_dwordx4 v[216:219], v[4:5], off offset:256
.LBB0_17:
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
	v_cmp_ne_u32_e32 vcc, s56, v3
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_19
	v_add_u32_e32 v4, s48, v150
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[8:9]
	global_load_dwordx4 v[220:223], v[4:5], off
	global_load_dwordx4 v[224:227], v[4:5], off offset:256
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
	v_cmp_ne_u32_e32 vcc, s56, v3
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_21
	s_waitcnt vmcnt(1)
	ds_write_b128 v151, v[212:215] offset:12288
	s_waitcnt vmcnt(0)
	ds_write_b128 v151, v[216:219] offset:18432
.LBB0_21:
	s_or_b64 exec, exec, s[44:45]
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v3, 0x180, v3
	v_cmp_ne_u32_e32 vcc, s56, v3
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_23
	v_add_u32_e32 v4, s48, v152
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[8:9]
	global_load_dwordx4 v[212:215], v[4:5], off
	global_load_dwordx4 v[216:219], v[4:5], off offset:256
.LBB0_23:
	s_or_b64 exec, exec, s[44:45]
	v_mul_f32_e32 v2, s53, v2
	s_sub_i32 s44, s46, s47
	s_lshl_b32 s45, s51, 7
	v_mul_f32_e32 v116, 0x3dd53b95, v2
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_add_i32 s44, s44, s45
	s_add_i32 s11, s44, s11
	s_addk_i32 s11, 0xff80
	s_add_i32 s66, s11, s50
	s_add_i32 s46, s66, 1
	s_ashr_i32 s44, s46, 31
	s_lshr_b32 s44, s44, 25
	s_add_i32 s44, s46, s44
	s_ashr_i32 s48, s44, 7
	s_and_b32 s44, s44, 0xffffff80
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
	s_cbranch_scc1 .LBB0_75
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v117, v116
	v_mov_b32_e32 v118, v116
	v_mov_b32_e32 v119, v116
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
	s_mov_b64 s[46:47], 0
	s_mov_b32 s68, 20
	v_mov_b32_e32 v114, 0xff800000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v161
	v_mov_b32_e32 v4, v161
	v_mov_b32_e32 v5, v161
	v_mov_b32_e32 v6, v161
	v_mov_b32_e32 v7, v161
	v_mov_b32_e32 v8, v161
	v_mov_b32_e32 v9, v161
	v_mov_b32_e32 v10, v161
	v_mov_b32_e32 v11, v161
	v_mov_b32_e32 v12, v161
	v_mov_b32_e32 v13, v161
	v_mov_b32_e32 v14, v161
	v_mov_b32_e32 v15, v161
	v_mov_b32_e32 v16, v161
	v_mov_b32_e32 v17, v161
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v161
	v_mov_b32_e32 v20, v161
	v_mov_b32_e32 v21, v161
	v_mov_b32_e32 v22, v161
	v_mov_b32_e32 v23, v161
	v_mov_b32_e32 v24, v161
	v_mov_b32_e32 v25, v161
	v_mov_b32_e32 v26, v161
	v_mov_b32_e32 v27, v161
	v_mov_b32_e32 v28, v161
	v_mov_b32_e32 v29, v161
	v_mov_b32_e32 v30, v161
	v_mov_b32_e32 v31, v161
	v_mov_b32_e32 v32, v161
	v_mov_b32_e32 v33, v161
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v161
	v_mov_b32_e32 v36, v161
	v_mov_b32_e32 v37, v161
	v_mov_b32_e32 v38, v161
	v_mov_b32_e32 v39, v161
	v_mov_b32_e32 v40, v161
	v_mov_b32_e32 v41, v161
	v_mov_b32_e32 v42, v161
	v_mov_b32_e32 v43, v161
	v_mov_b32_e32 v44, v161
	v_mov_b32_e32 v45, v161
	v_mov_b32_e32 v46, v161
	v_mov_b32_e32 v47, v161
	v_mov_b32_e32 v48, v161
	v_mov_b32_e32 v49, v161
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v161
	v_mov_b32_e32 v52, v161
	v_mov_b32_e32 v53, v161
	v_mov_b32_e32 v54, v161
	v_mov_b32_e32 v55, v161
	v_mov_b32_e32 v56, v161
	v_mov_b32_e32 v57, v161
	v_mov_b32_e32 v58, v161
	v_mov_b32_e32 v59, v161
	v_mov_b32_e32 v60, v161
	v_mov_b32_e32 v61, v161
	v_mov_b32_e32 v62, v161
	v_mov_b32_e32 v63, v161
	v_mov_b32_e32 v64, v161
	v_mov_b32_e32 v65, v161
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, v161
	v_mov_b32_e32 v68, v161
	v_mov_b32_e32 v69, v161
	v_mov_b32_e32 v70, v161
	v_mov_b32_e32 v71, v161
	v_mov_b32_e32 v72, v161
	v_mov_b32_e32 v73, v161
	v_mov_b32_e32 v74, v161
	v_mov_b32_e32 v75, v161
	v_mov_b32_e32 v76, v161
	v_mov_b32_e32 v77, v161
	v_mov_b32_e32 v78, v161
	v_mov_b32_e32 v79, v161
	v_mov_b32_e32 v80, v161
	v_mov_b32_e32 v81, v161
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, v161
	v_mov_b32_e32 v84, v161
	v_mov_b32_e32 v85, v161
	v_mov_b32_e32 v86, v161
	v_mov_b32_e32 v87, v161
	v_mov_b32_e32 v88, v161
	v_mov_b32_e32 v89, v161
	v_mov_b32_e32 v90, v161
	v_mov_b32_e32 v91, v161
	v_mov_b32_e32 v92, v161
	v_mov_b32_e32 v93, v161
	v_mov_b32_e32 v94, v161
	v_mov_b32_e32 v95, v161
	v_mov_b32_e32 v96, v161
	v_mov_b32_e32 v97, v161
	s_branch .LBB0_26
.LBB0_25:
	s_or_b64 exec, exec, s[48:49]
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
	v_fma_f32 v161, v160, v162, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v162
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
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v132, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s71, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s71
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[132:135], v[106:107], off offset:512
	global_load_dwordx4 v[136:139], v[106:107], off offset:1024
	global_load_dwordx4 v[140:143], v[106:107], off offset:1536
	global_load_dwordx4 v[144:147], v[106:107], off offset:2048
	global_load_dwordx4 v[228:231], v[106:107], off offset:2560
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
	v_mfma_f32_32x32x8_bf16 v[82:97], v[228:229], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
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
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[230:231], v[102:103], v[82:97]
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
	s_add_u32 s46, s46, 2
	s_addc_u32 s47, s47, 0
	v_mov_b64_e32 v[98:99], s[44:45]
	v_cmp_lt_i64_e32 vcc, s[46:47], v[98:99]
	s_add_i32 s68, s68, 8
	s_mov_b32 s64, s70
	s_mov_b32 s71, s69
	s_cbranch_vccz .LBB0_74
.LBB0_26:
	s_add_i32 s48, s68, -4
	v_mov_b32_e32 v98, s48
	buffer_load_dword v98, v98, s[4:7], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_mov_b32 s69, s1
	v_and_b32_e32 v99, 0x180, v99
	s_mov_b32 s70, s65
	v_cmp_ne_u32_e32 vcc, s56, v99
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s1, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_28
	ds_write_b128 v151, v[220:223]
	ds_write_b128 v151, v[224:227] offset:6144
.LBB0_28:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_30
	s_mul_i32 s65, s71, 0x3000
	v_add_u32_e32 v98, s65, v153
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[220:223], v[98:99], off
	global_load_dwordx4 v[224:227], v[98:99], off offset:256
.LBB0_30:
	s_or_b64 exec, exec, s[48:49]
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
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[166:167], v[98:113]
	v_add_u32_e32 v133, 0x1810, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[170:171], v[98:113]
	v_add_u32_e32 v133, 0x1820, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[174:175], v[98:113]
	v_add_u32_e32 v133, 0x1830, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[178:179], v[98:113]
	v_add_u32_e32 v133, 0x1840, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[182:183], v[98:113]
	v_add_u32_e32 v133, 0x1850, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[186:187], v[98:113]
	v_add_u32_e32 v133, 0x1860, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[190:191], v[98:113]
	v_add_u32_e32 v133, 0x1870, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[194:195], v[98:113]
	v_add_u32_e32 v133, 0x1880, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[198:199], v[98:113]
	v_add_u32_e32 v133, 0x1890, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[202:203], v[98:113]
	v_add_u32_e32 v133, 0x18a0, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	v_add_u32_e32 v132, 0x18b0, v132
	v_lshrrev_b32_e32 v133, 3, v132
	v_and_b32_e32 v133, 56, v133
	v_xor_b32_e32 v132, v133, v132
	v_lshlrev_b32_e32 v132, 1, v132
	ds_read_b128 v[132:135], v132
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[132:133], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
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
	ds_bpermute_b32 v133, v154, v132
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v133, v133, v133
	v_max_f32_e32 v133, v132, v133
	;;#ASMSTART
	v_add_f32 v132, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v133, v132
	v_mov_b32_e32 v132, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v133, v133, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v133
	v_exp_f32_e32 v132, v114
	v_mov_b32_e32 v114, v133
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
	v_pk_add_f32 v[104:105], v[104:105], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v100
	v_add_f32_e32 v133, v133, v101
	v_add_f32_e32 v133, v133, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v103
	v_add_f32_e32 v133, v133, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v105
	v_add_f32_e32 v133, v133, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v107
	v_add_f32_e32 v133, v133, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
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
	v_fma_f32 v160, v161, v132, v133
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
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v132, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s65, s71, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s65, s65, s0
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s65
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[132:135], v[106:107], off offset:512
	global_load_dwordx4 v[136:139], v[106:107], off offset:1024
	global_load_dwordx4 v[140:143], v[106:107], off offset:1536
	global_load_dwordx4 v[144:147], v[106:107], off offset:2048
	global_load_dwordx4 v[228:231], v[106:107], off offset:2560
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
	v_mfma_f32_32x32x8_bf16 v[82:97], v[228:229], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
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
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[230:231], v[102:103], v[82:97]
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
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_34
	ds_write_b128 v151, v[212:215] offset:12288
	ds_write_b128 v151, v[216:219] offset:18432
.LBB0_34:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mul_i32 s71, s64, 0x3000
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_36
	v_add_u32_e32 v98, s71, v149
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[212:215], v[98:99], off
	global_load_dwordx4 v[216:219], v[98:99], off offset:256
.LBB0_36:
	s_or_b64 exec, exec, s[48:49]
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
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[166:167], v[98:113]
	v_or_b32_e32 v134, 16, v132
	v_xor_b32_e32 v134, v134, v133
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[170:171], v[98:113]
	v_or_b32_e32 v134, 32, v132
	v_xor_b32_e32 v134, v134, v133
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[174:175], v[98:113]
	v_or_b32_e32 v134, 48, v132
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[178:179], v[98:113]
	v_add_u32_e32 v133, 64, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[182:183], v[98:113]
	v_add_u32_e32 v133, 0x50, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[186:187], v[98:113]
	v_add_u32_e32 v133, 0x60, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[190:191], v[98:113]
	v_add_u32_e32 v133, 0x70, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[194:195], v[98:113]
	v_add_u32_e32 v133, 0x80, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[198:199], v[98:113]
	v_add_u32_e32 v133, 0x90, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[202:203], v[98:113]
	v_add_u32_e32 v133, 0xa0, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	v_add_u32_e32 v132, 0xb0, v132
	v_lshrrev_b32_e32 v133, 3, v132
	v_and_b32_e32 v133, 56, v133
	v_xor_b32_e32 v132, v133, v132
	v_lshlrev_b32_e32 v132, 1, v132
	ds_read_b128 v[132:135], v132
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[132:133], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
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
	ds_bpermute_b32 v133, v154, v132
	v_mov_b32_e32 v161, 1.0
	v_mov_b32_e32 v134, v114
	v_mov_b32_e32 v135, v114
	v_mov_b32_e32 v136, v114
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v133, v133, v133
	v_max_f32_e32 v162, v132, v133
	;;#ASMSTART
	v_add_f32 v132, v114, v156
	;;#ASMEND
	v_mov_b32_e32 v133, v114
	v_cmp_gt_f32_e32 vcc, v162, v132
	v_mov_b32_e32 v132, v114
	v_mov_b32_e32 v137, v114
	v_mov_b32_e32 v138, v114
	v_mov_b32_e32 v139, v114
	v_mov_b32_e32 v140, v114
	v_mov_b32_e32 v141, v114
	v_mov_b32_e32 v142, v114
	v_mov_b32_e32 v143, v114
	v_mov_b32_e32 v144, v114
	v_mov_b32_e32 v145, v114
	v_mov_b32_e32 v146, v114
	v_mov_b32_e32 v147, v114
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_38
	;;#ASMSTART
	v_add_f32 v132, v162, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v161, v114
	v_mov_b32_e32 v114, v132
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
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, 0, v98
	v_add_f32_e32 v162, v162, v99
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
	v_add_f32_e32 v162, v162, v100
	v_add_f32_e32 v162, v162, v101
	v_add_f32_e32 v162, v162, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v103
	v_add_f32_e32 v162, v162, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v105
	v_add_f32_e32 v162, v162, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v107
	v_add_f32_e32 v162, v162, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v109
	v_add_f32_e32 v162, v162, v110
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
	v_add_f32_e32 v162, v162, v111
	v_add_f32_e32 v162, v162, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v162, v162, v113
	;;#ASMSTART
	v_fma_f32 v160, v160, v161, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v161
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v161, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v161, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
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
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_40
	ds_write_b128 v151, v[220:223]
	ds_write_b128 v151, v[224:227] offset:6144
.LBB0_40:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_42
	v_add_u32_e32 v98, s71, v150
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[220:223], v[98:99], off
	global_load_dwordx4 v[224:227], v[98:99], off offset:256
.LBB0_42:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v161, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v161
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_add_u32_e32 v162, 0x1810, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_add_u32_e32 v162, 0x1820, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_add_u32_e32 v162, 0x1830, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v162, 0x1840, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v162, 0x1850, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v162, 0x1860, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v162, 0x1870, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v162, 0x1880, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v162, 0x1890, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v162, 0x18a0, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v161, 0x18b0, v161
	v_lshrrev_b32_e32 v162, 3, v161
	v_and_b32_e32 v162, 56, v162
	v_xor_b32_e32 v161, v162, v161
	v_lshlrev_b32_e32 v161, 1, v161
	ds_read_b128 v[228:231], v161
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v161, v98, v99
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v161, v161, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v161, v161, v102, v103
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v161, v161, v104, v105
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v161, v161, v106, v107
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v161, v161, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v161, v161, v110, v111
	v_max3_f32 v161, v161, v112, v113
	ds_bpermute_b32 v162, v154, v161
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v162, v162, v162
	v_max_f32_e32 v162, v161, v162
	;;#ASMSTART
	v_add_f32 v161, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v162, v161
	v_mov_b32_e32 v161, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_44
	;;#ASMSTART
	v_add_f32 v132, v162, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v161, v114
	v_mov_b32_e32 v114, v132
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
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, 0, v98
	v_add_f32_e32 v162, v162, v99
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
	v_add_f32_e32 v162, v162, v100
	v_add_f32_e32 v162, v162, v101
	v_add_f32_e32 v162, v162, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v103
	v_add_f32_e32 v162, v162, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v105
	v_add_f32_e32 v162, v162, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v107
	v_add_f32_e32 v162, v162, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v109
	v_add_f32_e32 v162, v162, v110
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
	v_add_f32_e32 v162, v162, v111
	v_add_f32_e32 v162, v162, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v162, v162, v113
	;;#ASMSTART
	v_fma_f32 v160, v160, v161, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v161
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v161, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v161, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s48, s65, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s48
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_46
	ds_write_b128 v151, v[212:215] offset:12288
	ds_write_b128 v151, v[216:219] offset:18432
.LBB0_46:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_48
	v_add_u32_e32 v98, s71, v152
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[212:215], v[98:99], off
	global_load_dwordx4 v[216:219], v[98:99], off offset:256
.LBB0_48:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v161, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v162, 56, v98
	v_xor_b32_e32 v98, v162, v161
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_or_b32_e32 v163, 16, v161
	v_xor_b32_e32 v163, v163, v162
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_or_b32_e32 v163, 32, v161
	v_xor_b32_e32 v163, v163, v162
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_or_b32_e32 v163, 48, v161
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v162, 64, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v162, 0x50, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v162, 0x60, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v162, 0x70, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v162, 0x80, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v162, 0x90, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v162, 0xa0, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v161, 0xb0, v161
	v_lshrrev_b32_e32 v162, 3, v161
	v_and_b32_e32 v162, 56, v162
	v_xor_b32_e32 v161, v162, v161
	v_lshlrev_b32_e32 v161, 1, v161
	ds_read_b128 v[228:231], v161
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v161, v98, v99
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v161, v161, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v161, v161, v102, v103
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v161, v161, v104, v105
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v161, v161, v106, v107
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v161, v161, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v161, v161, v110, v111
	v_max3_f32 v161, v161, v112, v113
	ds_bpermute_b32 v162, v154, v161
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v162, v162, v162
	v_max_f32_e32 v162, v161, v162
	;;#ASMSTART
	v_add_f32 v161, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v162, v161
	v_mov_b32_e32 v161, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_50
	;;#ASMSTART
	v_add_f32 v132, v162, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v161, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_50:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, 0, v98
	v_add_f32_e32 v162, v162, v99
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
	v_add_f32_e32 v162, v162, v100
	v_add_f32_e32 v162, v162, v101
	v_add_f32_e32 v162, v162, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v103
	v_add_f32_e32 v162, v162, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v105
	v_add_f32_e32 v162, v162, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v107
	v_add_f32_e32 v162, v162, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v109
	v_add_f32_e32 v162, v162, v110
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
	v_add_f32_e32 v162, v162, v111
	v_add_f32_e32 v162, v162, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v162, v162, v113
	;;#ASMSTART
	v_fma_f32 v160, v160, v161, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v161
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v161, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v161, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s65, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s65
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	v_mov_b32_e32 v98, s68
	buffer_load_dword v98, v98, s[4:7], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s65, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s56, v99
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_52
	ds_write_b128 v151, v[220:223]
	ds_write_b128 v151, v[224:227] offset:6144
.LBB0_52:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_54
	v_add_u32_e32 v98, s71, v153
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[220:223], v[98:99], off
	global_load_dwordx4 v[224:227], v[98:99], off offset:256
.LBB0_54:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v161, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v161
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_add_u32_e32 v162, 0x1810, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_add_u32_e32 v162, 0x1820, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_add_u32_e32 v162, 0x1830, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v162, 0x1840, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v162, 0x1850, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v162, 0x1860, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v162, 0x1870, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v162, 0x1880, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v162, 0x1890, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v162, 0x18a0, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v161, 0x18b0, v161
	v_lshrrev_b32_e32 v162, 3, v161
	v_and_b32_e32 v162, 56, v162
	v_xor_b32_e32 v161, v162, v161
	v_lshlrev_b32_e32 v161, 1, v161
	ds_read_b128 v[228:231], v161
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v161, v98, v99
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v161, v161, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v161, v161, v102, v103
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v161, v161, v104, v105
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v161, v161, v106, v107
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v161, v161, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v161, v161, v110, v111
	v_max3_f32 v161, v161, v112, v113
	ds_bpermute_b32 v162, v154, v161
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v162, v162, v162
	v_max_f32_e32 v162, v161, v162
	;;#ASMSTART
	v_add_f32 v161, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v162, v161
	v_mov_b32_e32 v161, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_56
	;;#ASMSTART
	v_add_f32 v132, v162, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v161, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_56:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, 0, v98
	v_add_f32_e32 v162, v162, v99
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
	v_add_f32_e32 v162, v162, v100
	v_add_f32_e32 v162, v162, v101
	v_add_f32_e32 v162, v162, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v103
	v_add_f32_e32 v162, v162, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v105
	v_add_f32_e32 v162, v162, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v107
	v_add_f32_e32 v162, v162, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v109
	v_add_f32_e32 v162, v162, v110
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
	v_add_f32_e32 v162, v162, v111
	v_add_f32_e32 v162, v162, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v162, v162, v113
	;;#ASMSTART
	v_fma_f32 v160, v160, v161, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v161
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v161, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v161, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s71, s64, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s71, s71, s0
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s71
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_58
	ds_write_b128 v151, v[212:215] offset:12288
	ds_write_b128 v151, v[216:219] offset:18432
.LBB0_58:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mul_i32 s64, s69, 0x3000
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_60
	v_add_u32_e32 v98, s64, v149
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[212:215], v[98:99], off
	global_load_dwordx4 v[216:219], v[98:99], off offset:256
.LBB0_60:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v161, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v162, 56, v98
	v_xor_b32_e32 v98, v162, v161
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_or_b32_e32 v163, 16, v161
	v_xor_b32_e32 v163, v163, v162
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_or_b32_e32 v163, 32, v161
	v_xor_b32_e32 v163, v163, v162
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_or_b32_e32 v163, 48, v161
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v162, 64, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v162, 0x50, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v162, 0x60, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v162, 0x70, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v162, 0x80, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v162, 0x90, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v162, 0xa0, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v161, 0xb0, v161
	v_lshrrev_b32_e32 v162, 3, v161
	v_and_b32_e32 v162, 56, v162
	v_xor_b32_e32 v161, v162, v161
	v_lshlrev_b32_e32 v161, 1, v161
	ds_read_b128 v[228:231], v161
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v161, v98, v99
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v161, v161, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v161, v161, v102, v103
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v161, v161, v104, v105
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v161, v161, v106, v107
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v161, v161, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v161, v161, v110, v111
	v_max3_f32 v161, v161, v112, v113
	ds_bpermute_b32 v162, v154, v161
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v162, v162, v162
	v_max_f32_e32 v162, v161, v162
	;;#ASMSTART
	v_add_f32 v161, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v162, v161
	v_mov_b32_e32 v161, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_62
	;;#ASMSTART
	v_add_f32 v132, v162, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v161, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_62:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, 0, v98
	v_add_f32_e32 v162, v162, v99
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
	v_add_f32_e32 v162, v162, v100
	v_add_f32_e32 v162, v162, v101
	v_add_f32_e32 v162, v162, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v103
	v_add_f32_e32 v162, v162, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v105
	v_add_f32_e32 v162, v162, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v107
	v_add_f32_e32 v162, v162, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v109
	v_add_f32_e32 v162, v162, v110
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
	v_add_f32_e32 v162, v162, v111
	v_add_f32_e32 v162, v162, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v162, v162, v113
	;;#ASMSTART
	v_fma_f32 v160, v160, v161, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v161
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v161, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v161, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s71, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s71
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_64
	ds_write_b128 v151, v[220:223]
	ds_write_b128 v151, v[224:227] offset:6144
.LBB0_64:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_66
	v_add_u32_e32 v98, s64, v150
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[220:223], v[98:99], off
	global_load_dwordx4 v[224:227], v[98:99], off offset:256
.LBB0_66:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v161, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v161
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_add_u32_e32 v162, 0x1810, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_add_u32_e32 v162, 0x1820, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_add_u32_e32 v162, 0x1830, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v162, 0x1840, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v162, 0x1850, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v162, 0x1860, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v162, 0x1870, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v162, 0x1880, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v162, 0x1890, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v162, 0x18a0, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v161, 0x18b0, v161
	v_lshrrev_b32_e32 v162, 3, v161
	v_and_b32_e32 v162, 56, v162
	v_xor_b32_e32 v161, v162, v161
	v_lshlrev_b32_e32 v161, 1, v161
	ds_read_b128 v[228:231], v161
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v161, v98, v99
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v161, v161, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v161, v161, v102, v103
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v161, v161, v104, v105
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v161, v161, v106, v107
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v161, v161, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v161, v161, v110, v111
	v_max3_f32 v161, v161, v112, v113
	ds_bpermute_b32 v162, v154, v161
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v162, v162, v162
	v_max_f32_e32 v162, v161, v162
	;;#ASMSTART
	v_add_f32 v161, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v162, v161
	v_mov_b32_e32 v161, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_68
	;;#ASMSTART
	v_add_f32 v132, v162, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v161, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_68:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, 0, v98
	v_add_f32_e32 v162, v162, v99
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
	v_add_f32_e32 v162, v162, v100
	v_add_f32_e32 v162, v162, v101
	v_add_f32_e32 v162, v162, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v103
	v_add_f32_e32 v162, v162, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v105
	v_add_f32_e32 v162, v162, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v107
	v_add_f32_e32 v162, v162, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v162, v162, v109
	v_add_f32_e32 v162, v162, v110
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
	v_add_f32_e32 v162, v162, v111
	v_add_f32_e32 v162, v162, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v162, v162, v113
	;;#ASMSTART
	v_fma_f32 v160, v160, v161, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v161
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v161
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v161, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v161, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s48, s71, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s48
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_70
	ds_write_b128 v151, v[212:215] offset:12288
	ds_write_b128 v151, v[216:219] offset:18432
.LBB0_70:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_72
	v_add_u32_e32 v98, s64, v152
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[212:215], v[98:99], off
	global_load_dwordx4 v[216:219], v[98:99], off offset:256
.LBB0_72:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v161, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v162, 56, v98
	v_xor_b32_e32 v98, v162, v161
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_or_b32_e32 v163, 16, v161
	v_xor_b32_e32 v163, v163, v162
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_or_b32_e32 v163, 32, v161
	v_xor_b32_e32 v163, v163, v162
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_or_b32_e32 v163, 48, v161
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v162, 64, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v162, 0x50, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v162, 0x60, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v162, 0x70, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v162, 0x80, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v162, 0x90, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v162, 0xa0, v161
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v161, 0xb0, v161
	v_lshrrev_b32_e32 v162, 3, v161
	v_and_b32_e32 v162, 56, v162
	v_xor_b32_e32 v161, v162, v161
	v_lshlrev_b32_e32 v161, 1, v161
	ds_read_b128 v[228:231], v161
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_nop 7
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_max_f32_e32 v161, v98, v99
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_max3_f32 v161, v161, v100, v101
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_max3_f32 v161, v161, v102, v103
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_max3_f32 v161, v161, v104, v105
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_max3_f32 v161, v161, v106, v107
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_max3_f32 v161, v161, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v161, v161, v110, v111
	v_max3_f32 v161, v161, v112, v113
	ds_bpermute_b32 v162, v154, v161
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v162, v162, v162
	v_max_f32_e32 v161, v161, v162
	;;#ASMSTART
	v_add_f32 v162, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v161, v162
	v_mov_b32_e32 v162, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_25
	;;#ASMSTART
	v_add_f32 v132, v161, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v162, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_74:
	s_mov_b32 s64, s70
	s_mov_b32 s71, s69
	s_branch .LBB0_76
.LBB0_75:
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
	v_mov_b32_e32 v161, v2
.LBB0_76:
	s_add_i32 s46, s43, s66
	s_add_i32 s48, s46, 0x7f
	s_ashr_i32 s46, s48, 31
	s_lshr_b32 s46, s46, 25
	s_add_i32 s46, s48, s46
	s_ashr_i32 s66, s46, 7
	s_and_b32 s46, s46, 0xffffff80
	s_cmp_lg_u32 s48, s46
	s_cselect_b64 s[46:47], -1, 0
	s_cmp_lt_i32 s48, 0
	s_cselect_b64 s[48:49], -1, 0
	s_and_b64 s[46:47], s[48:49], s[46:47]
	s_subb_u32 s46, s66, 0
	s_min_i32 s46, s46, s51
	s_cmp_ge_i32 s44, s46
	s_cbranch_scc1 .LBB0_130
	s_lshl_b32 s48, s67, 8
	s_ashr_i32 s47, s46, 31
	v_mov_b32_e32 v117, v116
	v_mov_b32_e32 v118, v116
	v_mov_b32_e32 v119, v116
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
	v_or_b32_e32 v160, s50, v1
	s_or_b32 s66, s48, 0xf7
	s_lshl3_add_u32 s67, s67, 20
	s_branch .LBB0_80
.LBB0_78:
	s_or_b64 exec, exec, s[50:51]
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
	v_fma_f32 v161, v161, v162, v132
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v162
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
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v132, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s71, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s71
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[132:135], v[106:107], off offset:512
	global_load_dwordx4 v[136:139], v[106:107], off offset:1024
	global_load_dwordx4 v[140:143], v[106:107], off offset:1536
	global_load_dwordx4 v[144:147], v[106:107], off offset:2048
	global_load_dwordx4 v[228:231], v[106:107], off offset:2560
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
	v_mfma_f32_32x32x8_bf16 v[82:97], v[228:229], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
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
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[230:231], v[102:103], v[82:97]
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
.LBB0_79:
	s_and_b64 s[48:49], s[48:49], exec
	s_cselect_b32 s71, s1, s64
	s_cselect_b32 s64, s65, s1
	s_cselect_b32 s1, s68, s65
	s_add_u32 s44, s44, 2
	s_addc_u32 s45, s45, 0
	v_mov_b64_e32 v[98:99], s[46:47]
	v_cmp_lt_i64_e32 vcc, s[44:45], v[98:99]
	s_addk_i32 s66, 0x100
	s_add_i32 s67, s67, 8
	s_mov_b32 s65, s69
	s_cbranch_vccz .LBB0_130
.LBB0_80:
	s_add_i32 s48, s67, -4
	v_mov_b32_e32 v98, s48
	buffer_load_dword v98, v98, s[4:7], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s68, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s56, v99
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_82
	ds_write_b128 v151, v[220:223]
	ds_write_b128 v151, v[224:227] offset:6144
.LBB0_82:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_84
	s_mul_i32 s50, s71, 0x3000
	v_add_u32_e32 v98, s50, v153
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[220:223], v[98:99], off
	global_load_dwordx4 v[224:227], v[98:99], off offset:256
.LBB0_84:
	s_or_b64 exec, exec, s[48:49]
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
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[166:167], v[98:113]
	v_add_u32_e32 v133, 0x1810, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[170:171], v[98:113]
	v_add_u32_e32 v133, 0x1820, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[174:175], v[98:113]
	v_add_u32_e32 v133, 0x1830, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[178:179], v[98:113]
	v_add_u32_e32 v133, 0x1840, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[182:183], v[98:113]
	v_add_u32_e32 v133, 0x1850, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[186:187], v[98:113]
	v_add_u32_e32 v133, 0x1860, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[190:191], v[98:113]
	v_add_u32_e32 v133, 0x1870, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[194:195], v[98:113]
	v_add_u32_e32 v133, 0x1880, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[198:199], v[98:113]
	v_add_u32_e32 v133, 0x1890, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[202:203], v[98:113]
	v_add_u32_e32 v133, 0x18a0, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	v_add_u32_e32 v132, 0x18b0, v132
	v_lshrrev_b32_e32 v133, 3, v132
	v_and_b32_e32 v133, 56, v133
	v_xor_b32_e32 v132, v133, v132
	v_lshlrev_b32_e32 v132, 1, v132
	ds_read_b128 v[132:135], v132
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[132:133], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v132, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v133, 2, v132
	v_and_b32_e32 v133, 8, v133
	v_lshrrev_b32_e32 v132, 1, v132
	v_and_or_b32 v132, v132, s55, v160
	v_add_u32_e32 v141, s66, v133
	v_add_u32_e32 v140, s11, v132
	v_add_u32_e32 v132, 0xffffff09, v141
	v_cmp_lt_i32_e32 vcc, v132, v140
	s_nop 1
	v_cndmask_b32_e32 v133, v158, v99, vcc
	v_cmp_le_i32_e32 vcc, v132, v140
	v_add_u32_e32 v99, 0xffffff20, v141
	s_nop 0
	v_cndmask_b32_e32 v132, v158, v98, vcc
	v_add_u32_e32 v98, 0xffffff0b, v141
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff0c, v141
	s_nop 0
	v_cndmask_b32_e32 v134, v158, v100, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff0d, v141
	s_nop 0
	v_cndmask_b32_e32 v135, v158, v101, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff0e, v141
	s_nop 0
	v_cndmask_b32_e32 v136, v158, v102, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff0f, v141
	s_nop 0
	v_cndmask_b32_e32 v137, v158, v103, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff10, v141
	s_nop 0
	v_cndmask_b32_e32 v138, v158, v104, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff19, v141
	s_nop 0
	v_cndmask_b32_e32 v139, v158, v105, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff1a, v141
	s_nop 0
	v_cndmask_b32_e32 v104, v158, v106, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff1b, v141
	s_nop 0
	v_cndmask_b32_e32 v105, v158, v107, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff1c, v141
	v_pk_mul_f32 v[106:107], v[122:123], v[138:139]
	v_cndmask_b32_e32 v102, v158, v108, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff1d, v141
	v_pk_mul_f32 v[104:105], v[124:125], v[104:105]
	v_cndmask_b32_e32 v103, v158, v109, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff1e, v141
	v_pk_mul_f32 v[108:109], v[120:121], v[136:137]
	v_cndmask_b32_e32 v100, v158, v110, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_add_u32_e32 v98, 0xffffff1f, v141
	v_pk_mul_f32 v[102:103], v[126:127], v[102:103]
	v_cndmask_b32_e32 v101, v158, v111, vcc
	v_cmp_le_i32_e32 vcc, v98, v140
	v_pk_mul_f32 v[110:111], v[118:119], v[134:135]
	v_pk_mul_f32 v[100:101], v[128:129], v[100:101]
	v_cndmask_b32_e32 v98, v158, v112, vcc
	v_cmp_le_i32_e32 vcc, v99, v140
	s_nop 1
	v_cndmask_b32_e32 v99, v158, v113, vcc
	v_pk_mul_f32 v[112:113], v[116:117], v[132:133]
	v_pk_mul_f32 v[98:99], v[130:131], v[98:99]
	v_max_f32_e32 v132, v112, v113
	v_max3_f32 v132, v132, v110, v111
	v_max3_f32 v132, v132, v108, v109
	v_max3_f32 v132, v132, v106, v107
	v_max3_f32 v132, v132, v104, v105
	v_max3_f32 v132, v132, v102, v103
	v_max3_f32 v132, v132, v100, v101
	v_max3_f32 v132, v132, v98, v99
	ds_bpermute_b32 v133, v154, v132
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v133, v133, v133
	v_max_f32_e32 v133, v132, v133
	;;#ASMSTART
	v_add_f32 v132, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v133, v132
	v_mov_b32_e32 v132, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_86
	;;#ASMSTART
	v_add_f32 v133, v133, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v133
	v_exp_f32_e32 v132, v114
	v_mov_b32_e32 v114, v133
.LBB0_86:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[112:113], v[112:113], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[110:111], v[110:111], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[106:107], v[106:107], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v110
	v_add_f32_e32 v133, v133, v111
	v_add_f32_e32 v133, v133, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v109
	v_add_f32_e32 v133, v133, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v107
	v_add_f32_e32 v133, v133, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	v_pk_add_f32 v[100:101], v[100:101], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v133, v133, v105
	v_add_f32_e32 v133, v133, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v100, v100
	;;#ASMEND
	v_pk_add_f32 v[98:99], v[98:99], v[114:115] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
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
	v_fma_f32 v161, v161, v132, v133
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
	v_perm_b32 v100, v100, v109, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v102, v107, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v103, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v134, v135, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v98, v133, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v99, v132, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s50, s71, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s50, s50, s0
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s50
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[132:135], v[106:107], off offset:512
	global_load_dwordx4 v[136:139], v[106:107], off offset:1024
	global_load_dwordx4 v[140:143], v[106:107], off offset:1536
	global_load_dwordx4 v[144:147], v[106:107], off offset:2048
	global_load_dwordx4 v[228:231], v[106:107], off offset:2560
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
	v_mfma_f32_32x32x8_bf16 v[82:97], v[228:229], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
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
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[230:231], v[102:103], v[82:97]
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
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_88
	ds_write_b128 v151, v[212:215] offset:12288
	ds_write_b128 v151, v[216:219] offset:18432
.LBB0_88:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mul_i32 s70, s64, 0x3000
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_90
	v_add_u32_e32 v98, s70, v149
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[212:215], v[98:99], off
	global_load_dwordx4 v[216:219], v[98:99], off offset:256
.LBB0_90:
	s_or_b64 exec, exec, s[48:49]
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
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[166:167], v[98:113]
	v_or_b32_e32 v134, 16, v132
	v_xor_b32_e32 v134, v134, v133
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[170:171], v[98:113]
	v_or_b32_e32 v134, 32, v132
	v_xor_b32_e32 v134, v134, v133
	v_lshlrev_b32_e32 v134, 1, v134
	ds_read_b128 v[134:137], v134
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[174:175], v[98:113]
	v_or_b32_e32 v134, 48, v132
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[178:179], v[98:113]
	v_add_u32_e32 v133, 64, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[182:183], v[98:113]
	v_add_u32_e32 v133, 0x50, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[186:187], v[98:113]
	v_add_u32_e32 v133, 0x60, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[190:191], v[98:113]
	v_add_u32_e32 v133, 0x70, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[194:195], v[98:113]
	v_add_u32_e32 v133, 0x80, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[198:199], v[98:113]
	v_add_u32_e32 v133, 0x90, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[202:203], v[98:113]
	v_add_u32_e32 v133, 0xa0, v132
	v_lshrrev_b32_e32 v134, 3, v133
	v_and_b32_e32 v134, 56, v134
	v_xor_b32_e32 v133, v134, v133
	v_lshlrev_b32_e32 v133, 1, v133
	ds_read_b128 v[134:137], v133
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[136:137], v[206:207], v[98:113]
	v_add_u32_e32 v132, 0xb0, v132
	v_lshrrev_b32_e32 v133, 3, v132
	v_and_b32_e32 v133, 56, v133
	v_xor_b32_e32 v132, v133, v132
	v_lshlrev_b32_e32 v132, 1, v132
	ds_read_b128 v[132:135], v132
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[132:133], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[134:135], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v132, v0
	;;#ASMEND
	v_mov_b32_e32 v162, 1.0
	v_lshrrev_b32_e32 v133, 2, v132
	v_and_b32_e32 v133, 8, v133
	v_lshrrev_b32_e32 v132, 1, v132
	v_and_or_b32 v132, v132, s55, v160
	v_add_u32_e32 v133, s66, v133
	v_add_u32_e32 v132, s11, v132
	v_add_u32_e32 v134, 0xffffff29, v133
	v_cmp_lt_i32_e32 vcc, v134, v132
	v_mov_b32_e32 v135, v114
	v_mov_b32_e32 v136, v114
	v_cndmask_b32_e32 v99, v158, v99, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff2b, v133
	v_mov_b32_e32 v137, v114
	v_cndmask_b32_e32 v98, v158, v98, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff2c, v133
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v158, v100, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff2d, v133
	v_mov_b32_e32 v138, v114
	v_cndmask_b32_e32 v101, v158, v101, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff2e, v133
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_cndmask_b32_e32 v102, v158, v102, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff2f, v133
	v_mov_b32_e32 v139, v114
	v_cndmask_b32_e32 v103, v158, v103, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff30, v133
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_cndmask_b32_e32 v104, v158, v104, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff39, v133
	v_mov_b32_e32 v140, v114
	v_cndmask_b32_e32 v105, v158, v105, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff3a, v133
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_cndmask_b32_e32 v106, v158, v106, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff3b, v133
	v_mov_b32_e32 v141, v114
	v_cndmask_b32_e32 v107, v158, v107, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff3c, v133
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_cndmask_b32_e32 v108, v158, v108, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff3d, v133
	v_mov_b32_e32 v142, v114
	v_cndmask_b32_e32 v109, v158, v109, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff3e, v133
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_cndmask_b32_e32 v110, v158, v110, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_add_u32_e32 v134, 0xffffff3f, v133
	v_add_u32_e32 v133, 0xffffff40, v133
	v_cndmask_b32_e32 v111, v158, v111, vcc
	v_cmp_le_i32_e32 vcc, v134, v132
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	v_mov_b32_e32 v134, v114
	v_cndmask_b32_e32 v112, v158, v112, vcc
	v_cmp_le_i32_e32 vcc, v133, v132
	v_max_f32_e32 v132, v98, v99
	v_max3_f32 v132, v132, v100, v101
	v_max3_f32 v132, v132, v102, v103
	v_max3_f32 v132, v132, v104, v105
	v_max3_f32 v132, v132, v106, v107
	v_cndmask_b32_e32 v113, v158, v113, vcc
	v_max3_f32 v132, v132, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v132, v132, v110, v111
	v_max3_f32 v132, v132, v112, v113
	ds_bpermute_b32 v133, v154, v132
	v_mov_b32_e32 v143, v114
	v_mov_b32_e32 v144, v114
	v_mov_b32_e32 v145, v114
	v_mov_b32_e32 v146, v114
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v133, v133, v133
	v_max_f32_e32 v163, v132, v133
	;;#ASMSTART
	v_add_f32 v132, v114, v156
	;;#ASMEND
	v_mov_b32_e32 v133, v114
	v_cmp_gt_f32_e32 vcc, v163, v132
	v_mov_b32_e32 v132, v114
	v_mov_b32_e32 v147, v114
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_92
	;;#ASMSTART
	v_add_f32 v132, v163, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v162, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_92:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, 0, v98
	v_add_f32_e32 v163, v163, v99
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
	v_add_f32_e32 v163, v163, v100
	v_add_f32_e32 v163, v163, v101
	v_add_f32_e32 v163, v163, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v103
	v_add_f32_e32 v163, v163, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v105
	v_add_f32_e32 v163, v163, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v107
	v_add_f32_e32 v163, v163, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v109
	v_add_f32_e32 v163, v163, v110
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
	v_add_f32_e32 v163, v163, v111
	v_add_f32_e32 v163, v163, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v163, v163, v113
	;;#ASMSTART
	v_fma_f32 v161, v161, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v162
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v162, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v162, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
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
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_94
	ds_write_b128 v151, v[220:223]
	ds_write_b128 v151, v[224:227] offset:6144
.LBB0_94:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_96
	v_add_u32_e32 v98, s70, v150
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[220:223], v[98:99], off
	global_load_dwordx4 v[224:227], v[98:99], off offset:256
.LBB0_96:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v162, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v162
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_add_u32_e32 v163, 0x1810, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_add_u32_e32 v163, 0x1820, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_add_u32_e32 v163, 0x1830, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v163, 0x1840, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v163, 0x1850, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v163, 0x1860, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v163, 0x1870, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v163, 0x1880, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v163, 0x1890, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v163, 0x18a0, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v162, 0x18b0, v162
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v162, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v163, 2, v162
	v_and_b32_e32 v163, 8, v163
	v_lshrrev_b32_e32 v162, 1, v162
	v_and_or_b32 v162, v162, s55, v160
	v_add_u32_e32 v163, s66, v163
	v_add_u32_e32 v162, s11, v162
	v_add_u32_e32 v228, 0xffffff49, v163
	v_cmp_lt_i32_e32 vcc, v228, v162
	s_nop 1
	v_cndmask_b32_e32 v99, v158, v99, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff4b, v163
	s_nop 0
	v_cndmask_b32_e32 v98, v158, v98, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff4c, v163
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v158, v100, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff4d, v163
	s_nop 0
	v_cndmask_b32_e32 v101, v158, v101, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff4e, v163
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_cndmask_b32_e32 v102, v158, v102, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff4f, v163
	s_nop 0
	v_cndmask_b32_e32 v103, v158, v103, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff50, v163
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_cndmask_b32_e32 v104, v158, v104, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff59, v163
	s_nop 0
	v_cndmask_b32_e32 v105, v158, v105, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff5a, v163
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_cndmask_b32_e32 v106, v158, v106, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff5b, v163
	s_nop 0
	v_cndmask_b32_e32 v107, v158, v107, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff5c, v163
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_cndmask_b32_e32 v108, v158, v108, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff5d, v163
	s_nop 0
	v_cndmask_b32_e32 v109, v158, v109, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff5e, v163
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_cndmask_b32_e32 v110, v158, v110, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff5f, v163
	v_add_u32_e32 v163, 0xffffff60, v163
	v_cndmask_b32_e32 v111, v158, v111, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v158, v112, vcc
	v_cmp_le_i32_e32 vcc, v163, v162
	v_max_f32_e32 v162, v98, v99
	v_max3_f32 v162, v162, v100, v101
	v_max3_f32 v162, v162, v102, v103
	v_max3_f32 v162, v162, v104, v105
	v_max3_f32 v162, v162, v106, v107
	v_cndmask_b32_e32 v113, v158, v113, vcc
	v_max3_f32 v162, v162, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v162, v162, v110, v111
	v_max3_f32 v162, v162, v112, v113
	ds_bpermute_b32 v163, v154, v162
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v163, v163, v163
	v_max_f32_e32 v163, v162, v163
	;;#ASMSTART
	v_add_f32 v162, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v163, v162
	v_mov_b32_e32 v162, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_98
	;;#ASMSTART
	v_add_f32 v132, v163, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v162, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_98:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, 0, v98
	v_add_f32_e32 v163, v163, v99
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
	v_add_f32_e32 v163, v163, v100
	v_add_f32_e32 v163, v163, v101
	v_add_f32_e32 v163, v163, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v103
	v_add_f32_e32 v163, v163, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v105
	v_add_f32_e32 v163, v163, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v107
	v_add_f32_e32 v163, v163, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v109
	v_add_f32_e32 v163, v163, v110
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
	v_add_f32_e32 v163, v163, v111
	v_add_f32_e32 v163, v163, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v163, v163, v113
	;;#ASMSTART
	v_fma_f32 v161, v161, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v162
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v162, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v162, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s48, s50, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s48
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_100
	ds_write_b128 v151, v[212:215] offset:12288
	ds_write_b128 v151, v[216:219] offset:18432
.LBB0_100:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_102
	v_add_u32_e32 v98, s70, v152
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[212:215], v[98:99], off
	global_load_dwordx4 v[216:219], v[98:99], off offset:256
.LBB0_102:
	s_or_b64 exec, exec, s[48:49]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v162, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v163, 56, v98
	v_xor_b32_e32 v98, v163, v162
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_or_b32_e32 v228, 16, v162
	v_xor_b32_e32 v228, v228, v163
	v_lshlrev_b32_e32 v228, 1, v228
	ds_read_b128 v[228:231], v228
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_or_b32_e32 v228, 32, v162
	v_xor_b32_e32 v228, v228, v163
	v_lshlrev_b32_e32 v228, 1, v228
	ds_read_b128 v[228:231], v228
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_or_b32_e32 v228, 48, v162
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v163, 64, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v163, 0x50, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v163, 0x60, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v163, 0x70, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v163, 0x80, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v163, 0x90, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v163, 0xa0, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v162, 0xb0, v162
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v162, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v163, 2, v162
	v_and_b32_e32 v163, 8, v163
	v_lshrrev_b32_e32 v162, 1, v162
	v_and_or_b32 v162, v162, s55, v160
	v_add_u32_e32 v163, s66, v163
	v_add_u32_e32 v162, s11, v162
	v_add_u32_e32 v228, 0xffffff69, v163
	v_cmp_lt_i32_e32 vcc, v228, v162
	s_nop 1
	v_cndmask_b32_e32 v99, v158, v99, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff6b, v163
	s_nop 0
	v_cndmask_b32_e32 v98, v158, v98, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff6c, v163
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v158, v100, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff6d, v163
	s_nop 0
	v_cndmask_b32_e32 v101, v158, v101, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff6e, v163
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_cndmask_b32_e32 v102, v158, v102, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff6f, v163
	s_nop 0
	v_cndmask_b32_e32 v103, v158, v103, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff70, v163
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_cndmask_b32_e32 v104, v158, v104, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff79, v163
	s_nop 0
	v_cndmask_b32_e32 v105, v158, v105, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff7a, v163
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_cndmask_b32_e32 v106, v158, v106, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff7b, v163
	s_nop 0
	v_cndmask_b32_e32 v107, v158, v107, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff7c, v163
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_cndmask_b32_e32 v108, v158, v108, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff7d, v163
	s_nop 0
	v_cndmask_b32_e32 v109, v158, v109, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff7e, v163
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_cndmask_b32_e32 v110, v158, v110, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff7f, v163
	v_add_u32_e32 v163, 0xffffff80, v163
	v_cndmask_b32_e32 v111, v158, v111, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v158, v112, vcc
	v_cmp_le_i32_e32 vcc, v163, v162
	v_max_f32_e32 v162, v98, v99
	v_max3_f32 v162, v162, v100, v101
	v_max3_f32 v162, v162, v102, v103
	v_max3_f32 v162, v162, v104, v105
	v_max3_f32 v162, v162, v106, v107
	v_cndmask_b32_e32 v113, v158, v113, vcc
	v_max3_f32 v162, v162, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v162, v162, v110, v111
	v_max3_f32 v162, v162, v112, v113
	ds_bpermute_b32 v163, v154, v162
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v163, v163, v163
	v_max_f32_e32 v163, v162, v163
	;;#ASMSTART
	v_add_f32 v162, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v163, v162
	v_mov_b32_e32 v162, 1.0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_104
	;;#ASMSTART
	v_add_f32 v132, v163, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v162, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_104:
	s_or_b64 exec, exec, s[48:49]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, 0, v98
	v_add_f32_e32 v163, v163, v99
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
	v_add_f32_e32 v163, v163, v100
	v_add_f32_e32 v163, v163, v101
	v_add_f32_e32 v163, v163, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v103
	v_add_f32_e32 v163, v163, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v105
	v_add_f32_e32 v163, v163, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v107
	v_add_f32_e32 v163, v163, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v109
	v_add_f32_e32 v163, v163, v110
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
	v_add_f32_e32 v163, v163, v111
	v_add_f32_e32 v163, v163, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v163, v163, v113
	;;#ASMSTART
	v_fma_f32 v161, v161, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v162
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v162, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v162, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s50, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s50
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	s_add_i32 s50, s44, 1
	s_cmp_gt_i32 s46, s50
	s_cselect_b64 s[48:49], -1, 0
	s_cmp_le_i32 s46, s50
	s_cbranch_scc1 .LBB0_129
	v_mov_b32_e32 v98, s67
	buffer_load_dword v98, v98, s[4:7], 0 offen
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s69, v98
	v_and_b32_e32 v99, 0x180, v99
	v_cmp_ne_u32_e32 vcc, s56, v99
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_107
	ds_write_b128 v151, v[220:223]
	ds_write_b128 v151, v[224:227] offset:6144
.LBB0_107:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_109
	v_add_u32_e32 v98, s70, v153
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[220:223], v[98:99], off
	global_load_dwordx4 v[224:227], v[98:99], off offset:256
.LBB0_109:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v162, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v162
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_add_u32_e32 v163, 0x1810, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_add_u32_e32 v163, 0x1820, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_add_u32_e32 v163, 0x1830, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v163, 0x1840, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v163, 0x1850, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v163, 0x1860, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v163, 0x1870, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v163, 0x1880, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v163, 0x1890, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v163, 0x18a0, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v162, 0x18b0, v162
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v162, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v163, 2, v162
	v_and_b32_e32 v163, 8, v163
	v_lshrrev_b32_e32 v162, 1, v162
	v_and_or_b32 v162, v162, s55, v160
	v_add_u32_e32 v163, s66, v163
	v_add_u32_e32 v162, s11, v162
	v_add_u32_e32 v228, 0xffffff89, v163
	v_cmp_lt_i32_e32 vcc, v228, v162
	s_nop 1
	v_cndmask_b32_e32 v99, v158, v99, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff8b, v163
	s_nop 0
	v_cndmask_b32_e32 v98, v158, v98, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff8c, v163
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v158, v100, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff8d, v163
	s_nop 0
	v_cndmask_b32_e32 v101, v158, v101, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff8e, v163
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_cndmask_b32_e32 v102, v158, v102, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff8f, v163
	s_nop 0
	v_cndmask_b32_e32 v103, v158, v103, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff90, v163
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_cndmask_b32_e32 v104, v158, v104, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff99, v163
	s_nop 0
	v_cndmask_b32_e32 v105, v158, v105, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff9a, v163
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_cndmask_b32_e32 v106, v158, v106, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff9b, v163
	s_nop 0
	v_cndmask_b32_e32 v107, v158, v107, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff9c, v163
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_cndmask_b32_e32 v108, v158, v108, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff9d, v163
	s_nop 0
	v_cndmask_b32_e32 v109, v158, v109, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff9e, v163
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_cndmask_b32_e32 v110, v158, v110, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffff9f, v163
	v_add_u32_e32 v163, 0xffffffa0, v163
	v_cndmask_b32_e32 v111, v158, v111, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v158, v112, vcc
	v_cmp_le_i32_e32 vcc, v163, v162
	v_max_f32_e32 v162, v98, v99
	v_max3_f32 v162, v162, v100, v101
	v_max3_f32 v162, v162, v102, v103
	v_max3_f32 v162, v162, v104, v105
	v_max3_f32 v162, v162, v106, v107
	v_cndmask_b32_e32 v113, v158, v113, vcc
	v_max3_f32 v162, v162, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v162, v162, v110, v111
	v_max3_f32 v162, v162, v112, v113
	ds_bpermute_b32 v163, v154, v162
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v163, v163, v163
	v_max_f32_e32 v163, v162, v163
	;;#ASMSTART
	v_add_f32 v162, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v163, v162
	v_mov_b32_e32 v162, 1.0
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_111
	;;#ASMSTART
	v_add_f32 v132, v163, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v162, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_111:
	s_or_b64 exec, exec, s[50:51]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, 0, v98
	v_add_f32_e32 v163, v163, v99
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
	v_add_f32_e32 v163, v163, v100
	v_add_f32_e32 v163, v163, v101
	v_add_f32_e32 v163, v163, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v103
	v_add_f32_e32 v163, v163, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v105
	v_add_f32_e32 v163, v163, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v107
	v_add_f32_e32 v163, v163, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v109
	v_add_f32_e32 v163, v163, v110
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
	v_add_f32_e32 v163, v163, v111
	v_add_f32_e32 v163, v163, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v163, v163, v113
	;;#ASMSTART
	v_fma_f32 v161, v161, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v162
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v162, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v162, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s71, s64, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s71, s71, s0
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s71
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_113
	ds_write_b128 v151, v[212:215] offset:12288
	ds_write_b128 v151, v[216:219] offset:18432
.LBB0_113:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mul_i32 s70, s1, 0x3000
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_115
	v_add_u32_e32 v98, s70, v149
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[212:215], v[98:99], off
	global_load_dwordx4 v[216:219], v[98:99], off offset:256
.LBB0_115:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v162, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v163, 56, v98
	v_xor_b32_e32 v98, v163, v162
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_or_b32_e32 v228, 16, v162
	v_xor_b32_e32 v228, v228, v163
	v_lshlrev_b32_e32 v228, 1, v228
	ds_read_b128 v[228:231], v228
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_or_b32_e32 v228, 32, v162
	v_xor_b32_e32 v228, v228, v163
	v_lshlrev_b32_e32 v228, 1, v228
	ds_read_b128 v[228:231], v228
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_or_b32_e32 v228, 48, v162
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v163, 64, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v163, 0x50, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v163, 0x60, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v163, 0x70, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v163, 0x80, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v163, 0x90, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v163, 0xa0, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v162, 0xb0, v162
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v162, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v163, 2, v162
	v_and_b32_e32 v163, 8, v163
	v_lshrrev_b32_e32 v162, 1, v162
	v_and_or_b32 v162, v162, s55, v160
	v_add_u32_e32 v163, s66, v163
	v_add_u32_e32 v162, s11, v162
	v_add_u32_e32 v228, 0xffffffa9, v163
	v_cmp_lt_i32_e32 vcc, v228, v162
	s_nop 1
	v_cndmask_b32_e32 v99, v158, v99, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffab, v163
	s_nop 0
	v_cndmask_b32_e32 v98, v158, v98, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffac, v163
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v158, v100, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffad, v163
	s_nop 0
	v_cndmask_b32_e32 v101, v158, v101, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffae, v163
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_cndmask_b32_e32 v102, v158, v102, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffaf, v163
	s_nop 0
	v_cndmask_b32_e32 v103, v158, v103, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffb0, v163
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_cndmask_b32_e32 v104, v158, v104, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffb9, v163
	s_nop 0
	v_cndmask_b32_e32 v105, v158, v105, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffba, v163
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_cndmask_b32_e32 v106, v158, v106, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffbb, v163
	s_nop 0
	v_cndmask_b32_e32 v107, v158, v107, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffbc, v163
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_cndmask_b32_e32 v108, v158, v108, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffbd, v163
	s_nop 0
	v_cndmask_b32_e32 v109, v158, v109, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffbe, v163
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_cndmask_b32_e32 v110, v158, v110, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, 0xffffffbf, v163
	v_subrev_u32_e32 v163, 64, v163
	v_cndmask_b32_e32 v111, v158, v111, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v158, v112, vcc
	v_cmp_le_i32_e32 vcc, v163, v162
	v_max_f32_e32 v162, v98, v99
	v_max3_f32 v162, v162, v100, v101
	v_max3_f32 v162, v162, v102, v103
	v_max3_f32 v162, v162, v104, v105
	v_max3_f32 v162, v162, v106, v107
	v_cndmask_b32_e32 v113, v158, v113, vcc
	v_max3_f32 v162, v162, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v162, v162, v110, v111
	v_max3_f32 v162, v162, v112, v113
	ds_bpermute_b32 v163, v154, v162
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v163, v163, v163
	v_max_f32_e32 v163, v162, v163
	;;#ASMSTART
	v_add_f32 v162, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v163, v162
	v_mov_b32_e32 v162, 1.0
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_117
	;;#ASMSTART
	v_add_f32 v132, v163, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v162, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_117:
	s_or_b64 exec, exec, s[50:51]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, 0, v98
	v_add_f32_e32 v163, v163, v99
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
	v_add_f32_e32 v163, v163, v100
	v_add_f32_e32 v163, v163, v101
	v_add_f32_e32 v163, v163, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v103
	v_add_f32_e32 v163, v163, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v105
	v_add_f32_e32 v163, v163, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v107
	v_add_f32_e32 v163, v163, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v109
	v_add_f32_e32 v163, v163, v110
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
	v_add_f32_e32 v163, v163, v111
	v_add_f32_e32 v163, v163, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v163, v163, v113
	;;#ASMSTART
	v_fma_f32 v161, v161, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v162
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v162, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v162, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s71, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s71
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_119
	ds_write_b128 v151, v[220:223]
	ds_write_b128 v151, v[224:227] offset:6144
.LBB0_119:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_121
	v_add_u32_e32 v98, s70, v150
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[220:223], v[98:99], off
	global_load_dwordx4 v[224:227], v[98:99], off offset:256
.LBB0_121:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v162, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v98, 56, v98
	v_xor_b32_e32 v98, v98, v162
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98 offset:12288
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_add_u32_e32 v163, 0x1810, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_add_u32_e32 v163, 0x1820, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_add_u32_e32 v163, 0x1830, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v163, 0x1840, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v163, 0x1850, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v163, 0x1860, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v163, 0x1870, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v163, 0x1880, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v163, 0x1890, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v163, 0x18a0, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v162, 0x18b0, v162
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v162, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v163, 2, v162
	v_and_b32_e32 v163, 8, v163
	v_lshrrev_b32_e32 v162, 1, v162
	v_and_or_b32 v162, v162, s55, v160
	v_add_u32_e32 v163, s66, v163
	v_add_u32_e32 v162, s11, v162
	v_subrev_u32_e32 v228, 55, v163
	v_cmp_lt_i32_e32 vcc, v228, v162
	s_nop 1
	v_cndmask_b32_e32 v99, v158, v99, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 53, v163
	s_nop 0
	v_cndmask_b32_e32 v98, v158, v98, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 52, v163
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v158, v100, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 51, v163
	s_nop 0
	v_cndmask_b32_e32 v101, v158, v101, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 50, v163
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_cndmask_b32_e32 v102, v158, v102, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 49, v163
	s_nop 0
	v_cndmask_b32_e32 v103, v158, v103, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 48, v163
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_cndmask_b32_e32 v104, v158, v104, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 39, v163
	s_nop 0
	v_cndmask_b32_e32 v105, v158, v105, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 38, v163
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_cndmask_b32_e32 v106, v158, v106, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 37, v163
	s_nop 0
	v_cndmask_b32_e32 v107, v158, v107, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 36, v163
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_cndmask_b32_e32 v108, v158, v108, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 35, v163
	s_nop 0
	v_cndmask_b32_e32 v109, v158, v109, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 34, v163
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_cndmask_b32_e32 v110, v158, v110, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 33, v163
	v_subrev_u32_e32 v163, 32, v163
	v_cndmask_b32_e32 v111, v158, v111, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v158, v112, vcc
	v_cmp_le_i32_e32 vcc, v163, v162
	v_max_f32_e32 v162, v98, v99
	v_max3_f32 v162, v162, v100, v101
	v_max3_f32 v162, v162, v102, v103
	v_max3_f32 v162, v162, v104, v105
	v_max3_f32 v162, v162, v106, v107
	v_cndmask_b32_e32 v113, v158, v113, vcc
	v_max3_f32 v162, v162, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v162, v162, v110, v111
	v_max3_f32 v162, v162, v112, v113
	ds_bpermute_b32 v163, v154, v162
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v163, v163, v163
	v_max_f32_e32 v163, v162, v163
	;;#ASMSTART
	v_add_f32 v162, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v163, v162
	v_mov_b32_e32 v162, 1.0
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_123
	;;#ASMSTART
	v_add_f32 v132, v163, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v162, v114
	v_mov_b32_e32 v114, v132
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
.LBB0_123:
	s_or_b64 exec, exec, s[50:51]
	v_pk_add_f32 v[98:99], v[98:99], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, 0, v98
	v_add_f32_e32 v163, v163, v99
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
	v_add_f32_e32 v163, v163, v100
	v_add_f32_e32 v163, v163, v101
	v_add_f32_e32 v163, v163, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[140:141] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v103
	v_add_f32_e32 v163, v163, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[142:143] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v105
	v_add_f32_e32 v163, v163, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[144:145] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v107
	v_add_f32_e32 v163, v163, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[146:147] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v163, v163, v109
	v_add_f32_e32 v163, v163, v110
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
	v_add_f32_e32 v163, v163, v111
	v_add_f32_e32 v163, v163, v112
	v_add_u32_e32 v104, 0x8000, v104
	v_add_u32_e32 v103, 0x8000, v103
	v_add_u32_e32 v102, 0x8000, v102
	v_add_u32_e32 v101, 0x8000, v101
	v_add_u32_e32 v99, 0x8000, v99
	v_add_u32_e32 v98, 0x8000, v98
	v_add_f32_e32 v163, v163, v113
	;;#ASMSTART
	v_fma_f32 v161, v161, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v2, v2, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v3, v3, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v4, v4, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v5, v5, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v6, v6, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v7, v7, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v8, v8, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v9, v9, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v10, v10, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v11, v11, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v12, v12, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v13, v13, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v14, v14, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v15, v15, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v16, v16, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v17, v17, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v18, v18, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v19, v19, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v20, v20, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v21, v21, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v22, v22, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v23, v23, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v24, v24, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v25, v25, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v26, v26, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v27, v27, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v28, v28, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v29, v29, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v30, v30, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v31, v31, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v32, v32, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v33, v33, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v34, v34, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v35, v35, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v36, v36, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v37, v37, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v38, v38, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v39, v39, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v40, v40, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v41, v41, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v42, v42, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v43, v43, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v44, v44, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v45, v45, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v46, v46, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v47, v47, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v48, v48, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v49, v49, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v50, v50, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v51, v51, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v52, v52, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v53, v53, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v54, v54, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v55, v55, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v56, v56, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v57, v57, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v58, v58, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v59, v59, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v60, v60, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v61, v61, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v62, v62, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v63, v63, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v64, v64, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v65, v65, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v66, v66, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v67, v67, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v68, v68, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v69, v69, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v70, v70, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v71, v71, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v72, v72, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v73, v73, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v74, v74, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v75, v75, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v76, v76, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v77, v77, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v78, v78, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v79, v79, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v80, v80, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v81, v81, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v82, v82, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v83, v83, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v84, v84, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v85, v85, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v86, v86, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v87, v87, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v88, v88, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v89, v89, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v90, v90, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v91, v91, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v92, v92, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v93, v93, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v94, v94, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v95, v95, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v96, v96, v162
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32 v97, v97, v162
	;;#ASMEND
	v_add_u32_e32 v113, 0x8000, v113
	v_add_u32_e32 v112, 0x8000, v112
	v_add_u32_e32 v111, 0x8000, v111
	v_add_u32_e32 v110, 0x8000, v110
	v_add_u32_e32 v109, 0x8000, v109
	v_add_u32_e32 v108, 0x8000, v108
	v_add_u32_e32 v107, 0x8000, v107
	v_add_u32_e32 v106, 0x8000, v106
	v_add_u32_e32 v162, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v162, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s57
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s50, s71, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s50
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[22:23]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[228:231], v[106:107], off offset:512
	global_load_dwordx4 v[232:235], v[106:107], off offset:1024
	global_load_dwordx4 v[236:239], v[106:107], off offset:1536
	global_load_dwordx4 v[240:243], v[106:107], off offset:2048
	global_load_dwordx4 v[244:247], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[100:101], v[50:65]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[240:241], v[100:101], v[66:81]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[244:245], v[100:101], v[82:97]
	v_add_co_u32_e32 v100, vcc, s58, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[102:103], v[18:33]
	global_load_dwordx4 v[228:231], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[102:103], v[34:49]
	global_load_dwordx4 v[232:235], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[102:103], v[50:65]
	global_load_dwordx4 v[236:239], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s59, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[66:81], v[242:243], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[246:247], v[102:103], v[82:97]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[228:229], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[232:233], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[236:237], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[106:107], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[100:101], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[230:231], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[234:235], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[238:239], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[108:109], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[102:103], v[98:99], v[82:97]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_125
	ds_write_b128 v151, v[212:215] offset:12288
	ds_write_b128 v151, v[216:219] offset:18432
.LBB0_125:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v98, 0x180, v98
	v_cmp_ne_u32_e32 vcc, s56, v98
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_127
	v_add_u32_e32 v98, s70, v152
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	global_load_dwordx4 v[212:215], v[98:99], off
	global_load_dwordx4 v[216:219], v[98:99], off offset:256
.LBB0_127:
	s_or_b64 exec, exec, s[50:51]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v99, 31, v98
	v_mul_u32_u24_e32 v99, 0xc0, v99
	v_lshrrev_b32_e32 v98, 2, v98
	v_and_or_b32 v162, v98, 8, v99
	v_lshrrev_b32_e32 v98, 3, v99
	v_and_b32_e32 v163, 56, v98
	v_xor_b32_e32 v98, v163, v162
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[228:231], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[164:165], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[166:167], v[98:113]
	v_or_b32_e32 v228, 16, v162
	v_xor_b32_e32 v228, v228, v163
	v_lshlrev_b32_e32 v228, 1, v228
	ds_read_b128 v[228:231], v228
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[170:171], v[98:113]
	v_or_b32_e32 v228, 32, v162
	v_xor_b32_e32 v228, v228, v163
	v_lshlrev_b32_e32 v228, 1, v228
	ds_read_b128 v[228:231], v228
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[174:175], v[98:113]
	v_or_b32_e32 v228, 48, v162
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[178:179], v[98:113]
	v_add_u32_e32 v163, 64, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[182:183], v[98:113]
	v_add_u32_e32 v163, 0x50, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[186:187], v[98:113]
	v_add_u32_e32 v163, 0x60, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[190:191], v[98:113]
	v_add_u32_e32 v163, 0x70, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[192:193], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[194:195], v[98:113]
	v_add_u32_e32 v163, 0x80, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[196:197], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[198:199], v[98:113]
	v_add_u32_e32 v163, 0x90, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[200:201], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[202:203], v[98:113]
	v_add_u32_e32 v163, 0xa0, v162
	v_lshrrev_b32_e32 v228, 3, v163
	v_and_b32_e32 v228, 56, v228
	v_xor_b32_e32 v163, v228, v163
	v_lshlrev_b32_e32 v163, 1, v163
	ds_read_b128 v[228:231], v163
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[204:205], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[206:207], v[98:113]
	v_add_u32_e32 v162, 0xb0, v162
	v_lshrrev_b32_e32 v163, 3, v162
	v_and_b32_e32 v163, 56, v163
	v_xor_b32_e32 v162, v163, v162
	v_lshlrev_b32_e32 v162, 1, v162
	ds_read_b128 v[228:231], v162
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[228:229], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[230:231], v[210:211], v[98:113]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v162, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v163, 2, v162
	v_and_b32_e32 v163, 8, v163
	v_lshrrev_b32_e32 v162, 1, v162
	v_and_or_b32 v162, v162, s55, v160
	v_add_u32_e32 v163, s66, v163
	v_add_u32_e32 v162, s11, v162
	v_subrev_u32_e32 v228, 23, v163
	v_cmp_lt_i32_e32 vcc, v228, v162
	s_nop 1
	v_cndmask_b32_e32 v99, v158, v99, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 21, v163
	s_nop 0
	v_cndmask_b32_e32 v98, v158, v98, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 20, v163
	v_pk_mul_f32 v[98:99], v[116:117], v[98:99]
	v_cndmask_b32_e32 v100, v158, v100, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 19, v163
	s_nop 0
	v_cndmask_b32_e32 v101, v158, v101, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 18, v163
	v_pk_mul_f32 v[100:101], v[118:119], v[100:101]
	v_cndmask_b32_e32 v102, v158, v102, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_subrev_u32_e32 v228, 17, v163
	s_nop 0
	v_cndmask_b32_e32 v103, v158, v103, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, -16, v163
	v_pk_mul_f32 v[102:103], v[120:121], v[102:103]
	v_cndmask_b32_e32 v104, v158, v104, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, -7, v163
	s_nop 0
	v_cndmask_b32_e32 v105, v158, v105, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, -6, v163
	v_pk_mul_f32 v[104:105], v[122:123], v[104:105]
	v_cndmask_b32_e32 v106, v158, v106, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, -5, v163
	s_nop 0
	v_cndmask_b32_e32 v107, v158, v107, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, -4, v163
	v_pk_mul_f32 v[106:107], v[124:125], v[106:107]
	v_cndmask_b32_e32 v108, v158, v108, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, -3, v163
	s_nop 0
	v_cndmask_b32_e32 v109, v158, v109, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, -2, v163
	v_pk_mul_f32 v[108:109], v[126:127], v[108:109]
	v_cndmask_b32_e32 v110, v158, v110, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_add_u32_e32 v228, -1, v163
	s_nop 0
	v_cndmask_b32_e32 v111, v158, v111, vcc
	v_cmp_le_i32_e32 vcc, v228, v162
	v_pk_mul_f32 v[110:111], v[128:129], v[110:111]
	s_nop 0
	v_cndmask_b32_e32 v112, v158, v112, vcc
	v_cmp_le_i32_e32 vcc, v163, v162
	v_max_f32_e32 v162, v98, v99
	v_max3_f32 v162, v162, v100, v101
	v_max3_f32 v162, v162, v102, v103
	v_max3_f32 v162, v162, v104, v105
	v_max3_f32 v162, v162, v106, v107
	v_cndmask_b32_e32 v113, v158, v113, vcc
	v_max3_f32 v162, v162, v108, v109
	v_pk_mul_f32 v[112:113], v[130:131], v[112:113]
	v_max3_f32 v162, v162, v110, v111
	v_max3_f32 v162, v162, v112, v113
	ds_bpermute_b32 v163, v154, v162
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v163, v163, v163
	v_max_f32_e32 v163, v162, v163
	;;#ASMSTART
	v_add_f32 v162, v114, v156
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v163, v162
	v_mov_b32_e32 v162, 1.0
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_78
	;;#ASMSTART
	v_add_f32 v132, v163, v157
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v114, v114, v132
	v_exp_f32_e32 v162, v114
	v_mov_b32_e32 v114, v132
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
	s_branch .LBB0_78
.LBB0_129:
	s_mov_b32 s69, s68
	s_branch .LBB0_79
.LBB0_130:
	;;#ASMSTART
	v_mov_b32 v100, v0
	;;#ASMEND
	ds_bpermute_b32 v98, v154, v161
	v_lshrrev_b32_e32 v99, 1, v100
	v_and_b32_e32 v101, 31, v100
	v_and_or_b32 v99, v99, s55, v101
	v_and_b32_e32 v100, 32, v100
	v_cmp_eq_u32_e32 vcc, 0, v100
	v_cmp_gt_i32_e64 s[0:1], s43, v99
	s_and_b64 s[4:5], vcc, s[0:1]
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v98, v161, v98
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_132
	v_cmp_gt_f32_e32 vcc, s60, v98
	s_ashr_i32 s43, s42, 31
	s_lshl_b64 s[4:5], s[42:43], 6
	v_cndmask_b32_e64 v101, 0, 32, vcc
	v_ldexp_f32 v101, v98, v101
	v_log_f32_e32 v101, v101
	v_cndmask_b32_e32 v100, 0, v159, vcc
	s_add_u32 s6, s36, s4
	s_addc_u32 s8, s37, s5
	v_sub_f32_e32 v100, v101, v100
	v_add_f32_e32 v100, v114, v100
	s_lshl_b64 s[4:5], s[26:27], 2
	v_mul_f32_e32 v100, 0x3f317218, v100
	v_cmp_lt_f32_e32 vcc, 0, v98
	s_add_u32 s4, s6, s4
	s_addc_u32 s5, s8, s5
	v_cndmask_b32_e32 v100, v158, v100, vcc
	v_lshlrev_b32_e32 v99, 6, v99
	global_store_dword v99, v100, s[4:5]
.LBB0_132:
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
	v_perm_b32 v48, v3, v2, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v49, v5, v4, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v46, v7, v6, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v47, v9, v8, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v44, v11, v10, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v45, v13, v12, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v42, v15, v14, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v43, v17, v16, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v40, v19, v18, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v41, v21, v20, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v38, v23, v22, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v39, v25, v24, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v36, v27, v26, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v37, v29, v28, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v34, v31, v30, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v35, v33, v32, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v112, v113, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v110, v111, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v108, v109, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v106, v107, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v28, v104, v105, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v102, v103, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v100, v101, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v98, v99, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v51, v50, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v53, v52, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v55, v54, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v57, v56, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v59, v58, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v61, v60, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v63, v62, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v65, v64, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v67, v66, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v69, v68, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v71, v70, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v73, v72, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v75, v74, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v77, v76, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v79, v78, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v81, v80, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v83, v82, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v85, v84, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v87, v86, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v89, v88, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v91, v90, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v93, v92, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v95, v94, s57
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v97, v96, s57
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v50, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v50, 0x100, v50
	v_cmp_eq_u32_e32 vcc, 0, v50
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_134
	s_barrier
.LBB0_134:
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
	s_cbranch_execz .LBB0_136
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
.LBB0_136:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v88, 0x1ff, v91
	s_add_u32 s8, s34, s40
	s_addc_u32 s0, s35, s41
	v_lshlrev_b32_e32 v92, 3, v88
	s_and_b32 s9, s0, 0xffff
	s_mov_b32 s11, s7
	v_and_b32_e32 v91, 56, v91
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_137:
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
	v_add_u32_e32 v96, s17, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[8:11], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_137
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 1, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_140
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
.LBB0_140:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s4, s17, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_141:
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
	v_add_u32_e32 v96, s4, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[8:11], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_141
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 2, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_144
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
.LBB0_144:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_145:
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
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[8:11], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_145
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 3, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_148
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
.LBB0_148:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x30000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_149:
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
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[8:11], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_149
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 4, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_152
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
.LBB0_152:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x48000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_153:
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
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[8:11], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_153
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 5, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_156
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
.LBB0_156:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x60000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_157:
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
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[8:11], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_157
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 6, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_160
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
.LBB0_160:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x78000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_161:
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
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[8:11], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_161
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 7, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_164
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
.LBB0_164:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s4, s4, 0x90000
	s_mov_b64 s[0:1], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_165:
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
	v_add_u32_e32 v3, s4, v3
	v_add_u32_e32 v92, 0x1000, v92
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v88, v4
	v_add_lshl_u32 v2, v3, v2, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v2, s[8:11], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_165
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
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_170
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_169
	s_bcnt1_i32_b64 s6, s[10:11]
	v_mov_b32_e32 v3, s6
	global_atomic_add v3, v155, v3, s[38:39] sc0
.LBB0_169:
	s_or_b64 exec, exec, s[8:9]
	s_lshl_b64 s[8:9], s[0:1], 2
	s_add_u32 s8, s38, s8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s6, v3
	s_addc_u32 s9, s39, s9
	s_nop 0
	v_add_u32_e32 v2, s6, v2
	global_store_dword v155, v2, s[8:9]
	s_waitcnt vmcnt(0)
.LBB0_170:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s38, s0
	s_addc_u32 s1, s39, s1
	s_barrier
	global_load_dword v2, v155, s[0:1]
	s_sub_i32 s0, s33, s2
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s2, v2
	s_add_i32 s33, s0, s2
	s_cmp_ge_i32 s16, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s52
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_173
	s_branch .LBB0_12
.LBB0_171:
	s_mov_b32 s16, s5
.LBB0_172:
	s_sub_i32 s33, s33, s52
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s26, 0, s4
	s_cmp_ge_i32 s16, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s52, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_173:
	s_add_i32 s4, s26, 1
	s_cmp_gt_i32 s4, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s4, 16
	s_cbranch_scc1 .LBB0_176
	s_add_i32 s5, s16, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s52
	s_cbranch_scc1 .LBB0_171
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[8:9], s[16:17], 2
	s_add_u32 s8, s12, s8
	s_addc_u32 s9, s13, s9
	global_load_dwordx2 v[2:3], v155, s[8:9] offset:4
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
	s_branch .LBB0_171
.LBB0_176:
	s_mov_b32 s6, s52
	s_branch .LBB0_172
.LBB0_177:
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
		.amdhsa_next_free_vgpr 248
		.amdhsa_next_free_sgpr 96
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

	.set .Lattn_kernel_0.num_vgpr, 248
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
    .sgpr_count:     78
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     248
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
