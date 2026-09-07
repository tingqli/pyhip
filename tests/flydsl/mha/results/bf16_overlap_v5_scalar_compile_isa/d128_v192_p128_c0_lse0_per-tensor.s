	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	attn_kernel_0
	.p2align	8
	.type	attn_kernel_0,@function
attn_kernel_0:
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dword s3, s[0:1], 0x38
	s_load_dwordx8 s[36:43], s[0:1], 0x70
	s_mov_b32 s48, 0
	s_waitcnt lgkmcnt(0)
	s_load_dwordx2 s[4:5], s[34:35], 0x0
	s_add_i32 s3, s3, -1
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s4, s5, s4
	s_add_i32 s6, s4, 0xff
	s_ashr_i32 s4, s6, 31
	s_lshr_b32 s4, s4, 24
	s_add_i32 s4, s6, s4
	s_ashr_i32 s8, s4, 8
	s_and_b32 s4, s4, 0xffffff00
	s_cmp_lg_u32 s6, s4
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s6, 0
	s_cselect_b64 s[6:7], -1, 0
	s_and_b64 s[4:5], s[6:7], s[4:5]
	s_subb_u32 s6, s8, 0
	s_cmp_lt_i32 s3, 1
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s2, s6
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[4:5], s[4:5], s[8:9]
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_8
	s_mov_b32 s73, 0
	s_mov_b32 s33, s2
	s_branch .LBB0_4
.LBB0_2:
	s_mov_b32 s48, s8
.LBB0_3:
	s_sub_i32 s33, s33, s6
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s73, 0, s7
	s_cmp_ge_i32 s48, s3
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s33, s70
	s_cselect_b64 s[6:7], -1, 0
	s_or_b64 s[4:5], s[4:5], s[6:7]
	s_and_b64 vcc, exec, s[4:5]
	s_mov_b32 s6, s70
	s_cbranch_vccnz .LBB0_9
.LBB0_4:
	s_add_i32 s7, s73, 1
	s_cmp_gt_i32 s7, 15
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s7, 16
	s_cbranch_scc1 .LBB0_7
	s_add_i32 s8, s48, 1
	s_cmp_ge_i32 s8, s3
	s_mov_b32 s70, s6
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s49, s48, 31
	s_lshl_b64 s[10:11], s[48:49], 2
	s_add_u32 s10, s34, s10
	s_addc_u32 s11, s35, s11
	s_load_dwordx2 s[12:13], s[10:11], 0x4
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s9, s13, s12
	s_addk_i32 s9, 0xff
	s_ashr_i32 s10, s9, 31
	s_lshr_b32 s10, s10, 24
	s_add_i32 s10, s9, s10
	s_ashr_i32 s14, s10, 8
	s_and_b32 s10, s10, 0xffffff00
	s_cmp_lg_u32 s9, s10
	s_cselect_b64 s[10:11], -1, 0
	s_cmp_lt_i32 s9, 0
	s_cselect_b64 s[12:13], -1, 0
	s_and_b64 s[10:11], s[12:13], s[10:11]
	s_subb_u32 s70, s14, 0
	s_branch .LBB0_2
.LBB0_7:
	s_mov_b32 s70, s6
	s_branch .LBB0_3
.LBB0_8:
	s_mov_b32 s70, s6
	s_mov_b32 s73, 0
	s_mov_b32 s33, s2
.LBB0_9:
	s_load_dword s71, s[38:39], 0x0
	s_load_dword s72, s[40:41], 0x0
	s_cmp_ge_i32 s48, s3
	s_cbranch_scc1 .LBB0_135
	s_load_dwordx2 s[38:39], s[0:1], 0x0
	s_load_dwordx2 s[40:41], s[0:1], 0x10
	s_load_dwordx2 s[50:51], s[0:1], 0x20
	s_load_dwordx2 s[52:53], s[0:1], 0x50
	s_load_dwordx2 s[54:55], s[0:1], 0x60
	s_load_dwordx2 s[56:57], s[0:1], 0x98
	s_load_dwordx2 s[58:59], s[0:1], 0xb8
	v_lshrrev_b32_e32 v2, 2, v0
	v_lshlrev_b32_e32 v3, 10, v0
	v_lshl_or_b32 v1, v0, 11, v2
	v_and_b32_e32 v3, 0x70000, v3
	s_mov_b32 s0, 0xf808
	v_lshrrev_b32_e32 v5, 1, v0
	v_lshrrev_b32_e32 v6, 3, v0
	v_and_or_b32 v1, v1, s0, v3
	v_lshlrev_b32_e32 v3, 9, v0
	v_and_b32_e32 v4, 12, v2
	v_and_b32_e32 v5, 32, v5
	v_and_b32_e32 v6, 16, v6
	v_and_b32_e32 v3, 0x1e00, v3
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
	v_mov_b32_e32 v115, 0
	v_xor_b32_e32 v154, 0x80, v2
	s_mov_b32 s47, 0x27000
	s_movk_i32 s74, 0xf80
	v_mov_b32_e32 v155, 0x40e00000
	v_mov_b32_e32 v156, 1.0
	s_mov_b32 s75, 0x7060302
	s_movk_i32 s76, 0x1000
	s_movk_i32 s77, 0x2000
	s_mov_b32 s78, 0xaaab
	s_movk_i32 s79, 0xff
	s_mov_b32 s80, s2
	v_mov_b32_e32 v157, 0xff800000
	s_branch .LBB0_13
.LBB0_11:
	s_mov_b32 s70, s6
.LBB0_12:
	s_mov_b32 s2, s4
	s_cmp_ge_i32 s48, s3
	s_cbranch_scc1 .LBB0_135
.LBB0_13:
	s_ashr_i32 s49, s48, 31
	s_lshl_b32 s6, s33, 8
	s_lshl_b64 s[0:1], s[48:49], 2
	s_add_u32 s4, s34, s0
	s_addc_u32 s5, s35, s1
	global_load_dwordx2 v[4:5], v115, s[4:5]
	global_load_dword v2, v115, s[36:37]
	v_lshl_add_u32 v3, s73, 8, v1
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s81, v4
	s_add_i32 s81, s81, s6
	v_readfirstlane_b32 s4, v5
	s_add_i32 s5, s81, 0x100
	s_min_i32 s4, s5, s4
	s_sub_i32 s49, s4, s81
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s52, s0
	s_addc_u32 s5, s53, s1
	s_add_u32 s0, s42, s0
	s_addc_u32 s1, s43, s1
	s_lshl_b32 s6, s81, 11
	s_ashr_i32 s7, s6, 31
	global_load_dwordx2 v[4:5], v115, s[4:5]
	global_load_dword v6, v115, s[0:1]
	s_lshl_b64 s[0:1], s[6:7], 1
	s_add_u32 s44, s38, s0
	s_addc_u32 s0, s39, s1
	s_lshl_b32 s46, s49, 12
	s_and_b32 s45, s0, 0xffff
	buffer_load_dwordx4 v[160:163], v3, s[44:47], 0 offen
	buffer_load_dwordx4 v[164:167], v3, s[44:47], 0 offen offset:32
	buffer_load_dwordx4 v[168:171], v3, s[44:47], 0 offen offset:64
	buffer_load_dwordx4 v[172:175], v3, s[44:47], 0 offen offset:96
	buffer_load_dwordx4 v[176:179], v3, s[44:47], 0 offen offset:128
	buffer_load_dwordx4 v[180:183], v3, s[44:47], 0 offen offset:160
	buffer_load_dwordx4 v[184:187], v3, s[44:47], 0 offen offset:192
	buffer_load_dwordx4 v[188:191], v3, s[44:47], 0 offen offset:224
	;;#ASMSTART
	v_mov_b32 v3, v0
	;;#ASMEND
	s_waitcnt vmcnt(9)
	v_readfirstlane_b32 s0, v4
	v_and_b32_e32 v3, 0x100, v3
	v_readfirstlane_b32 s1, v5
	s_waitcnt vmcnt(8)
	v_readfirstlane_b32 s8, v6
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_15
	s_barrier
.LBB0_15:
	s_or_b64 exec, exec, s[4:5]
	s_sub_i32 s60, s1, s0
	s_ashr_i32 s1, s73, 31
	s_lshr_b32 s1, s1, 28
	s_add_i32 s1, s73, s1
	s_add_i32 s9, s60, -1
	s_ashr_i32 s10, s1, 4
	s_and_b32 s1, s1, -16
	s_cmp_lg_u32 s73, s1
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s73, 0
	s_cselect_b64 s[6:7], -1, 0
	s_and_b64 s[4:5], s[6:7], s[4:5]
	s_subb_u32 s82, s10, 0
	s_lshl_b32 s4, s82, 14
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s62, s40, s4
	s_addc_u32 s63, s41, s5
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s44, s54, s0
	s_addc_u32 s0, s55, s1
	s_lshl_b32 s46, s60, 2
	s_and_b32 s45, s0, 0xffff
	buffer_load_dwordx2 v[4:5], off, s[44:47], 0
	v_mul_f32_e32 v2, s71, v2
	v_mul_f32_e32 v116, 0x3e0293ee, v2
	s_mulk_i32 s82, 0x6000
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s66, v4
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
	buffer_load_dword v2, off, s[44:47], 0 offset:8
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s83, v2
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
	buffer_load_dword v2, off, s[44:47], 0 offset:12
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s85, v2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_lshl_b32 s0, s66, 13
	v_or_b32_e32 v2, s0, v114
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 2, s[62:63]
	global_load_dwordx4 v[4:7], v[4:5], off
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_ashr_i32 s0, s0, 31
	v_mov_b32_e32 v3, s0
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[62:63]
	global_load_dwordx4 v[192:195], v[2:3], off offset:512
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	global_load_dwordx4 v[196:199], v[2:3], off offset:1024
	ds_write_b128 v119, v[4:7] offset:8192
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	s_barrier
	s_setprio 1
	s_add_i32 s0, s60, -2
	s_bitcmp0_b32 s60, 0
	s_cselect_b32 s64, s0, s9
	s_ashr_i32 s65, s64, 31
	s_cmp_lt_i32 s64, 1
	s_cbranch_scc1 .LBB0_51
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v120, v116
	v_mov_b32_e32 v121, v116
	s_mov_b64 s[0:1], 0
	s_mov_b32 s10, 20
	v_mov_b32_e32 v118, 0xff800000
	v_mov_b32_e32 v122, v116
	v_mov_b32_e32 v123, v116
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v158
	v_mov_b32_e32 v4, v158
	v_mov_b32_e32 v5, v158
	v_mov_b32_e32 v6, v158
	v_mov_b32_e32 v7, v158
	v_mov_b32_e32 v8, v158
	v_mov_b32_e32 v9, v158
	v_mov_b32_e32 v10, v158
	v_mov_b32_e32 v11, v158
	v_mov_b32_e32 v12, v158
	v_mov_b32_e32 v13, v158
	v_mov_b32_e32 v14, v158
	v_mov_b32_e32 v15, v158
	v_mov_b32_e32 v16, v158
	v_mov_b32_e32 v17, v158
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v158
	v_mov_b32_e32 v20, v158
	v_mov_b32_e32 v21, v158
	v_mov_b32_e32 v22, v158
	v_mov_b32_e32 v23, v158
	v_mov_b32_e32 v24, v158
	v_mov_b32_e32 v25, v158
	v_mov_b32_e32 v26, v158
	v_mov_b32_e32 v27, v158
	v_mov_b32_e32 v28, v158
	v_mov_b32_e32 v29, v158
	v_mov_b32_e32 v30, v158
	v_mov_b32_e32 v31, v158
	v_mov_b32_e32 v32, v158
	v_mov_b32_e32 v33, v158
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v158
	v_mov_b32_e32 v36, v158
	v_mov_b32_e32 v37, v158
	v_mov_b32_e32 v38, v158
	v_mov_b32_e32 v39, v158
	v_mov_b32_e32 v40, v158
	v_mov_b32_e32 v41, v158
	v_mov_b32_e32 v42, v158
	v_mov_b32_e32 v43, v158
	v_mov_b32_e32 v44, v158
	v_mov_b32_e32 v45, v158
	v_mov_b32_e32 v46, v158
	v_mov_b32_e32 v47, v158
	v_mov_b32_e32 v48, v158
	v_mov_b32_e32 v49, v158
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, v158
	v_mov_b32_e32 v84, v158
	v_mov_b32_e32 v85, v158
	v_mov_b32_e32 v86, v158
	v_mov_b32_e32 v87, v158
	v_mov_b32_e32 v88, v158
	v_mov_b32_e32 v89, v158
	v_mov_b32_e32 v90, v158
	v_mov_b32_e32 v91, v158
	v_mov_b32_e32 v92, v158
	v_mov_b32_e32 v93, v158
	v_mov_b32_e32 v94, v158
	v_mov_b32_e32 v95, v158
	v_mov_b32_e32 v96, v158
	v_mov_b32_e32 v97, v158
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v158
	v_mov_b32_e32 v52, v158
	v_mov_b32_e32 v53, v158
	v_mov_b32_e32 v54, v158
	v_mov_b32_e32 v55, v158
	v_mov_b32_e32 v56, v158
	v_mov_b32_e32 v57, v158
	v_mov_b32_e32 v58, v158
	v_mov_b32_e32 v59, v158
	v_mov_b32_e32 v60, v158
	v_mov_b32_e32 v61, v158
	v_mov_b32_e32 v62, v158
	v_mov_b32_e32 v63, v158
	v_mov_b32_e32 v64, v158
	v_mov_b32_e32 v65, v158
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, v158
	v_mov_b32_e32 v68, v158
	v_mov_b32_e32 v69, v158
	v_mov_b32_e32 v70, v158
	v_mov_b32_e32 v71, v158
	v_mov_b32_e32 v72, v158
	v_mov_b32_e32 v73, v158
	v_mov_b32_e32 v74, v158
	v_mov_b32_e32 v75, v158
	v_mov_b32_e32 v76, v158
	v_mov_b32_e32 v77, v158
	v_mov_b32_e32 v78, v158
	v_mov_b32_e32 v79, v158
	v_mov_b32_e32 v80, v158
	v_mov_b32_e32 v81, v158
	s_branch .LBB0_18
.LBB0_17:
	s_or_b64 exec, exec, s[4:5]
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
	v_perm_b32 v98, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v101, v100, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v100, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v140, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v141, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v104, v0
	;;#ASMEND
	s_addk_i32 s6, 0x3000
	v_lshlrev_b32_e32 v105, 3, v104
	v_bfe_i32 v104, v104, 5, 1
	v_and_b32_e32 v105, 0xf8, v105
	v_and_b32_e32 v104, 0x600, v104
	v_or3_b32 v104, v105, v104, s6
	v_ashrrev_i32_e32 v105, 31, v104
	v_lshl_add_u64 v[104:105], v[104:105], 1, s[50:51]
	global_load_dwordx4 v[106:109], v[104:105], off
	global_load_dwordx4 v[110:113], v[104:105], off offset:512
	global_load_dwordx4 v[124:127], v[104:105], off offset:1024
	global_load_dwordx4 v[128:131], v[104:105], off offset:1536
	global_load_dwordx4 v[132:135], v[104:105], off offset:2048
	global_load_dwordx4 v[136:139], v[104:105], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[106:107], v[98:99], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[110:111], v[98:99], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[124:125], v[98:99], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[128:129], v[98:99], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[132:133], v[98:99], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[136:137], v[98:99], v[66:81]
	v_add_co_u32_e32 v98, vcc, s76, v104
	s_nop 1
	v_addc_co_u32_e32 v99, vcc, 0, v105, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	global_load_dwordx4 v[106:109], v[98:99], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[112:113], v[100:101], v[18:33]
	global_load_dwordx4 v[110:113], v[98:99], off offset:3584
	v_mfma_f32_32x32x8_bf16 v[34:49], v[126:127], v[100:101], v[34:49]
	global_load_dwordx4 v[124:127], v[98:99], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[82:97], v[130:131], v[100:101], v[82:97]
	global_load_dwordx4 v[130:133], v[98:99], off offset:3072
	v_add_co_u32_e32 v98, vcc, s77, v104
	s_nop 1
	v_addc_co_u32_e32 v99, vcc, 0, v105, vcc
	global_load_dwordx4 v[142:145], v[98:99], off
	v_mfma_f32_32x32x8_bf16 v[50:65], v[134:135], v[100:101], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[138:139], v[100:101], v[66:81]
	global_load_dwordx4 v[98:101], v[98:99], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[106:107], v[102:103], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[102:103], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[142:143], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[98:99], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[140:141], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[112:113], v[140:141], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[126:127], v[140:141], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[132:133], v[140:141], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[144:145], v[140:141], v[50:65]
	s_add_u32 s0, s0, 2
	s_addc_u32 s1, s1, 0
	v_mov_b64_e32 v[140:141], s[64:65]
	v_cmp_lt_i64_e32 vcc, s[0:1], v[140:141]
	s_add_i32 s10, s10, 8
	s_mov_b32 s84, s12
	s_mov_b32 s66, s11
	s_cbranch_vccz .LBB0_50
.LBB0_18:
	s_add_i32 s4, s10, -4
	v_mov_b32_e32 v98, s4
	s_lshl_b32 s4, s66, 13
	s_ashr_i32 s5, s4, 31
	buffer_load_dword v117, v98, s[44:47], 0 offen
	v_or_b32_e32 v98, s4, v114
	v_mov_b32_e32 v99, s5
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[62:63]
	global_load_dwordx4 v[148:151], v[98:99], off offset:1536
	ds_write_b128 v119, v[192:195]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_mov_b32 s11, s83
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v124, 48, v98
	v_and_or_b32 v125, v99, s74, v100
	v_or_b32_e32 v98, v125, v124
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[126:129], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[160:161], 0
	s_mov_b32 s12, s85
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s83, v117
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[162:163], v[98:113]
	v_or_b32_e32 v117, 0x1010, v125
	v_xor_b32_e32 v117, v117, v124
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[166:167], v[98:113]
	v_or_b32_e32 v117, 0x1020, v125
	v_xor_b32_e32 v117, v117, v124
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[170:171], v[98:113]
	v_or_b32_e32 v117, 0x1030, v125
	v_xor_b32_e32 v117, v117, v124
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[174:175], v[98:113]
	v_or_b32_e32 v117, 0x1040, v125
	v_lshrrev_b32_e32 v124, 3, v117
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v117, v124, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[178:179], v[98:113]
	v_or_b32_e32 v117, 0x1050, v125
	v_lshrrev_b32_e32 v124, 3, v117
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v117, v124, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[182:183], v[98:113]
	v_or_b32_e32 v117, 0x1060, v125
	v_lshrrev_b32_e32 v124, 3, v117
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v117, v124, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[126:129], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x1070, v125
	v_lshrrev_b32_e32 v124, 3, v117
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v117, v124, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[190:191], v[98:113]
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
	ds_bpermute_b32 v124, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v124, v124, v124
	v_max_f32_e32 v124, v117, v124
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v124, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_20
	;;#ASMSTART
	v_add_f32 v124, v124, v156
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v124
	v_exp_f32_e32 v117, v117
	v_mov_b32_e32 v118, v124
.LBB0_20:
	s_or_b64 exec, exec, s[4:5]
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
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v124, v124, v111
	v_add_f32_e32 v124, v124, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v124, v124, v113
	;;#ASMSTART
	v_fma_f32 v140, v158, v117, v124
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_22
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
.LBB0_22:
	s_or_b64 exec, exec, s[4:5]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s5, s66, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s5, s5, s82
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s5
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[124:127], v[106:107], off offset:512
	global_load_dwordx4 v[128:131], v[106:107], off offset:1024
	global_load_dwordx4 v[132:135], v[106:107], off offset:1536
	global_load_dwordx4 v[136:139], v[106:107], off offset:2048
	global_load_dwordx4 v[142:145], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[132:133], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[136:137], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[126:127], v[102:103], v[18:33]
	global_load_dwordx4 v[124:127], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[102:103], v[34:49]
	global_load_dwordx4 v[128:131], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[134:135], v[102:103], v[82:97]
	global_load_dwordx4 v[132:135], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[138:139], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[124:125], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[128:129], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[132:133], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[126:127], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[134:135], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	s_lshl_b32 s4, s84, 13
	v_or_b32_e32 v98, s4, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[62:63]
	global_load_dwordx4 v[144:147], v[98:99], off
	ds_write_b128 v119, v[196:199] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v124, 48, v98
	v_or_b32_e32 v98, v117, v124
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[126:129], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[162:163], v[98:113]
	v_or_b32_e32 v125, 16, v117
	v_xor_b32_e32 v125, v125, v124
	v_lshlrev_b32_e32 v125, 1, v125
	ds_read_b128 v[126:129], v125
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[166:167], v[98:113]
	v_or_b32_e32 v125, 32, v117
	v_xor_b32_e32 v125, v125, v124
	v_lshlrev_b32_e32 v125, 1, v125
	ds_read_b128 v[126:129], v125
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[128:129], v[170:171], v[98:113]
	v_or_b32_e32 v125, 48, v117
	v_xor_b32_e32 v124, v125, v124
	v_lshlrev_b32_e32 v124, 1, v124
	ds_read_b128 v[124:127], v124
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[174:175], v[98:113]
	v_or_b32_e32 v124, 64, v117
	v_lshrrev_b32_e32 v125, 3, v124
	v_and_b32_e32 v125, 56, v125
	v_xor_b32_e32 v124, v125, v124
	v_lshlrev_b32_e32 v124, 1, v124
	ds_read_b128 v[124:127], v124
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[178:179], v[98:113]
	v_or_b32_e32 v124, 0x50, v117
	v_lshrrev_b32_e32 v125, 3, v124
	v_and_b32_e32 v125, 56, v125
	v_xor_b32_e32 v124, v125, v124
	v_lshlrev_b32_e32 v124, 1, v124
	ds_read_b128 v[124:127], v124
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[182:183], v[98:113]
	v_or_b32_e32 v124, 0x60, v117
	v_lshrrev_b32_e32 v125, 3, v124
	v_and_b32_e32 v125, 56, v125
	v_xor_b32_e32 v124, v125, v124
	v_lshlrev_b32_e32 v124, 1, v124
	ds_read_b128 v[124:127], v124
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x70, v117
	v_lshrrev_b32_e32 v124, 3, v117
	v_and_b32_e32 v124, 56, v124
	v_xor_b32_e32 v117, v124, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[190:191], v[98:113]
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
	ds_bpermute_b32 v124, v154, v117
	v_mov_b32_e32 v125, v118
	v_mov_b32_e32 v126, v118
	v_mov_b32_e32 v127, v118
	v_mov_b32_e32 v128, v118
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v124, v124, v124
	v_max_f32_e32 v141, v117, v124
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	v_mov_b32_e32 v124, v118
	v_cmp_gt_f32_e32 vcc, v141, v117
	v_mov_b32_e32 v117, 1.0
	v_mov_b32_e32 v129, v118
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
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_24
	;;#ASMSTART
	v_add_f32 v124, v141, v156
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
	s_or_b64 exec, exec, s[6:7]
	v_pk_add_f32 v[98:99], v[98:99], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, 0, v98
	v_add_f32_e32 v141, v141, v99
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
	v_add_f32_e32 v141, v141, v100
	v_add_f32_e32 v141, v141, v101
	v_add_f32_e32 v141, v141, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v103
	v_add_f32_e32 v141, v141, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v105
	v_add_f32_e32 v141, v141, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v107
	v_add_f32_e32 v141, v141, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v109
	v_add_f32_e32 v141, v141, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v141, v141, v111
	v_add_f32_e32 v141, v141, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v141, v141, v113
	;;#ASMSTART
	v_fma_f32 v142, v140, v117, v141
	;;#ASMEND
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_26
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
.LBB0_26:
	s_or_b64 exec, exec, s[6:7]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s6, s5, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s6
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[192:195], v[106:107], off offset:512
	global_load_dwordx4 v[196:199], v[106:107], off offset:1024
	global_load_dwordx4 v[200:203], v[106:107], off offset:1536
	global_load_dwordx4 v[204:207], v[106:107], off offset:2048
	global_load_dwordx4 v[208:211], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[200:201], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[194:195], v[102:103], v[18:33]
	global_load_dwordx4 v[192:195], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[102:103], v[34:49]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[202:203], v[102:103], v[82:97]
	global_load_dwordx4 v[200:203], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[200:201], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[194:195], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[202:203], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	s_ashr_i32 s5, s4, 31
	v_lshl_add_u64 v[98:99], s[4:5], 0, v[114:115]
	v_lshl_add_u64 v[140:141], v[98:99], 2, s[62:63]
	global_load_dwordx4 v[192:195], v[140:141], off offset:512
	ds_write_b128 v119, v[148:151]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v143, 48, v98
	v_or_b32_e32 v98, v117, v143
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[148:151], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[162:163], v[98:113]
	v_or_b32_e32 v148, 0x1010, v117
	v_xor_b32_e32 v148, v148, v143
	v_lshlrev_b32_e32 v148, 1, v148
	ds_read_b128 v[148:151], v148
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[166:167], v[98:113]
	v_or_b32_e32 v148, 0x1020, v117
	v_xor_b32_e32 v148, v148, v143
	v_lshlrev_b32_e32 v148, 1, v148
	ds_read_b128 v[148:151], v148
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[170:171], v[98:113]
	v_or_b32_e32 v148, 0x1030, v117
	v_xor_b32_e32 v143, v148, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[148:151], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[174:175], v[98:113]
	v_or_b32_e32 v143, 0x1040, v117
	v_lshrrev_b32_e32 v148, 3, v143
	v_and_b32_e32 v148, 56, v148
	v_xor_b32_e32 v143, v148, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[148:151], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[178:179], v[98:113]
	v_or_b32_e32 v143, 0x1050, v117
	v_lshrrev_b32_e32 v148, 3, v143
	v_and_b32_e32 v148, 56, v148
	v_xor_b32_e32 v143, v148, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[148:151], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[182:183], v[98:113]
	v_or_b32_e32 v143, 0x1060, v117
	v_lshrrev_b32_e32 v148, 3, v143
	v_and_b32_e32 v148, 56, v148
	v_xor_b32_e32 v143, v148, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[148:151], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x1070, v117
	v_lshrrev_b32_e32 v143, 3, v117
	v_and_b32_e32 v143, 56, v143
	v_xor_b32_e32 v117, v143, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[148:151], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[190:191], v[98:113]
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
	ds_bpermute_b32 v143, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v143, v143, v143
	v_max_f32_e32 v143, v117, v143
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v143, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_28
	;;#ASMSTART
	v_add_f32 v124, v143, v156
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
.LBB0_28:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[98:99], v[98:99], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, 0, v98
	v_add_f32_e32 v143, v143, v99
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
	v_add_f32_e32 v143, v143, v100
	v_add_f32_e32 v143, v143, v101
	v_add_f32_e32 v143, v143, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v103
	v_add_f32_e32 v143, v143, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v105
	v_add_f32_e32 v143, v143, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v107
	v_add_f32_e32 v143, v143, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v109
	v_add_f32_e32 v143, v143, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v143, v143, v111
	v_add_f32_e32 v143, v143, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v143, v143, v113
	;;#ASMSTART
	v_fma_f32 v142, v142, v117, v143
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_30
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
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s4, s6, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s4
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[148:151], v[106:107], off offset:512
	global_load_dwordx4 v[196:199], v[106:107], off offset:1024
	global_load_dwordx4 v[200:203], v[106:107], off offset:1536
	global_load_dwordx4 v[204:207], v[106:107], off offset:2048
	global_load_dwordx4 v[208:211], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[148:149], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[200:201], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[150:151], v[102:103], v[18:33]
	global_load_dwordx4 v[148:151], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[102:103], v[34:49]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[202:203], v[102:103], v[82:97]
	global_load_dwordx4 v[200:203], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[148:149], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[200:201], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[150:151], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[202:203], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	global_load_dwordx4 v[148:151], v[140:141], off offset:1024
	ds_write_b128 v119, v[144:147] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v143, 48, v98
	v_or_b32_e32 v98, v117, v143
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[144:147], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[162:163], v[98:113]
	v_or_b32_e32 v144, 16, v117
	v_xor_b32_e32 v144, v144, v143
	v_lshlrev_b32_e32 v144, 1, v144
	ds_read_b128 v[144:147], v144
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[166:167], v[98:113]
	v_or_b32_e32 v144, 32, v117
	v_xor_b32_e32 v144, v144, v143
	v_lshlrev_b32_e32 v144, 1, v144
	ds_read_b128 v[144:147], v144
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[170:171], v[98:113]
	v_or_b32_e32 v144, 48, v117
	v_xor_b32_e32 v143, v144, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[144:147], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[174:175], v[98:113]
	v_or_b32_e32 v143, 64, v117
	v_lshrrev_b32_e32 v144, 3, v143
	v_and_b32_e32 v144, 56, v144
	v_xor_b32_e32 v143, v144, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[144:147], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[178:179], v[98:113]
	v_or_b32_e32 v143, 0x50, v117
	v_lshrrev_b32_e32 v144, 3, v143
	v_and_b32_e32 v144, 56, v144
	v_xor_b32_e32 v143, v144, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[144:147], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[182:183], v[98:113]
	v_or_b32_e32 v143, 0x60, v117
	v_lshrrev_b32_e32 v144, 3, v143
	v_and_b32_e32 v144, 56, v144
	v_xor_b32_e32 v143, v144, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[144:147], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x70, v117
	v_lshrrev_b32_e32 v143, 3, v117
	v_and_b32_e32 v143, 56, v143
	v_xor_b32_e32 v117, v143, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[144:147], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[190:191], v[98:113]
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
	ds_bpermute_b32 v143, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v143, v143, v143
	v_max_f32_e32 v143, v117, v143
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v143, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_32
	;;#ASMSTART
	v_add_f32 v124, v143, v156
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
.LBB0_32:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[98:99], v[98:99], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, 0, v98
	v_add_f32_e32 v143, v143, v99
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
	v_add_f32_e32 v143, v143, v100
	v_add_f32_e32 v143, v143, v101
	v_add_f32_e32 v143, v143, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v103
	v_add_f32_e32 v143, v143, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v105
	v_add_f32_e32 v143, v143, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v107
	v_add_f32_e32 v143, v143, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v109
	v_add_f32_e32 v143, v143, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v143, v143, v111
	v_add_f32_e32 v143, v143, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v143, v143, v113
	;;#ASMSTART
	v_fma_f32 v142, v142, v117, v143
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_34
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
.LBB0_34:
	s_or_b64 exec, exec, s[4:5]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s6, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s6
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[144:147], v[106:107], off offset:512
	global_load_dwordx4 v[196:199], v[106:107], off offset:1024
	global_load_dwordx4 v[200:203], v[106:107], off offset:1536
	global_load_dwordx4 v[204:207], v[106:107], off offset:2048
	global_load_dwordx4 v[208:211], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[144:145], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[200:201], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[146:147], v[102:103], v[18:33]
	global_load_dwordx4 v[144:147], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[102:103], v[34:49]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[202:203], v[102:103], v[82:97]
	global_load_dwordx4 v[200:203], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[144:145], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[200:201], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[146:147], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[202:203], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	v_mov_b32_e32 v98, s10
	buffer_load_dword v117, v98, s[44:47], 0 offen
	global_load_dwordx4 v[144:147], v[140:141], off offset:1536
	ds_write_b128 v119, v[192:195]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s85, v117
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v140, 48, v98
	v_and_or_b32 v141, v99, s74, v100
	v_or_b32_e32 v98, v141, v140
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[192:195], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[162:163], v[98:113]
	v_or_b32_e32 v117, 0x1010, v141
	v_xor_b32_e32 v117, v117, v140
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[192:195], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[166:167], v[98:113]
	v_or_b32_e32 v117, 0x1020, v141
	v_xor_b32_e32 v117, v117, v140
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[192:195], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[170:171], v[98:113]
	v_or_b32_e32 v117, 0x1030, v141
	v_xor_b32_e32 v117, v117, v140
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[192:195], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[174:175], v[98:113]
	v_or_b32_e32 v117, 0x1040, v141
	v_lshrrev_b32_e32 v140, 3, v117
	v_and_b32_e32 v140, 56, v140
	v_xor_b32_e32 v117, v140, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[192:195], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[178:179], v[98:113]
	v_or_b32_e32 v117, 0x1050, v141
	v_lshrrev_b32_e32 v140, 3, v117
	v_and_b32_e32 v140, 56, v140
	v_xor_b32_e32 v117, v140, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[192:195], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[182:183], v[98:113]
	v_or_b32_e32 v117, 0x1060, v141
	v_lshrrev_b32_e32 v140, 3, v117
	v_and_b32_e32 v140, 56, v140
	v_xor_b32_e32 v117, v140, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[192:195], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x1070, v141
	v_lshrrev_b32_e32 v140, 3, v117
	v_and_b32_e32 v140, 56, v140
	v_xor_b32_e32 v117, v140, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[192:195], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[192:193], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[194:195], v[190:191], v[98:113]
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
	ds_bpermute_b32 v140, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v140, v140, v140
	v_max_f32_e32 v140, v117, v140
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v140, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_36
	;;#ASMSTART
	v_add_f32 v124, v140, v156
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
.LBB0_36:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[98:99], v[98:99], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v140, 0, v98
	v_add_f32_e32 v140, v140, v99
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
	v_add_f32_e32 v140, v140, v100
	v_add_f32_e32 v140, v140, v101
	v_add_f32_e32 v140, v140, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v140, v140, v103
	v_add_f32_e32 v140, v140, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v140, v140, v105
	v_add_f32_e32 v140, v140, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v140, v140, v107
	v_add_f32_e32 v140, v140, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v140, v140, v109
	v_add_f32_e32 v140, v140, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v140, v140, v111
	v_add_f32_e32 v140, v140, v112
	v_add_f32_e32 v140, v140, v113
	;;#ASMSTART
	v_fma_f32 v140, v142, v117, v140
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_38
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
.LBB0_38:
	s_or_b64 exec, exec, s[4:5]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s5, s84, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s5, s5, s82
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s5
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[192:195], v[106:107], off offset:512
	global_load_dwordx4 v[196:199], v[106:107], off offset:1024
	global_load_dwordx4 v[200:203], v[106:107], off offset:1536
	global_load_dwordx4 v[204:207], v[106:107], off offset:2048
	global_load_dwordx4 v[208:211], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[200:201], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[194:195], v[102:103], v[18:33]
	global_load_dwordx4 v[192:195], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[102:103], v[34:49]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[202:203], v[102:103], v[82:97]
	global_load_dwordx4 v[200:203], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[192:193], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[200:201], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[194:195], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[202:203], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	s_lshl_b32 s4, s11, 13
	v_or_b32_e32 v98, s4, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[62:63]
	global_load_dwordx4 v[200:203], v[98:99], off
	ds_write_b128 v119, v[148:151] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v141, 48, v98
	v_or_b32_e32 v98, v117, v141
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[148:151], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[162:163], v[98:113]
	v_or_b32_e32 v142, 16, v117
	v_xor_b32_e32 v142, v142, v141
	v_lshlrev_b32_e32 v142, 1, v142
	ds_read_b128 v[148:151], v142
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[166:167], v[98:113]
	v_or_b32_e32 v142, 32, v117
	v_xor_b32_e32 v142, v142, v141
	v_lshlrev_b32_e32 v142, 1, v142
	ds_read_b128 v[148:151], v142
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[170:171], v[98:113]
	v_or_b32_e32 v142, 48, v117
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[148:151], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[174:175], v[98:113]
	v_or_b32_e32 v141, 64, v117
	v_lshrrev_b32_e32 v142, 3, v141
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[148:151], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[178:179], v[98:113]
	v_or_b32_e32 v141, 0x50, v117
	v_lshrrev_b32_e32 v142, 3, v141
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[148:151], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[182:183], v[98:113]
	v_or_b32_e32 v141, 0x60, v117
	v_lshrrev_b32_e32 v142, 3, v141
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[148:151], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x70, v117
	v_lshrrev_b32_e32 v141, 3, v117
	v_and_b32_e32 v141, 56, v141
	v_xor_b32_e32 v117, v141, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[148:151], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[190:191], v[98:113]
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
	ds_bpermute_b32 v141, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v141, v141, v141
	v_max_f32_e32 v141, v117, v141
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v141, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_40
	;;#ASMSTART
	v_add_f32 v124, v141, v156
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
.LBB0_40:
	s_or_b64 exec, exec, s[6:7]
	v_pk_add_f32 v[98:99], v[98:99], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, 0, v98
	v_add_f32_e32 v141, v141, v99
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
	v_add_f32_e32 v141, v141, v100
	v_add_f32_e32 v141, v141, v101
	v_add_f32_e32 v141, v141, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v103
	v_add_f32_e32 v141, v141, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v105
	v_add_f32_e32 v141, v141, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v107
	v_add_f32_e32 v141, v141, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v109
	v_add_f32_e32 v141, v141, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v141, v141, v111
	v_add_f32_e32 v141, v141, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v141, v141, v113
	;;#ASMSTART
	v_fma_f32 v142, v140, v117, v141
	;;#ASMEND
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_42
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
.LBB0_42:
	s_or_b64 exec, exec, s[6:7]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s6, s5, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s6
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[148:151], v[106:107], off offset:512
	global_load_dwordx4 v[192:195], v[106:107], off offset:1024
	global_load_dwordx4 v[196:199], v[106:107], off offset:1536
	global_load_dwordx4 v[204:207], v[106:107], off offset:2048
	global_load_dwordx4 v[208:211], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[148:149], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[196:197], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[150:151], v[102:103], v[18:33]
	global_load_dwordx4 v[148:151], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[102:103], v[34:49]
	global_load_dwordx4 v[192:195], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[198:199], v[102:103], v[82:97]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[148:149], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[196:197], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[150:151], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[198:199], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	s_ashr_i32 s5, s4, 31
	v_lshl_add_u64 v[98:99], s[4:5], 0, v[114:115]
	v_lshl_add_u64 v[140:141], v[98:99], 2, s[62:63]
	global_load_dwordx4 v[192:195], v[140:141], off offset:512
	ds_write_b128 v119, v[144:147]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v143, 48, v98
	v_or_b32_e32 v98, v117, v143
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[144:147], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[162:163], v[98:113]
	v_or_b32_e32 v144, 0x1010, v117
	v_xor_b32_e32 v144, v144, v143
	v_lshlrev_b32_e32 v144, 1, v144
	ds_read_b128 v[144:147], v144
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[166:167], v[98:113]
	v_or_b32_e32 v144, 0x1020, v117
	v_xor_b32_e32 v144, v144, v143
	v_lshlrev_b32_e32 v144, 1, v144
	ds_read_b128 v[144:147], v144
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[170:171], v[98:113]
	v_or_b32_e32 v144, 0x1030, v117
	v_xor_b32_e32 v143, v144, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[144:147], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[174:175], v[98:113]
	v_or_b32_e32 v143, 0x1040, v117
	v_lshrrev_b32_e32 v144, 3, v143
	v_and_b32_e32 v144, 56, v144
	v_xor_b32_e32 v143, v144, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[144:147], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[178:179], v[98:113]
	v_or_b32_e32 v143, 0x1050, v117
	v_lshrrev_b32_e32 v144, 3, v143
	v_and_b32_e32 v144, 56, v144
	v_xor_b32_e32 v143, v144, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[144:147], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[182:183], v[98:113]
	v_or_b32_e32 v143, 0x1060, v117
	v_lshrrev_b32_e32 v144, 3, v143
	v_and_b32_e32 v144, 56, v144
	v_xor_b32_e32 v143, v144, v143
	v_lshlrev_b32_e32 v143, 1, v143
	ds_read_b128 v[144:147], v143
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x1070, v117
	v_lshrrev_b32_e32 v143, 3, v117
	v_and_b32_e32 v143, 56, v143
	v_xor_b32_e32 v117, v143, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[144:147], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[190:191], v[98:113]
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
	ds_bpermute_b32 v143, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v143, v143, v143
	v_max_f32_e32 v143, v117, v143
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v143, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_44
	;;#ASMSTART
	v_add_f32 v124, v143, v156
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
.LBB0_44:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[98:99], v[98:99], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, 0, v98
	v_add_f32_e32 v143, v143, v99
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
	v_add_f32_e32 v143, v143, v100
	v_add_f32_e32 v143, v143, v101
	v_add_f32_e32 v143, v143, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v103
	v_add_f32_e32 v143, v143, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v105
	v_add_f32_e32 v143, v143, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v107
	v_add_f32_e32 v143, v143, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v143, v143, v109
	v_add_f32_e32 v143, v143, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v143, v143, v111
	v_add_f32_e32 v143, v143, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v143, v143, v113
	;;#ASMSTART
	v_fma_f32 v142, v142, v117, v143
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_46
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
.LBB0_46:
	s_or_b64 exec, exec, s[4:5]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s4, s6, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s4
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[144:147], v[106:107], off offset:512
	global_load_dwordx4 v[148:151], v[106:107], off offset:1024
	global_load_dwordx4 v[196:199], v[106:107], off offset:1536
	global_load_dwordx4 v[204:207], v[106:107], off offset:2048
	global_load_dwordx4 v[208:211], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[100:101], v[2:17]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[144:145], v[100:101], v[18:33]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[148:149], v[100:101], v[34:49]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[196:197], v[100:101], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[204:205], v[100:101], v[50:65]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[208:209], v[100:101], v[66:81]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[102:103], v[2:17]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[18:33], v[146:147], v[102:103], v[18:33]
	global_load_dwordx4 v[144:147], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[34:49], v[150:151], v[102:103], v[34:49]
	global_load_dwordx4 v[148:151], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[82:97], v[198:199], v[102:103], v[82:97]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[50:65], v[206:207], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[210:211], v[102:103], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[108:109], v[104:105], v[2:17]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[144:145], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[148:149], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[196:197], v[104:105], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[110:111], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[106:107], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[100:101], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[146:147], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[150:151], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[198:199], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[108:109], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[102:103], v[98:99], v[66:81]
	global_load_dwordx4 v[196:199], v[140:141], off offset:1024
	ds_write_b128 v119, v[200:203] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v140, 48, v98
	v_or_b32_e32 v98, v117, v140
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[144:147], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[162:163], v[98:113]
	v_or_b32_e32 v141, 16, v117
	v_xor_b32_e32 v141, v141, v140
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[144:147], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[166:167], v[98:113]
	v_or_b32_e32 v141, 32, v117
	v_xor_b32_e32 v141, v141, v140
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[144:147], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[170:171], v[98:113]
	v_or_b32_e32 v141, 48, v117
	v_xor_b32_e32 v140, v141, v140
	v_lshlrev_b32_e32 v140, 1, v140
	ds_read_b128 v[144:147], v140
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[174:175], v[98:113]
	v_or_b32_e32 v140, 64, v117
	v_lshrrev_b32_e32 v141, 3, v140
	v_and_b32_e32 v141, 56, v141
	v_xor_b32_e32 v140, v141, v140
	v_lshlrev_b32_e32 v140, 1, v140
	ds_read_b128 v[144:147], v140
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[178:179], v[98:113]
	v_or_b32_e32 v140, 0x50, v117
	v_lshrrev_b32_e32 v141, 3, v140
	v_and_b32_e32 v141, 56, v141
	v_xor_b32_e32 v140, v141, v140
	v_lshlrev_b32_e32 v140, 1, v140
	ds_read_b128 v[144:147], v140
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[182:183], v[98:113]
	v_or_b32_e32 v140, 0x60, v117
	v_lshrrev_b32_e32 v141, 3, v140
	v_and_b32_e32 v141, 56, v141
	v_xor_b32_e32 v140, v141, v140
	v_lshlrev_b32_e32 v140, 1, v140
	ds_read_b128 v[144:147], v140
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x70, v117
	v_lshrrev_b32_e32 v140, 3, v117
	v_and_b32_e32 v140, 56, v140
	v_xor_b32_e32 v117, v140, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[144:147], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[190:191], v[98:113]
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
	ds_bpermute_b32 v140, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v140, v140, v140
	v_max_f32_e32 v140, v117, v140
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v140, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_48
	;;#ASMSTART
	v_add_f32 v124, v140, v156
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
.LBB0_48:
	s_or_b64 exec, exec, s[4:5]
	v_pk_add_f32 v[98:99], v[98:99], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[128:129] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[104:105], v[104:105], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v124, v124, v100
	v_add_f32_e32 v124, v124, v101
	v_add_f32_e32 v124, v124, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v124, v124, v103
	v_add_f32_e32 v124, v124, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v124, v124, v105
	v_add_f32_e32 v124, v124, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v124, v124, v107
	v_add_f32_e32 v124, v124, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[138:139] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v124, v124, v109
	v_add_f32_e32 v124, v124, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v124, v124, v111
	v_add_f32_e32 v124, v124, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v124, v124, v113
	;;#ASMSTART
	v_fma_f32 v158, v142, v117, v124
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_17
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
	v_mov_b32_e32 v125, v97
	v_mov_b32_e32 v124, v96
	v_mov_b32_e32 v127, v95
	v_mov_b32_e32 v126, v94
	v_mov_b32_e32 v129, v93
	v_mov_b32_e32 v128, v92
	v_mov_b32_e32 v131, v91
	v_mov_b32_e32 v130, v90
	v_mov_b32_e32 v133, v89
	v_mov_b32_e32 v132, v88
	v_mov_b32_e32 v135, v87
	v_mov_b32_e32 v134, v86
	v_mov_b32_e32 v137, v85
	v_mov_b32_e32 v136, v84
	v_mov_b32_e32 v139, v83
	v_mov_b32_e32 v138, v82
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
	v_mov_b32_e32 v122, v34
	v_mov_b32_e32 v123, v35
	v_mov_b32_e32 v140, v36
	v_mov_b32_e32 v141, v37
	v_mov_b32_e32 v142, v38
	v_mov_b32_e32 v143, v39
	v_mov_b32_e32 v144, v40
	v_mov_b32_e32 v145, v41
	v_mov_b32_e32 v146, v42
	v_mov_b32_e32 v147, v43
	v_mov_b32_e32 v148, v44
	v_mov_b32_e32 v149, v45
	v_mov_b32_e32 v150, v46
	v_mov_b32_e32 v151, v47
	v_mov_b32_e32 v152, v48
	v_mov_b32_e32 v153, v49
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
	s_mov_b32 s84, s12
	s_mov_b32 s66, s11
	s_cmp_ge_i32 s64, s60
	s_cbranch_scc0 .LBB0_52
	s_branch .LBB0_90
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
	v_mov_b32_e32 v122, v82
	v_mov_b32_e32 v123, v82
	v_mov_b32_e32 v140, v82
	v_mov_b32_e32 v141, v82
	v_mov_b32_e32 v142, v82
	v_mov_b32_e32 v143, v82
	v_mov_b32_e32 v144, v82
	v_mov_b32_e32 v145, v82
	v_mov_b32_e32 v146, v82
	v_mov_b32_e32 v147, v82
	v_mov_b32_e32 v148, v82
	v_mov_b32_e32 v149, v82
	v_mov_b32_e32 v150, v82
	v_mov_b32_e32 v151, v82
	v_mov_b32_e32 v152, v82
	v_mov_b32_e32 v153, v82
	v_mov_b32_e32 v138, v82
	v_mov_b32_e32 v139, v82
	v_mov_b32_e32 v136, v82
	v_mov_b32_e32 v137, v82
	v_mov_b32_e32 v134, v82
	v_mov_b32_e32 v135, v82
	v_mov_b32_e32 v132, v82
	v_mov_b32_e32 v133, v82
	v_mov_b32_e32 v130, v82
	v_mov_b32_e32 v131, v82
	v_mov_b32_e32 v128, v82
	v_mov_b32_e32 v129, v82
	v_mov_b32_e32 v126, v82
	v_mov_b32_e32 v127, v82
	v_mov_b32_e32 v124, v82
	v_mov_b32_e32 v125, v82
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
	v_mov_b32_e32 v158, v82
	v_mov_b32_e32 v118, 0xff800000
	s_cmp_ge_i32 s64, s60
	s_cbranch_scc1 .LBB0_90
.LBB0_52:
	s_lshl_b32 s86, s9, 7
	s_lshl_b32 s0, s64, 7
	s_ashr_i32 s61, s60, 31
	s_add_i32 s86, s86, s8
	v_mov_b32_e32 v120, v116
	v_mov_b32_e32 v121, v116
	s_add_i32 s87, s0, 0xf7
	s_add_i32 s88, s64, 1
	s_lshl2_add_u32 s89, s64, 20
	v_mov_b32_e32 v50, v122
	v_mov_b32_e32 v51, v123
	v_mov_b32_e32 v52, v140
	v_mov_b32_e32 v53, v141
	v_mov_b32_e32 v54, v142
	v_mov_b32_e32 v55, v143
	v_mov_b32_e32 v56, v144
	v_mov_b32_e32 v57, v145
	v_mov_b32_e32 v58, v146
	v_mov_b32_e32 v59, v147
	v_mov_b32_e32 v60, v148
	v_mov_b32_e32 v61, v149
	v_mov_b32_e32 v62, v150
	v_mov_b32_e32 v63, v151
	v_mov_b32_e32 v64, v152
	v_mov_b32_e32 v65, v153
	v_mov_b32_e32 v34, v138
	v_mov_b32_e32 v35, v139
	v_mov_b32_e32 v36, v136
	v_mov_b32_e32 v37, v137
	v_mov_b32_e32 v38, v134
	v_mov_b32_e32 v39, v135
	v_mov_b32_e32 v40, v132
	v_mov_b32_e32 v41, v133
	v_mov_b32_e32 v42, v130
	v_mov_b32_e32 v43, v131
	v_mov_b32_e32 v44, v128
	v_mov_b32_e32 v45, v129
	v_mov_b32_e32 v46, v126
	v_mov_b32_e32 v47, v127
	v_mov_b32_e32 v48, v124
	v_mov_b32_e32 v49, v125
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
	s_branch .LBB0_55
.LBB0_53:
	s_or_b64 exec, exec, s[0:1]
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
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s92, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s92
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[122:125], v[106:107], off offset:512
	global_load_dwordx4 v[126:129], v[106:107], off offset:1024
	global_load_dwordx4 v[130:133], v[106:107], off offset:1536
	global_load_dwordx4 v[134:137], v[106:107], off offset:2048
	global_load_dwordx4 v[138:141], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[126:127], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[138:139], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[102:103], v[66:81]
	global_load_dwordx4 v[122:125], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[102:103], v[50:65]
	global_load_dwordx4 v[126:129], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[132:133], v[102:103], v[34:49]
	global_load_dwordx4 v[130:133], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[136:137], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[140:141], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[126:127], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[132:133], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
.LBB0_54:
	s_and_b64 s[0:1], s[66:67], exec
	s_cselect_b32 s66, s83, s84
	s_cselect_b32 s84, s85, s83
	s_cselect_b32 s83, s90, s85
	s_add_u32 s64, s64, 2
	s_addc_u32 s65, s65, 0
	v_mov_b64_e32 v[98:99], s[60:61]
	v_cmp_lt_i64_e32 vcc, s[64:65], v[98:99]
	s_addk_i32 s87, 0x100
	s_add_i32 s88, s88, 2
	s_add_i32 s89, s89, 8
	s_mov_b32 s85, s91
	s_cbranch_vccz .LBB0_89
.LBB0_55:
	s_add_i32 s0, s89, -4
	v_mov_b32_e32 v98, s0
	s_lshl_b32 s0, s66, 13
	s_ashr_i32 s1, s0, 31
	buffer_load_dword v117, v98, s[44:47], 0 offen
	v_or_b32_e32 v98, s0, v114
	v_mov_b32_e32 v99, s1
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[62:63]
	global_load_dwordx4 v[146:149], v[98:99], off offset:1536
	ds_write_b128 v119, v[192:195]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s90, v117
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v122, 48, v98
	v_and_or_b32 v123, v99, s74, v100
	v_or_b32_e32 v98, v123, v122
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[124:127], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[162:163], v[98:113]
	v_or_b32_e32 v117, 0x1010, v123
	v_xor_b32_e32 v117, v117, v122
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[166:167], v[98:113]
	v_or_b32_e32 v117, 0x1020, v123
	v_xor_b32_e32 v117, v117, v122
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[170:171], v[98:113]
	v_or_b32_e32 v117, 0x1030, v123
	v_xor_b32_e32 v117, v117, v122
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[174:175], v[98:113]
	v_or_b32_e32 v117, 0x1040, v123
	v_lshrrev_b32_e32 v122, 3, v117
	v_and_b32_e32 v122, 56, v122
	v_xor_b32_e32 v117, v122, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[178:179], v[98:113]
	v_or_b32_e32 v117, 0x1050, v123
	v_lshrrev_b32_e32 v122, 3, v117
	v_and_b32_e32 v122, 56, v122
	v_xor_b32_e32 v117, v122, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[182:183], v[98:113]
	v_or_b32_e32 v117, 0x1060, v123
	v_lshrrev_b32_e32 v122, 3, v117
	v_and_b32_e32 v122, 56, v122
	v_xor_b32_e32 v117, v122, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[124:127], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x1070, v123
	v_lshrrev_b32_e32 v122, 3, v117
	v_and_b32_e32 v122, 56, v122
	v_xor_b32_e32 v117, v122, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[122:125], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[190:191], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v117
	v_and_b32_e32 v117, 8, v117
	v_add_u32_e32 v117, s87, v117
	v_add_u32_e32 v122, 0xffffff09, v117
	v_cmp_gt_i32_e32 vcc, s86, v122
	v_add_u32_e32 v122, 0xffffff0a, v117
	v_cmp_gt_i32_e64 s[0:1], s86, v122
	v_add_u32_e32 v122, 0xffffff0b, v117
	v_cmp_gt_i32_e64 s[4:5], s86, v122
	v_add_u32_e32 v122, 0xffffff0c, v117
	v_cmp_gt_i32_e64 s[6:7], s86, v122
	v_add_u32_e32 v122, 0xffffff0d, v117
	v_cmp_gt_i32_e64 s[8:9], s86, v122
	v_add_u32_e32 v122, 0xffffff0e, v117
	v_cmp_gt_i32_e64 s[10:11], s86, v122
	v_add_u32_e32 v122, 0xffffff0f, v117
	v_cmp_gt_i32_e64 s[12:13], s86, v122
	v_add_u32_e32 v122, 0xffffff10, v117
	v_cmp_gt_i32_e64 s[14:15], s86, v122
	v_add_u32_e32 v122, 0xffffff19, v117
	v_cmp_gt_i32_e64 s[16:17], s86, v122
	v_add_u32_e32 v122, 0xffffff1a, v117
	v_cmp_gt_i32_e64 s[18:19], s86, v122
	v_add_u32_e32 v122, 0xffffff1b, v117
	v_cmp_gt_i32_e64 s[20:21], s86, v122
	v_add_u32_e32 v122, 0xffffff1c, v117
	v_cmp_gt_i32_e64 s[22:23], s86, v122
	v_add_u32_e32 v122, 0xffffff1d, v117
	v_cmp_gt_i32_e64 s[24:25], s86, v122
	v_add_u32_e32 v122, 0xffffff1e, v117
	v_cmp_gt_i32_e64 s[26:27], s86, v122
	v_add_u32_e32 v122, 0xffffff1f, v117
	v_add_u32_e32 v117, 0xffffff20, v117
	v_cmp_gt_i32_e64 s[28:29], s86, v122
	v_cmp_gt_i32_e64 s[30:31], s86, v117
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
	v_cndmask_b32_e64 v101, v157, v101, s[6:7]
	v_cndmask_b32_e64 v100, v157, v100, s[4:5]
	v_cndmask_b32_e64 v123, v157, v99, s[0:1]
	v_cndmask_b32_e32 v122, v157, v98, vcc
	v_mov_b32_e32 v117, v116
	v_cndmask_b32_e64 v113, v157, v113, s[30:31]
	v_cndmask_b32_e64 v112, v157, v112, s[28:29]
	v_cndmask_b32_e64 v111, v157, v111, s[26:27]
	v_cndmask_b32_e64 v110, v157, v110, s[24:25]
	v_cndmask_b32_e64 v109, v157, v109, s[22:23]
	v_cndmask_b32_e64 v108, v157, v108, s[20:21]
	v_cndmask_b32_e64 v107, v157, v107, s[18:19]
	v_cndmask_b32_e64 v106, v157, v106, s[16:17]
	v_cndmask_b32_e64 v105, v157, v105, s[14:15]
	v_cndmask_b32_e64 v104, v157, v104, s[12:13]
	v_cndmask_b32_e64 v103, v157, v103, s[10:11]
	v_cndmask_b32_e64 v102, v157, v102, s[8:9]
	v_pk_mul_f32 v[98:99], v[116:117], v[100:101]
	v_pk_mul_f32 v[100:101], v[120:121], v[122:123]
	v_pk_mul_f32 v[112:113], v[116:117], v[112:113]
	v_pk_mul_f32 v[110:111], v[116:117], v[110:111]
	v_pk_mul_f32 v[108:109], v[116:117], v[108:109]
	v_pk_mul_f32 v[106:107], v[116:117], v[106:107]
	v_pk_mul_f32 v[104:105], v[116:117], v[104:105]
	v_pk_mul_f32 v[102:103], v[116:117], v[102:103]
	v_max_f32_e32 v117, v100, v101
	v_max3_f32 v117, v117, v98, v99
	v_max3_f32 v117, v117, v102, v103
	v_max3_f32 v117, v117, v104, v105
	v_max3_f32 v117, v117, v106, v107
	v_max3_f32 v117, v117, v108, v109
	v_max3_f32 v117, v117, v110, v111
	v_max3_f32 v117, v117, v112, v113
	ds_bpermute_b32 v122, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v122, v122, v122
	v_max_f32_e32 v122, v117, v122
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v122, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_57
	;;#ASMSTART
	v_add_f32 v122, v122, v156
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v122
	v_exp_f32_e32 v117, v117
	v_mov_b32_e32 v118, v122
.LBB0_57:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[122:123], v[98:99], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[98:99], v[100:101], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v100, v122
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v101, v123
	;;#ASMEND
	v_pk_add_f32 v[104:105], v[104:105], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, 0, v98
	v_add_f32_e32 v122, v122, v99
	v_add_f32_e32 v122, v122, v100
	v_add_f32_e32 v122, v122, v101
	;;#ASMSTART
	v_exp_f32 v102, v102
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v102
	v_add_f32_e32 v122, v122, v103
	v_add_f32_e32 v122, v122, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v105
	v_add_f32_e32 v122, v122, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v107
	v_add_f32_e32 v122, v122, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[118:119] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v109
	v_add_f32_e32 v122, v122, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v122, v122, v111
	v_add_f32_e32 v122, v122, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v122, v122, v113
	;;#ASMSTART
	v_fma_f32 v138, v158, v117, v122
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_59
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
.LBB0_59:
	s_or_b64 exec, exec, s[0:1]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s68, s66, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s68, s68, s82
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s68
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[122:125], v[106:107], off offset:512
	global_load_dwordx4 v[126:129], v[106:107], off offset:1024
	global_load_dwordx4 v[130:133], v[106:107], off offset:1536
	global_load_dwordx4 v[134:137], v[106:107], off offset:2048
	global_load_dwordx4 v[140:143], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[126:127], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[134:135], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[140:141], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[102:103], v[66:81]
	global_load_dwordx4 v[122:125], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[102:103], v[50:65]
	global_load_dwordx4 v[126:129], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[132:133], v[102:103], v[34:49]
	global_load_dwordx4 v[130:133], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[136:137], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[142:143], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[122:123], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[126:127], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[130:131], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[124:125], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[128:129], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[132:133], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	s_lshl_b32 s66, s84, 13
	v_or_b32_e32 v98, s66, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[62:63]
	global_load_dwordx4 v[142:145], v[98:99], off
	ds_write_b128 v119, v[196:199] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v122, 48, v98
	v_or_b32_e32 v98, v117, v122
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[124:127], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[162:163], v[98:113]
	v_or_b32_e32 v123, 16, v117
	v_xor_b32_e32 v123, v123, v122
	v_lshlrev_b32_e32 v123, 1, v123
	ds_read_b128 v[124:127], v123
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[166:167], v[98:113]
	v_or_b32_e32 v123, 32, v117
	v_xor_b32_e32 v123, v123, v122
	v_lshlrev_b32_e32 v123, 1, v123
	ds_read_b128 v[124:127], v123
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[126:127], v[170:171], v[98:113]
	v_or_b32_e32 v123, 48, v117
	v_xor_b32_e32 v122, v123, v122
	v_lshlrev_b32_e32 v122, 1, v122
	ds_read_b128 v[122:125], v122
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[174:175], v[98:113]
	v_or_b32_e32 v122, 64, v117
	v_lshrrev_b32_e32 v123, 3, v122
	v_and_b32_e32 v123, 56, v123
	v_xor_b32_e32 v122, v123, v122
	v_lshlrev_b32_e32 v122, 1, v122
	ds_read_b128 v[122:125], v122
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[178:179], v[98:113]
	v_or_b32_e32 v122, 0x50, v117
	v_lshrrev_b32_e32 v123, 3, v122
	v_and_b32_e32 v123, 56, v123
	v_xor_b32_e32 v122, v123, v122
	v_lshlrev_b32_e32 v122, 1, v122
	ds_read_b128 v[122:125], v122
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[182:183], v[98:113]
	v_or_b32_e32 v122, 0x60, v117
	v_lshrrev_b32_e32 v123, 3, v122
	v_and_b32_e32 v123, 56, v123
	v_xor_b32_e32 v122, v123, v122
	v_lshlrev_b32_e32 v122, 1, v122
	ds_read_b128 v[122:125], v122
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x70, v117
	v_lshrrev_b32_e32 v122, 3, v117
	v_and_b32_e32 v122, 56, v122
	v_xor_b32_e32 v117, v122, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[122:125], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[122:123], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[124:125], v[190:191], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	v_mov_b32_e32 v123, v118
	v_lshrrev_b32_e32 v117, 2, v117
	v_and_b32_e32 v117, 8, v117
	v_add_u32_e32 v117, s87, v117
	v_add_u32_e32 v122, 0xffffff29, v117
	v_cmp_gt_i32_e32 vcc, s86, v122
	v_add_u32_e32 v122, 0xffffff2a, v117
	v_cmp_gt_i32_e64 s[0:1], s86, v122
	v_add_u32_e32 v122, 0xffffff2b, v117
	v_cmp_gt_i32_e64 s[4:5], s86, v122
	v_add_u32_e32 v122, 0xffffff2c, v117
	v_cmp_gt_i32_e64 s[6:7], s86, v122
	v_add_u32_e32 v122, 0xffffff2d, v117
	v_cmp_gt_i32_e64 s[8:9], s86, v122
	v_add_u32_e32 v122, 0xffffff2e, v117
	v_cmp_gt_i32_e64 s[10:11], s86, v122
	v_add_u32_e32 v122, 0xffffff2f, v117
	v_cmp_gt_i32_e64 s[12:13], s86, v122
	v_add_u32_e32 v122, 0xffffff30, v117
	v_cmp_gt_i32_e64 s[14:15], s86, v122
	v_add_u32_e32 v122, 0xffffff39, v117
	v_cmp_gt_i32_e64 s[16:17], s86, v122
	v_add_u32_e32 v122, 0xffffff3a, v117
	v_cmp_gt_i32_e64 s[18:19], s86, v122
	v_add_u32_e32 v122, 0xffffff3b, v117
	v_cmp_gt_i32_e64 s[20:21], s86, v122
	v_add_u32_e32 v122, 0xffffff3c, v117
	v_cmp_gt_i32_e64 s[22:23], s86, v122
	v_add_u32_e32 v122, 0xffffff3d, v117
	v_cmp_gt_i32_e64 s[24:25], s86, v122
	v_add_u32_e32 v122, 0xffffff3e, v117
	v_cmp_gt_i32_e64 s[26:27], s86, v122
	v_add_u32_e32 v122, 0xffffff3f, v117
	v_add_u32_e32 v117, 0xffffff40, v117
	v_cmp_gt_i32_e64 s[28:29], s86, v122
	v_cmp_gt_i32_e64 s[30:31], s86, v117
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
	v_cndmask_b32_e64 v99, v157, v99, s[0:1]
	v_cndmask_b32_e32 v98, v157, v98, vcc
	v_cndmask_b32_e64 v113, v157, v113, s[30:31]
	v_cndmask_b32_e64 v112, v157, v112, s[28:29]
	v_cndmask_b32_e64 v111, v157, v111, s[26:27]
	v_cndmask_b32_e64 v110, v157, v110, s[24:25]
	v_cndmask_b32_e64 v109, v157, v109, s[22:23]
	v_cndmask_b32_e64 v108, v157, v108, s[20:21]
	v_cndmask_b32_e64 v107, v157, v107, s[18:19]
	v_cndmask_b32_e64 v106, v157, v106, s[16:17]
	v_cndmask_b32_e64 v105, v157, v105, s[14:15]
	v_cndmask_b32_e64 v104, v157, v104, s[12:13]
	v_cndmask_b32_e64 v103, v157, v103, s[10:11]
	v_cndmask_b32_e64 v102, v157, v102, s[8:9]
	v_cndmask_b32_e64 v101, v157, v101, s[6:7]
	v_cndmask_b32_e64 v100, v157, v100, s[4:5]
	v_mov_b32_e32 v117, v116
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
	ds_bpermute_b32 v122, v154, v117
	v_mov_b32_e32 v124, v118
	v_mov_b32_e32 v125, v118
	v_mov_b32_e32 v126, v118
	v_mov_b32_e32 v127, v118
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v122, v122, v122
	v_max_f32_e32 v139, v117, v122
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	v_mov_b32_e32 v122, v118
	v_cmp_gt_f32_e32 vcc, v139, v117
	v_mov_b32_e32 v117, 1.0
	v_mov_b32_e32 v128, v118
	v_mov_b32_e32 v129, v118
	v_mov_b32_e32 v130, v118
	v_mov_b32_e32 v131, v118
	v_mov_b32_e32 v132, v118
	v_mov_b32_e32 v133, v118
	v_mov_b32_e32 v134, v118
	v_mov_b32_e32 v135, v118
	v_mov_b32_e32 v136, v118
	v_mov_b32_e32 v137, v118
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_61
	;;#ASMSTART
	v_add_f32 v122, v139, v156
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v122
	v_exp_f32_e32 v117, v117
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
.LBB0_61:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[98:99], v[98:99], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v139, 0, v98
	v_add_f32_e32 v139, v139, v99
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
	v_add_f32_e32 v139, v139, v100
	v_add_f32_e32 v139, v139, v101
	v_add_f32_e32 v139, v139, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v139, v139, v103
	v_add_f32_e32 v139, v139, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v139, v139, v105
	v_add_f32_e32 v139, v139, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v139, v139, v107
	v_add_f32_e32 v139, v139, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v139, v139, v109
	v_add_f32_e32 v139, v139, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v139, v139, v111
	v_add_f32_e32 v139, v139, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v139, v139, v113
	;;#ASMSTART
	v_fma_f32 v140, v138, v117, v139
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_63
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
.LBB0_63:
	s_or_b64 exec, exec, s[0:1]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s68, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s68
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[150:153], v[106:107], off offset:512
	global_load_dwordx4 v[192:195], v[106:107], off offset:1024
	global_load_dwordx4 v[196:199], v[106:107], off offset:1536
	global_load_dwordx4 v[200:203], v[106:107], off offset:2048
	global_load_dwordx4 v[204:207], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[150:151], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[200:201], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[152:153], v[102:103], v[66:81]
	global_load_dwordx4 v[150:153], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[194:195], v[102:103], v[50:65]
	global_load_dwordx4 v[192:195], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[102:103], v[34:49]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[202:203], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[206:207], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[150:151], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[152:153], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[194:195], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	s_ashr_i32 s67, s66, 31
	v_lshl_add_u64 v[98:99], s[66:67], 0, v[114:115]
	v_lshl_add_u64 v[138:139], v[98:99], 2, s[62:63]
	global_load_dwordx4 v[192:195], v[138:139], off offset:512
	ds_write_b128 v119, v[146:149]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v141, 48, v98
	v_or_b32_e32 v98, v117, v141
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[146:149], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[162:163], v[98:113]
	v_or_b32_e32 v146, 0x1010, v117
	v_xor_b32_e32 v146, v146, v141
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[146:149], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[166:167], v[98:113]
	v_or_b32_e32 v146, 0x1020, v117
	v_xor_b32_e32 v146, v146, v141
	v_lshlrev_b32_e32 v146, 1, v146
	ds_read_b128 v[146:149], v146
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[170:171], v[98:113]
	v_or_b32_e32 v146, 0x1030, v117
	v_xor_b32_e32 v141, v146, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[146:149], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[174:175], v[98:113]
	v_or_b32_e32 v141, 0x1040, v117
	v_lshrrev_b32_e32 v146, 3, v141
	v_and_b32_e32 v146, 56, v146
	v_xor_b32_e32 v141, v146, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[146:149], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[178:179], v[98:113]
	v_or_b32_e32 v141, 0x1050, v117
	v_lshrrev_b32_e32 v146, 3, v141
	v_and_b32_e32 v146, 56, v146
	v_xor_b32_e32 v141, v146, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[146:149], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[182:183], v[98:113]
	v_or_b32_e32 v141, 0x1060, v117
	v_lshrrev_b32_e32 v146, 3, v141
	v_and_b32_e32 v146, 56, v146
	v_xor_b32_e32 v141, v146, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[146:149], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x1070, v117
	v_lshrrev_b32_e32 v141, 3, v117
	v_and_b32_e32 v141, 56, v141
	v_xor_b32_e32 v117, v141, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[146:149], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[190:191], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v117
	v_and_b32_e32 v117, 8, v117
	v_add_u32_e32 v117, s87, v117
	v_add_u32_e32 v141, 0xffffff49, v117
	v_cmp_gt_i32_e32 vcc, s86, v141
	v_add_u32_e32 v141, 0xffffff4a, v117
	v_cmp_gt_i32_e64 s[0:1], s86, v141
	v_add_u32_e32 v141, 0xffffff4b, v117
	v_cmp_gt_i32_e64 s[4:5], s86, v141
	v_add_u32_e32 v141, 0xffffff4c, v117
	v_cmp_gt_i32_e64 s[6:7], s86, v141
	v_add_u32_e32 v141, 0xffffff4d, v117
	v_cmp_gt_i32_e64 s[8:9], s86, v141
	v_add_u32_e32 v141, 0xffffff4e, v117
	v_cmp_gt_i32_e64 s[10:11], s86, v141
	v_add_u32_e32 v141, 0xffffff4f, v117
	v_cmp_gt_i32_e64 s[12:13], s86, v141
	v_add_u32_e32 v141, 0xffffff50, v117
	v_cmp_gt_i32_e64 s[14:15], s86, v141
	v_add_u32_e32 v141, 0xffffff59, v117
	v_cmp_gt_i32_e64 s[16:17], s86, v141
	v_add_u32_e32 v141, 0xffffff5a, v117
	v_cmp_gt_i32_e64 s[18:19], s86, v141
	v_add_u32_e32 v141, 0xffffff5b, v117
	v_cmp_gt_i32_e64 s[20:21], s86, v141
	v_add_u32_e32 v141, 0xffffff5c, v117
	v_cmp_gt_i32_e64 s[22:23], s86, v141
	v_add_u32_e32 v141, 0xffffff5d, v117
	v_cmp_gt_i32_e64 s[24:25], s86, v141
	v_add_u32_e32 v141, 0xffffff5e, v117
	v_cmp_gt_i32_e64 s[26:27], s86, v141
	v_add_u32_e32 v141, 0xffffff5f, v117
	v_add_u32_e32 v117, 0xffffff60, v117
	v_cmp_gt_i32_e64 s[28:29], s86, v141
	v_cmp_gt_i32_e64 s[30:31], s86, v117
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
	v_cndmask_b32_e64 v99, v157, v99, s[0:1]
	v_cndmask_b32_e32 v98, v157, v98, vcc
	v_cndmask_b32_e64 v113, v157, v113, s[30:31]
	v_cndmask_b32_e64 v112, v157, v112, s[28:29]
	v_cndmask_b32_e64 v111, v157, v111, s[26:27]
	v_cndmask_b32_e64 v110, v157, v110, s[24:25]
	v_cndmask_b32_e64 v109, v157, v109, s[22:23]
	v_cndmask_b32_e64 v108, v157, v108, s[20:21]
	v_cndmask_b32_e64 v107, v157, v107, s[18:19]
	v_cndmask_b32_e64 v106, v157, v106, s[16:17]
	v_cndmask_b32_e64 v105, v157, v105, s[14:15]
	v_cndmask_b32_e64 v104, v157, v104, s[12:13]
	v_cndmask_b32_e64 v103, v157, v103, s[10:11]
	v_cndmask_b32_e64 v102, v157, v102, s[8:9]
	v_cndmask_b32_e64 v101, v157, v101, s[6:7]
	v_cndmask_b32_e64 v100, v157, v100, s[4:5]
	v_mov_b32_e32 v117, v116
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
	ds_bpermute_b32 v141, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v141, v141, v141
	v_max_f32_e32 v141, v117, v141
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v141, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_65
	;;#ASMSTART
	v_add_f32 v122, v141, v156
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v122
	v_exp_f32_e32 v117, v117
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
.LBB0_65:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[98:99], v[98:99], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, 0, v98
	v_add_f32_e32 v141, v141, v99
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
	v_add_f32_e32 v141, v141, v100
	v_add_f32_e32 v141, v141, v101
	v_add_f32_e32 v141, v141, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v103
	v_add_f32_e32 v141, v141, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v105
	v_add_f32_e32 v141, v141, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v107
	v_add_f32_e32 v141, v141, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v109
	v_add_f32_e32 v141, v141, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v141, v141, v111
	v_add_f32_e32 v141, v141, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v141, v141, v113
	;;#ASMSTART
	v_fma_f32 v140, v140, v117, v141
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_67
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
.LBB0_67:
	s_or_b64 exec, exec, s[0:1]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s0, s68, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s0
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[146:149], v[106:107], off offset:512
	global_load_dwordx4 v[150:153], v[106:107], off offset:1024
	global_load_dwordx4 v[196:199], v[106:107], off offset:1536
	global_load_dwordx4 v[200:203], v[106:107], off offset:2048
	global_load_dwordx4 v[204:207], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[200:201], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[148:149], v[102:103], v[66:81]
	global_load_dwordx4 v[146:149], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[152:153], v[102:103], v[50:65]
	global_load_dwordx4 v[150:153], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[102:103], v[34:49]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[202:203], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[206:207], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[148:149], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[152:153], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	global_load_dwordx4 v[196:199], v[138:139], off offset:1024
	ds_write_b128 v119, v[142:145] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v141, 48, v98
	v_or_b32_e32 v98, v117, v141
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[142:145], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[162:163], v[98:113]
	v_or_b32_e32 v142, 16, v117
	v_xor_b32_e32 v142, v142, v141
	v_lshlrev_b32_e32 v142, 1, v142
	ds_read_b128 v[142:145], v142
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[166:167], v[98:113]
	v_or_b32_e32 v142, 32, v117
	v_xor_b32_e32 v142, v142, v141
	v_lshlrev_b32_e32 v142, 1, v142
	ds_read_b128 v[142:145], v142
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[170:171], v[98:113]
	v_or_b32_e32 v142, 48, v117
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[142:145], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[174:175], v[98:113]
	v_or_b32_e32 v141, 64, v117
	v_lshrrev_b32_e32 v142, 3, v141
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[142:145], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[178:179], v[98:113]
	v_or_b32_e32 v141, 0x50, v117
	v_lshrrev_b32_e32 v142, 3, v141
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[142:145], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[182:183], v[98:113]
	v_or_b32_e32 v141, 0x60, v117
	v_lshrrev_b32_e32 v142, 3, v141
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[142:145], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x70, v117
	v_lshrrev_b32_e32 v141, 3, v117
	v_and_b32_e32 v141, 56, v141
	v_xor_b32_e32 v117, v141, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[142:145], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[190:191], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v117
	v_and_b32_e32 v117, 8, v117
	v_add_u32_e32 v117, s87, v117
	v_add_u32_e32 v141, 0xffffff69, v117
	v_cmp_gt_i32_e32 vcc, s86, v141
	v_add_u32_e32 v141, 0xffffff6a, v117
	v_cmp_gt_i32_e64 s[0:1], s86, v141
	v_add_u32_e32 v141, 0xffffff6b, v117
	v_cmp_gt_i32_e64 s[4:5], s86, v141
	v_add_u32_e32 v141, 0xffffff6c, v117
	v_cmp_gt_i32_e64 s[6:7], s86, v141
	v_add_u32_e32 v141, 0xffffff6d, v117
	v_cmp_gt_i32_e64 s[8:9], s86, v141
	v_add_u32_e32 v141, 0xffffff6e, v117
	v_cmp_gt_i32_e64 s[10:11], s86, v141
	v_add_u32_e32 v141, 0xffffff6f, v117
	v_cmp_gt_i32_e64 s[12:13], s86, v141
	v_add_u32_e32 v141, 0xffffff70, v117
	v_cmp_gt_i32_e64 s[14:15], s86, v141
	v_add_u32_e32 v141, 0xffffff79, v117
	v_cmp_gt_i32_e64 s[16:17], s86, v141
	v_add_u32_e32 v141, 0xffffff7a, v117
	v_cmp_gt_i32_e64 s[18:19], s86, v141
	v_add_u32_e32 v141, 0xffffff7b, v117
	v_cmp_gt_i32_e64 s[20:21], s86, v141
	v_add_u32_e32 v141, 0xffffff7c, v117
	v_cmp_gt_i32_e64 s[22:23], s86, v141
	v_add_u32_e32 v141, 0xffffff7d, v117
	v_cmp_gt_i32_e64 s[24:25], s86, v141
	v_add_u32_e32 v141, 0xffffff7e, v117
	v_cmp_gt_i32_e64 s[26:27], s86, v141
	v_add_u32_e32 v141, 0xffffff7f, v117
	v_add_u32_e32 v117, 0xffffff80, v117
	v_cmp_gt_i32_e64 s[28:29], s86, v141
	v_cmp_gt_i32_e64 s[30:31], s86, v117
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
	v_cndmask_b32_e64 v99, v157, v99, s[0:1]
	v_cndmask_b32_e32 v98, v157, v98, vcc
	v_cndmask_b32_e64 v113, v157, v113, s[30:31]
	v_cndmask_b32_e64 v112, v157, v112, s[28:29]
	v_cndmask_b32_e64 v111, v157, v111, s[26:27]
	v_cndmask_b32_e64 v110, v157, v110, s[24:25]
	v_cndmask_b32_e64 v109, v157, v109, s[22:23]
	v_cndmask_b32_e64 v108, v157, v108, s[20:21]
	v_cndmask_b32_e64 v107, v157, v107, s[18:19]
	v_cndmask_b32_e64 v106, v157, v106, s[16:17]
	v_cndmask_b32_e64 v105, v157, v105, s[14:15]
	v_cndmask_b32_e64 v104, v157, v104, s[12:13]
	v_cndmask_b32_e64 v103, v157, v103, s[10:11]
	v_cndmask_b32_e64 v102, v157, v102, s[8:9]
	v_cndmask_b32_e64 v101, v157, v101, s[6:7]
	v_cndmask_b32_e64 v100, v157, v100, s[4:5]
	v_mov_b32_e32 v117, v116
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
	ds_bpermute_b32 v141, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v141, v141, v141
	v_max_f32_e32 v141, v117, v141
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v141, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_69
	;;#ASMSTART
	v_add_f32 v122, v141, v156
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v122
	v_exp_f32_e32 v117, v117
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
.LBB0_69:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[98:99], v[98:99], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, 0, v98
	v_add_f32_e32 v141, v141, v99
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
	v_add_f32_e32 v141, v141, v100
	v_add_f32_e32 v141, v141, v101
	v_add_f32_e32 v141, v141, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v103
	v_add_f32_e32 v141, v141, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v105
	v_add_f32_e32 v141, v141, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v107
	v_add_f32_e32 v141, v141, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v109
	v_add_f32_e32 v141, v141, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v141, v141, v111
	v_add_f32_e32 v141, v141, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v141, v141, v113
	;;#ASMSTART
	v_fma_f32 v158, v140, v117, v141
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_71
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
.LBB0_71:
	s_or_b64 exec, exec, s[0:1]
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
	v_add_u32_e32 v117, 0x8000, v100
	;;#ASMSTART
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_addk_i32 s68, 0x3000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s68
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[140:143], v[106:107], off offset:512
	global_load_dwordx4 v[144:147], v[106:107], off offset:1024
	global_load_dwordx4 v[148:151], v[106:107], off offset:1536
	global_load_dwordx4 v[200:203], v[106:107], off offset:2048
	global_load_dwordx4 v[204:207], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[140:141], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[144:145], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[148:149], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[200:201], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[102:103], v[66:81]
	global_load_dwordx4 v[140:143], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[146:147], v[102:103], v[50:65]
	global_load_dwordx4 v[144:147], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[150:151], v[102:103], v[34:49]
	global_load_dwordx4 v[148:151], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[202:203], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[206:207], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[140:141], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[144:145], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[148:149], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[146:147], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[150:151], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	s_cmp_gt_i32 s60, s88
	s_cselect_b64 s[66:67], -1, 0
	s_cmp_le_i32 s60, s88
	s_cbranch_scc1 .LBB0_88
	v_mov_b32_e32 v98, s89
	buffer_load_dword v117, v98, s[44:47], 0 offen
	global_load_dwordx4 v[142:145], v[138:139], off offset:1536
	ds_write_b128 v119, v[192:195]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s91, v117
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_b32_e32 v100, 8, v100
	v_and_b32_e32 v138, 48, v98
	v_and_or_b32 v139, v99, s74, v100
	v_or_b32_e32 v98, v139, v138
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[146:149], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[162:163], v[98:113]
	v_or_b32_e32 v117, 0x1010, v139
	v_xor_b32_e32 v117, v117, v138
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[146:149], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[166:167], v[98:113]
	v_or_b32_e32 v117, 0x1020, v139
	v_xor_b32_e32 v117, v117, v138
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[146:149], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[170:171], v[98:113]
	v_or_b32_e32 v117, 0x1030, v139
	v_xor_b32_e32 v117, v117, v138
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[146:149], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[174:175], v[98:113]
	v_or_b32_e32 v117, 0x1040, v139
	v_lshrrev_b32_e32 v138, 3, v117
	v_and_b32_e32 v138, 56, v138
	v_xor_b32_e32 v117, v138, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[146:149], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[178:179], v[98:113]
	v_or_b32_e32 v117, 0x1050, v139
	v_lshrrev_b32_e32 v138, 3, v117
	v_and_b32_e32 v138, 56, v138
	v_xor_b32_e32 v117, v138, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[146:149], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[182:183], v[98:113]
	v_or_b32_e32 v117, 0x1060, v139
	v_lshrrev_b32_e32 v138, 3, v117
	v_and_b32_e32 v138, 56, v138
	v_xor_b32_e32 v117, v138, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[146:149], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[146:147], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x1070, v139
	v_lshrrev_b32_e32 v138, 3, v117
	v_and_b32_e32 v138, 56, v138
	v_xor_b32_e32 v117, v138, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[138:141], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[138:139], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[140:141], v[190:191], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v117
	v_and_b32_e32 v117, 8, v117
	v_add_u32_e32 v117, s87, v117
	v_add_u32_e32 v138, 0xffffff89, v117
	v_cmp_gt_i32_e32 vcc, s86, v138
	v_add_u32_e32 v138, 0xffffff8a, v117
	v_cmp_gt_i32_e64 s[0:1], s86, v138
	v_add_u32_e32 v138, 0xffffff8b, v117
	v_cmp_gt_i32_e64 s[4:5], s86, v138
	v_add_u32_e32 v138, 0xffffff8c, v117
	v_cmp_gt_i32_e64 s[6:7], s86, v138
	v_add_u32_e32 v138, 0xffffff8d, v117
	v_cmp_gt_i32_e64 s[8:9], s86, v138
	v_add_u32_e32 v138, 0xffffff8e, v117
	v_cmp_gt_i32_e64 s[10:11], s86, v138
	v_add_u32_e32 v138, 0xffffff8f, v117
	v_cmp_gt_i32_e64 s[12:13], s86, v138
	v_add_u32_e32 v138, 0xffffff90, v117
	v_cmp_gt_i32_e64 s[14:15], s86, v138
	v_add_u32_e32 v138, 0xffffff99, v117
	v_cmp_gt_i32_e64 s[16:17], s86, v138
	v_add_u32_e32 v138, 0xffffff9a, v117
	v_cmp_gt_i32_e64 s[18:19], s86, v138
	v_add_u32_e32 v138, 0xffffff9b, v117
	v_cmp_gt_i32_e64 s[20:21], s86, v138
	v_add_u32_e32 v138, 0xffffff9c, v117
	v_cmp_gt_i32_e64 s[22:23], s86, v138
	v_add_u32_e32 v138, 0xffffff9d, v117
	v_cmp_gt_i32_e64 s[24:25], s86, v138
	v_add_u32_e32 v138, 0xffffff9e, v117
	v_cmp_gt_i32_e64 s[26:27], s86, v138
	v_add_u32_e32 v138, 0xffffff9f, v117
	v_add_u32_e32 v117, 0xffffffa0, v117
	v_cmp_gt_i32_e64 s[28:29], s86, v138
	v_cmp_gt_i32_e64 s[30:31], s86, v117
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
	v_cndmask_b32_e64 v99, v157, v99, s[0:1]
	v_cndmask_b32_e32 v98, v157, v98, vcc
	v_cndmask_b32_e64 v113, v157, v113, s[30:31]
	v_cndmask_b32_e64 v112, v157, v112, s[28:29]
	v_cndmask_b32_e64 v111, v157, v111, s[26:27]
	v_cndmask_b32_e64 v110, v157, v110, s[24:25]
	v_cndmask_b32_e64 v109, v157, v109, s[22:23]
	v_cndmask_b32_e64 v108, v157, v108, s[20:21]
	v_cndmask_b32_e64 v107, v157, v107, s[18:19]
	v_cndmask_b32_e64 v106, v157, v106, s[16:17]
	v_cndmask_b32_e64 v105, v157, v105, s[14:15]
	v_cndmask_b32_e64 v104, v157, v104, s[12:13]
	v_cndmask_b32_e64 v103, v157, v103, s[10:11]
	v_cndmask_b32_e64 v102, v157, v102, s[8:9]
	v_cndmask_b32_e64 v101, v157, v101, s[6:7]
	v_cndmask_b32_e64 v100, v157, v100, s[4:5]
	v_mov_b32_e32 v117, v116
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
	ds_bpermute_b32 v138, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v138, v138, v138
	v_max_f32_e32 v138, v117, v138
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v138, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_74
	;;#ASMSTART
	v_add_f32 v122, v138, v156
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v122
	v_exp_f32_e32 v117, v117
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
.LBB0_74:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[98:99], v[98:99], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v138, 0, v98
	v_add_f32_e32 v138, v138, v99
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
	v_add_f32_e32 v138, v138, v100
	v_add_f32_e32 v138, v138, v101
	v_add_f32_e32 v138, v138, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v138, v138, v103
	v_add_f32_e32 v138, v138, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v138, v138, v105
	v_add_f32_e32 v138, v138, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v138, v138, v107
	v_add_f32_e32 v138, v138, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v138, v138, v109
	v_add_f32_e32 v138, v138, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v138, v138, v111
	v_add_f32_e32 v138, v138, v112
	v_add_f32_e32 v138, v138, v113
	;;#ASMSTART
	v_fma_f32 v138, v158, v117, v138
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_76
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
.LBB0_76:
	s_or_b64 exec, exec, s[0:1]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_mul_i32 s92, s84, 0x6000
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	s_add_i32 s92, s92, s82
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s92
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[146:149], v[106:107], off offset:512
	global_load_dwordx4 v[150:153], v[106:107], off offset:1024
	global_load_dwordx4 v[192:195], v[106:107], off offset:1536
	global_load_dwordx4 v[200:203], v[106:107], off offset:2048
	global_load_dwordx4 v[204:207], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[200:201], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[148:149], v[102:103], v[66:81]
	global_load_dwordx4 v[146:149], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[152:153], v[102:103], v[50:65]
	global_load_dwordx4 v[150:153], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[102:103], v[34:49]
	global_load_dwordx4 v[192:195], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[202:203], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[206:207], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[192:193], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[148:149], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[152:153], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[194:195], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	s_lshl_b32 s68, s83, 13
	v_or_b32_e32 v98, s68, v114
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[62:63]
	global_load_dwordx4 v[146:149], v[98:99], off
	ds_write_b128 v119, v[196:199] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v139, 48, v98
	v_or_b32_e32 v98, v117, v139
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[150:153], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[152:153], v[162:163], v[98:113]
	v_or_b32_e32 v140, 16, v117
	v_xor_b32_e32 v140, v140, v139
	v_lshlrev_b32_e32 v140, 1, v140
	ds_read_b128 v[150:153], v140
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[152:153], v[166:167], v[98:113]
	v_or_b32_e32 v140, 32, v117
	v_xor_b32_e32 v140, v140, v139
	v_lshlrev_b32_e32 v140, 1, v140
	ds_read_b128 v[150:153], v140
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[152:153], v[170:171], v[98:113]
	v_or_b32_e32 v140, 48, v117
	v_xor_b32_e32 v139, v140, v139
	v_lshlrev_b32_e32 v139, 1, v139
	ds_read_b128 v[150:153], v139
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[152:153], v[174:175], v[98:113]
	v_or_b32_e32 v139, 64, v117
	v_lshrrev_b32_e32 v140, 3, v139
	v_and_b32_e32 v140, 56, v140
	v_xor_b32_e32 v139, v140, v139
	v_lshlrev_b32_e32 v139, 1, v139
	ds_read_b128 v[150:153], v139
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[152:153], v[178:179], v[98:113]
	v_or_b32_e32 v139, 0x50, v117
	v_lshrrev_b32_e32 v140, 3, v139
	v_and_b32_e32 v140, 56, v140
	v_xor_b32_e32 v139, v140, v139
	v_lshlrev_b32_e32 v139, 1, v139
	ds_read_b128 v[150:153], v139
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[152:153], v[182:183], v[98:113]
	v_or_b32_e32 v139, 0x60, v117
	v_lshrrev_b32_e32 v140, 3, v139
	v_and_b32_e32 v140, 56, v140
	v_xor_b32_e32 v139, v140, v139
	v_lshlrev_b32_e32 v139, 1, v139
	ds_read_b128 v[150:153], v139
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[152:153], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x70, v117
	v_lshrrev_b32_e32 v139, 3, v117
	v_and_b32_e32 v139, 56, v139
	v_xor_b32_e32 v117, v139, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[150:153], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[150:151], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[152:153], v[190:191], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v117
	v_and_b32_e32 v117, 8, v117
	v_add_u32_e32 v117, s87, v117
	v_add_u32_e32 v139, 0xffffffa9, v117
	v_cmp_gt_i32_e32 vcc, s86, v139
	v_add_u32_e32 v139, 0xffffffaa, v117
	v_cmp_gt_i32_e64 s[0:1], s86, v139
	v_add_u32_e32 v139, 0xffffffab, v117
	v_cmp_gt_i32_e64 s[4:5], s86, v139
	v_add_u32_e32 v139, 0xffffffac, v117
	v_cmp_gt_i32_e64 s[6:7], s86, v139
	v_add_u32_e32 v139, 0xffffffad, v117
	v_cmp_gt_i32_e64 s[8:9], s86, v139
	v_add_u32_e32 v139, 0xffffffae, v117
	v_cmp_gt_i32_e64 s[10:11], s86, v139
	v_add_u32_e32 v139, 0xffffffaf, v117
	v_cmp_gt_i32_e64 s[12:13], s86, v139
	v_add_u32_e32 v139, 0xffffffb0, v117
	v_cmp_gt_i32_e64 s[14:15], s86, v139
	v_add_u32_e32 v139, 0xffffffb9, v117
	v_cmp_gt_i32_e64 s[16:17], s86, v139
	v_add_u32_e32 v139, 0xffffffba, v117
	v_cmp_gt_i32_e64 s[18:19], s86, v139
	v_add_u32_e32 v139, 0xffffffbb, v117
	v_cmp_gt_i32_e64 s[20:21], s86, v139
	v_add_u32_e32 v139, 0xffffffbc, v117
	v_cmp_gt_i32_e64 s[22:23], s86, v139
	v_add_u32_e32 v139, 0xffffffbd, v117
	v_cmp_gt_i32_e64 s[24:25], s86, v139
	v_add_u32_e32 v139, 0xffffffbe, v117
	v_cmp_gt_i32_e64 s[26:27], s86, v139
	v_add_u32_e32 v139, 0xffffffbf, v117
	v_subrev_u32_e32 v117, 64, v117
	v_cmp_gt_i32_e64 s[28:29], s86, v139
	v_cmp_gt_i32_e64 s[30:31], s86, v117
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
	v_cndmask_b32_e64 v99, v157, v99, s[0:1]
	v_cndmask_b32_e32 v98, v157, v98, vcc
	v_cndmask_b32_e64 v113, v157, v113, s[30:31]
	v_cndmask_b32_e64 v112, v157, v112, s[28:29]
	v_cndmask_b32_e64 v111, v157, v111, s[26:27]
	v_cndmask_b32_e64 v110, v157, v110, s[24:25]
	v_cndmask_b32_e64 v109, v157, v109, s[22:23]
	v_cndmask_b32_e64 v108, v157, v108, s[20:21]
	v_cndmask_b32_e64 v107, v157, v107, s[18:19]
	v_cndmask_b32_e64 v106, v157, v106, s[16:17]
	v_cndmask_b32_e64 v105, v157, v105, s[14:15]
	v_cndmask_b32_e64 v104, v157, v104, s[12:13]
	v_cndmask_b32_e64 v103, v157, v103, s[10:11]
	v_cndmask_b32_e64 v102, v157, v102, s[8:9]
	v_cndmask_b32_e64 v101, v157, v101, s[6:7]
	v_cndmask_b32_e64 v100, v157, v100, s[4:5]
	v_mov_b32_e32 v117, v116
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
	ds_bpermute_b32 v139, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v139, v139, v139
	v_max_f32_e32 v139, v117, v139
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v139, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_78
	;;#ASMSTART
	v_add_f32 v122, v139, v156
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v122
	v_exp_f32_e32 v117, v117
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
.LBB0_78:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[98:99], v[98:99], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v139, 0, v98
	v_add_f32_e32 v139, v139, v99
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
	v_add_f32_e32 v139, v139, v100
	v_add_f32_e32 v139, v139, v101
	v_add_f32_e32 v139, v139, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v139, v139, v103
	v_add_f32_e32 v139, v139, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v139, v139, v105
	v_add_f32_e32 v139, v139, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v139, v139, v107
	v_add_f32_e32 v139, v139, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v139, v139, v109
	v_add_f32_e32 v139, v139, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v139, v139, v111
	v_add_f32_e32 v139, v139, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v139, v139, v113
	;;#ASMSTART
	v_fma_f32 v140, v138, v117, v139
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_80
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
.LBB0_80:
	s_or_b64 exec, exec, s[0:1]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
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
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[150:153], v[106:107], off offset:512
	global_load_dwordx4 v[192:195], v[106:107], off offset:1024
	global_load_dwordx4 v[196:199], v[106:107], off offset:1536
	global_load_dwordx4 v[200:203], v[106:107], off offset:2048
	global_load_dwordx4 v[204:207], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[150:151], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[200:201], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[152:153], v[102:103], v[66:81]
	global_load_dwordx4 v[150:153], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[194:195], v[102:103], v[50:65]
	global_load_dwordx4 v[192:195], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[102:103], v[34:49]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[202:203], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[206:207], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[150:151], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[192:193], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[152:153], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[194:195], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	s_ashr_i32 s69, s68, 31
	v_lshl_add_u64 v[98:99], s[68:69], 0, v[114:115]
	v_lshl_add_u64 v[138:139], v[98:99], 2, s[62:63]
	global_load_dwordx4 v[192:195], v[138:139], off offset:512
	ds_write_b128 v119, v[142:145]
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v141, 48, v98
	v_or_b32_e32 v98, v117, v141
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[142:145], v98 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[162:163], v[98:113]
	v_or_b32_e32 v142, 0x1010, v117
	v_xor_b32_e32 v142, v142, v141
	v_lshlrev_b32_e32 v142, 1, v142
	ds_read_b128 v[142:145], v142
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[166:167], v[98:113]
	v_or_b32_e32 v142, 0x1020, v117
	v_xor_b32_e32 v142, v142, v141
	v_lshlrev_b32_e32 v142, 1, v142
	ds_read_b128 v[142:145], v142
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[170:171], v[98:113]
	v_or_b32_e32 v142, 0x1030, v117
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[142:145], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[174:175], v[98:113]
	v_or_b32_e32 v141, 0x1040, v117
	v_lshrrev_b32_e32 v142, 3, v141
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[142:145], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[178:179], v[98:113]
	v_or_b32_e32 v141, 0x1050, v117
	v_lshrrev_b32_e32 v142, 3, v141
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[142:145], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[182:183], v[98:113]
	v_or_b32_e32 v141, 0x1060, v117
	v_lshrrev_b32_e32 v142, 3, v141
	v_and_b32_e32 v142, 56, v142
	v_xor_b32_e32 v141, v142, v141
	v_lshlrev_b32_e32 v141, 1, v141
	ds_read_b128 v[142:145], v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x1070, v117
	v_lshrrev_b32_e32 v141, 3, v117
	v_and_b32_e32 v141, 56, v141
	v_xor_b32_e32 v117, v141, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[142:145], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[190:191], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v117
	v_and_b32_e32 v117, 8, v117
	v_add_u32_e32 v117, s87, v117
	v_subrev_u32_e32 v141, 55, v117
	v_cmp_gt_i32_e32 vcc, s86, v141
	v_subrev_u32_e32 v141, 54, v117
	v_cmp_gt_i32_e64 s[0:1], s86, v141
	v_subrev_u32_e32 v141, 53, v117
	v_cmp_gt_i32_e64 s[4:5], s86, v141
	v_subrev_u32_e32 v141, 52, v117
	v_cmp_gt_i32_e64 s[6:7], s86, v141
	v_subrev_u32_e32 v141, 51, v117
	v_cmp_gt_i32_e64 s[8:9], s86, v141
	v_subrev_u32_e32 v141, 50, v117
	v_cmp_gt_i32_e64 s[10:11], s86, v141
	v_subrev_u32_e32 v141, 49, v117
	v_cmp_gt_i32_e64 s[12:13], s86, v141
	v_subrev_u32_e32 v141, 48, v117
	v_cmp_gt_i32_e64 s[14:15], s86, v141
	v_subrev_u32_e32 v141, 39, v117
	v_cmp_gt_i32_e64 s[16:17], s86, v141
	v_subrev_u32_e32 v141, 38, v117
	v_cmp_gt_i32_e64 s[18:19], s86, v141
	v_subrev_u32_e32 v141, 37, v117
	v_cmp_gt_i32_e64 s[20:21], s86, v141
	v_subrev_u32_e32 v141, 36, v117
	v_cmp_gt_i32_e64 s[22:23], s86, v141
	v_subrev_u32_e32 v141, 35, v117
	v_cmp_gt_i32_e64 s[24:25], s86, v141
	v_subrev_u32_e32 v141, 34, v117
	v_cmp_gt_i32_e64 s[26:27], s86, v141
	v_subrev_u32_e32 v141, 33, v117
	v_subrev_u32_e32 v117, 32, v117
	v_cmp_gt_i32_e64 s[28:29], s86, v141
	v_cmp_gt_i32_e64 s[30:31], s86, v117
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
	v_cndmask_b32_e64 v99, v157, v99, s[0:1]
	v_cndmask_b32_e32 v98, v157, v98, vcc
	v_cndmask_b32_e64 v113, v157, v113, s[30:31]
	v_cndmask_b32_e64 v112, v157, v112, s[28:29]
	v_cndmask_b32_e64 v111, v157, v111, s[26:27]
	v_cndmask_b32_e64 v110, v157, v110, s[24:25]
	v_cndmask_b32_e64 v109, v157, v109, s[22:23]
	v_cndmask_b32_e64 v108, v157, v108, s[20:21]
	v_cndmask_b32_e64 v107, v157, v107, s[18:19]
	v_cndmask_b32_e64 v106, v157, v106, s[16:17]
	v_cndmask_b32_e64 v105, v157, v105, s[14:15]
	v_cndmask_b32_e64 v104, v157, v104, s[12:13]
	v_cndmask_b32_e64 v103, v157, v103, s[10:11]
	v_cndmask_b32_e64 v102, v157, v102, s[8:9]
	v_cndmask_b32_e64 v101, v157, v101, s[6:7]
	v_cndmask_b32_e64 v100, v157, v100, s[4:5]
	v_mov_b32_e32 v117, v116
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
	ds_bpermute_b32 v141, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v141, v141, v141
	v_max_f32_e32 v141, v117, v141
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v141, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_82
	;;#ASMSTART
	v_add_f32 v122, v141, v156
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v122
	v_exp_f32_e32 v117, v117
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
.LBB0_82:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[98:99], v[98:99], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, 0, v98
	v_add_f32_e32 v141, v141, v99
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
	v_add_f32_e32 v141, v141, v100
	v_add_f32_e32 v141, v141, v101
	v_add_f32_e32 v141, v141, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v103
	v_add_f32_e32 v141, v141, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v105
	v_add_f32_e32 v141, v141, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v107
	v_add_f32_e32 v141, v141, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v141, v141, v109
	v_add_f32_e32 v141, v141, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v141, v141, v111
	v_add_f32_e32 v141, v141, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v141, v141, v113
	;;#ASMSTART
	v_fma_f32 v140, v140, v117, v141
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_84
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
.LBB0_84:
	s_or_b64 exec, exec, s[0:1]
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
	v_perm_b32 v100, v99, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v101, v101, v117, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v102, v103, v102, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v103, v105, v104, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v104, v107, v106, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v105, v109, v108, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v98, v111, v110, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v99, v113, v112, s75
	;;#ASMEND
	s_barrier
	s_setprio 1
	;;#ASMSTART
	v_mov_b32 v106, v0
	;;#ASMEND
	s_add_i32 s0, s92, 0x1800
	v_lshlrev_b32_e32 v107, 3, v106
	v_bfe_i32 v106, v106, 5, 1
	v_and_b32_e32 v107, 0xf8, v107
	v_and_b32_e32 v106, 0x600, v106
	v_or3_b32 v106, v107, v106, s0
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshl_add_u64 v[106:107], v[106:107], 1, s[50:51]
	global_load_dwordx4 v[108:111], v[106:107], off
	global_load_dwordx4 v[142:145], v[106:107], off offset:512
	global_load_dwordx4 v[150:153], v[106:107], off offset:1024
	global_load_dwordx4 v[196:199], v[106:107], off offset:1536
	global_load_dwordx4 v[200:203], v[106:107], off offset:2048
	global_load_dwordx4 v[204:207], v[106:107], off offset:2560
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[100:101], v[82:97]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[100:101], v[66:81]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[100:101], v[50:65]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[100:101], v[34:49]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[200:201], v[100:101], v[18:33]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[204:205], v[100:101], v[2:17]
	v_add_co_u32_e32 v100, vcc, s76, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[102:103], v[82:97]
	global_load_dwordx4 v[108:111], v[100:101], off offset:2048
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[102:103], v[66:81]
	global_load_dwordx4 v[142:145], v[100:101], off offset:2560
	v_mfma_f32_32x32x8_bf16 v[50:65], v[152:153], v[102:103], v[50:65]
	global_load_dwordx4 v[150:153], v[100:101], off offset:3072
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[102:103], v[34:49]
	global_load_dwordx4 v[196:199], v[100:101], off offset:3584
	v_add_co_u32_e32 v100, vcc, s77, v106
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v107, vcc
	v_mfma_f32_32x32x8_bf16 v[18:33], v[202:203], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[206:207], v[102:103], v[2:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_bf16 v[82:97], v[108:109], v[104:105], v[82:97]
	global_load_dwordx4 v[106:109], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off offset:512
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[66:81], v[142:143], v[104:105], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[196:197], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[110:111], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[106:107], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[100:101], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[152:153], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[198:199], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[108:109], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[102:103], v[98:99], v[2:17]
	global_load_dwordx4 v[196:199], v[138:139], off offset:1024
	ds_write_b128 v119, v[146:149] offset:8192
	;;#ASMSTART
	v_mov_b32 v98, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v100, 2, v98
	v_lshlrev_b32_e32 v99, 7, v98
	v_and_b32_e32 v100, 8, v100
	v_lshlrev_b32_e32 v98, 4, v98
	v_and_or_b32 v117, v99, s74, v100
	v_and_b32_e32 v138, 48, v98
	v_or_b32_e32 v98, v117, v138
	v_lshlrev_b32_e32 v98, 1, v98
	ds_read_b128 v[142:145], v98
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[160:161], 0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[162:163], v[98:113]
	v_or_b32_e32 v139, 16, v117
	v_xor_b32_e32 v139, v139, v138
	v_lshlrev_b32_e32 v139, 1, v139
	ds_read_b128 v[142:145], v139
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[164:165], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[166:167], v[98:113]
	v_or_b32_e32 v139, 32, v117
	v_xor_b32_e32 v139, v139, v138
	v_lshlrev_b32_e32 v139, 1, v139
	ds_read_b128 v[142:145], v139
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[168:169], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[170:171], v[98:113]
	v_or_b32_e32 v139, 48, v117
	v_xor_b32_e32 v138, v139, v138
	v_lshlrev_b32_e32 v138, 1, v138
	ds_read_b128 v[142:145], v138
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[172:173], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[174:175], v[98:113]
	v_or_b32_e32 v138, 64, v117
	v_lshrrev_b32_e32 v139, 3, v138
	v_and_b32_e32 v139, 56, v139
	v_xor_b32_e32 v138, v139, v138
	v_lshlrev_b32_e32 v138, 1, v138
	ds_read_b128 v[142:145], v138
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[176:177], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[178:179], v[98:113]
	v_or_b32_e32 v138, 0x50, v117
	v_lshrrev_b32_e32 v139, 3, v138
	v_and_b32_e32 v139, 56, v139
	v_xor_b32_e32 v138, v139, v138
	v_lshlrev_b32_e32 v138, 1, v138
	ds_read_b128 v[142:145], v138
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[180:181], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[182:183], v[98:113]
	v_or_b32_e32 v138, 0x60, v117
	v_lshrrev_b32_e32 v139, 3, v138
	v_and_b32_e32 v139, 56, v139
	v_xor_b32_e32 v138, v139, v138
	v_lshlrev_b32_e32 v138, 1, v138
	ds_read_b128 v[142:145], v138
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[184:185], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[186:187], v[98:113]
	v_or_b32_e32 v117, 0x70, v117
	v_lshrrev_b32_e32 v138, 3, v117
	v_and_b32_e32 v138, 56, v138
	v_xor_b32_e32 v117, v138, v117
	v_lshlrev_b32_e32 v117, 1, v117
	ds_read_b128 v[142:145], v117
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[98:113], v[142:143], v[188:189], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[144:145], v[190:191], v[98:113]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	s_barrier
	s_setprio 0
	;;#ASMSTART
	v_mov_b32 v117, v0
	;;#ASMEND
	s_nop 0
	v_lshrrev_b32_e32 v117, 2, v117
	v_and_b32_e32 v117, 8, v117
	v_add_u32_e32 v117, s87, v117
	v_subrev_u32_e32 v138, 23, v117
	v_cmp_gt_i32_e32 vcc, s86, v138
	v_subrev_u32_e32 v138, 22, v117
	v_cmp_gt_i32_e64 s[0:1], s86, v138
	v_subrev_u32_e32 v138, 21, v117
	v_cmp_gt_i32_e64 s[4:5], s86, v138
	v_subrev_u32_e32 v138, 20, v117
	v_cmp_gt_i32_e64 s[6:7], s86, v138
	v_subrev_u32_e32 v138, 19, v117
	v_cmp_gt_i32_e64 s[8:9], s86, v138
	v_subrev_u32_e32 v138, 18, v117
	v_cmp_gt_i32_e64 s[10:11], s86, v138
	v_subrev_u32_e32 v138, 17, v117
	v_cmp_gt_i32_e64 s[12:13], s86, v138
	v_add_u32_e32 v138, -16, v117
	v_cmp_gt_i32_e64 s[14:15], s86, v138
	v_add_u32_e32 v138, -7, v117
	v_cmp_gt_i32_e64 s[16:17], s86, v138
	v_add_u32_e32 v138, -6, v117
	v_cmp_gt_i32_e64 s[18:19], s86, v138
	v_add_u32_e32 v138, -5, v117
	v_cmp_gt_i32_e64 s[20:21], s86, v138
	v_add_u32_e32 v138, -4, v117
	v_cmp_gt_i32_e64 s[22:23], s86, v138
	v_add_u32_e32 v138, -3, v117
	v_cmp_gt_i32_e64 s[24:25], s86, v138
	v_add_u32_e32 v138, -2, v117
	v_cmp_gt_i32_e64 s[26:27], s86, v138
	v_add_u32_e32 v138, -1, v117
	v_cmp_gt_i32_e64 s[28:29], s86, v138
	v_cmp_gt_i32_e64 s[30:31], s86, v117
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
	v_cndmask_b32_e64 v99, v157, v99, s[0:1]
	v_cndmask_b32_e32 v98, v157, v98, vcc
	v_cndmask_b32_e64 v113, v157, v113, s[30:31]
	v_cndmask_b32_e64 v112, v157, v112, s[28:29]
	v_cndmask_b32_e64 v111, v157, v111, s[26:27]
	v_cndmask_b32_e64 v110, v157, v110, s[24:25]
	v_cndmask_b32_e64 v109, v157, v109, s[22:23]
	v_cndmask_b32_e64 v108, v157, v108, s[20:21]
	v_cndmask_b32_e64 v107, v157, v107, s[18:19]
	v_cndmask_b32_e64 v106, v157, v106, s[16:17]
	v_cndmask_b32_e64 v105, v157, v105, s[14:15]
	v_cndmask_b32_e64 v104, v157, v104, s[12:13]
	v_cndmask_b32_e64 v103, v157, v103, s[10:11]
	v_cndmask_b32_e64 v102, v157, v102, s[8:9]
	v_cndmask_b32_e64 v101, v157, v101, s[6:7]
	v_cndmask_b32_e64 v100, v157, v100, s[4:5]
	v_mov_b32_e32 v117, v116
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
	ds_bpermute_b32 v138, v154, v117
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v138, v138, v138
	v_max_f32_e32 v138, v117, v138
	;;#ASMSTART
	v_add_f32 v117, v118, v155
	;;#ASMEND
	s_nop 0
	v_cmp_gt_f32_e32 vcc, v138, v117
	v_mov_b32_e32 v117, 1.0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_86
	;;#ASMSTART
	v_add_f32 v122, v138, v156
	;;#ASMEND
	s_nop 0
	v_sub_f32_e32 v117, v118, v122
	v_exp_f32_e32 v117, v117
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
.LBB0_86:
	s_or_b64 exec, exec, s[0:1]
	v_pk_add_f32 v[98:99], v[98:99], v[122:123] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101], v[100:101], v[124:125] neg_lo:[0,1] neg_hi:[0,1]
	;;#ASMSTART
	v_exp_f32 v98, v98
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v99, v99
	;;#ASMEND
	v_pk_add_f32 v[102:103], v[102:103], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, 0, v98
	v_add_f32_e32 v122, v122, v99
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
	v_add_f32_e32 v122, v122, v100
	v_add_f32_e32 v122, v122, v101
	v_add_f32_e32 v122, v122, v102
	;;#ASMSTART
	v_exp_f32 v103, v103
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v104, v104
	;;#ASMEND
	v_pk_add_f32 v[106:107], v[106:107], v[130:131] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v103
	v_add_f32_e32 v122, v122, v104
	;;#ASMSTART
	v_exp_f32 v105, v105
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v106, v106
	;;#ASMEND
	v_pk_add_f32 v[108:109], v[108:109], v[132:133] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v105
	v_add_f32_e32 v122, v122, v106
	;;#ASMSTART
	v_exp_f32 v107, v107
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v108, v108
	;;#ASMEND
	v_pk_add_f32 v[110:111], v[110:111], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v107
	v_add_f32_e32 v122, v122, v108
	;;#ASMSTART
	v_exp_f32 v109, v109
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v110, v110
	;;#ASMEND
	v_pk_add_f32 v[112:113], v[112:113], v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	v_add_f32_e32 v122, v122, v109
	v_add_f32_e32 v122, v122, v110
	;;#ASMSTART
	v_exp_f32 v111, v111
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32 v112, v112
	;;#ASMEND
	v_cmp_gt_f32_e32 vcc, 1.0, v117
	v_add_f32_e32 v122, v122, v111
	v_add_f32_e32 v122, v122, v112
	;;#ASMSTART
	v_exp_f32 v113, v113
	;;#ASMEND
	s_nop 0
	v_add_f32_e32 v122, v122, v113
	;;#ASMSTART
	v_fma_f32 v158, v140, v117, v122
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_53
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
	s_branch .LBB0_53
.LBB0_88:
	s_mov_b32 s91, s90
	s_branch .LBB0_54
.LBB0_89:
	v_mov_b32_e32 v122, v50
	v_mov_b32_e32 v123, v51
	v_mov_b32_e32 v140, v52
	v_mov_b32_e32 v141, v53
	v_mov_b32_e32 v142, v54
	v_mov_b32_e32 v143, v55
	v_mov_b32_e32 v144, v56
	v_mov_b32_e32 v145, v57
	v_mov_b32_e32 v146, v58
	v_mov_b32_e32 v147, v59
	v_mov_b32_e32 v148, v60
	v_mov_b32_e32 v149, v61
	v_mov_b32_e32 v150, v62
	v_mov_b32_e32 v151, v63
	v_mov_b32_e32 v152, v64
	v_mov_b32_e32 v153, v65
	v_mov_b32_e32 v138, v34
	v_mov_b32_e32 v139, v35
	v_mov_b32_e32 v136, v36
	v_mov_b32_e32 v137, v37
	v_mov_b32_e32 v134, v38
	v_mov_b32_e32 v135, v39
	v_mov_b32_e32 v132, v40
	v_mov_b32_e32 v133, v41
	v_mov_b32_e32 v130, v42
	v_mov_b32_e32 v131, v43
	v_mov_b32_e32 v128, v44
	v_mov_b32_e32 v129, v45
	v_mov_b32_e32 v126, v46
	v_mov_b32_e32 v127, v47
	v_mov_b32_e32 v124, v48
	v_mov_b32_e32 v125, v49
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
.LBB0_90:
	ds_bpermute_b32 v2, v154, v158
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_add_f32 v2, v158, v2
	;;#ASMEND
	s_nop 0
	v_div_scale_f32 v3, s[0:1], v2, v2, s72
	v_rcp_f32_e32 v4, v3
	v_div_scale_f32 v5, vcc, s72, v2, s72
	v_fma_f32 v6, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v6, v4
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, -v3, v6, v5
	v_fmac_f32_e32 v6, v7, v4
	v_fma_f32 v3, -v3, v6, v5
	v_div_fmas_f32 v3, v3, v4, v6
	v_div_fixup_f32 v2, v3, v2, s72
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
	v_pk_mul_f32 v[52:53], v[122:123], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[140:141], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[142:143], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[144:145], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[146:147], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[148:149], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[150:151], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[152:153], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[138:139], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[136:137], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[134:135], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[132:133], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[130:131], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[128:129], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[126:127], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[124:125], v[2:3] op_sel_hi:[1,0]
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
	v_perm_b32 v48, v5, v4, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v49, v7, v6, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v46, v9, v8, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v47, v11, v10, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v44, v13, v12, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v45, v15, v14, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v42, v17, v16, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v43, v35, v34, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v40, v37, v36, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v41, v39, v38, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v38, v69, v71, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v39, v65, v67, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v36, v61, v63, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v37, v57, v59, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v34, v53, v55, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v35, v51, v50, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v32, v32, v52, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v33, v33, v54, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v30, v30, v56, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v31, v31, v58, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v28, v28, v60, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v29, v29, v62, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v26, v26, v64, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v27, v27, v66, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v24, v24, v68, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v25, v25, v70, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v22, v22, v72, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v23, v23, v74, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v20, v20, v76, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v21, v21, v78, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v18, v18, v80, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v19, v19, v82, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v112, v113, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v17, v110, v111, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v108, v109, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v106, v107, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v104, v105, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v13, v102, v103, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v100, v101, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v97, v99, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v85, v84, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v87, v86, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v89, v88, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v91, v90, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v93, v92, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v95, v94, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v2, v96, s75
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v3, v98, s75
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v50, v0
	;;#ASMEND
	s_nop 0
	v_and_b32_e32 v50, 0x100, v50
	v_cmp_eq_u32_e32 vcc, 0, v50
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_92
	s_barrier
.LBB0_92:
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
	s_cbranch_execz .LBB0_94
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
.LBB0_94:
	s_or_b64 exec, exec, s[0:1]
	s_mul_i32 s0, s81, 0xc00
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_and_b32_e32 v88, 0x1ff, v91
	s_add_u32 s44, s56, s0
	s_addc_u32 s0, s57, s1
	v_lshlrev_b32_e32 v92, 3, v88
	s_mul_i32 s46, s49, 0x1800
	s_and_b32 s45, s0, 0xffff
	s_mul_i32 s4, s73, 0xc0
	v_and_b32_e32 v91, 56, v91
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_95:
	v_mul_u32_u24_sdwa v95, v94, s78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s79, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s4, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[44:47], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_95
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 1, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_98
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
.LBB0_98:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s4, s4, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_99:
	v_mul_u32_u24_sdwa v95, v94, s78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s79, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s4, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[44:47], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_99
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 2, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_102
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
.LBB0_102:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x18000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_103:
	v_mul_u32_u24_sdwa v95, v94, s78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s79, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[44:47], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_103
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 3, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_106
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
.LBB0_106:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x30000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_107:
	v_mul_u32_u24_sdwa v95, v94, s78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s79, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[44:47], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_107
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 4, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_110
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
.LBB0_110:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x48000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_111:
	v_mul_u32_u24_sdwa v95, v94, s78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s79, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[44:47], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_111
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 5, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_114
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
.LBB0_114:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x60000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_115:
	v_mul_u32_u24_sdwa v95, v94, s78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s79, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[44:47], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_115
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 6, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_118
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
.LBB0_118:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s5, s4, 0x78000
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v93, v92
	v_mov_b32_e32 v94, v88
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_119:
	v_mul_u32_u24_sdwa v95, v94, s78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v96, v93, v91
	v_lshrrev_b32_e32 v95, 20, v95
	v_lshlrev_b32_e32 v96, 1, v96
	v_mul_lo_u16_e32 v98, 24, v95
	ds_read_b128 v[100:103], v96
	v_sub_u16_e32 v96, v94, v98
	v_lshlrev_b16_e32 v96, 3, v96
	v_add_u32_e32 v97, 0x200, v94
	v_cmp_lt_u32_e32 vcc, s79, v94
	v_mul_u32_u24_e32 v95, 0xc00, v95
	v_add_u32_e32 v96, s5, v96
	v_add_u32_e32 v93, 0x1000, v93
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v94, v97
	v_add_lshl_u32 v95, v96, v95, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[100:103], v95, s[44:47], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_119
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e32 vcc, 7, v87
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_122
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
.LBB0_122:
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s4, s4, 0x90000
	s_mov_b64 s[0:1], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB0_123:
	v_mul_u32_u24_sdwa v2, v88, s78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_xor_b32_e32 v3, v92, v91
	v_lshrrev_b32_e32 v2, 20, v2
	v_lshlrev_b32_e32 v3, 1, v3
	v_mul_lo_u16_e32 v5, 24, v2
	ds_read_b128 v[6:9], v3
	v_sub_u16_e32 v3, v88, v5
	v_lshlrev_b16_e32 v3, 3, v3
	v_add_u32_e32 v4, 0x200, v88
	v_cmp_lt_u32_e32 vcc, s79, v88
	v_mul_u32_u24_e32 v2, 0xc00, v2
	v_add_u32_e32 v3, s4, v3
	v_add_u32_e32 v92, 0x1000, v92
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b32_e32 v88, v4
	v_add_lshl_u32 v2, v3, v2, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v2, s[44:47], 0 offen
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_123
	s_or_b64 exec, exec, s[0:1]
	s_barrier
	;;#ASMSTART
	s_mov_b32 s0, s80
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v0
	;;#ASMEND
	s_add_i32 s0, s0, 1
	v_and_b32_e32 v2, 0x1ff, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_ashr_i32 s1, s0, 31
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_128
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_127
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v3, s8
	global_atomic_add v3, v115, v3, s[58:59] sc0
.LBB0_127:
	s_or_b64 exec, exec, s[6:7]
	s_lshl_b64 s[6:7], s[0:1], 2
	s_add_u32 s6, s58, s6
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v3
	s_addc_u32 s7, s59, s7
	s_nop 0
	v_add_u32_e32 v2, s8, v2
	global_store_dword v115, v2, s[6:7]
	s_waitcnt vmcnt(0)
.LBB0_128:
	s_or_b64 exec, exec, s[4:5]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s58, s0
	s_addc_u32 s1, s59, s1
	s_barrier
	global_load_dword v2, v115, s[0:1]
	s_waitcnt vmcnt(0)
	s_barrier
	v_readfirstlane_b32 s4, v2
	s_add_i32 s0, s4, s33
	s_sub_i32 s33, s0, s2
	s_cmp_ge_i32 s48, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s70
	s_cselect_b64 s[6:7], -1, 0
	s_or_b64 s[0:1], s[0:1], s[6:7]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_131
	s_branch .LBB0_12
.LBB0_129:
	s_mov_b32 s48, s5
.LBB0_130:
	s_sub_i32 s33, s33, s70
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s73, 0, s2
	s_cmp_ge_i32 s48, s3
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s33, s6
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s70, s6
	s_cbranch_vccnz .LBB0_11
.LBB0_131:
	s_add_i32 s2, s73, 1
	s_cmp_gt_i32 s2, 15
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s2, 16
	s_cbranch_scc1 .LBB0_134
	s_add_i32 s5, s48, 1
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s6, s70
	s_cbranch_scc1 .LBB0_129
	s_ashr_i32 s49, s48, 31
	s_lshl_b64 s[6:7], s[48:49], 2
	s_add_u32 s6, s34, s6
	s_addc_u32 s7, s35, s7
	global_load_dwordx2 v[2:3], v115, s[6:7] offset:4
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
	s_branch .LBB0_129
.LBB0_134:
	s_mov_b32 s6, s70
	s_branch .LBB0_130
.LBB0_135:
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
		.amdhsa_next_free_vgpr 212
		.amdhsa_next_free_sgpr 93
		.amdhsa_accum_offset 212
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

	.set .Lattn_kernel_0.num_vgpr, 212
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
    .sgpr_count:     99
    .sgpr_spill_count: 0
    .symbol:         attn_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     212
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
