	; _ZN2ck59kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffleINS_58GridwiseGemmMultiD_blockscale_xdl_cshuffle_v3_b_preshuffleINS_13tensor_layout4gemm8RowMajorENS3_11ColumnMajorENS_5TupleIJEEES4_NS_9f8_fnuz_tES8_ffS7_tNS_16tensor_operation12element_wise11PassThroughESB_SB_LNS9_6device18GemmSpecializationE0ELi256ELi1ELi128ELi128ELi64ELi256ELi128ELi16ELi16ELi16ELi16ELi4ELi4ENS_8SequenceIJLi8ELi32ELi1EEEENSE_IJLi1ELi0ELi2EEEESG_Li2ELi16ELi16ELb0ELi0ESF_SG_SG_Li2ELi16ELi16ELb0ELi0ELi2ELi1ENSE_IJLi1ELi32ELi1ELi8EEEENSE_IJLi8EEEELNS_26BlockGemmPipelineSchedulerE0ELNS_24BlockGemmPipelineVersionE0ES8_S8_S8_S8_EELb1ELNS_25InMemoryDataOperationEnumE0ELi2ELNS_10TailNumberE1EEEvNT_8ArgumentE
	s_load_dwordx4 s[12:15], s[0:1], 0x10            ;	s[12:15] = load_dwordx4_from(s[0:1] + 0x10, glc=0);  // 8.2.1.1. Scalar Memory Addressing
	s_load_dword s18, s[0:1], 0x40                   ;	s18 = load_dword_from(s[0:1] + 0x40, glc=0);  // 8.2.1.1. Scalar Memory Addressing
	s_load_dwordx2 s[16:17], s[0:1], 0x78            ;	s[16:17] = load_dwordx2_from(s[0:1] + 0x78, glc=0);  // 8.2.1.1. Scalar Memory Addressing
	s_load_dwordx4 s[4:7], s[0:1], 0x50              ;	s[4:7] = load_dwordx4_from(s[0:1] + 0x50, glc=0);  // 8.2.1.1. Scalar Memory Addressing
	s_load_dwordx4 s[8:11], s[0:1], 0x68             ;	s[8:11] = load_dwordx4_from(s[0:1] + 0x68, glc=0);  // 8.2.1.1. Scalar Memory Addressing
	s_getpc_b64 s[20:21]
	s_add_u32 s20, s20, 0xffff7ed8                   ;	s20.u32 = s20 + 0xffff7ed8; scc=overflow_or_carry
	s_addc_u32 s21, s21, -1                          ;	s21.u32 = s21 + -1 + scc; scc=overflow_or_carry
	s_load_dword s19, s[20:21], 0x0                  ;	s19 = load_dword_from(s[20:21] + 0x0, glc=0);  // 8.2.1.1. Scalar Memory Addressing
	s_waitcnt lgkmcnt(0)
	s_add_i32 s33, s12, -1                           ;	s33.i32 = s12 + -1; scc=overflow_or_carry
	s_cmp_lt_u32 s33, 64                             ;	scc = (s33.u32 < 64.u32)
	s_mov_b32 s34, 0                                 ;	s34 = 0
	s_cbranch_scc1 36                                ;	jump to 36 if scc1
	s_add_i32 s3, s13, -1                            ;	s3.i32 = s13 + -1; scc=overflow_or_carry
	s_cmpk_lt_u32 s3, 0x100                          ;	scc = (s3.u32 < extend_as_u32(0x100))
	s_mov_b32 s20, 0                                 ;	s20 = 0
	s_cbranch_scc1 113                               ;	jump to 113 if scc1
	s_add_i32 s3, s12, 63                            ;	s3.i32 = s12 + 63; scc=overflow_or_carry
	s_ashr_i32 s20, s3, 31
	s_lshr_b32 s20, s20, 26
	s_add_i32 s3, s3, s20                            ;	s3.i32 = s3 + s20; scc=overflow_or_carry
	s_ashr_i32 s21, s3, 6
	s_add_i32 s3, s13, 0xff                          ;	s3.i32 = s13 + 0xff; scc=overflow_or_carry
	s_ashr_i32 s20, s3, 31
	s_lshr_b32 s20, s20, 24
	s_add_i32 s3, s3, s20                            ;	s3.i32 = s3 + s20; scc=overflow_or_carry
	s_ashr_i32 s20, s3, 8
	s_mul_i32 s3, s20, s21                           ;	s3 = s20 * s21
	s_add_i32 s23, s3, 7                             ;	s23.i32 = s3 + 7; scc=overflow_or_carry
	s_ashr_i32 s22, s23, 31
	s_lshr_b32 s22, s22, 29
	s_add_i32 s23, s23, s22                          ;	s23.i32 = s23 + s22; scc=overflow_or_carry
	s_ashr_i32 s22, s23, 3
	s_and_b32 s23, s23, -8                           ;	s23 = s23 & -8
	s_sub_i32 s23, s3, s23                           ;	s23.i32 = (s3.i32 - s23.i32); scc=(overflow or carry-out of last arith);
	s_add_i32 s23, s23, 8                            ;	s23.i32 = s23 + 8; scc=overflow_or_carry
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 29
	s_add_i32 s26, s2, s3                            ;	s26.i32 = s2 + s3; scc=overflow_or_carry
	s_and_b32 s3, s26, -8                            ;	s3 = s26 & -8
	s_sub_i32 s25, s2, s3                            ;	s25.i32 = (s2.i32 - s3.i32); scc=(overflow or carry-out of last arith);
	s_cmp_gt_i32 s25, s23                            ;	scc = (s25.i32 > s23.i32)
	s_cbranch_scc1 7                                 ;	jump to 7 if scc1
	s_mul_i32 s24, s22, s25                          ;	s24 = s22 * s25
	s_mov_b64 vcc, exec                              ;	vcc = exec
	s_ashr_i32 s2, s26, 3
	s_cbranch_execz 5                                ;	jump to 5 if execz
	s_branch 7
	s_mov_b32 s3, 0                                  ;	s3 = 0
	s_branch 81
	s_mov_b64 vcc, 0                                 ;	vcc = 0
	s_ashr_i32 s2, s26, 3
	s_add_i32 s3, s22, -1                            ;	s3.i32 = s22 + -1; scc=overflow_or_carry
	s_mul_i32 s3, s3, s25                            ;	s3 = s3 * s25
	s_add_i32 s24, s23, s3                           ;	s24.i32 = s23 + s3; scc=overflow_or_carry
	s_abs_i32 s25, s20
	v_cvt_f32_u32_e32 v1, s25
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s22, s24, s2                           ;	s22.i32 = s24 + s2; scc=overflow_or_carry
	s_xor_b32 s2, s22, s20                           ;	s2 = s22 ^ s20;  scc=(s2!=0);
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_abs_i32 s23, s22
	s_sub_i32 s24, 0, s25                            ;	s24.i32 = (0.i32 - s25.i32); scc=(overflow or carry-out of last arith);
	s_ashr_i32 s3, s2, 31
	v_readfirstlane_b32 s2, v1
	s_mul_i32 s24, s24, s2                           ;	s24 = s24 * s2
	s_mul_hi_u32 s24, s2, s24                        ;	s24 = s2 * s24
	s_add_i32 s2, s2, s24                            ;	s2.i32 = s2 + s24; scc=overflow_or_carry
	s_mul_hi_u32 s2, s23, s2                         ;	s2 = s23 * s2
	s_mul_i32 s24, s2, s25                           ;	s24 = s2 * s25
	s_add_i32 s26, s2, 1                             ;	s26.i32 = s2 + 1; scc=overflow_or_carry
	s_sub_i32 s23, s23, s24                          ;	s23.i32 = (s23.i32 - s24.i32); scc=(overflow or carry-out of last arith);
	s_sub_i32 s24, s23, s25                          ;	s24.i32 = (s23.i32 - s25.i32); scc=(overflow or carry-out of last arith);
	s_cmp_ge_u32 s23, s25                            ;	scc = (s23.u32 >= s25.u32)
	s_cselect_b32 s2, s26, s2                        ;	s2 = scc ? s26 : s2
	s_cselect_b32 s24, s24, s23                      ;	s24 = scc ? s24 : s23
	s_add_i32 s23, s2, 1                             ;	s23.i32 = s2 + 1; scc=overflow_or_carry
	s_cmp_ge_u32 s24, s25                            ;	scc = (s24.u32 >= s25.u32)
	s_cselect_b32 s23, s23, s2                       ;	s23 = scc ? s23 : s2
	s_lshr_b32 s2, s21, 30
	s_xor_b32 s23, s23, s3                           ;	s23 = s23 ^ s3;  scc=(s23!=0);
	s_add_i32 s2, s21, s2                            ;	s2.i32 = s21 + s2; scc=overflow_or_carry
	s_sub_i32 s3, s23, s3                            ;	s3.i32 = (s23.i32 - s3.i32); scc=(overflow or carry-out of last arith);
	s_mul_i32 s23, s3, s20                           ;	s23 = s3 * s20
	s_sub_i32 s23, s22, s23                          ;	s23.i32 = (s22.i32 - s23.i32); scc=(overflow or carry-out of last arith);
	s_and_b32 s22, s2, -4                            ;	s22 = s2 & -4
	s_sub_i32 s2, s21, s22                           ;	s2.i32 = (s21.i32 - s22.i32); scc=(overflow or carry-out of last arith);
	s_cmp_ge_i32 s3, s22                             ;	scc = (s3.i32 >= s22.i32)
	s_cselect_b32 s22, s2, 4                         ;	s22 = scc ? s2 : 4
	s_ashr_i32 s2, s3, 31
	s_lshr_b32 s2, s2, 30
	s_abs_i32 s25, s22
	v_cvt_f32_u32_e32 v1, s25
	s_add_i32 s2, s3, s2                             ;	s2.i32 = s3 + s2; scc=overflow_or_carry
	s_and_b32 s2, s2, -4                             ;	s2 = s2 & -4
	s_sub_i32 s2, s3, s2                             ;	s2.i32 = (s3.i32 - s2.i32); scc=(overflow or carry-out of last arith);
	v_rcp_iflag_f32_e32 v1, v1
	s_mul_i32 s21, s2, s20                           ;	s21 = s2 * s20
	s_add_i32 s21, s21, s23                          ;	s21.i32 = s21 + s23; scc=overflow_or_carry
	s_xor_b32 s20, s21, s22                          ;	s20 = s21 ^ s22;  scc=(s20!=0);
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_ashr_i32 s20, s20, 31
	s_abs_i32 s24, s21
	s_sub_i32 s26, 0, s25                            ;	s26.i32 = (0.i32 - s25.i32); scc=(overflow or carry-out of last arith);
	v_readfirstlane_b32 s23, v1
	s_mul_i32 s26, s26, s23                          ;	s26 = s26 * s23
	s_mul_hi_u32 s26, s23, s26                       ;	s26 = s23 * s26
	s_add_i32 s23, s23, s26                          ;	s23.i32 = s23 + s26; scc=overflow_or_carry
	s_mul_hi_u32 s23, s24, s23                       ;	s23 = s24 * s23
	s_mul_i32 s26, s23, s25                          ;	s26 = s23 * s25
	s_sub_i32 s24, s24, s26                          ;	s24.i32 = (s24.i32 - s26.i32); scc=(overflow or carry-out of last arith);
	s_add_i32 s27, s23, 1                            ;	s27.i32 = s23 + 1; scc=overflow_or_carry
	s_sub_i32 s26, s24, s25                          ;	s26.i32 = (s24.i32 - s25.i32); scc=(overflow or carry-out of last arith);
	s_cmp_ge_u32 s24, s25                            ;	scc = (s24.u32 >= s25.u32)
	s_cselect_b32 s23, s27, s23                      ;	s23 = scc ? s27 : s23
	s_cselect_b32 s26, s26, s24                      ;	s26 = scc ? s26 : s24
	s_add_i32 s24, s23, 1                            ;	s24.i32 = s23 + 1; scc=overflow_or_carry
	s_cmp_ge_u32 s26, s25                            ;	scc = (s26.u32 >= s25.u32)
	s_cselect_b32 s23, s24, s23                      ;	s23 = scc ? s24 : s23
	s_xor_b32 s23, s23, s20                          ;	s23 = s23 ^ s20;  scc=(s23!=0);
	s_sub_i32 s20, s23, s20                          ;	s20.i32 = (s23.i32 - s20.i32); scc=(overflow or carry-out of last arith);
	s_mul_i32 s22, s20, s22                          ;	s22 = s20 * s22
	s_sub_i32 s21, s21, s22                          ;	s21.i32 = (s21.i32 - s22.i32); scc=(overflow or carry-out of last arith);
	s_add_i32 s3, s21, s3                            ;	s3.i32 = s21 + s3; scc=overflow_or_carry
	s_sub_i32 s2, s3, s2                             ;	s2.i32 = (s3.i32 - s2.i32); scc=(overflow or carry-out of last arith);
	s_mov_b32 s3, s2                                 ;	s3 = s2
	s_mov_b32 s2, s20                                ;	s2 = s20
	s_add_i32 s20, s13, 15                           ;	s20.i32 = s13 + 15; scc=overflow_or_carry
	s_add_i32 s21, s14, 63                           ;	s21.i32 = s14 + 63; scc=overflow_or_carry
	s_ashr_i32 s22, s21, 31
	s_lshr_b32 s22, s22, 26
	s_add_i32 s21, s21, s22                          ;	s21.i32 = s21 + s22; scc=overflow_or_carry
	s_ashr_i32 s44, s21, 6
	s_mul_i32 s23, s33, s15                          ;	s23 = s33 * s15
	s_ashr_i32 s21, s20, 31
	s_lshr_b32 s21, s21, 26
	s_add_i32 s20, s20, s21                          ;	s20.i32 = s20 + s21; scc=overflow_or_carry
	s_ashr_i32 s20, s20, 6
	s_lshl_b32 s35, s44, 12                          ;	s35 = s44 << 12[4:0]; scc=(s35!=0);
	s_lshl_b32 s25, s44, 10                          ;	s25 = s44 << 10[4:0]; scc=(s25!=0);
	s_add_i32 s20, s20, -1                           ;	s20.i32 = s20 + -1; scc=overflow_or_carry
	s_mul_i32 s20, s35, s20                          ;	s20 = s35 * s20
	s_add_i32 s19, s19, -1                           ;	s19.i32 = s19 + -1; scc=overflow_or_carry
	s_mul_i32 s19, s19, s25                          ;	s19 = s19 * s25
	s_add_i32 s20, s25, s20                          ;	s20.i32 = s25 + s20; scc=overflow_or_carry
	s_add_i32 s22, s20, s19                          ;	s22.i32 = s20 + s19; scc=overflow_or_carry
	s_add_i32 s19, s14, 0x7f                         ;	s19.i32 = s14 + 0x7f; scc=overflow_or_carry
	s_ashr_i32 s20, s19, 31
	s_lshr_b32 s20, s20, 25
	s_add_i32 s19, s19, s20                          ;	s19.i32 = s19 + s20; scc=overflow_or_carry
	s_ashr_i32 s20, s19, 7
	s_add_i32 s19, s13, 0x7f                         ;	s19.i32 = s13 + 0x7f; scc=overflow_or_carry
	s_ashr_i32 s21, s19, 31
	s_lshr_b32 s21, s21, 25
	s_add_i32 s19, s19, s21                          ;	s19.i32 = s19 + s21; scc=overflow_or_carry
	s_ashr_i32 s19, s19, 7
	s_lshl_b32 s21, s3, 6                            ;	s21 = s3 << 6[4:0]; scc=(s21!=0);
	v_and_b32_e32 v155, 7, v0                        ;	v155.u32 = (7 & v0.u32)
	v_lshrrev_b32_e32 v1, 2, v0                      ;	v1.b32 = v0 >> 2;
	v_and_b32_e32 v49, 62, v1                        ;	v49.u32 = (62 & v1.u32)
	v_add_u32_e32 v2, s21, v49                       ;	v2 = s21 + v49
	v_lshlrev_b32_e32 v156, 4, v155                  ;	v156.b32 = v155 << 4;
	v_mul_lo_u32 v157, v2, s15
	v_add_u32_e32 v51, v157, v156                    ;	v51 = v157 + v156
	v_lshrrev_b32_e32 v47, 6, v0                     ;	v47.b32 = v0 >> 6;
	v_lshlrev_b32_e32 v2, 4, v0                      ;	v2.b32 = v0 << 4;
	v_and_b32_e32 v153, 0x3f0, v2                    ;	v153.u32 = (0x3f0 & v2.u32)
	s_mul_i32 s24, s44, s2                           ;	s24 = s44 * s2
	v_mul_lo_u32 v2, s25, v47
	v_or_b32_e32 v2, v2, v153
	v_lshl_add_u32 v46, s24, 14, v2                  ;	v46.u32 = (s24.u32 << 14.u32[2 : 0].u32) + v2.u32
	v_and_b32_e32 v43, 15, v0                        ;	v43.u32 = (15 & v0.u32)
	s_ashr_i32 s24, s18, 31
	s_lshr_b32 s24, s24, 29
	s_add_i32 s18, s18, s24                          ;	s18.i32 = s18 + s24; scc=overflow_or_carry
	s_ashr_i32 s45, s18, 3
	v_or_b32_e32 v54, s21, v43
	s_lshl_b32 s21, s2, 1                            ;	s21 = s2 << 1[4:0]; scc=(s21!=0);
	s_mul_i32 s46, s21, s20                          ;	s46 = s21 * s20
	s_add_u32 s18, 0, 0                              ;	s18.u32 = 0 + 0; scc=overflow_or_carry
	s_addc_u32 s26, s23, s14                         ;	s26.u32 = s23 + s14 + scc; scc=overflow_or_carry
	s_mov_b32 s27, 0x20000                           ;	s27 = 0x20000
	s_and_b32 s25, s5, 0xffff                        ;	s25 = s5 & 0xffff
	s_mov_b32 s24, s4                                ;	s24 = s4
	v_add_u32_e32 v55, s15, v51                      ;	v55 = s15 + v51
	buffer_load_dwordx4 v[38:41], v51, s[24:27], 0 offen
	buffer_load_dwordx4 v[34:37], v55, s[24:27], 0 offen
	s_and_b32 s29, s7, 0xffff                        ;	s29 = s7 & 0xffff
	s_mov_b32 s28, s6                                ;	s28 = s6
	s_mov_b32 s30, s22                               ;	s30 = s22
	s_mov_b32 s31, s27                               ;	s31 = s27
	v_add_u32_e32 v44, s35, v46                      ;	v44 = s35 + v46
	v_add_u32_e32 v45, s35, v44                      ;	v45 = s35 + v44
	v_add_u32_e32 v42, s35, v45                      ;	v42 = s35 + v45
	buffer_load_dwordx4 v[2:5], v42, s[28:31], 0 offen
	buffer_load_dwordx4 v[6:9], v42, s[28:31], 0 offen offset:1024
	buffer_load_dwordx4 v[10:13], v45, s[28:31], 0 offen
	buffer_load_dwordx4 v[14:17], v45, s[28:31], 0 offen offset:1024
	buffer_load_dwordx4 v[18:21], v44, s[28:31], 0 offen
	buffer_load_dwordx4 v[22:25], v44, s[28:31], 0 offen offset:1024
	buffer_load_dwordx4 v[26:29], v46, s[28:31], 0 offen
	buffer_load_dwordx4 v[30:33], v46, s[28:31], 0 offen offset:1024
	s_mul_i32 s5, s12, s20                           ;	s5 = s12 * s20
	s_lshl_b32 s30, s5, 2                            ;	s30 = s5 << 2[4:0]; scc=(s30!=0);
	s_and_b32 s41, s11, 0xffff                       ;	s41 = s11 & 0xffff
	s_mov_b32 s40, s10                               ;	s40 = s10
	s_mov_b32 s42, s30                               ;	s42 = s30
	s_mov_b32 s43, s27                               ;	s43 = s27
	v_lshlrev_b32_e32 v44, 2, v54                    ;	v44.b32 = v54 << 2;
	buffer_load_dword v50, v44, s[40:43], 0 offen
	buffer_load_dword v48, v44, s[40:43], 0 offen offset:64
	buffer_load_dword v46, v44, s[40:43], 0 offen offset:128
	buffer_load_dword v42, v44, s[40:43], 0 offen offset:192
	s_mul_i32 s5, s20, s19                           ;	s5 = s20 * s19
	s_lshl_b32 s18, s5, 2                            ;	s18 = s5 << 2[4:0]; scc=(s18!=0);
	s_and_b32 s37, s17, 0xffff                       ;	s37 = s17 & 0xffff
	s_mov_b32 s36, s16                               ;	s36 = s16
	s_mov_b32 s38, s18                               ;	s38 = s18
	s_mov_b32 s39, s27                               ;	s39 = s27
	s_lshl_b32 s5, s46, 2                            ;	s5 = s46 << 2[4:0]; scc=(s5!=0);
	v_mov_b32_e32 v52, s5                            ;	v52 = s5;
	s_add_i32 s5, s46, s20                           ;	s5.i32 = s46 + s20; scc=overflow_or_carry
	s_lshl_b32 s5, s5, 2                             ;	s5 = s5 << 2[4:0]; scc=(s5!=0);
	v_mov_b32_e32 v53, s5                            ;	v53 = s5;
	buffer_load_dword v45, v52, s[36:39], 0 offen
	buffer_load_dword v44, v53, s[36:39], 0 offen
	s_load_dword s14, s[0:1], 0x28                   ;	s14 = load_dword_from(s[0:1] + 0x28, glc=0);  // 8.2.1.1. Scalar Memory Addressing
	v_and_b32_e32 v57, 6, v1                         ;	v57.u32 = (6 & v1.u32)
	v_bitop3_b32 v162, v1, v155, 6 bitop3:0x6c
	v_lshlrev_b32_e32 v52, 7, v49                    ;	v52.b32 = v49 << 7;
	v_lshl_or_b32 v56, v162, 4, v52                  ;	v56.u32 = (v162.u32 << 4[4:0].u32) | v52.u32;
	v_lshlrev_b32_e32 v58, 7, v43                    ;	v58.b32 = v43 << 7;
	v_lshrrev_b32_e32 v52, 4, v0                     ;	v52.b32 = v0 >> 4;
	v_bfe_u32 v53, v0, 4, 2
	v_bitop3_b32 v52, v52, v155, 3 bitop3:0x6c
	v_lshl_or_b32 v152, v52, 4, v58                  ;	v152.u32 = (v52.u32 << 4[4:0].u32) | v58.u32;
	s_mov_b32 s23, s27                               ;	s23 = s27
	s_mov_b32 s0, s29                                ;	s0 = s29
	s_mov_b32 s29, s41                               ;	s29 = s41
	s_mov_b32 s19, s27                               ;	s19 = s27
	s_mov_b32 s17, s37                               ;	s17 = s37
	s_or_b32 s5, s46, 1                              ;	s5 = s46 | 1;  scc=(s5!=0);
	s_waitcnt vmcnt(15)
	ds_write_b128 v56, v[38:41]                      ;	LDS_MEM[v56 + 0].b128 = v[38:41].b128
	v_bitop3_b32 v39, v57, v155, 1 bitop3:0x36
	v_sub_u32_e32 v38, v39, v162
	v_lshl_add_u32 v38, v38, 4, v56                  ;	v38.u32 = (v38.u32 << 4.u32[2 : 0].u32) + v56.u32
	s_waitcnt vmcnt(14)
	ds_write_b128 v38, v[34:37] offset:128           ;	LDS_MEM[v38 + 128].b128 = v[34:37].b128
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[86:87], v[50:51], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[48:49], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[46:47], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[42:43], v[44:45] op_sel_hi:[0,1]
	v_sub_u32_e32 v34, v162, v39
	v_lshl_add_u32 v171, v34, 4, v38                 ;	v171.u32 = (v34.u32 << 4.u32[2 : 0].u32) + v38.u32
	s_movk_i32 s1, 0x80
	v_add_u32_e32 v36, 0x80, v55                     ;	v36 = 0x80 + v55
	v_add_lshl_u32 v34, v54, s12, 2                  ;	v34 = (v54 + s12) << 2[4:0]
	buffer_load_dword v79, v34, s[40:43], 0 offen
	buffer_load_dword v78, v34, s[40:43], 0 offen offset:64
	buffer_load_dword v75, v34, s[40:43], 0 offen offset:128
	buffer_load_dword v74, v34, s[40:43], 0 offen offset:192
	s_lshl_b32 s7, s5, 2                             ;	s7 = s5 << 2[4:0]; scc=(s7!=0);
	v_bitop3_b32 v34, v53, v155, 4 bitop3:0x36
	v_sub_u32_e32 v34, v34, v52
	v_mov_b32_e32 v37, s7                            ;	v37 = s7;
	s_add_i32 s5, s5, s20                            ;	s5.i32 = s5 + s20; scc=overflow_or_carry
	s_lshl_b32 s5, s5, 2                             ;	s5 = s5 << 2[4:0]; scc=(s5!=0);
	v_mov_b32_e32 v35, s5                            ;	v35 = s5;
	buffer_load_dword v77, v37, s[36:39], 0 offen
	buffer_load_dword v76, v35, s[36:39], 0 offen
	buffer_load_dwordx4 v[66:69], v51, s[24:27], 0 offen offset:128
	buffer_load_dwordx4 v[70:73], v36, s[24:27], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v34, 4, v34                    ;	v34.b32 = v34 << 4;
	v_add_u32_e32 v154, v152, v34                    ;	v154 = v152 + v34
	s_add_i32 s5, s45, -2                            ;	s5.i32 = s45 + -2; scc=overflow_or_carry
	v_or_b32_e32 v163, 1, v1
	v_sub_u32_e32 v34, v163, v49
	v_lshlrev_b32_e32 v35, 10, v47                   ;	v35.b32 = v47 << 10;
	v_lshl_or_b32 v35, s2, 14, v35                   ;	v35.u32 = (s2.u32 << 14[4:0].u32) | v35.u32;
	v_or_b32_e32 v36, 0x3000, v35
	v_mul_lo_u32 v160, s44, v36
	v_or_b32_e32 v36, 0x2000, v35
	v_mul_lo_u32 v158, s44, v36
	v_or_b32_e32 v36, 0x1000, v35
	v_mul_lo_u32 v159, s44, v36
	v_mul_lo_u32 v161, s44, v35
	v_lshlrev_b32_e32 v164, 7, v34                   ;	v164.b32 = v34 << 7;
	v_add_u32_e32 v165, -1, v34                      ;	v165 = -1 + v34
	v_lshlrev_b32_e32 v166, 2, v43                   ;	v166.b32 = v43 << 2;
	s_lshl_b32 s37, s3, 8                            ;	s37 = s3 << 8[4:0]; scc=(s37!=0);
	s_mul_i32 s7, s12, 12                            ;	s7 = s12 * 12
	s_add_i32 s38, s37, s7                           ;	s38.i32 = s37 + s7; scc=overflow_or_carry
	s_lshl_b32 s7, s12, 3                            ;	s7 = s12 << 3[4:0]; scc=(s7!=0);
	s_add_i32 s37, s37, s7                           ;	s37.i32 = s37 + s7; scc=overflow_or_carry
	s_or_b32 s11, s21, 1                             ;	s11 = s21 | 1;  scc=(s11!=0);
	s_mul_i32 s11, s20, s11                          ;	s11 = s20 * s11
	s_lshl_b32 s11, s11, 2                           ;	s11 = s11 << 2[4:0]; scc=(s11!=0);
	s_mul_i32 s20, s2, s20                           ;	s20 = s2 * s20
	s_lshl_b32 s36, s20, 3                           ;	s36 = s20 << 3[4:0]; scc=(s36!=0);
	s_mov_b32 s20, s6                                ;	s20 = s6
	s_mov_b32 s21, s0                                ;	s21 = s0
	s_mov_b32 s28, s10                               ;	s28 = s10
	s_mov_b32 s4, 0                                  ;	s4 = 0
	v_mov_b32_e32 v167, s38                          ;	v167 = s38;
	v_mov_b32_e32 v168, s37                          ;	v168 = s37;
	v_add_u32_e32 v169, s15, v157                    ;	v169 = s15 + v157
	v_mov_b32_e32 v88, 0                             ;	v88 = 0;
	v_mov_b32_e32 v170, 0xffffff80                   ;	v170 = 0xffffff80;
	v_mov_b32_e32 v89, v88                           ;	v89 = v88;
	v_mov_b32_e32 v92, v88                           ;	v92 = v88;
	v_mov_b32_e32 v93, v88                           ;	v93 = v88;
	v_mov_b32_e32 v94, v88                           ;	v94 = v88;
	v_mov_b32_e32 v95, v88                           ;	v95 = v88;
	v_mov_b32_e32 v98, v88                           ;	v98 = v88;
	v_mov_b32_e32 v99, v88                           ;	v99 = v88;
	v_mov_b32_e32 v90, v88                           ;	v90 = v88;
	v_mov_b32_e32 v91, v88                           ;	v91 = v88;
	v_mov_b32_e32 v100, v88                          ;	v100 = v88;
	v_mov_b32_e32 v101, v88                          ;	v101 = v88;
	v_mov_b32_e32 v104, v88                          ;	v104 = v88;
	v_mov_b32_e32 v105, v88                          ;	v105 = v88;
	v_mov_b32_e32 v106, v88                          ;	v106 = v88;
	v_mov_b32_e32 v107, v88                          ;	v107 = v88;
	v_mov_b32_e32 v96, v88                           ;	v96 = v88;
	v_mov_b32_e32 v97, v88                           ;	v97 = v88;
	v_mov_b32_e32 v102, v88                          ;	v102 = v88;
	v_mov_b32_e32 v103, v88                          ;	v103 = v88;
	v_mov_b32_e32 v108, v88                          ;	v108 = v88;
	v_mov_b32_e32 v109, v88                          ;	v109 = v88;
	v_mov_b32_e32 v112, v88                          ;	v112 = v88;
	v_mov_b32_e32 v113, v88                          ;	v113 = v88;
	v_mov_b32_e32 v110, v88                          ;	v110 = v88;
	v_mov_b32_e32 v111, v88                          ;	v111 = v88;
	v_mov_b32_e32 v116, v88                          ;	v116 = v88;
	v_mov_b32_e32 v117, v88                          ;	v117 = v88;
	v_mov_b32_e32 v120, v88                          ;	v120 = v88;
	v_mov_b32_e32 v121, v88                          ;	v121 = v88;
	v_mov_b32_e32 v124, v88                          ;	v124 = v88;
	v_mov_b32_e32 v125, v88                          ;	v125 = v88;
	v_mov_b32_e32 v114, v88                          ;	v114 = v88;
	v_mov_b32_e32 v115, v88                          ;	v115 = v88;
	v_mov_b32_e32 v118, v88                          ;	v118 = v88;
	v_mov_b32_e32 v119, v88                          ;	v119 = v88;
	v_mov_b32_e32 v122, v88                          ;	v122 = v88;
	v_mov_b32_e32 v123, v88                          ;	v123 = v88;
	v_mov_b32_e32 v126, v88                          ;	v126 = v88;
	v_mov_b32_e32 v127, v88                          ;	v127 = v88;
	v_mov_b32_e32 v128, v88                          ;	v128 = v88;
	v_mov_b32_e32 v129, v88                          ;	v129 = v88;
	v_mov_b32_e32 v130, v88                          ;	v130 = v88;
	v_mov_b32_e32 v131, v88                          ;	v131 = v88;
	v_mov_b32_e32 v132, v88                          ;	v132 = v88;
	v_mov_b32_e32 v133, v88                          ;	v133 = v88;
	v_mov_b32_e32 v134, v88                          ;	v134 = v88;
	v_mov_b32_e32 v135, v88                          ;	v135 = v88;
	v_mov_b32_e32 v136, v88                          ;	v136 = v88;
	v_mov_b32_e32 v137, v88                          ;	v137 = v88;
	v_mov_b32_e32 v138, v88                          ;	v138 = v88;
	v_mov_b32_e32 v139, v88                          ;	v139 = v88;
	v_mov_b32_e32 v140, v88                          ;	v140 = v88;
	v_mov_b32_e32 v141, v88                          ;	v141 = v88;
	v_mov_b32_e32 v142, v88                          ;	v142 = v88;
	v_mov_b32_e32 v143, v88                          ;	v143 = v88;
	v_mov_b32_e32 v144, v88                          ;	v144 = v88;
	v_mov_b32_e32 v145, v88                          ;	v145 = v88;
	v_mov_b32_e32 v146, v88                          ;	v146 = v88;
	v_mov_b32_e32 v147, v88                          ;	v147 = v88;
	ds_read_b128 v[58:61], v152                      ;	v[58:61] = LDS_MEM[v152 + 0].b128; // read w/o any type convertion
	ds_read_b128 v[50:53], v152 offset:2048          ;	v[50:53] = LDS_MEM[v152 + 2048].b128; // read w/o any type convertion
	ds_read_b128 v[62:65], v154                      ;	v[62:65] = LDS_MEM[v154 + 0].b128; // read w/o any type convertion
	ds_read_b128 v[54:57], v154 offset:2048          ;	v[54:57] = LDS_MEM[v154 + 2048].b128; // read w/o any type convertion
	ds_read_b128 v[42:45], v152 offset:4096          ;	v[42:45] = LDS_MEM[v152 + 4096].b128; // read w/o any type convertion
	ds_read_b128 v[34:37], v152 offset:6144          ;	v[34:37] = LDS_MEM[v152 + 6144].b128; // read w/o any type convertion
	ds_read_b128 v[46:49], v154 offset:4096          ;	v[46:49] = LDS_MEM[v154 + 4096].b128; // read w/o any type convertion
	ds_read_b128 v[38:41], v154 offset:6144          ;	v[38:41] = LDS_MEM[v154 + 6144].b128; // read w/o any type convertion
	v_mov_b32_e32 v148, v88                          ;	v148 = v88;
	v_mov_b32_e32 v149, v88                          ;	v149 = v88;
	v_mov_b32_e32 v150, v88                          ;	v150 = v88;
	v_mov_b32_e32 v151, v88                          ;	v151 = v88;
	s_waitcnt vmcnt(8) lgkmcnt(5)
	v_mfma_f32_16x16x128_f8f6f4 v[248:251], v[26:33], v[58:65], 0
	v_add_u32_e32 v172, v153, v161                   ;	v172 = v153 + v161
	buffer_load_dwordx4 v[176:179], v172, s[20:23], 0 offen offset:2048
	v_mfma_f32_16x16x128_f8f6f4 v[244:247], v[18:25], v[58:65], 0
	v_add_u32_e32 v255, v153, v159                   ;	v255 = v153 + v159
	buffer_load_dwordx4 v[200:203], v255, s[20:23], 0 offen offset:2048
	v_mfma_f32_16x16x128_f8f6f4 v[240:243], v[10:17], v[58:65], 0
	v_add_u32_e32 v254, v153, v158                   ;	v254 = v153 + v158
	buffer_load_dwordx4 v[192:195], v254, s[20:23], 0 offen offset:2048
	v_mfma_f32_16x16x128_f8f6f4 v[236:239], v[2:9], v[58:65], 0
	v_add_u32_e32 v174, v153, v160                   ;	v174 = v153 + v160
	buffer_load_dwordx4 v[184:187], v174, s[20:23], 0 offen offset:2048
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x128_f8f6f4 v[232:235], v[26:33], v[50:57], 0
	buffer_load_dwordx4 v[188:191], v174, s[20:23], 0 offen offset:3072
	v_mfma_f32_16x16x128_f8f6f4 v[228:231], v[18:25], v[50:57], 0
	buffer_load_dwordx4 v[196:199], v254, s[20:23], 0 offen offset:3072
	v_mfma_f32_16x16x128_f8f6f4 v[224:227], v[10:17], v[50:57], 0
	buffer_load_dwordx4 v[204:207], v255, s[20:23], 0 offen offset:3072
	v_mfma_f32_16x16x128_f8f6f4 v[220:223], v[2:9], v[50:57], 0
	buffer_load_dwordx4 v[180:183], v172, s[20:23], 0 offen offset:3072
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x128_f8f6f4 v[216:219], v[26:33], v[42:49], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(9)
	ds_write_b128 v171, v[66:69]                     ;	LDS_MEM[v171 + 0].b128 = v[66:69].b128
	v_mfma_f32_16x16x128_f8f6f4 v[212:215], v[18:25], v[42:49], 0
	v_add_u32_e32 v253, v156, v157                   ;	v253 = v156 + v157
	v_ashrrev_i32_e32 v50, 31, v163
	v_add_u32_e32 v252, v156, v169                   ;	v252 = v156 + v169
	v_lshrrev_b32_e32 v50, 29, v50                   ;	v50.b32 = v50 >> 29;
	buffer_load_dwordx4 v[208:211], v253, s[24:27], 0 offen offset:256
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[10:17], v[42:49], 0
	v_add_u32_e32 v50, v163, v50                     ;	v50 = v163 + v50
	v_and_b32_e32 v50, -8, v50                       ;	v50.u32 = (-8 & v50.u32)
	v_sub_u32_e32 v50, v163, v50
	v_xor_b32_e32 v173, v50, v155                    ;	v173.u32 = (v50 ^ v155.u32)
	v_sub_u32_e32 v50, v173, v162
	v_lshlrev_b32_e32 v50, 4, v50                    ;	v50.b32 = v50 << 4;
	v_add3_u32 v50, v171, v164, v50
	s_waitcnt vmcnt(9)
	ds_write_b128 v50, v[70:73]                      ;	LDS_MEM[v50 + 0].b128 = v[70:73].b128
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[2:9], v[42:49], 0
	buffer_load_dwordx4 v[66:69], v252, s[24:27], 0 offen offset:256
	v_add_u32_e32 v42, -1, v163                      ;	v42 = -1 + v163
	v_add_u32_e32 v71, 0x1000, v172                  ;	v71 = 0x1000 + v172
	v_ashrrev_i32_e32 v43, 31, v42
	v_lshrrev_b32_e32 v43, 29, v43                   ;	v43.b32 = v43 >> 29;
	v_add_u32_e32 v43, v42, v43                      ;	v43 = v42 + v43
	v_and_b32_e32 v43, -8, v43                       ;	v43.u32 = (-8 & v43.u32)
	v_sub_u32_e32 v42, v42, v43
	v_xor_b32_e32 v162, v42, v155                    ;	v162.u32 = (v42 ^ v155.u32)
	v_sub_u32_e32 v42, v162, v173
	v_lshl_add_u32 v172, v42, 4, v170                ;	v172.u32 = (v42.u32 << 4.u32[2 : 0].u32) + v170.u32
	v_pk_fma_f32 v[150:151], v[248:249], v[86:87], v[150:151] op_sel:[0,1,0]
	v_pk_fma_f32 v[148:149], v[250:251], v[86:87], v[148:149] op_sel:[0,1,0]
	v_add_u32_e32 v70, v50, v172                     ;	v70 = v50 + v172
	v_pk_fma_f32 v[146:147], v[244:245], v[86:87], v[146:147] op_sel:[0,1,0]
	v_pk_fma_f32 v[144:145], v[246:247], v[86:87], v[144:145] op_sel:[0,1,0]
	v_pk_fma_f32 v[142:143], v[240:241], v[86:87], v[142:143] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[140:141], v[242:243], v[86:87], v[140:141] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[138:139], v[236:237], v[86:87], v[138:139] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[136:137], v[238:239], v[86:87], v[136:137] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[134:135], v[232:233], v[84:85], v[134:135] op_sel:[0,1,0]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[26:33], v[34:41], 0
	v_fma_f32 v132, v234, v85, v132
	v_fma_f32 v133, v235, v85, v133
	v_fma_f32 v130, v228, v85, v130
	v_fma_f32 v131, v229, v85, v131
	v_fma_f32 v128, v230, v85, v128
	v_fma_f32 v129, v231, v85, v129
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[26:29], v152                      ;	v[26:29] = LDS_MEM[v152 + 0].b128; // read w/o any type convertion
	ds_read_b128 v[30:33], v154                      ;	v[30:33] = LDS_MEM[v154 + 0].b128; // read w/o any type convertion
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[18:25], v[34:41], 0
	v_fma_f32 v126, v224, v84, v126
	v_fma_f32 v127, v225, v84, v127
	v_fma_f32 v122, v226, v84, v122
	v_fma_f32 v123, v227, v84, v123
	v_fma_f32 v118, v220, v84, v118
	v_fma_f32 v119, v221, v84, v119
	v_pk_fma_f32 v[114:115], v[222:223], v[84:85], v[114:115] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[124:125], v[216:217], v[82:83], v[124:125] op_sel:[0,1,0]
	ds_read_b128 v[18:21], v152 offset:2048          ;	v[18:21] = LDS_MEM[v152 + 2048].b128; // read w/o any type convertion
	ds_read_b128 v[22:25], v154 offset:2048          ;	v[22:25] = LDS_MEM[v154 + 2048].b128; // read w/o any type convertion
	v_mfma_f32_16x16x128_f8f6f4 v[10:13], v[10:17], v[34:41], 0
	v_fma_f32 v120, v218, v83, v120
	v_fma_f32 v121, v219, v83, v121
	v_fma_f32 v116, v212, v83, v116
	v_fma_f32 v117, v213, v83, v117
	v_fma_f32 v110, v214, v83, v110
	v_fma_f32 v111, v215, v83, v111
	v_pk_fma_f32 v[112:113], v[62:63], v[82:83], v[112:113] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[108:109], v[64:65], v[82:83], v[108:109] op_sel_hi:[1,0,1]
	ds_read_b128 v[42:45], v152 offset:4096          ;	v[42:45] = LDS_MEM[v152 + 4096].b128; // read w/o any type convertion
	ds_read_b128 v[46:49], v154 offset:4096          ;	v[46:49] = LDS_MEM[v154 + 4096].b128; // read w/o any type convertion
	v_mfma_f32_16x16x128_f8f6f4 v[2:5], v[2:9], v[34:41], 0
	ds_read_b128 v[34:37], v152 offset:6144          ;	v[34:37] = LDS_MEM[v152 + 6144].b128; // read w/o any type convertion
	ds_read_b128 v[38:41], v154 offset:6144          ;	v[38:41] = LDS_MEM[v154 + 6144].b128; // read w/o any type convertion
	v_fma_f32 v240, v58, v82, v102
	v_fma_f32 v241, v59, v82, v103
	v_fma_f32 v238, v60, v82, v96
	v_fma_f32 v239, v61, v82, v97
	v_pk_fma_f32 v[236:237], v[54:55], v[80:81], v[106:107] op_sel:[0,1,0]
	v_pk_fma_f32 v[234:235], v[56:57], v[80:81], v[104:105] op_sel:[0,1,0]
	v_pk_fma_f32 v[232:233], v[50:51], v[80:81], v[100:101] op_sel:[0,1,0]
	v_pk_fma_f32 v[230:231], v[52:53], v[80:81], v[90:91] op_sel:[0,1,0]
	v_pk_fma_f32 v[228:229], v[10:11], v[80:81], v[98:99] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[226:227], v[12:13], v[80:81], v[94:95] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[224:225], v[2:3], v[80:81], v[92:93] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[222:223], v[4:5], v[80:81], v[88:89] op_sel_hi:[1,0,1]
	s_waitcnt vmcnt(2) lgkmcnt(6)
	v_mfma_f32_16x16x128_f8f6f4 v[104:107], v[176:183], v[26:33], 0
	v_add_u32_e32 v2, v166, v168                     ;	v2 = v166 + v168
	buffer_load_dword v220, v2, s[28:31], 0 offen
	v_mfma_f32_16x16x128_f8f6f4 v[100:103], v[200:207], v[26:33], 0
	buffer_load_dword v218, v2, s[28:31], 0 offen offset:64
	v_mfma_f32_16x16x128_f8f6f4 v[88:91], v[192:199], v[26:33], 0
	buffer_load_dword v216, v2, s[28:31], 0 offen offset:128
	v_mfma_f32_16x16x128_f8f6f4 v[96:99], v[184:191], v[26:33], 0
	buffer_load_dword v212, v2, s[28:31], 0 offen offset:192
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x128_f8f6f4 v[92:95], v[176:183], v[18:25], 0
	s_add_i32 s10, s36, s34                          ;	s10.i32 = s36 + s34; scc=overflow_or_carry
	v_mov_b32_e32 v213, s10                          ;	v213 = s10;
	buffer_load_dword v215, v213, s[16:19], 0 offen offset:8
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[200:207], v[18:25], 0
	s_add_i32 s10, s11, s34                          ;	s10.i32 = s11 + s34; scc=overflow_or_carry
	v_mov_b32_e32 v175, s10                          ;	v175 = s10;
	buffer_load_dword v214, v175, s[16:19], 0 offen offset:8
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[192:199], v[18:25], 0
	buffer_load_dwordx4 v[26:29], v71, s[20:23], 0 offen
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[184:191], v[18:25], 0
	v_add_u32_e32 v22, 0x1000, v255                  ;	v22 = 0x1000 + v255
	buffer_load_dwordx4 v[18:21], v22, s[20:23], 0 offen
	v_add_u32_e32 v14, 0x1000, v254                  ;	v14 = 0x1000 + v254
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[176:183], v[42:49], 0
	buffer_load_dwordx4 v[10:13], v14, s[20:23], 0 offen
	v_add_u32_e32 v6, 0x1000, v174                   ;	v6 = 0x1000 + v174
	buffer_load_dwordx4 v[2:5], v6, s[20:23], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[6:9], v6, s[20:23], 0 offen offset:1024
	s_nop 0
	buffer_load_dwordx4 v[14:17], v14, s[20:23], 0 offen offset:1024
	s_nop 0
	buffer_load_dwordx4 v[22:25], v22, s[20:23], 0 offen offset:1024
	s_nop 0
	buffer_load_dwordx4 v[30:33], v71, s[20:23], 0 offen offset:1024
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(15)
	ds_write_b128 v70, v[208:211]                    ;	LDS_MEM[v70 + 0].b128 = v[208:211].b128
	v_sub_u32_e32 v71, v173, v162
	v_lshlrev_b32_e32 v173, 4, v71                   ;	v173.b32 = v71 << 4;
	v_add3_u32 v174, v173, s1, v70
	s_waitcnt vmcnt(14)
	ds_write_b128 v174, v[66:69]                     ;	LDS_MEM[v174 + 0].b128 = v[66:69].b128
	buffer_load_dwordx4 v[66:69], v253, s[24:27], 0 offen offset:384
	v_mfma_f32_16x16x128_f8f6f4 v[80:83], v[184:191], v[34:41], 0
	buffer_load_dwordx4 v[70:73], v252, s[24:27], 0 offen offset:384
	v_mul_f32_e64 v210, v78, v76
	v_mul_f32_e64 v211, v79, v77
	v_mul_f32_e64 v208, v74, v76
	v_mul_f32_e64 v209, v75, v77
	v_add_u32_e32 v171, v174, v172                   ;	v171 = v174 + v172
	v_pk_fma_f32 v[150:151], v[104:105], v[210:211], v[150:151] op_sel:[0,1,0]
	v_mfma_f32_16x16x128_f8f6f4 v[84:87], v[192:199], v[34:41], 0
	v_fma_f32 v148, v106, v211, v148
	v_fma_f32 v149, v107, v211, v149
	v_fma_f32 v146, v100, v211, v146
	v_fma_f32 v147, v101, v211, v147
	v_fma_f32 v144, v102, v211, v144
	v_fma_f32 v145, v103, v211, v145
	v_mul_f32_e32 v100, v79, v76
	v_pk_fma_f32 v[142:143], v[88:89], v[100:101], v[142:143] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[140:141], v[90:91], v[100:101], v[140:141] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[138:139], v[96:97], v[100:101], v[138:139] op_sel_hi:[1,0,1]
	v_mfma_f32_16x16x128_f8f6f4 v[88:91], v[200:207], v[34:41], 0
	v_fma_f32 v136, v98, v100, v136
	v_fma_f32 v137, v99, v100, v137
	v_mul_f32_e32 v78, v78, v77
	v_fma_f32 v134, v92, v78, v134
	v_fma_f32 v135, v93, v78, v135
	v_pk_fma_f32 v[132:133], v[94:95], v[78:79], v[132:133] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[130:131], v[50:51], v[78:79], v[130:131] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[128:129], v[52:53], v[78:79], v[128:129] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[126:127], v[62:63], v[210:211], v[126:127] op_sel_hi:[1,0,1]
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[200:207], v[42:49], 0
	v_fma_f32 v122, v64, v210, v122
	v_fma_f32 v123, v65, v210, v123
	v_fma_f32 v118, v58, v210, v118
	v_fma_f32 v119, v59, v210, v119
	v_fma_f32 v114, v60, v210, v114
	v_fma_f32 v115, v61, v210, v115
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[58:61], v152                      ;	v[58:61] = LDS_MEM[v152 + 0].b128; // read w/o any type convertion
	ds_read_b128 v[62:65], v154                      ;	v[62:65] = LDS_MEM[v154 + 0].b128; // read w/o any type convertion
	v_mfma_f32_16x16x128_f8f6f4 v[92:95], v[192:199], v[42:49], 0
	v_fma_f32 v124, v54, v209, v124
	v_fma_f32 v125, v55, v209, v125
	v_fma_f32 v120, v56, v209, v120
	v_fma_f32 v121, v57, v209, v121
	v_fma_f32 v116, v50, v209, v116
	v_fma_f32 v117, v51, v209, v117
	v_pk_fma_f32 v[110:111], v[52:53], v[208:209], v[110:111] op_sel:[0,1,0]
	v_mul_f32_e32 v76, v75, v76
	ds_read_b128 v[50:53], v152 offset:2048          ;	v[50:53] = LDS_MEM[v152 + 2048].b128; // read w/o any type convertion
	ds_read_b128 v[54:57], v154 offset:2048          ;	v[54:57] = LDS_MEM[v154 + 2048].b128; // read w/o any type convertion
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[184:191], v[42:49], 0
	s_nop 0
	v_fma_f32 v112, v92, v76, v112
	v_fma_f32 v113, v93, v76, v113
	v_fma_f32 v108, v94, v76, v108
	v_fma_f32 v109, v95, v76, v109
	s_nop 6
	v_pk_fma_f32 v[102:103], v[42:43], v[76:77], v[240:241] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[96:97], v[44:45], v[76:77], v[238:239] op_sel_hi:[1,0,1]
	v_mul_f32_e32 v74, v74, v77
	ds_read_b128 v[42:45], v152 offset:4096          ;	v[42:45] = LDS_MEM[v152 + 4096].b128; // read w/o any type convertion
	ds_read_b128 v[46:49], v154 offset:4096          ;	v[46:49] = LDS_MEM[v154 + 4096].b128; // read w/o any type convertion
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[176:183], v[34:41], 0
	s_nop 11
	v_pk_fma_f32 v[106:107], v[34:35], v[74:75], v[236:237] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[104:105], v[36:37], v[74:75], v[234:235] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[100:101], v[88:89], v[74:75], v[232:233] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[90:91], v[90:91], v[74:75], v[230:231] op_sel_hi:[1,0,1]
	ds_read_b128 v[34:37], v152 offset:6144          ;	v[34:37] = LDS_MEM[v152 + 6144].b128; // read w/o any type convertion
	ds_read_b128 v[38:41], v154 offset:6144          ;	v[38:41] = LDS_MEM[v154 + 6144].b128; // read w/o any type convertion
	v_pk_fma_f32 v[98:99], v[84:85], v[208:209], v[228:229] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[94:95], v[86:87], v[208:209], v[226:227] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[92:93], v[80:81], v[208:209], v[224:225] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[88:89], v[82:83], v[208:209], v[222:223] op_sel_hi:[1,0,1]
	v_add_u32_e32 v74, v166, v167                    ;	v74 = v166 + v167
	buffer_load_dword v77, v213, s[16:19], 0 offen offset:12
	buffer_load_dword v76, v175, s[16:19], 0 offen offset:12
	buffer_load_dword v79, v74, s[28:31], 0 offen
	buffer_load_dword v78, v74, s[28:31], 0 offen offset:64
	buffer_load_dword v75, v74, s[28:31], 0 offen offset:128
	s_nop 0
	buffer_load_dword v74, v74, s[28:31], 0 offen offset:192
	s_waitcnt vmcnt(16)
	v_pk_mul_f32 v[86:87], v[220:221], v[214:215] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[218:219], v[214:215] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[216:217], v[214:215] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[212:213], v[214:215] op_sel_hi:[0,1]
	s_add_i32 s4, s4, 2                              ;	s4.i32 = s4 + 2; scc=overflow_or_carry
	v_add_u32_e32 v160, 0x1000, v160                 ;	v160 = 0x1000 + v160
	v_add_u32_e32 v158, 0x1000, v158                 ;	v158 = 0x1000 + v158
	v_add_u32_e32 v159, 0x1000, v159                 ;	v159 = 0x1000 + v159
	v_add_u32_e32 v161, 0x1000, v161                 ;	v161 = 0x1000 + v161
	v_add_u32_e32 v163, v163, v165                   ;	v163 = v163 + v165
	v_add_u32_e32 v167, s7, v167                     ;	v167 = s7 + v167
	v_add_u32_e32 v168, s7, v168                     ;	v168 = s7 + v168
	s_add_i32 s34, s34, 8                            ;	s34.i32 = s34 + 8; scc=overflow_or_carry
	v_add_u32_e32 v169, 0x100, v169                  ;	v169 = 0x100 + v169
	s_cmp_lt_i32 s4, s5                              ;	scc = (s4.i32 < s5.i32)
	v_add_u32_e32 v157, 0x100, v157                  ;	v157 = 0x100 + v157
	s_cbranch_scc1 65080                             ;	jump to 65080 if scc1
	s_waitcnt vmcnt(8) lgkmcnt(6)
	v_mfma_f32_16x16x128_f8f6f4 v[220:223], v[26:33], v[58:65], 0
	v_mfma_f32_16x16x128_f8f6f4 v[216:219], v[18:25], v[58:65], 0
	v_mfma_f32_16x16x128_f8f6f4 v[212:215], v[10:17], v[58:65], 0
	v_mfma_f32_16x16x128_f8f6f4 v[208:211], v[2:9], v[58:65], 0
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x128_f8f6f4 v[204:207], v[26:33], v[50:57], 0
	v_mfma_f32_16x16x128_f8f6f4 v[200:203], v[18:25], v[50:57], 0
	v_mfma_f32_16x16x128_f8f6f4 v[196:199], v[10:17], v[50:57], 0
	v_mfma_f32_16x16x128_f8f6f4 v[192:195], v[2:9], v[50:57], 0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[188:191], v[26:33], v[42:49], 0
	v_add_u32_e32 v50, v153, v161                    ;	v50 = v153 + v161
	v_add_u32_e32 v155, v153, v160                   ;	v155 = v153 + v160
	v_mfma_f32_16x16x128_f8f6f4 v[184:187], v[18:25], v[42:49], 0
	s_mov_b32 s20, s6                                ;	s20 = s6
	s_mov_b32 s21, s0                                ;	s21 = s0
	buffer_load_dwordx4 v[50:53], v50, s[20:23], 0 offen offset:2048
	v_subrev_u32_e32 v54, s35, v155
	v_add_u32_e32 v156, 0xc00, v54                   ;	v156 = 0xc00 + v54
	v_subrev_u32_e32 v58, s35, v156
	v_subrev_u32_e32 v54, s35, v58
	v_mfma_f32_16x16x128_f8f6f4 v[180:183], v[10:17], v[42:49], 0
	buffer_load_dwordx4 v[54:57], v54, s[20:23], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[62:65], v58, s[20:23], 0 offen
	v_add_u32_e32 v58, v153, v159                    ;	v58 = v153 + v159
	v_mfma_f32_16x16x128_f8f6f4 v[176:179], v[2:9], v[42:49], 0
	buffer_load_dwordx4 v[58:61], v58, s[20:23], 0 offen offset:2048
	v_add_u32_e32 v42, v153, v158                    ;	v42 = v153 + v158
	buffer_load_dwordx4 v[42:45], v42, s[20:23], 0 offen offset:2048
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[168:171], v[26:33], v[34:41], 0
	buffer_load_dwordx4 v[46:49], v156, s[20:23], 0 offen
	buffer_load_dwordx4 v[26:29], v155, s[20:23], 0 offen offset:3072
	v_mfma_f32_16x16x128_f8f6f4 v[164:167], v[18:25], v[34:41], 0
	buffer_load_dwordx4 v[22:25], v155, s[20:23], 0 offen offset:2048
	v_mfma_f32_16x16x128_f8f6f4 v[160:163], v[10:17], v[34:41], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v10, v174, v172                    ;	v10 = v174 + v172
	s_waitcnt vmcnt(15)
	ds_write_b128 v10, v[66:69]                      ;	LDS_MEM[v10 + 0].b128 = v[66:69].b128
	v_add_u32_e32 v10, v10, v173                     ;	v10 = v10 + v173
	s_waitcnt vmcnt(14)
	ds_write_b128 v10, v[70:73] offset:128           ;	LDS_MEM[v10 + 128].b128 = v[70:73].b128
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[2:9], v[34:41], 0
	s_barrier
	ds_read_b128 v[14:17], v154                      ;	v[14:17] = LDS_MEM[v154 + 0].b128; // read w/o any type convertion
	ds_read_b128 v[10:13], v152                      ;	v[10:13] = LDS_MEM[v152 + 0].b128; // read w/o any type convertion
	ds_read_b128 v[2:5], v152 offset:2048            ;	v[2:5] = LDS_MEM[v152 + 2048].b128; // read w/o any type convertion
	ds_read_b128 v[6:9], v154 offset:2048            ;	v[6:9] = LDS_MEM[v154 + 2048].b128; // read w/o any type convertion
	s_waitcnt vmcnt(6) lgkmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[156:159], v[50:57], v[10:17], 0
	s_waitcnt vmcnt(4)
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[58:65], v[10:17], 0
	s_mul_i32 s0, s33, s14                           ;	s0 = s33 * s14
	s_mul_hi_u32 s4, 0, s14                          ;	s4 = 0 * s14
	s_waitcnt vmcnt(2)
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[42:49], v[10:17], 0
	s_add_i32 s4, s4, s0                             ;	s4.i32 = s4 + s0; scc=overflow_or_carry
	v_fma_f32 v224, v220, v87, v150
	v_fma_f32 v225, v221, v87, v151
	v_fma_f32 v222, v222, v87, v148
	v_fma_f32 v223, v223, v87, v149
	v_pk_fma_f32 v[220:221], v[216:217], v[86:87], v[146:147] op_sel:[0,1,0]
	v_pk_fma_f32 v[218:219], v[218:219], v[86:87], v[144:145] op_sel:[0,1,0]
	v_pk_fma_f32 v[216:217], v[212:213], v[86:87], v[142:143] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[214:215], v[214:215], v[86:87], v[140:141] op_sel_hi:[1,0,1]
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[22:29], v[10:17], 0
	v_fma_f32 v212, v208, v86, v138
	v_fma_f32 v213, v209, v86, v139
	v_fma_f32 v208, v210, v86, v136
	v_fma_f32 v209, v211, v86, v137
	v_fma_f32 v174, v204, v85, v134
	v_fma_f32 v175, v205, v85, v135
	v_pk_fma_f32 v[172:173], v[206:207], v[84:85], v[132:133] op_sel:[0,1,0]
	v_pk_fma_f32 v[86:87], v[200:201], v[84:85], v[130:131] op_sel:[0,1,0]
	v_pk_fma_f32 v[150:151], v[202:203], v[84:85], v[128:129] op_sel:[0,1,0]
	v_pk_fma_f32 v[148:149], v[196:197], v[84:85], v[126:127] op_sel_hi:[1,0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[50:57], v[2:9], 0
	v_fma_f32 v146, v198, v84, v122
	v_fma_f32 v147, v199, v84, v123
	v_fma_f32 v144, v192, v84, v118
	v_fma_f32 v145, v193, v84, v119
	v_fma_f32 v140, v194, v84, v114
	v_fma_f32 v141, v195, v84, v115
	v_pk_fma_f32 v[138:139], v[188:189], v[82:83], v[124:125] op_sel:[0,1,0]
	v_pk_fma_f32 v[136:137], v[190:191], v[82:83], v[120:121] op_sel:[0,1,0]
	v_pk_fma_f32 v[134:135], v[184:185], v[82:83], v[116:117] op_sel:[0,1,0]
	v_pk_fma_f32 v[132:133], v[186:187], v[82:83], v[110:111] op_sel:[0,1,0]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[58:65], v[2:9], 0
	v_fma_f32 v130, v180, v82, v112
	v_fma_f32 v131, v181, v82, v113
	v_fma_f32 v124, v182, v82, v108
	v_fma_f32 v125, v183, v82, v109
	v_fma_f32 v122, v176, v82, v102
	v_fma_f32 v123, v177, v82, v103
	v_pk_fma_f32 v[114:115], v[178:179], v[82:83], v[96:97] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[106:107], v[168:169], v[80:81], v[106:107] op_sel:[0,1,0]
	v_pk_fma_f32 v[102:103], v[170:171], v[80:81], v[104:105] op_sel:[0,1,0]
	v_pk_fma_f32 v[108:109], v[164:165], v[80:81], v[100:101] op_sel:[0,1,0]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[42:49], v[2:9], 0
	v_fma_f32 v104, v166, v81, v90
	v_fma_f32 v105, v167, v81, v91
	v_fma_f32 v120, v160, v80, v98
	v_fma_f32 v121, v161, v80, v99
	v_fma_f32 v110, v162, v80, v94
	v_fma_f32 v111, v163, v80, v95
	v_pk_fma_f32 v[118:119], v[34:35], v[80:81], v[92:93] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[116:117], v[36:37], v[80:81], v[88:89] op_sel_hi:[1,0,1]
	v_pk_mul_f32 v[142:143], v[78:79], v[76:77]
	v_pk_mul_f32 v[112:113], v[74:75], v[76:77]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[22:29], v[2:9], 0
	ds_read_b128 v[10:13], v152 offset:4096          ;	v[10:13] = LDS_MEM[v152 + 4096].b128; // read w/o any type convertion
	ds_read_b128 v[2:5], v152 offset:6144            ;	v[2:5] = LDS_MEM[v152 + 6144].b128; // read w/o any type convertion
	ds_read_b128 v[14:17], v154 offset:4096          ;	v[14:17] = LDS_MEM[v154 + 4096].b128; // read w/o any type convertion
	ds_read_b128 v[6:9], v154 offset:6144            ;	v[6:9] = LDS_MEM[v154 + 6144].b128; // read w/o any type convertion
	v_fma_f32 v98, v156, v143, v224
	v_fma_f32 v99, v157, v143, v225
	v_pk_fma_f32 v[100:101], v[158:159], v[142:143], v[222:223] op_sel:[0,1,0]
	v_pk_fma_f32 v[90:91], v[18:19], v[142:143], v[220:221] op_sel:[0,1,0]
	v_pk_fma_f32 v[92:93], v[20:21], v[142:143], v[218:219] op_sel:[0,1,0]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[50:57], v[10:17], 0
	v_mul_f32_e32 v80, v79, v76
	v_fma_f32 v82, v70, v80, v216
	v_fma_f32 v83, v71, v80, v217
	v_fma_f32 v84, v72, v80, v214
	v_fma_f32 v85, v73, v80, v215
	v_pk_fma_f32 v[70:71], v[30:31], v[80:81], v[212:213] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[72:73], v[32:33], v[80:81], v[208:209] op_sel_hi:[1,0,1]
	v_mul_f32_e32 v78, v77, v78
	v_pk_fma_f32 v[94:95], v[126:127], v[78:79], v[174:175] op_sel_hi:[1,0,1]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[58:65], v[10:17], 0
	v_fma_f32 v96, v128, v78, v172
	v_fma_f32 v97, v129, v78, v173
	v_fma_f32 v86, v66, v78, v86
	v_fma_f32 v87, v67, v78, v87
	v_fma_f32 v88, v68, v78, v150
	v_fma_f32 v89, v69, v78, v151
	v_pk_fma_f32 v[78:79], v[38:39], v[142:143], v[148:149] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[80:81], v[40:41], v[142:143], v[146:147] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[66:67], v[34:35], v[142:143], v[144:145] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[68:69], v[36:37], v[142:143], v[140:141] op_sel_hi:[1,0,1]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[42:49], v[10:17], 0
	v_fma_f32 v18, v18, v113, v138
	v_fma_f32 v19, v19, v113, v139
	v_fma_f32 v20, v20, v113, v136
	v_fma_f32 v21, v21, v113, v137
	v_fma_f32 v30, v30, v113, v134
	v_fma_f32 v31, v31, v113, v135
	v_pk_fma_f32 v[32:33], v[32:33], v[112:113], v[132:133] op_sel:[0,1,0]
	v_mul_f32_e32 v40, v75, v76
	s_nop 3
	v_pk_fma_f32 v[34:35], v[34:35], v[40:41], v[130:131] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[36:37], v[36:37], v[40:41], v[124:125] op_sel_hi:[1,0,1]
	v_mfma_f32_16x16x128_f8f6f4 v[10:13], v[22:29], v[10:17], 0
	s_nop 11
	v_pk_fma_f32 v[38:39], v[10:11], v[40:41], v[122:123] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[40:41], v[12:13], v[40:41], v[114:115] op_sel_hi:[1,0,1]
	v_mul_f32_e32 v76, v77, v74
	v_and_b32_e32 v11, 60, v1                        ;	v11.u32 = (60 & v1.u32)
	v_lshrrev_b32_e32 v1, 3, v0                      ;	v1.b32 = v0 >> 3;
	v_lshlrev_b32_e32 v10, 3, v0                     ;	v10.b32 = v0 << 3;
	v_and_b32_e32 v10, 56, v10                       ;	v10.u32 = (56 & v10.u32)
	v_lshl_or_b32 v114, s2, 8, v10                   ;	v114.u32 = (s2.u32 << 8[4:0].u32) | v10.u32;
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v0, 8, v0                      ;	v0.b32 = v0 << 8;
	v_and_b32_e32 v0, 0xf00, v0                      ;	v0.u32 = (0xf00 & v0.u32)
	v_lshl_or_b32 v115, v11, 2, v0                   ;	v115.u32 = (v11.u32 << 2[4:0].u32) | v0.u32;
	ds_write_b128 v115, v[98:101]                    ;	LDS_MEM[v115 + 0].b128 = v[98:101].b128
	ds_write_b128 v115, v[94:97] offset:4096         ;	LDS_MEM[v115 + 4096].b128 = v[94:97].b128
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v0, 2, v10                     ;	v0.b32 = v10 << 2;
	v_lshl_or_b32 v77, v1, 8, v0                     ;	v77.u32 = (v1.u32 << 8[4:0].u32) | v0.u32;
	ds_read_b96 v[10:12], v77                        ;	v[10:12] = LDS_MEM[v77 + 0].b96; // read w/o any type convertion
	v_lshl_or_b32 v94, s3, 6, v1                     ;	v94.u32 = (s3.u32 << 6[4:0].u32) | v1.u32;
	v_mad_u64_u32 v[74:75], s[0:1], v94, s14, v[114:115]
	ds_read2_b32 v[14:15], v77 offset0:3 offset1:4
	s_waitcnt lgkmcnt(1)
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_cvt_pk_bf16_f32 v11, v11, v12
	ds_read_b32 v13, v77 offset:28                   ;	v13 = LDS_MEM[v77 + 28].b32; // read w/o any type convertion
	ds_read2_b32 v[0:1], v77 offset0:5 offset1:6
	s_mov_b32 s15, 0x5040100                         ;	s15 = 0x5040100
	v_perm_b32 v10, v11, v10, s15
	s_waitcnt lgkmcnt(2)
	v_cvt_pk_bf16_f32 v12, v14, v15
	v_alignbit_b32 v11, v12, v11, 16
	s_waitcnt lgkmcnt(0)
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_alignbit_b32 v12, v0, v12, 16
	v_cvt_pk_bf16_f32 v1, v13, s0
	v_alignbit_b32 v13, v1, v0, 16
	v_cmp_gt_i32_e32 vcc, s13, v114                  ;	vcc.u64[laneId] = (s13.i32  > v114.i32 )
	v_cmp_gt_i32_e64 s[0:1], s12, v94                ;	s[0:1].u64[laneId] = (s12.i32  > v94.i32 )
	s_add_i32 s2, s13, s4                            ;	s2.i32 = s13 + s4; scc=overflow_or_carry
	s_lshl_b32 s10, s2, 1                            ;	s10 = s2 << 1[4:0]; scc=(s10!=0);
	s_and_b32 s9, s9, 0xffff                         ;	s9 = s9 & 0xffff
	s_mov_b32 s11, 0x20000                           ;	s11 = 0x20000
	v_lshlrev_b32_e32 v95, 1, v74                    ;	v95.b32 = v74 << 1;
	v_bfrev_b32_e32 v75, 1
	s_and_b64 s[2:3], s[0:1], vcc                    ;	s[2:3] = s[0:1] & vcc
	v_cndmask_b32_e64 v0, v75, 0, s[2:3]             ;	v0.b32 = s[2:3].u64[laneId] ? 0.u32 : v75.u32
	v_add_u32_e32 v0, v0, v95                        ;	v0 = v0 + v95
	buffer_store_dwordx4 v[10:13], v0, s[8:11], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v115, v[90:93]                     ;	LDS_MEM[v115 + 0].b128 = v[90:93].b128
	ds_write_b128 v115, v[86:89] offset:4096         ;	LDS_MEM[v115 + 4096].b128 = v[86:89].b128
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b96 v[14:16], v77                        ;	v[14:16] = LDS_MEM[v77 + 0].b96; // read w/o any type convertion
	v_mfma_f32_16x16x128_f8f6f4 v[10:13], v[42:49], v[2:9], 0
	s_nop 11
	v_pk_fma_f32 v[10:11], v[10:11], v[112:113], v[120:121] op_sel_hi:[1,0,1]
	v_or_b32_e32 v44, 64, v114
	ds_read2_b32 v[42:43], v77 offset0:3 offset1:4
	s_waitcnt lgkmcnt(1)
	v_cvt_pk_bf16_f32 v14, v14, s0
	ds_read_b32 v17, v77 offset:28                   ;	v17 = LDS_MEM[v77 + 28].b32; // read w/o any type convertion
	ds_read2_b32 v[0:1], v77 offset0:5 offset1:6
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_perm_b32 v14, v15, v14, s15
	s_waitcnt lgkmcnt(2)
	v_cvt_pk_bf16_f32 v16, v42, v43
	v_alignbit_b32 v15, v16, v15, 16
	s_waitcnt lgkmcnt(0)
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_alignbit_b32 v16, v0, v16, 16
	v_cvt_pk_bf16_f32 v1, v17, s0
	v_alignbit_b32 v17, v1, v0, 16
	v_cmp_gt_i32_e64 s[2:3], s13, v44                ;	s[2:3].u64[laneId] = (s13.i32  > v44.i32 )
	s_and_b64 s[4:5], s[0:1], s[2:3]                 ;	s[4:5] = s[0:1] & s[2:3]
	v_cndmask_b32_e64 v0, v75, 0, s[4:5]             ;	v0.b32 = s[4:5].u64[laneId] ? 0.u32 : v75.u32
	v_add_u32_e32 v0, v95, v0                        ;	v0 = v95 + v0
	buffer_store_dwordx4 v[14:17], v0, s[8:11], 0 offen offset:128
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v115, v[82:85]                     ;	LDS_MEM[v115 + 0].b128 = v[82:85].b128
	ds_write_b128 v115, v[78:81] offset:4096         ;	LDS_MEM[v115 + 4096].b128 = v[78:81].b128
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b96 v[42:44], v77                        ;	v[42:44] = LDS_MEM[v77 + 0].b96; // read w/o any type convertion
	v_mfma_f32_16x16x128_f8f6f4 v[14:17], v[22:29], v[2:9], 0
	s_nop 11
	v_pk_fma_f32 v[14:15], v[14:15], v[112:113], v[118:119] op_sel_hi:[1,0,1]
	v_or_b32_e32 v26, 0x80, v114
	ds_read2_b32 v[24:25], v77 offset0:3 offset1:4
	s_waitcnt lgkmcnt(1)
	v_cvt_pk_bf16_f32 v22, v42, s0
	ds_read_b32 v27, v77 offset:28                   ;	v27 = LDS_MEM[v77 + 28].b32; // read w/o any type convertion
	ds_read2_b32 v[0:1], v77 offset0:5 offset1:6
	v_cvt_pk_bf16_f32 v23, v43, v44
	v_perm_b32 v22, v23, v22, s15
	s_waitcnt lgkmcnt(2)
	v_cvt_pk_bf16_f32 v24, v24, v25
	v_alignbit_b32 v23, v24, v23, 16
	s_waitcnt lgkmcnt(0)
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_alignbit_b32 v24, v0, v24, 16
	v_cvt_pk_bf16_f32 v1, v27, s0
	v_alignbit_b32 v25, v1, v0, 16
	v_cmp_gt_i32_e64 s[4:5], s13, v26                ;	s[4:5].u64[laneId] = (s13.i32  > v26.i32 )
	s_and_b64 s[6:7], s[0:1], s[4:5]                 ;	s[6:7] = s[0:1] & s[4:5]
	v_cndmask_b32_e64 v0, v75, 0, s[6:7]             ;	v0.b32 = s[6:7].u64[laneId] ? 0.u32 : v75.u32
	v_add_u32_e32 v0, v95, v0                        ;	v0 = v95 + v0
	buffer_store_dwordx4 v[22:25], v0, s[8:11], 0 offen offset:256
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v115, v[70:73]                     ;	LDS_MEM[v115 + 0].b128 = v[70:73].b128
	ds_write_b128 v115, v[66:69] offset:4096         ;	LDS_MEM[v115 + 4096].b128 = v[66:69].b128
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b96 v[22:24], v77                        ;	v[22:24] = LDS_MEM[v77 + 0].b96; // read w/o any type convertion
	v_pk_fma_f32 v[16:17], v[16:17], v[112:113], v[116:117] op_sel_hi:[1,0,1]
	v_or_b32_e32 v29, 0xc0, v114
	v_add_u32_e32 v28, 0xc0, v74                     ;	v28 = 0xc0 + v74
	ds_read2_b32 v[26:27], v77 offset0:3 offset1:4
	s_waitcnt lgkmcnt(1)
	v_cvt_pk_bf16_f32 v22, v22, s0
	ds_read_b32 v25, v77 offset:28                   ;	v25 = LDS_MEM[v77 + 28].b32; // read w/o any type convertion
	ds_read2_b32 v[0:1], v77 offset0:5 offset1:6
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_perm_b32 v22, v23, v22, s15
	s_waitcnt lgkmcnt(2)
	v_cvt_pk_bf16_f32 v24, v26, v27
	v_alignbit_b32 v23, v24, v23, 16
	s_waitcnt lgkmcnt(0)
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_alignbit_b32 v24, v0, v24, 16
	v_cvt_pk_bf16_f32 v1, v25, s0
	v_alignbit_b32 v25, v1, v0, 16
	v_cmp_gt_i32_e64 s[6:7], s13, v29                ;	s[6:7].u64[laneId] = (s13.i32  > v29.i32 )
	s_and_b64 s[0:1], s[0:1], s[6:7]                 ;	s[0:1] = s[0:1] & s[6:7]
	v_cndmask_b32_e64 v0, v75, 0, s[0:1]             ;	v0.b32 = s[0:1].u64[laneId] ? 0.u32 : v75.u32
	v_lshl_add_u32 v0, v28, 1, v0                    ;	v0.u32 = (v28.u32 << 1.u32[2 : 0].u32) + v0.u32
	buffer_store_dwordx4 v[22:25], v0, s[8:11], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v115, v[38:41]                     ;	LDS_MEM[v115 + 0].b128 = v[38:41].b128
	ds_write_b128 v115, v[14:17] offset:4096         ;	LDS_MEM[v115 + 4096].b128 = v[14:17].b128
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b96 v[14:16], v77                        ;	v[14:16] = LDS_MEM[v77 + 0].b96; // read w/o any type convertion
	v_pk_fma_f32 v[12:13], v[12:13], v[112:113], v[110:111] op_sel_hi:[1,0,1]
	v_or_b32_e32 v24, 32, v94
	s_lshl_b32 s13, s14, 5                           ;	s13 = s14 << 5[4:0]; scc=(s13!=0);
	ds_read2_b32 v[22:23], v77 offset0:3 offset1:4
	s_waitcnt lgkmcnt(1)
	v_cvt_pk_bf16_f32 v14, v14, s0
	ds_read_b32 v17, v77 offset:28                   ;	v17 = LDS_MEM[v77 + 28].b32; // read w/o any type convertion
	ds_read2_b32 v[0:1], v77 offset0:5 offset1:6
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_perm_b32 v14, v15, v14, s15
	s_waitcnt lgkmcnt(2)
	v_cvt_pk_bf16_f32 v16, v22, v23
	v_alignbit_b32 v15, v16, v15, 16
	s_waitcnt lgkmcnt(0)
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_alignbit_b32 v16, v0, v16, 16
	v_cvt_pk_bf16_f32 v1, v17, s0
	v_alignbit_b32 v17, v1, v0, 16
	v_cmp_gt_i32_e64 s[0:1], s12, v24                ;	s[0:1].u64[laneId] = (s12.i32  > v24.i32 )
	v_add_lshl_u32 v24, v28, s13, 1                  ;	v24 = (v28 + s13) << 1[4:0]
	s_and_b64 s[6:7], s[0:1], s[6:7]                 ;	s[6:7] = s[0:1] & s[6:7]
	v_cndmask_b32_e64 v0, v75, 0, s[6:7]             ;	v0.b32 = s[6:7].u64[laneId] ? 0.u32 : v75.u32
	v_add_u32_e32 v0, v24, v0                        ;	v0 = v24 + v0
	buffer_store_dwordx4 v[14:17], v0, s[8:11], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v115, v[34:37]                     ;	LDS_MEM[v115 + 0].b128 = v[34:37].b128
	ds_write_b128 v115, v[10:13] offset:4096         ;	LDS_MEM[v115 + 4096].b128 = v[10:13].b128
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b96 v[14:16], v77                        ;	v[14:16] = LDS_MEM[v77 + 0].b96; // read w/o any type convertion
	v_mfma_f32_16x16x128_f8f6f4 v[10:13], v[58:65], v[2:9], 0
	s_nop 11
	v_pk_fma_f32 v[10:11], v[10:11], v[76:77], v[108:109] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[12:13], v[12:13], v[76:77], v[104:105] op_sel_hi:[1,0,1]
	ds_read2_b32 v[22:23], v77 offset0:3 offset1:4
	s_waitcnt lgkmcnt(1)
	v_cvt_pk_bf16_f32 v14, v14, s0
	ds_read_b32 v17, v77 offset:28                   ;	v17 = LDS_MEM[v77 + 28].b32; // read w/o any type convertion
	ds_read2_b32 v[0:1], v77 offset0:5 offset1:6
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_perm_b32 v14, v15, v14, s15
	s_waitcnt lgkmcnt(2)
	v_cvt_pk_bf16_f32 v16, v22, v23
	v_alignbit_b32 v15, v16, v15, 16
	s_waitcnt lgkmcnt(0)
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_alignbit_b32 v16, v0, v16, 16
	v_cvt_pk_bf16_f32 v1, v17, s0
	v_alignbit_b32 v17, v1, v0, 16
	v_mov_b32_e32 v0, 0x7fffff80                     ;	v0 = 0x7fffff80;
	v_mov_b32_e32 v1, 0xffffff80                     ;	v1 = 0xffffff80;
	s_and_b64 s[4:5], s[0:1], s[4:5]                 ;	s[4:5] = s[0:1] & s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]             ;	v0.b32 = s[4:5].u64[laneId] ? v1.u32 : v0.u32
	v_add_u32_e32 v0, v24, v0                        ;	v0 = v24 + v0
	buffer_store_dwordx4 v[14:17], v0, s[8:11], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v115, v[30:33]                     ;	LDS_MEM[v115 + 0].b128 = v[30:33].b128
	ds_write_b128 v115, v[10:13] offset:4096         ;	LDS_MEM[v115 + 4096].b128 = v[10:13].b128
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b96 v[10:12], v77                        ;	v[10:12] = LDS_MEM[v77 + 0].b96; // read w/o any type convertion
	v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[50:57], v[2:9], 0
	s_nop 11
	v_pk_fma_f32 v[0:1], v[0:1], v[76:77], v[106:107] op_sel_hi:[1,0,1]
	v_pk_fma_f32 v[2:3], v[2:3], v[76:77], v[102:103] op_sel_hi:[1,0,1]
	ds_read2_b32 v[8:9], v77 offset0:3 offset1:4
	s_waitcnt lgkmcnt(1)
	v_cvt_pk_bf16_f32 v4, v10, s0
	ds_read_b32 v10, v77 offset:28                   ;	v10 = LDS_MEM[v77 + 28].b32; // read w/o any type convertion
	ds_read2_b32 v[6:7], v77 offset0:5 offset1:6
	v_cvt_pk_bf16_f32 v5, v11, v12
	v_perm_b32 v4, v5, v4, s15
	s_waitcnt lgkmcnt(2)
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_alignbit_b32 v5, v8, v5, 16
	s_waitcnt lgkmcnt(0)
	v_cvt_pk_bf16_f32 v7, v6, v7
	v_alignbit_b32 v6, v7, v8, 16
	v_cvt_pk_bf16_f32 v8, v10, s0
	v_alignbit_b32 v7, v8, v7, 16
	v_mov_b32_e32 v8, 0x7fffff00                     ;	v8 = 0x7fffff00;
	v_mov_b32_e32 v9, 0xffffff00                     ;	v9 = 0xffffff00;
	s_and_b64 s[2:3], s[0:1], s[2:3]                 ;	s[2:3] = s[0:1] & s[2:3]
	v_cndmask_b32_e64 v8, v8, v9, s[2:3]             ;	v8.b32 = s[2:3].u64[laneId] ? v9.u32 : v8.u32
	v_add_u32_e32 v8, v24, v8                        ;	v8 = v24 + v8
	buffer_store_dwordx4 v[4:7], v8, s[8:11], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v115, v[18:21]                     ;	LDS_MEM[v115 + 0].b128 = v[18:21].b128
	ds_write_b128 v115, v[0:3] offset:4096           ;	LDS_MEM[v115 + 4096].b128 = v[0:3].b128
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b96 v[0:2], v77                          ;	v[0:2] = LDS_MEM[v77 + 0].b96; // read w/o any type convertion
	v_add_u32_e32 v8, s13, v74                       ;	v8 = s13 + v74
	ds_read2_b32 v[6:7], v77 offset0:3 offset1:4
	s_waitcnt lgkmcnt(1)
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_read_b32 v9, v77 offset:28                    ;	v9 = LDS_MEM[v77 + 28].b32; // read w/o any type convertion
	ds_read2_b32 v[4:5], v77 offset0:5 offset1:6
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_perm_b32 v0, v1, v0, s15
	s_waitcnt lgkmcnt(2)
	v_cvt_pk_bf16_f32 v2, v6, v7
	v_alignbit_b32 v1, v2, v1, 16
	s_waitcnt lgkmcnt(0)
	v_cvt_pk_bf16_f32 v3, v4, v5
	v_alignbit_b32 v2, v3, v2, 16
	v_cvt_pk_bf16_f32 v4, v9, s0
	v_alignbit_b32 v3, v4, v3, 16
	s_and_b64 s[0:1], vcc, s[0:1]                    ;	s[0:1] = vcc & s[0:1]
	v_cndmask_b32_e64 v4, v75, 0, s[0:1]             ;	v4.b32 = s[0:1].u64[laneId] ? 0.u32 : v75.u32
	v_lshl_add_u32 v4, v8, 1, v4                     ;	v4.u32 = (v8.u32 << 1.u32[2 : 0].u32) + v4.u32
	buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen
	s_endpgm
