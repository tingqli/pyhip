def de_shuffle_weight(weight, mfma_MN = 16):
    org_dtype = weight.dtype
    if org_dtype == torch.float4_e2m1fn_x2:
        weight = weight.view(torch.int8)
    M, K = weight.shape
    K_bytes = K * weight.itemsize
    sizeof_DW4 = 16
    mfma_K_lanes = 64 // mfma_MN
    mfma_K_L = sizeof_DW4//weight.itemsize
    mfma_K = mfma_K_lanes * mfma_K_L 

    assert M % mfma_MN == 0
    mfma_K_bytes = mfma_K_lanes * sizeof_DW4
    assert K_bytes % mfma_K_bytes == 0
    #x = x.reshape(M//mfma_MN, mfma_MN, K//mfma_K, mfma_K_lanes, mfma_K_L)
    #x = x.permute(0,2,3,1,4)

    assert K % mfma_K == 0
    weight = weight.reshape(M//mfma_MN, K//mfma_K, mfma_K_lanes, mfma_MN, mfma_K_L)
    weight = weight.permute(0,3,1,2,4)
    weight = weight.reshape(M, K).contiguous()
    return weight.view(org_dtype)

def moe_gemm_ref(activation, quant_type, topk, block_m, sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                 input, input_scale, weight, w_scale, output, is_gateup, is_shuffled):

    global torch, aiter
    import torch
    import aiter
    assert quant_type in [aiter.QuantType.No, aiter.QuantType.per_128x128, aiter.QuantType.per_1x32]
    from aiter.utility import fp4_utils

    #assert weight.dtype == torch.bfloat16
    if is_gateup:
        NUM_TOKENS, NUM_STATES = input.shape
    else:
        NUM_TOKENS, TOPK, NUM_STATES = input.shape

    if str(input.dtype).endswith("_x2"): NUM_STATES *= 2

    NUM_EXPERTS, OC, IC = weight.shape
    if str(weight.dtype).endswith("_x2"): IC *= 2

    NUM_BLOCKS = sorted_expert_ids.shape[0]

    weight_de_shuffle = weight.clone()
    if is_shuffled:
        for i in range(NUM_EXPERTS):
            weight_de_shuffle[i, ...] = de_shuffle_weight(weight[i, ...])

    if w_scale is not None:
        # dequantize
        if quant_type == aiter.QuantType.per_128x128:
            weight_de_shuffle = weight_de_shuffle.view(NUM_EXPERTS, OC//128, 128, IC//128, 128).to(w_scale.dtype) * w_scale.view(NUM_EXPERTS, OC//128, 1, IC//128, 1)
            weight_de_shuffle = weight_de_shuffle.view(NUM_EXPERTS, OC, IC)
        elif quant_type == aiter.QuantType.per_1x32:
            # m//32, (k//32)//8, [(4n,16m), (2n,2m)]
            # sm//32, sn//8, [(4n,16m), (2n,2m)]
            sm, sn = w_scale.shape
            w_scale_e8m0_deshuffle =  fp4_utils.e8m0_to_f32(w_scale).to(output.dtype).view(sm//32, sn//8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2)
            # print(sm, sn, w_scale_e8m0_deshuffle.shape, w_scale_e8m0_deshuffle.dtype, NUM_EXPERTS, OC, IC//32)
            w_scale_e8m0_deshuffle = w_scale_e8m0_deshuffle.reshape(sm, sn)
            w_scale_e8m0_deshuffle = w_scale_e8m0_deshuffle[:NUM_EXPERTS*OC, :IC//32]
            w_scale_e8m0_deshuffle = w_scale_e8m0_deshuffle.view(NUM_EXPERTS, OC, IC//32, 1)
            weight_de_shuffle = fp4_utils.mxfp4_to_f32(weight_de_shuffle).to(output.dtype)
            # print(weight_de_shuffle.shape, NUM_EXPERTS, OC, IC, w_scale_e8m0_deshuffle.shape)
            weight_de_shuffle = weight_de_shuffle.view(NUM_EXPERTS, OC, IC//32, 32) * w_scale_e8m0_deshuffle
            weight_de_shuffle = weight_de_shuffle.view(NUM_EXPERTS, OC, IC)
        else:
            assert 0, f"unknown {quant_type=}"

    if input_scale is not None:
        # dequantize
        # print(input_scale.dtype, input_scale.shape, NUM_TOKENS, NUM_STATES)
        if quant_type == aiter.QuantType.per_128x128:
            if is_gateup:
                input = input.view(NUM_TOKENS, NUM_STATES//128, 128).to(input_scale.dtype) * input_scale.view(-1, NUM_TOKENS).t().contiguous().view(NUM_TOKENS, NUM_STATES//128, 1)
                input = input.view(NUM_TOKENS, NUM_STATES)
            else:
                input = input.view(NUM_TOKENS*TOPK, NUM_STATES//128, 128).to(input_scale.dtype) * input_scale.view(NUM_STATES//128, NUM_TOKENS*TOPK).t().contiguous().view(NUM_TOKENS*TOPK, NUM_STATES//128, 1)
                input = input.view(NUM_TOKENS, TOPK, NUM_STATES)
        elif quant_type == aiter.QuantType.per_1x32:
            #torch.cuda.synchronize();print("=========a", input.dtype, input.shape, sorted_ids.shape, input_scale.shape, input_scale.dtype)
            #assert 0
            input_deq = fp4_utils.mxfp4_to_f32(input).view(-1, NUM_STATES//32, 32).to(output.dtype) # * input_scale.view(-1, NUM_STATES//32, 1).to(output.dtype)
            if is_gateup:
                input = input_deq.view(NUM_TOKENS, NUM_STATES)
            else:
                input = input_deq.view(NUM_TOKENS, TOPK, NUM_STATES)
        else:
            assert 0, f"unknown {quant_type=}"

    for e_idx in range(NUM_BLOCKS):
        max_id = num_valid_ids[0]
        if e_idx * block_m >= max_id:
            break

        s_e_id = sorted_expert_ids[e_idx]
        i0 = e_idx*block_m
        i1 = i0 + block_m

        ids = sorted_ids[i0:i1].clone()
        tok_ids = ids & 0xFFFFFF
        tok_topk = ids >> 24

        valid_mask = tok_topk < torch.tensor(topk)

        expert_w = weight_de_shuffle[s_e_id, ...]
        if is_gateup:
            src = input[tok_ids[valid_mask],...]
        else:
            src = input[tok_ids[valid_mask], tok_topk[valid_mask], ...]

        if quant_type == aiter.QuantType.per_1x32:
            # moe_mxfp4_sort sort input scales in sorted_ids order
            # [32x256] =>  [2x2,16x128] ==== scale ====> [2x2, 4x16]e8m0 === pack2x2intou32 ===> [4x16]u32
            iscales = input_scale[i0:i1,...]
            sm, sn = iscales.shape
            input_mxfp4_scales = fp4_utils.e8m0_to_f32(input_scale[i0:i1,...]).to(torch.float)
            input_mxfp4_scales = input_mxfp4_scales.view(sm//32, sn//8, 4,16, 2,2).permute(0, 5, 3, 1, 4, 2).reshape(sm, sn, 1)
            # print(src.shape, input_mxfp4_scales.shape)
            src = (src.to(torch.float).view(-1, sn, 32) * input_mxfp4_scales[valid_mask, ...])
            src = src.view(-1, sn*32)

        act = src.to(torch.float) @ expert_w.t().to(torch.float)

        if is_gateup:
            if activation == aiter.ActivationType.Silu:
                act_gate = act[:, :(OC//2)]
                act_up = act[:,(OC//2):]
                act = torch.nn.functional.silu(act_gate) * act_up
            elif activation == aiter.ActivationType.Gelu:
                act = torch.nn.functional.gelu(act)
            else:
                assert 0, f"Unknown activation: {activation}"

            output[tok_ids[valid_mask], tok_topk[valid_mask], :] = act.to(output.dtype)
        else:
            tok_w = sorted_weights[i0:i1]
            cur_out = act[...].to(output.dtype) * tok_w[valid_mask, None]
            output[tok_ids[valid_mask], ...] += cur_out
            #output[tok_ids[valid_mask], tok_topk[valid_mask], :] = cur_out.to(output.dtype)
