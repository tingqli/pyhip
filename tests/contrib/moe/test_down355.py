import pyhip
from pyhip import calc_diff
from pyhip.contrib.moe_gemm_down_tp import moe_gemm_down_tp
import torch

def test_down(target_file):
    args_dict = torch.load(target_file)
    num_e_blocks = args_dict["num_e_blocks"]
    is_output_over_4GB = args_dict["is_output_over_4GB"]
    AB_dtype = args_dict["AB_dtype"]
    wg_M = args_dict["wg_M"]
    E = args_dict["E"]
    model_dim = args_dict["model_dim"]
    inter_dim = args_dict["inter_dim"]
    w2_is_shuffled = args_dict["w2_is_shuffled"]
    topk = args_dict["topk"]
    sorted_ids = args_dict["sorted_ids"]
    sorted_weights = args_dict["sorted_weights"]
    sorted_expert_ids = args_dict["sorted_expert_ids"]
    num_valid_ids = args_dict["num_valid_ids"]
    w2 = args_dict["w2"]
    w2_scale = args_dict["w2_scale"]
    a2 = args_dict["a2"]
    a2_scale = args_dict["a2_scale"]
    stage2_out = args_dict["stage2_out"]
    token_num = args_dict["token_num"]

    ref = stage2_out
    ret = torch.empty_like(ref)
    moe_gemm_down_tp([1, num_e_blocks], [4*64],
                    is_output_over_4GB,
                    AB_dtype, wg_M, 64,
                    E, model_dim, inter_dim, 
                    False, w2_is_shuffled, topk,
                    sorted_ids.data_ptr(),
                    sorted_weights.data_ptr(),
                    sorted_expert_ids.data_ptr(),
                    num_valid_ids.data_ptr(),
                    w2.data_ptr(), None if w2_scale is None else w2_scale.data_ptr(),
                    a2.data_ptr(), None if a2_scale is None else a2_scale.data_ptr(),
                    ret.data_ptr(),
                    token_num)
    
    print(calc_diff(ref, ret))

test_down("moe_gemm_down_16384_256_6144_256_True.pt")