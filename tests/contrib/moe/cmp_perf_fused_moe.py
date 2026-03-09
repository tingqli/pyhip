import subprocess
import pandas as pd
from tqdm import tqdm
import argparse
from aiter import dtypes

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)

parser.add_argument(
    "-dim",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=(4096,1024,512,10,0),
    help="""Model dimension.
    e.g.: -dim 6144,4096""",
)
args = parser.parse_args()

model_dim, inter_dim, num_experts, num_experts_per_tok, quant_mode = args.dim

def get_fused_moe_time(model_dim, inter_dim, num_tokens, experts, topk, quant_mode, use_jit):
    cmd = f"python test_fused_moe.py -dim {model_dim},{inter_dim} -t {num_tokens} -a silu -s f -e {experts} -k {topk} -p t -q {quant_mode}"
    if use_jit:
        cmd += " -j"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("last-time-us:"):
            return float(line.split(":")[1]), cmd
    #print(result.stdout)
    #print(cmd)
    return -1, cmd

test_cases = [
    # EP8 num-tokens reducing
    (model_dim, inter_dim, 1024, num_experts//8, num_experts_per_tok, 0),
    (model_dim, inter_dim, 512, num_experts//8, num_experts_per_tok, 0),
    (model_dim, inter_dim, 256, num_experts//8, num_experts_per_tok, 0),
    (model_dim, inter_dim, 128, num_experts//8, num_experts_per_tok, 0),
    (model_dim, inter_dim, 64, num_experts//8, num_experts_per_tok, 0),

    # TP8
    (model_dim, inter_dim//8, 4, num_experts, num_experts_per_tok, 0),
    (model_dim, inter_dim//8, 8, num_experts, num_experts_per_tok, 0),
    (model_dim, inter_dim//8, 16, num_experts, num_experts_per_tok, 0),
    (model_dim, inter_dim//8, 32, num_experts, num_experts_per_tok, 0),
    (model_dim, inter_dim//8, 64, num_experts, num_experts_per_tok, 0),
    (model_dim, inter_dim//8, 128, num_experts, num_experts_per_tok, 0),
    (model_dim, inter_dim//8, 256, num_experts, num_experts_per_tok, 0),
    (model_dim, inter_dim//8, 512, num_experts, num_experts_per_tok, 0),
    (model_dim, inter_dim//8, 5120, num_experts, num_experts_per_tok, 0),
    (model_dim, inter_dim//8, 8000, num_experts, num_experts_per_tok, 0),
]

df = []    
for model_dim,inter_dim,num_tokens,experts,topk,_ in tqdm(test_cases):
    us_aiter, cmd_aiter = get_fused_moe_time(model_dim, inter_dim, num_tokens, experts, topk, quant_mode, False)
    us_asmjit, cmd_jit = get_fused_moe_time(model_dim, inter_dim, num_tokens, experts, topk, quant_mode, True)
    ret = {}
    # \u2713 tick   \u2717 cross
    if us_aiter < us_asmjit:
        ret["us(aiter)"] = f"\033[92m \u2713 {'N/A' if us_aiter <= 0 else us_aiter}\033[0m"
        ret["us(amsjit)"] = f"{us_asmjit}"
    else:
        ret["us(aiter)"] = f"{us_aiter}"
        ret["us(amsjit)"] = f"\033[92m \u2713 {'N/A' if us_asmjit <= 0 else us_asmjit}\033[0m"

    ret["model_dim"] = model_dim
    ret["inter_dim"] = inter_dim
    ret["num_tokens"] = num_tokens
    ret["experts"] = experts
    ret["topk"] = topk
    ret["cmd"] = cmd_jit

    df.append(ret)

    df_md = pd.DataFrame(df).to_markdown(index=False)
    print(f"========== {quant_mode=} ============")
    print(df_md)
