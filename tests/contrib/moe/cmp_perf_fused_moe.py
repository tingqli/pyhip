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

parser.add_argument(
    "-tp",
    type=int,
    nargs="?",
    const=None,
    default=0,
    help="""TP.
    e.g.: -tp 8""",
)

parser.add_argument(
    "-ep",
    type=int,
    nargs="?",
    const=None,
    default=0,
    help="""EP.
    e.g.: -ep 8""",
)

args = parser.parse_args()

model_dim, inter_dim, num_experts, num_experts_per_tok, quant_mode = args.dim

def get_fused_moe_time(model_dim, inter_dim, num_tokens, experts, topk, quant_mode, ep_size, use_jit):
    cmd = f"python test_fused_moe.py -dim {model_dim},{inter_dim} -t {num_tokens} -a silu -s f -e {experts} -k {topk} -p t -q {quant_mode} -ep {ep_size}"
    if use_jit:
        cmd += " -j"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("last-time-us:"):
            return float(line.split(":")[1]), cmd
    #print(result.stdout)
    #print(cmd)
    return -1, cmd


def do_test(test_cases, ep_size=1):
    df = []
    for model_dim,inter_dim,num_tokens,experts,topk in tqdm(test_cases):
        us_aiter, cmd_aiter = get_fused_moe_time(model_dim, inter_dim, num_tokens, experts, topk, quant_mode, ep_size, False)
        us_asmjit, cmd_jit = get_fused_moe_time(model_dim, inter_dim, num_tokens, experts, topk, quant_mode, ep_size, True)
        ret = {}
        quant_2_wdtype_size = {0:2, 5:1}
        wdtype_size = quant_2_wdtype_size[quant_mode] 
        weight_size = min(num_tokens * topk, experts) * (model_dim * inter_dim * 3 * wdtype_size)
        act_size = num_tokens * model_dim * wdtype_size + num_tokens * topk * model_dim * 2 * wdtype_size
        
        def us2bw(us):
            if us <= 0:
                return 0
            return (weight_size + act_size)/(us * 1e-6) * 1e-9 # GB/s

        # \u2713 tick   \u2717 cross
        if us_aiter < us_asmjit:
            ret["us(aiter)"] = f"\033[92m \u2713 {'N/A' if us_aiter <= 0 else us_aiter}\033[0m"
            ret["us(amsjit)"] = f"{us_asmjit}"
        else:
            ret["us(aiter)"] = f"{us_aiter}"
            ret["us(amsjit)"] = f"\033[92m \u2713 {'N/A' if us_asmjit <= 0 else us_asmjit}\033[0m"
        ret["GB/s(aiter)"] = f"{us2bw(us_aiter):.0f}"
        ret["GB/s(amsjit)"] = f"{us2bw(us_asmjit):.0f}"

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

if args.ep > 0:
    do_test([
        (model_dim, inter_dim, 1024, num_experts, num_experts_per_tok),
        (model_dim, inter_dim, 512, num_experts, num_experts_per_tok),
        (model_dim, inter_dim, 256, num_experts, num_experts_per_tok),
        (model_dim, inter_dim, 128, num_experts, num_experts_per_tok),
        (model_dim, inter_dim, 64, num_experts, num_experts_per_tok),
    ],
    ep_size=args.ep)

if args.tp > 0:
    do_test([
        (model_dim, inter_dim//args.tp, 4, num_experts, num_experts_per_tok),
        (model_dim, inter_dim//args.tp, 8, num_experts, num_experts_per_tok),
        (model_dim, inter_dim//args.tp, 16, num_experts, num_experts_per_tok),
        (model_dim, inter_dim//args.tp, 32, num_experts, num_experts_per_tok),
        (model_dim, inter_dim//args.tp, 64, num_experts, num_experts_per_tok),
        (model_dim, inter_dim//args.tp, 128, num_experts, num_experts_per_tok),
        (model_dim, inter_dim//args.tp, 256, num_experts, num_experts_per_tok),
        (model_dim, inter_dim//args.tp, 512, num_experts, num_experts_per_tok),
        (model_dim, inter_dim//args.tp, 5120, num_experts, num_experts_per_tok),
        (model_dim, inter_dim//args.tp, 8000, num_experts, num_experts_per_tok),
    ],
    ep_size=1)
