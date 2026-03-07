import subprocess
import pandas as pd
from tqdm import tqdm

def get_fused_moe_time(model_dim, inter_dim, num_tokens, experts, topk, quant_mode, use_jit):
    cmd = f"python test_fused_moe.py -dim {model_dim},{inter_dim} -t {num_tokens} -a silu -s f -e {experts} -k {topk} -p t -q {quant_mode}"
    if use_jit:
        cmd += " -j"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("last-time-us:"):
            return float(line.split(":")[1])
    return None

df = []
test_cases = [
    # EP num-tokens reducing
    (4096, 1536, 512, 8, 8, 0),
    (4096, 1536, 256, 8, 8, 0),
    (4096, 1536, 128, 8, 8, 0),
    (4096, 1536, 64, 8, 8, 0),
    # TP1
    (4096, 1536, 512, 128, 8, 0),
    (4096, 1536, 5120, 128, 8, 0),
    # TP8
    (4096, 128, 512, 8, 8, 0),
    (4096, 128, 512, 128, 8, 0),
    (4096, 128, 5120, 128, 8, 0),
]
for model_dim,inter_dim,num_tokens,experts,topk,quant_mode in tqdm(test_cases):
    ret = {"model_dim":model_dim, "inter_dim":inter_dim, "num_tokens":num_tokens, "experts":experts, "topk":topk}
    us_aiter = get_fused_moe_time(model_dim, inter_dim, num_tokens, experts, topk, quant_mode, False)
    us_asmjit = get_fused_moe_time(model_dim, inter_dim, num_tokens, experts, topk, quant_mode, True)
    # \u2713 tick   \u2717 cross
    if us_aiter < us_asmjit:
        ret["us(aiter)"] = f"\033[92m{us_aiter}\033[0m"
        ret["us(amsjit)"] = f"{us_asmjit}"
    else:
        ret["us(aiter)"] = f"{us_aiter}"
        ret["us(amsjit)"] = f"\033[92m{us_asmjit}\033[0m"
    df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
print(df_md)