import torch
import json

# for each wg:
#  0  start in us
#  1  start in cycle
#  2  loop start in us
#  3  loop start in cycle
#  4  loop end in us
#  5  loop end in cycle
#  6  end in us
#  7  end in cycle
#  8  xcc_id|se_id|cu_id|slot_id etc
def gen_timing(p_debug_buf: torch.Tensor, access_size_wg, flops_wg):
    p_debug_buf = p_debug_buf.reshape(-1, 9)
    cu_buf = p_debug_buf[:, 8]
    cu_ids_unique = torch.unique(cu_buf)
    cu_ids_logic_lookup = {}
    for i in range(cu_ids_unique.shape[0]):
        cu_ids_logic_lookup[cu_ids_unique[i].item()] = i
    p_debug_buf = p_debug_buf[:, 0:8].reshape(-1, 2)
    wg_size = p_debug_buf.shape[0] // 4
    min_value = torch.min(p_debug_buf, dim=0, keepdim=True)[0]
    normed_buf: torch.Tensor = p_debug_buf - min_value
    # S_MEMREALTIME: the time value is from a constant 100MHz clock
    buf_us = normed_buf[:, 0].to(torch.float64) / 100 # unit is us
    buf_cycle = normed_buf[:, 1]
    infos = []
    buf_us = buf_us.reshape(-1, 4)
    buf_cycle = buf_cycle.reshape(-1, 4)
    freqs = (buf_cycle[:, 3] - buf_cycle[:, 0]) / (buf_us[:, 3] - buf_us[:, 0]) / 1000
    durs = buf_us[:, 3] - buf_us[:, 0]
    cyc_durs = buf_cycle[:, 3] - buf_cycle[:, 0]
    cyc_pros = buf_cycle[:, 1] - buf_cycle[:, 0]
    cyc_loops = buf_cycle[:, 2] - buf_cycle[:, 1]
    cyc_epis = buf_cycle[:, 3] - buf_cycle[:, 2]
    time_pros = buf_us[:, 1] - buf_us[:, 0]
    time_loops = buf_us[:, 2] - buf_us[:, 1]
    time_epis = buf_us[:, 3] - buf_us[:, 2]

    bws = access_size_wg / (durs * 1000)  # unit is GB/s
    gflops = flops_wg / (durs * 1000)     # unit is GFLOPS
    for i in range(buf_us.shape[0]):
        hw_ids = cu_buf[i].item()
        logic_cu_id = cu_ids_logic_lookup[hw_ids]
        slot_id = hw_ids & 0xff
        cu_id = (hw_ids & 0xff00) >> 8
        se_id = (hw_ids & 0xff0000) >> 16
        xcc_id = (hw_ids & 0xff000000) >> 24
        info = f'''{{
                "ph": "X",
                "name": "xcc{xcc_id}_se{se_id}_cu{cu_id}_slot{slot_id}",
                "pid": {0},
                "tid": {logic_cu_id},
                "ts": {buf_us[i, 0]},
                "dur": {durs[i]:.2f},
                "args": {{ "cyc.pro":{cyc_pros[i]},
                           "cyc.loop":{cyc_loops[i]}, 
                           "cyc.epi":{cyc_epis[i]},
                           "time.pro":{time_pros[i]:.2f},
                           "time.loop":{time_loops[i]:.2f},
                           "time.epi":{time_epis[i]:.2f},
                           "freq(G)":{freqs[i]:.2f},
                           "bw(GB/s)": {bws[i]:.2f},
                           "flops(GF/s)":{gflops[i]:.2f}
                            }}
                }}'''
        infos.append(info)

        info = f'''{{ "ph": "B", "name": "pro", "pid": {0}, "tid": {logic_cu_id}, "ts": {buf_us[i, 0]}, "args": {{"cyc":{cyc_pros[i]}, "pro/all":"{time_pros[i] / durs[i] * 100:.2f}%"}} }}'''
        infos.append(info)
        info = f'''{{ "ph": "E", "name": "pro", "pid": {0}, "tid": {logic_cu_id}, "ts": {buf_us[i, 1]}, "args": {{}} }}'''
        infos.append(info)
        info = f'''{{ "ph": "B", "name": "loop", "pid": {0}, "tid": {logic_cu_id}, "ts": {buf_us[i, 1]}, "args": {{"cyc":{cyc_loops[i]}, "loop/all":"{time_loops[i] / durs[i] * 100:.2f}%"}} }}'''
        infos.append(info)
        info = f'''{{ "ph": "E", "name": "loop", "pid": {0}, "tid": {logic_cu_id}, "ts": {buf_us[i, 2]}, "args": {{}} }}'''
        infos.append(info)
        info = f'''{{ "ph": "B", "name": "epi", "pid": {0}, "tid": {logic_cu_id}, "ts": {buf_us[i, 2]}, "args": {{"cyc":{cyc_epis[i]}, "epi/all":"{time_epis[i] / durs[i] * 100:.2f}%"}} }}'''
        infos.append(info)
        info = f'''{{ "ph": "E", "name": "epi", "pid": {0}, "tid": {logic_cu_id}, "ts": {buf_us[i, 3]}, "args": {{}} }}'''
        infos.append(info)
        # info = f'''{{ "ph": "C", "name": "freq", "pid": 0, "ts": {buf_us[i, 0]}, "args": {{"{logic_cu_id}": {freqs[i]:.2f}}} }}'''
        # infos.append(info)

    durs_mean = torch.mean(durs).item()
    cyc_durs_mean = torch.mean(cyc_durs.to(torch.float32)).item()
    freqs_mean = torch.mean(freqs)
    durs_max = torch.max(durs, dim=0, keepdim=False)
    durs_min = torch.min(durs, dim=0, keepdim=False)
    durs_median = torch.median(durs, dim=0, keepdim=False)
    print(f'\nmemory access size per wg: {access_size_wg / 1024:.2f} KB, flops per wg: {flops_wg / 1e6:.2f} MFlops, per wg statis:')
    print(f'{"item":<13s} {"all(us)":>10s} {"prolog(us)":>15s} {"loop(us)":>18s} {"epi(us)":>15s} {"freq(GHz)":>10s} {"bw(GB/s)":>10s} {"GFlops/s":>10s} {"all.cyc":>15s} {"pro.cyc":>10s} {"loop.cyc":>15s} {"epi.cyc":>10s}')
    str_pros = f'{time_pros.mean():.2f}({time_pros.mean() / durs_mean * 100:.2f}%)'
    str_loops = f'{time_loops.mean():.2f}({time_loops.mean() / durs_mean * 100:.2f}%)'
    str_epis = f'{time_epis.mean():.2f}({time_epis.mean() / durs_mean * 100:.2f}%)'
    print(f'{"mean":<13s} {durs_mean:>10.2f} {str_pros:>15s} {str_loops:>18s} {str_epis:>15s} {freqs_mean:>10.2f} {access_size_wg / 1024 / durs_mean:>10.2f} {flops_wg / 1000 / durs_mean:>10.2f} {cyc_durs_mean:>15,.0f} {cyc_pros.to(torch.float32).mean():>10,.0f} {cyc_loops.to(torch.float32).mean():>15,.0f} {cyc_epis.to(torch.float32).mean():>10,.0f}')
    detail_idx = durs_median[1]
    detail_val = durs_median[0].item()
    hw_ids = cu_buf[detail_idx].item()
    logic_cu_id = cu_ids_logic_lookup[hw_ids]
    header = f'median({logic_cu_id})'
    str_pros = f'{time_pros[detail_idx]:.2f}({time_pros[detail_idx] / detail_val * 100:.2f}%)'
    str_loops = f'{time_loops[detail_idx]:.2f}({time_loops[detail_idx] / detail_val * 100:.2f}%)'
    str_epis = f'{time_epis[detail_idx]:.2f}({time_epis[detail_idx] / detail_val * 100:.2f}%)'
    print(f'{header:<13s} {detail_val:>10.2f} {str_pros:>15s} {str_loops:>18s} {str_epis:>15s} {freqs[detail_idx]:>10.2f} {access_size_wg / 1000 / detail_val:>10.2f} {flops_wg / 1000 / detail_val:>10.2f} {cyc_durs[detail_idx]:>15,.0f} {cyc_pros[detail_idx]:>10,.0f} {cyc_loops[detail_idx]:>15,.0f} {cyc_epis[detail_idx]:>10,.0f}')

    detail_idx = durs_max[1]
    detail_val = durs_max[0].item()
    hw_ids = cu_buf[detail_idx].item()
    logic_cu_id = cu_ids_logic_lookup[hw_ids]
    header = f'max({logic_cu_id})'
    str_pros = f'{time_pros[detail_idx]:.2f}({time_pros[detail_idx] / detail_val * 100:.2f}%)'
    str_loops = f'{time_loops[detail_idx]:.2f}({time_loops[detail_idx] / detail_val * 100:.2f}%)'
    str_epis = f'{time_epis[detail_idx]:.2f}({time_epis[detail_idx] / detail_val * 100:.2f}%)'
    print(f'{header:<13s} {detail_val:>10.2f} {str_pros:>15s} {str_loops:>18s} {str_epis:>15s} {freqs[detail_idx]:>10.2f} {access_size_wg / 1000 / detail_val:>10.2f} {flops_wg / 1000 / detail_val:>10.2f} {cyc_durs[detail_idx]:>15,.0f} {cyc_pros[detail_idx]:>10,.0f} {cyc_loops[detail_idx]:>15,.0f} {cyc_epis[detail_idx]:>10,.0f}')
    detail_idx = durs_min[1]
    detail_val = durs_min[0].item()
    hw_ids = cu_buf[detail_idx].item()
    logic_cu_id = cu_ids_logic_lookup[hw_ids]
    header = f'min({logic_cu_id})'
    str_pros = f'{time_pros[detail_idx]:.2f}({time_pros[detail_idx] / detail_val * 100:.2f}%)'
    str_loops = f'{time_loops[detail_idx]:.2f}({time_loops[detail_idx] / detail_val * 100:.2f}%)'
    str_epis = f'{time_epis[detail_idx]:.2f}({time_epis[detail_idx] / detail_val * 100:.2f}%)'
    print(f'{header:<13s} {detail_val:>10.2f} {str_pros:>15s} {str_loops:>18s} {str_epis:>15s} {freqs[detail_idx]:>10.2f} {access_size_wg / 1000 / detail_val:>10.2f} {flops_wg / 1000 / detail_val:>10.2f} {cyc_durs[detail_idx]:>15,.0f} {cyc_pros[detail_idx]:>10,.0f} {cyc_loops[detail_idx]:>15,.0f} {cyc_epis[detail_idx]:>10,.0f}')
    with open('statis.json', 'w') as f:
        s = '{"traceEvents":[' + ','.join(infos) + "]}"
        f.write(s)
    print(f'statis.json is dumped.\n')
