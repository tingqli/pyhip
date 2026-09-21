# SPDX-License-Identifier: MIT
"""CPU-only README 6.1/6.2 diagrams. Requires Pillow and a CJK --font.

All animation frames are teaching steps, not GPU timestamps. No torch,
kernel imports, queue execution, or hardware-placement assumptions.
"""

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
TASKS, N, SPLITS, PACKETS = 9256, 6144, 8, 12
RUNTIME = ("moe_multistage_down.py", "moe_multistage_down_m128.py", "moe_multistage_reduce.py",
           "moe_multistage_pipeline.py", "test_blockscaled.py", "moe_8wave_down.py", "moe_8wave_down_utils.py")
BG, INK, MUTED, BORDER = "#f1f5fb", "#18283f", "#52637b", "#d7e1ee"
BLUE, TEAL, PURPLE, ORANGE = "#285dde", "#007c85", "#6c49bf", "#b85b17"
PAIR_COLORS = ("#dce9fd", "#cef0eb", "#efe2fc", "#ffebd0", "#faddE6", "#e5ecc6")
PAIR_INKS = (BLUE, TEAL, PURPLE, ORANGE, "#ae3c63", "#627b20")
MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


def task_of(worker, rank, tasks=TASKS):
    shard = worker % 8
    virtual = rank * 8 + shard
    if virtual >= tasks:
        return None
    chunk = tasks // 4
    task = virtual % 4 * chunk + virtual // 4 if virtual < chunk * 4 else virtual
    m, oc = divmod(task, 8)
    return {"worker": worker, "h": shard, "r": rank, "v": virtual, "t": task, "m": m, "oc": oc,
            "phi": worker // 8}


def packet_of(worker, q, packets=PACKETS):
    return 2 * ((q // 2 + worker // 8) % (packets // 2)) + q % 2


def packet_address(worker, rank, q):
    task = task_of(worker, rank)
    p = packet_of(worker, q)
    n64 = task["oc"] * PACKETS + p
    return {"q": q, "p": p, "slot": q % 4, "global_n64": n64,
            "column_begin": n64 * 64, "column_end": (n64 + 1) * 64,
            "global_n128_scale": n64 // 2}


def self_test():
    claims = [task_of(worker, rank) for worker, rank in ((0, 0), (8, 1), (0, 2), (16, 3))]
    assert [(r["v"], r["t"], r["m"], r["oc"], r["phi"]) for r in claims] == [
        (0, 0, 0, 0, 0), (8, 2, 0, 2, 1), (16, 4, 0, 4, 0), (24, 6, 0, 6, 2)]
    for h in range(8):
        assert len([worker for worker in range(512) if worker % 8 == h]) == 64
    mapped = [task_of(h, rank)["t"] for h in range(8) for rank in range(TASKS // 8)]
    assert sorted(mapped) == list(range(TASKS))
    assert all(task_of(h, TASKS // 8) is None for h in range(8))
    checked_orders = 0
    for packets in (2, 4, 6, 10, 12, 16):
        for worker in range(512):
            order = [packet_of(worker, q, packets) for q in range(packets)]
            assert sorted(order) == list(range(packets))
            assert all(order[q] % 2 == 0 and order[q + 1] == order[q] + 1 for q in range(0, packets, 2))
            checked_orders += 1
    addresses = [packet_address(8, 1, q) for q in range(PACKETS)]
    assert [r["p"] for r in addresses] == list(range(2, 12)) + [0, 1]
    assert addresses[0] == {"q": 0, "p": 2, "slot": 0, "global_n64": 26,
                            "column_begin": 1664, "column_end": 1728, "global_n128_scale": 13}
    assert addresses[10]["p"] == 0 and addresses[10]["slot"] == 2
    return {"tasks": TASKS, "n": N, "example_batch": 16384, "claims": claims,
            "worker8_packets": addresses, "verified_packet_orders": checked_orders,
            "semantics": "Logical examples only; neither placement nor concurrent execution is measured."}


class Canvas:
    def __init__(self, font_path, height):
        self.font_path = str(font_path)
        self.image = Image.new("RGB", (1600, height), BG)
        self.draw = ImageDraw.Draw(self.image)

    @lru_cache(maxsize=None)
    def font(self, size, mono=False):
        return ImageFont.truetype(str(MONO) if mono else self.font_path, size)

    def text(self, x, y, value, size=24, color=INK, mono=False):
        for i, line in enumerate(str(value).splitlines()):
            self.draw.text((x, y + i * (size + 11)), line, font=self.font(size, mono), fill=color, anchor="lt")

    def center(self, box, value, size=26, color=INK, mono=False):
        x0, y0, x1, y1 = box
        self.draw.text(((x0 + x1) / 2, (y0 + y1) / 2), str(value),
                       font=self.font(size, mono), fill=color, anchor="mm")

    def panel(self, box, fill="white", outline=BORDER, radius=18, width=2):
        self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def arrow(self, start, end, color=BLUE, width=4):
        self.draw.line((start, end), fill=color, width=width)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        points = [end, (end[0] - 13 * math.cos(angle - 0.45), end[1] - 13 * math.sin(angle - 0.45)),
                  (end[0] - 13 * math.cos(angle + 0.45), end[1] - 13 * math.sin(angle + 0.45))]
        self.draw.polygon(points, fill=color)

    def header(self, title, subtitle):
        self.text(50, 33, title, 43)
        self.text(52, 99, subtitle, 23, MUTED)

    def card(self, box, title, lines, accent=BLUE, active=False):
        self.panel(box, outline=accent if active else BORDER, width=4 if active else 2)
        x, y, _, _ = box
        self.text(x + 22, y + 20, title, 27, accent)
        self.text(x + 22, y + 71, lines, 23)

    def footer(self, text):
        self.text(53, self.image.height - 59, text, 21, MUTED)


def packet_grid(c, y, worker=8, active=None):
    """Address order, not a snapshot of all outstanding ring/refill packets."""
    left, stride, width = 239, 108, 96
    labels = ("q：处理顺序", "p：OC内包号", "S：读取槽号")
    order = [packet_of(worker, q) for q in range(PACKETS)]
    for row, label in enumerate(labels):
        c.text(75, y + row * 77 + 14, label, 23, MUTED)
    for q, p in enumerate(order):
        x = left + q * stride
        accent = PAIR_INKS[p // 2]
        selected = q == active
        for row, value in enumerate((q, p, f"S{q % 4}")):
            box = (x, y + row * 77, x + width, y + row * 77 + 54)
            fill = PAIR_COLORS[p // 2] if row == 1 else "#f5f8fc"
            c.panel(box, fill, accent if selected else BORDER, radius=10, width=4 if selected else 1)
            c.center(box, value, 26, accent if row == 1 or selected else INK, mono=True)
        c.arrow((x + width / 2, y + 58), (x + width / 2, y + 71), accent, 2)


def queue_diagram(font):
    c = Canvas(font, 1210)
    c.header("6.1  一次领号，如何变成 GEMM 任务？", "默认 B16k：T=9256  ·  512 CTA workers  ·  8 shards  ·  每worker 4 waves / 256 threads")
    c.panel((50, 150, 1550, 345))
    c.text(75, 169, "① 选分片：worker8 → h = 8 % 8 = 0 → 共享 head0", 29, TEAL)
    c.text(75, 217, "同组64个CTA：worker0、8、16、…、504。私有buffer有256个int32，但只有8个head。", 24)
    for h in range(8):
        x = 80 + h * 182
        c.panel((x, 264, x + 165, 324), "#cef0eb" if h == 0 else "#f5f8fc", TEAL if h == 0 else BORDER, 10)
        c.text(x + 13, 273, f"head{h}  [{h * 32}]", 22, TEAL if h == 0 else MUTED, mono=True)
        c.text(x + 13, 302, f"字节偏移 {h * 128}", 16, MUTED)
    cards = [
        ("② leader领号", "仅 tid0 执行 atomicAdd\n假设 head0 原值为1\n返回 r=1；head0 变2", TEAL),
        ("③ CTA内广播", "leader → LDS task_slot\nCTA barrier 后读取\n4个wave获得同一个 r", BLUE),
        ("④ width4置换", "v = 8×1+0 = 8\nc = 9256/4 = 2314\nt = (v%4)×c+v//4 = 2", PURPLE),
        ("⑤ 解码实际task", "m = 2//8 = 0\noc = 2%8 = 2\nM0 × 列[1536,2304)", ORANGE),
    ]
    for i, (title, body, accent) in enumerate(cards):
        x = 50 + i * 385
        c.card((x, 385, x + 345, 596), title, body, accent)
        if i < 3:
            c.arrow((x + 352, 490), (x + 379, 490), accent)
    c.panel((50, 633, 1550, 786))
    c.text(75, 651, "⑥ 四个wave合作完成一个128×768任务，全部12包结束后才再次领号", 27, TEAL)
    for wave in range(4):
        x = 75 + wave * 183
        c.panel((x, 703, x + 168, 762), "#e7edfe", radius=10)
        c.text(x + 12, 711, f"wave{wave}", 21, BLUE, mono=True)
        c.text(x + 12, 740, f"行 {wave * 32}～{wave * 32 + 31}", 17, MUTED)
    c.arrow((813, 731), (855, 731), TEAL)
    c.text(877, 702, "task完成 → 等待必要DMA/LDS → 回到②", 25, TEAL)
    c.text(877, 741, "每包N64×K256；一个packet不是一次claim。", 22, MUTED)
    c.panel((50, 820, 1550, 1128))
    c.text(75, 838, "同一head的一种可能领取顺序：worker可再次出现，rank不重复", 27)
    columns = (85, 402, 553, 698, 868, 1113, 1310)
    for x, value in zip(columns, ("领取者", "r", "v", "t", "(m,oc)", "worker的φ", "起始p(0)")):
        c.text(x, 890, value, 22, MUTED)
    examples = ((0, 0), (8, 1), (0, 2), (16, 3))
    for i, (worker, rank) in enumerate(examples):
        y = 931 + i * 43
        if i == 2:
            c.panel((69, y - 5, 1530, y + 33), "#fff0d9", outline="#fff0d9", radius=8)
        r = task_of(worker, rank)
        values = (f"worker{worker}" + (" 再次领取" if i == 2 else ""), rank, r["v"], r["t"],
                  f"({r['m']},{r['oc']})", r["phi"], packet_of(worker, 0))
        for x, value in zip(columns, values):
            c.text(x, y, value, 22)
    c.footer("逻辑示例，不是GPU时间线；只画部分worker。shard≠OC，width4≠4 waves，也不代表物理XCD绑定。")
    return c.image


def phase_diagram(font):
    c = Canvas(font, 1240)
    c.header("6.2  领到task以后，从哪一对 N64 开始？", "接上图：worker8 取得 r=1 → task(M0, OC2)；这里仅改变列访问顺序，不再领号。")
    context = [
        ("① 固定worker相位", "h = 8%8 = 0\nφ = 8//8 = 1\n与本次rank无关", TEAL),
        ("任务给出OC列区间", "N=6144，OC2起点1536\n每OC 768列 → Q=12包\n相邻两包一对 → G=6对", BLUE),
        ("按N128 pair旋转", "pair顺序：1,2,3,4,5,0\n每对的偶数/奇数半不交换\n每个新task的 q 从0开始", PURPLE),
    ]
    for i, (title, lines, accent) in enumerate(context):
        x = 50 + i * 508
        c.card((x, 150, x + 484, 342), title, lines, accent)
    c.panel((50, 375, 1550, 704))
    c.text(75, 394, "② 旋转的是实际包号p，不是LDS槽S", 28, PURPLE)
    c.text(665, 399, "p(q) = 2*((q//2 + 1)%6) + q%2", 26, PURPLE, mono=True)
    packet_grid(c, 450, active=0)
    c.text(75, 673, "同色两个p属于同一N128对；最后q10/11绕回p0/1。S=q%4，不能用p%4替代。", 21, MUTED)
    a = packet_address(8, 1, 0)
    c.text(55, 735, f"③ 放大 q=0：p={a['p']}，对应全局N64块{a['global_n64']}，逻辑列[{a['column_begin']},{a['column_end']})", 29)
    consumers = [
        ("B：按p找权重", "OC起点1536 + p2×64\n读取 E 的列[1664,1728)\n消费时从S0读取：q%4=0", TEAL),
        ("scale：按p找N128组", "全局组 = OC2×6+p2//2 = 13\nB_scale[E, 13, 0:2]\n两个scale分别用于两个K128", PURPLE),
        ("C：按p找正确落点", "global_N64 = OC2×12+p2 = 26\nC[M0, 26, row, col64]\n逻辑列仍是[1664,1728)", ORANGE),
    ]
    for i, (title, lines, accent) in enumerate(consumers):
        x = 50 + i * 508
        c.card((x, 787, x + 484, 979), title, lines, accent)
    c.panel((50, 1014, 1550, 1156), "#e9f0ff", "#cbdafb")
    c.text(76, 1032, "④ 再次领取也不改φ：worker0 的 r=0→2，task=(M0,OC0)→(M0,OC4)，但φ始终0。", 25, BLUE)
    c.text(76, 1076, "两次task都从p0开始；worker8无论rank是多少，都从p2开始。相同p不代表相同B地址。", 24)
    c.text(76, 1118, "本图只连地址含义：B预取更早、C写回更晚；并非三者在同一时刻执行。E = expert_ids[m]。", 21, MUTED)
    c.footer("默认B16k / T9256；B32k须使用其自己的T重新做task映射。q、p、φ、S是四个不同量。")
    return c.image


def animation_frame(font, kind, q, index, count):
    c = Canvas(font, 1060)
    c.header("分步播放：先领task，再按worker相位访问N", "逻辑教学步骤 · 非GPU时间线 · 每帧不是一次真实DMA/store · 未展开预取与延迟写回")
    repeat = kind == "repeat"
    worker, rank = (0, 2) if repeat else (8, 1)
    r = task_of(worker, rank)
    stages = {"shard": 0, "claim": 1, "broadcast": 1, "map": 2, "decode": 3, "packet": 3, "repeat": 0}
    cards = [
        (f"worker{worker}  ·  4 waves", f"h={r['h']} → 共享 head{r['h']}\nφ={worker}//8={r['phi']}（固定）\n512 workers / 8 shards", TEAL),
        ("leader atomic → CTA", f"head{r['h']}: {rank} → {rank + 1}\n返回旧值 r={rank}\nLDS + barrier → 4 waves", BLUE),
        ("virtual → width4 → t", f"v=8×{rank}+{r['h']}={r['v']}\nt=F4({r['v']})={r['t']}\nT=9256，c=2314", PURPLE),
        (f"task (M{r['m']}, OC{r['oc']})", f"列[{r['oc'] * 768},{(r['oc'] + 1) * 768})\n128×768，一共12包\n完成整个task后才再claim", ORANGE),
    ]
    for i, (title, body, accent) in enumerate(cards):
        x = 50 + i * 385
        c.card((x, 153, x + 345, 359), title, body, accent, stages[kind] == i)
        if i < 3:
            c.arrow((x + 352, 255), (x + 379, 255), accent)
    c.panel((50, 389, 1550, 732))
    title = f"task内顺序：worker{worker} 的 φ={r['phi']}，q每次从0开始"
    c.text(75, 408, title, 27, TEAL)
    if kind == "packet":
        c.text(1155, 410, f"处理 q={q:02d} / 11", 28, TEAL)
    elif repeat:
        c.text(1224, 410, "新task：q=0", 27, TEAL)
    packet_grid(c, 464, worker, q if kind == "packet" else 0 if repeat else None)
    c.text(75, 692, "p是当前OC内N64包号；S是消费该包的LDS槽。整张表是映射，不是同一时刻的ring内容。", 21, MUTED)
    c.panel((50, 762, 1550, 945), "#e9f0ff", "#cbdafb")
    descriptions = {
        "shard": ("① 选共享队列", "worker0、8、16、…、504 属于shard0：64个CTA共享一个head0。\n其余7个head也各由64个worker共享，彼此独立。"),
        "claim": ("② 只由tid0领号", "假设这一次 head0=1；atomicAdd返回r=1，再把head0变成2。\n不是4个wave各领一个task，也不是256个线程各领一次。"),
        "broadcast": ("③ CTA内广播", "leader把r=1存到该CTA的LDS task_slot；barrier后4个wave读取同一个r。\nwave0/1/2/3分别处理task内的32行，合计M128。"),
        "map": ("④ rank不等于task编号", "r=1、h=0 → v=8；F4(8)=(8%4)×2314+8//4=2。\n再解码：m=2//8=0，oc=2%8=2。所以shard0也能处理OC2。"),
        "decode": ("⑤ 一次claim覆盖全部12包", "worker8已拿到M0、OC2的128×768任务；φ=worker//8=1决定N起点。\n接下来q逐包推进，不再atomic领号；每包已经含完整K256。"),
        "repeat": ("⑦ worker0完成后再次领号：rank变，φ不变", "先前worker0：r=0 → task(M0,OC0)，φ=0；再次：r=2 → task(M0,OC4)，φ仍为0。\n新task的q重置0，仍从p0开始；不是用新rank=2把N起点改成p4。"),
    }
    if kind == "packet":
        a = packet_address(worker, rank, q)
        title = f"⑥ q={q} → p={a['p']} → 读取槽S{a['slot']}" + ("：绕回首个N128对" if q == 10 else "")
        body = (f"B对应逻辑列[{a['column_begin']},{a['column_end']})；scale对应全局N128组{a['global_n128_scale']}的两个K128分块。\n"
                f"C稍后写回 packed global_N64={a['global_n64']}，不是按q={q}写错位置。")
    else:
        title, body = descriptions[kind]
    c.text(75, 780, title, 28, BLUE)
    c.text(75, 830, body, 24)
    c.draw.rounded_rectangle((52, 968, 1548, 980), radius=6, fill=BORDER)
    c.draw.rounded_rectangle((52, 968, 52 + 1496 * (index + 1) / count, 980), radius=6, fill=TEAL)
    c.footer(f"{index + 1:02d}/{count:02d}  ·  自动循环，静态步骤图可逐项对照  ·  领取顺序只是例子，不承诺CTA并发/物理放置。")
    return c.image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_assets():
    manifest = json.loads((HERE / "schedule_examples.json").read_text())
    assert manifest["model"] == self_test()
    for name, info in manifest["assets"].items():
        path = HERE / name
        assert sha(path) == info["sha256"]
        with Image.open(path) as image:
            assert list(image.size) == info["size"]
            assert image.n_frames == info["frames"]
            if image.is_animated:
                assert image.info["loop"] == 0
                durations = []
                for frame in range(image.n_frames):
                    image.seek(frame)
                    durations.append(image.info["duration"])
                assert durations == info["durations_ms"]
    assert manifest["source_sha256"] == {name: sha(MAIN / name) for name in RUNTIME}
    print("CPU_DIAGRAM_CHECK_PASS", json.dumps({"tasks": TASKS, "orders": manifest["model"]["verified_packet_orders"],
                                               "assets": list(manifest["assets"]), "kernel_sources_unchanged": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, help="CJK font path; font is not copied to the repository")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        validate_assets()
        return
    if args.font is None or not args.font.is_file():
        parser.error("--font must name a readable CJK font (e.g. Noto Sans CJK SC)")
    sources = {name: sha(MAIN / name) for name in RUNTIME}
    model = self_test()
    queue_diagram(args.font).save(HERE / "queue_steps.png", optimize=True)
    phase_diagram(args.font).save(HERE / "n_phase_steps.png", optimize=True)
    scenes = [(name, None, duration) for name, duration in
              (("shard", 2200), ("claim", 2200), ("broadcast", 2200), ("map", 2600), ("decode", 2400))]
    scenes += [("packet", q, 1600 if q in (0, 10, 11) else 800) for q in range(PACKETS)]
    scenes += [("repeat", None, 3800)]
    frames = [animation_frame(args.font, kind, q, index, len(scenes))
              for index, (kind, q, _) in enumerate(scenes)]
    palette = frames[0].quantize(colors=192)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    durations = [duration for _, _, duration in scenes]
    frames[0].save(HERE / "queue_n_phase.gif", save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=2)
    assets = {}
    for name in ("queue_steps.png", "n_phase_steps.png", "queue_n_phase.gif"):
        path = HERE / name
        with Image.open(path) as image:
            assets[name] = {"sha256": sha(path), "size": list(image.size), "frames": image.n_frames,
                            "bytes": path.stat().st_size}
            if image.is_animated:
                assets[name]["durations_ms"] = durations
    assert sources == {name: sha(MAIN / name) for name in RUNTIME}
    (HERE / "schedule_examples.json").write_text(json.dumps(
        {"model": model, "assets": assets, "source_sha256": sources,
         "font": ImageFont.truetype(str(args.font), 24).getname(), "generator_sha256": sha(Path(__file__))},
        ensure_ascii=False, indent=2) + "\n")
    validate_assets()


if __name__ == "__main__":
    main()