"""Which edits invalidate FlyDSL's JIT disk cache? (FlyDSL 0.3.2; re-run after an upgrade)

    cd src/contrib/flydsl && HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261115_flydsl_cache_key/run_probe.py

Copies probe.py into a temp dir, then per case: one run fills a fresh cache, one input is
changed, a NEW process runs again with the same cache. The probe kernel writes
1000 * MODE (module global) + 100 * LST[0] (closure list or tuple) + 10 * SCALAR (closure
int) + helper_other() (plain helper, another directory) + helper_same() (plain helper, same
directory); "stale" = the second run printed the old value.
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.abspath(os.path.join(HERE, "../../.."))  # holds a8w4/


def main():
    d = tempfile.mkdtemp(prefix="flykey_")
    cache = os.path.join(d, "cache")
    env = dict(os.environ, FLYDSL_RUNTIME_CACHE_DIR=cache, AITER_LOG_LEVEL="ERROR",
               PYTHONPATH=PKG_PARENT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    probe = os.path.join(d, "probe.py")

    def write(path, text):
        with open(path, "w") as f:
            f.write(text)

    def reset():
        shutil.rmtree(cache, ignore_errors=True)
        os.makedirs(os.path.join(d, "other"), exist_ok=True)
        write(os.path.join(d, "other", "helper_other.py"), "def helper_other():\n    return 1\n")
        write(os.path.join(d, "helper_same.py"), "def helper_same():\n    return 1\n")
        shutil.copy(os.path.join(HERE, "probe.py"), probe)

    def run(*args):
        r = subprocess.run([sys.executable, probe, *map(str, args)], env=env,
                           capture_output=True, text=True)
        out = r.stdout.strip().splitlines()
        return out[-1] if out else "ERR " + r.stderr[-300:]

    def edit(path, old, new):
        s = open(path).read()
        assert old in s, old
        write(path, s.replace(old, new))

    cases = []
    reset()
    cases.append(("baseline", run(1, 1), "1112"))
    reset(); run(1, 1); edit(probe, "MODE = 1", "MODE = 2")
    cases.append(("module global MODE 1 -> 2", run(1, 1), "2112"))
    reset(); run(1, 1)
    cases.append(("closure list [1] -> [2]", run(2, 1), "1212"))
    reset(); run(1, 1, "tuple")
    cases.append(("closure tuple (1,) -> (2,)", run(2, 1, "tuple"), "1212"))
    reset(); run(1, 1)
    cases.append(("closure int 1 -> 2", run(1, 2), "1122"))
    reset(); run(1, 1)
    write(os.path.join(d, "other", "helper_other.py"), "def helper_other():\n    return 5\n")
    cases.append(("helper in another directory", run(1, 1), "1116"))
    reset(); run(1, 1)
    write(os.path.join(d, "helper_same.py"), "def helper_same():\n    return 5\n")
    cases.append(("helper in the same directory", run(1, 1), "1116"))
    reset(); run(1, 1); n0 = sum(len(f) for _, _, f in os.walk(cache))
    edit(probe, "MODE = 1", "# a module comment\nMODE = 1")
    run(1, 1)
    n1 = sum(len(f) for _, _, f in os.walk(cache))
    cases.append(("comment outside functions", "rebuilt" if n1 > n0 else "reused", "reused"))
    for name, got, fresh in cases:
        verdict = "in the key" if got == fresh else "STALE (not in the key)"
        if name.startswith("comment"):
            verdict = "no rebuild (the kernel did not change)" if got == "reused" else "rebuilt"
        print(f"{name:32s} -> {got:8s} (fresh: {fresh:7s}) {verdict}")
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
