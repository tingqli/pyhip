#!/usr/bin/env bash
# Validate an Aiter ref against this PyHIP checkout, retaining a debug environment.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "Run this script with bash; do not source it." >&2
    return 2
fi
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/validate_aiter.sh [options] AITER_REF [-- PYTEST_ARGS...]

Create a venv, reuse the calling Python's Torch, install the selected Aiter
and current PyHIP, run pytest, then open an activated Bash for debugging.

Options:
  --python PYTHON  Outer Python to reuse (default: python3 on PATH)
  --repo URL      Aiter repository URL or local clone (default: ROCm/aiter)
  --work-dir DIR  New directory for the checkout and venv; must not exist
  --no-shell      Report the result without opening an interactive shell
  -h, --help      Show this help

By default a unique directory is created under ~/.cache/pyhip/aiter-envs.
Torch is shared, never explicitly installed; incompatible Torch requirements
fail rather than replace it. Other outer packages are visible as fallbacks.
Installation failures also retain the environment and open a debug shell.
Without a terminal, the shell is skipped. Nothing is automatically deleted.
After the debug shell exits, return the installation/pytest exit code.
EOF
}

python=python3
repo=https://github.com/ROCm/aiter.git
work_dir=
aiter_ref=
open_shell=1
while (( $# )); do
    case $1 in
        -h|--help) usage; exit 0 ;;
        --python|--repo|--work-dir)
            if (( $# < 2 )) || [[ -z $2 || $2 == --* ]]; then
                echo "Missing value for $1" >&2; exit 2
            fi
            case $1 in
                --python) python=$2 ;;
                --repo) repo=$2 ;;
                --work-dir) work_dir=$2 ;;
            esac
            shift 2 ;;
        --no-shell) open_shell=0; shift ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if [[ -n $aiter_ref ]]; then
                echo "Put pytest arguments after --" >&2; exit 2
            fi
            aiter_ref=$1; shift ;;
    esac
done
if [[ -z $aiter_ref ]]; then usage >&2; exit 2; fi

pyhip_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
# Do not resolve the executable's symlink: that would lose an outer venv.
python=$(command -v "$python")
[[ $python == /* ]] || python=$PWD/$python
if [[ -d $repo ]]; then repo=$(cd -- "$repo" && pwd -P); fi
if [[ -z $work_dir ]]; then
    mkdir -p -- "$HOME/.cache/pyhip/aiter-envs"
    work_dir=$(mktemp -d "$HOME/.cache/pyhip/aiter-envs/run.XXXXXX")
else
    mkdir -p -- "$(dirname -- "$work_dir")"
    mkdir -- "$work_dir"
fi
work_dir=$(cd -- "$work_dir" && pwd -P)
venv_dir=$work_dir/venv
aiter_dir=$work_dir/aiter
activate=$work_dir/activate.sh
stage="create environment"
aiter_commit="not resolved"
pytest_status="not run"

finish() {
    local status=$?
    trap - EXIT
    set +e
    {
        printf '\nAiter ref: %s\nAiter commit: %s\n' "$aiter_ref" "$aiter_commit"
        printf 'Stage: %s\nExit code: %s\npytest exit code: %s\n' "$stage" "$status" "$pytest_status"
        printf 'Venv: %s\nAiter checkout: %s\n' "$venv_dir" "$aiter_dir"
        printf 'Reactivate: source %q\n' "$activate"
    } | tee "$work_dir/result.txt"
    if (( open_shell )) && [[ -t 0 && -t 1 && -f $venv_dir/bin/activate ]]; then
        echo "Entering debug Bash in the PyHIP checkout; use exit to return."
        # Avoid a user's .bashrc reactivating the outer environment.
        bash --noprofile --rcfile "$activate" -i
    else
        echo "Debug shell skipped (--no-shell, no terminal, or venv creation failed)."
    fi
    exit "$status"
}
trap finish EXIT

# This file also restores the same source selection and caches for later debug.
{
    printf 'unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX PIP_USER CK_DIR HIP_KITTENS_DIR OPUS_GEN_CO_DIR\n'
    printf 'source %q\n' "$venv_dir/bin/activate"
    printf 'export PYTHONNOUSERSITE=1 PIP_REQUIRE_VIRTUALENV=1\n'
    printf 'export PIP_CONSTRAINT=%q\n' "$work_dir/torch-constraint.txt"
    printf 'export AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0\n'
    printf 'export AITER_JIT_DIR=%q\n' "$work_dir/cache/aiter"
    printf 'export PYHIP_CACHE_DIR=%q\n' "$work_dir/cache/pyhip"
    printf 'export FLYDSL_RUNTIME_CACHE_DIR=%q\n' "$work_dir/cache/flydsl"
    printf 'export FLYDSL_AUTOTUNE_CACHE_DIR=%q\n' "$work_dir/cache/autotune"
    printf 'export TRITON_CACHE_DIR=%q\n' "$work_dir/cache/triton"
    printf 'export TORCHINDUCTOR_CACHE_DIR=%q\n' "$work_dir/cache/inductor"
    printf 'export TORCH_EXTENSIONS_DIR=%q\n' "$work_dir/cache/torch-extensions"
    printf 'cd -- %q\n' "$pyhip_dir"
} > "$activate"

echo "Creating environment: $venv_dir (outer Python: $python)"
"$python" -m venv --system-site-packages "$venv_dir"
venv_site=$("$venv_dir/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
# --system-site-packages alone does not inherit packages from a parent venv.
# Append its site directories WITHOUT executing its editable-install .pth files.
# Sort after the new environment's editable .pth files so local installs win.
outer_torch=$("$python" - "$venv_site/zzzz_outer_site.pth" "$work_dir/torch-constraint.txt" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
import torch

if not torch.version.hip:
    raise SystemExit("The outer Python must already contain ROCm Torch.")
paths = [p for p in sys.path if Path(p).name in ("site-packages", "dist-packages")]
paths.append(str(Path(torch.__file__).resolve().parent.parent))
Path(sys.argv[1]).write_text("\n".join(dict.fromkeys(paths)) + "\n")
Path(sys.argv[2]).write_text("torch===" + metadata.version("torch") + "\n")
print(Path(torch.__file__).resolve())
PY
)
# shellcheck disable=SC1090
source "$activate"

stage="clone Aiter"
git clone --no-checkout -- "$repo" "$aiter_dir"
stage="resolve Aiter ref"
if aiter_commit=$(git -C "$aiter_dir" rev-parse --verify --end-of-options "${aiter_ref}^{commit}" 2>/dev/null); then
    :
elif aiter_commit=$(git -C "$aiter_dir" rev-parse --verify --end-of-options "origin/${aiter_ref}^{commit}" 2>/dev/null); then
    :
else
    # Also allow a full SHA/ref which was not advertised during clone.
    git -C "$aiter_dir" fetch origin "$aiter_ref"
    aiter_commit=$(git -C "$aiter_dir" rev-parse 'FETCH_HEAD^{commit}')
fi
git -C "$aiter_dir" checkout --detach "$aiter_commit"
stage="Aiter submodules"
git -C "$aiter_dir" submodule update --init --recursive
echo "Testing Aiter $aiter_commit against $pyhip_dir"

stage="install build dependencies"
cd -- "$aiter_dir"
python -m pip install --upgrade pip
python -m pip install 'setuptools>=64' wheel 'setuptools-scm>=8' packaging 'tomli; python_version < "3.11"'
# Aiter builds need the shared ROCm Torch, hence no isolated build environment.
# Install the selected ref's declared build requirements before disabling it.
python - <<'PY'
from pathlib import Path
import subprocess
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib

path = Path("pyproject.toml")
if path.is_file():
    requirements = tomllib.loads(path.read_text()).get("build-system", {}).get("requires", [])
    if requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])
PY

stage="install Aiter dependencies"
if [[ -f requirements.txt ]]; then python -m pip install -r requirements.txt; fi
if [[ -f .github/scripts/install_triton.sh ]]; then
    bash .github/scripts/install_triton.sh
fi
# FlyDSL's version is controlled solely by the selected Aiter ref.
python -m pip install pytest
stage="install Aiter and current PyHIP"
# Compat editable installs use ordinary paths before the shared outer packages,
# rather than meta-path finders that could lose to an outer Aiter installation.
python -m pip install --no-build-isolation --config-settings editable_mode=compat -e "$aiter_dir" -e "$pyhip_dir"

stage="check imports and shared Torch"
cd -- "$pyhip_dir"
python - "$outer_torch" "$aiter_dir" "$pyhip_dir" <<'PY'
from pathlib import Path
import sys
import torch
import aiter
import pyhip

assert Path(torch.__file__).resolve() == Path(sys.argv[1]), "Torch was not reused from the outer Python"
for module, root in ((aiter, sys.argv[2]), (pyhip, sys.argv[3])):
    assert Path(module.__file__).resolve().is_relative_to(Path(root)), f"Unexpected {module.__name__}: {module.__file__}"
print(f"Torch: {torch.__version__} ({torch.__file__})")
print(f"Aiter: {aiter.__file__}")
print(f"PyHIP: {pyhip.__file__}")
PY

stage="pytest"
set +e
python -m pytest "$@"
pytest_status=$?
set -e
stage="pytest completed"
exit "$pytest_status"