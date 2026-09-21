rm -rf ./ck_test/*
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rocprofv3 -i "$SCRIPT_DIR/trace.yaml" --att-library-path=./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux -- python "$SCRIPT_DIR/../../tests/ops/gemm/test_4wave_cdna4_slicing.py"
python /mywork/luwei/gfx9-gluon-tutorials/scripts/process_json.py ./ck_test/ui*_234
