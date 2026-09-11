 rm -rf ./ck_test/*
 rocprofv3 -i trace.yaml --att-library-path=./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux -- python ./test_moe.py --tokens 8192 --gate-up-size 8192 --hidden-size 16384 --topk 1 --experts 1 --warmup 5 --iterations 20 --data-clones 20
 python /mywork/luwei/gfx9-gluon-tutorials/scripts/process_json.py ./ck_test/ui*_234
 