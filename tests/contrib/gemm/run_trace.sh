 rm -rf ./ck_test/*
 rocprofv3 -i trace.yaml --att-library-path=./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux -- python ./test_4wave_cdna4_slicing.py
 python /mywork/luwei/gfx9-gluon-tutorials/scripts/process_json.py ./ck_test/ui*_234
 