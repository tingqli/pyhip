
# Build vllm from source

```bash
# find an existing docker image with all dependencies installed
$ docker run -it --cap-add=SYS_ADMIN --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v /home/tingqli/raid-user/vllm:/my-vllm --name tq-whisper2 --entrypoint /bin/bash rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909

# replace vllm in docker with custom editable build
pip uninstall vllm
pip install --verbose --no-build-isolation --editable .

# install xformers： there is no ROCm pre-built xformers, need to build from source:
#   https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html#xformers

git clone https://github.com/ROCm/xformers.git
cd xformers/
git submodule update --init --recursive
PYTORCH_ROCM_ARCH=gfx942 python setup.py install
```

