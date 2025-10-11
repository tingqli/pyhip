

```bash

docker run -it --rm --cap-add=SYS_ADMIN --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v /raid/users/tingqli/test-triton/pyhip:/pyhip --name thread-trace  vllm/vllm-openai:v0.11.0 bash

```
