
from .hiptools import module
import argparse
import os

if __name__ == "__main__":
    # accept a single source cpp(HIP) file with python embedded as comments
    # extract python code PYHIP
    # in this mode, pyhip knows the kernel source code, thus pyhip.module will
    # be created automatically
    parser = argparse.ArgumentParser(
            description="Run a single source HIP with python code embedded",
            )
    parser.add_argument('input_file', help='HIP source file')
    parser.add_argument('-v','--verbose', action="store_true", help='HIP source file')
    args = parser.parse_args()

    src_file_path = args.input_file
    if not os.path.isabs(src_file_path):
        cwd = os.getcwd()
        src_file_path = os.path.join(cwd, src_file_path)

    # extract python code
    python_code = ""
    with open(src_file_path) as file:
        in_python = 0
        for line in file:
            if line.startswith("*/"): in_python += 1
            if in_python & 1:
                python_code += line
            if line.startswith("/*PYHIP"): in_python += 1

    if args.verbose:
        print("================= PYHIP source code:")
        print(python_code)
        print("================= PYHIP ends")

    hip = module(src_file_path)
    exec(python_code, {"hip":hip})
