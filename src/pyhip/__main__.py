
from .hiptools import module
import argparse
import os
import tempfile

def extract_by_line_marker(src_file_path, marker_start, marker_end):
    code = ""
    with open(src_file_path) as file:
        in_code = 0
        for line in file:
            if in_code and line.startswith(marker_end):
                break
            if in_code:
                code += line
            else:
                code += "\n" # to make line number consistent with src
            if line.startswith(marker_start): in_code = 1
    if not in_code:
        return ""
    return code

def extract_sources(src_file_path):
    if src_file_path.endswith(".cpp"):
        with open(src_file_path) as file:
            hip_code = file.read()
        python_code = extract_by_line_marker(src_file_path, "/*PYHIP", "*/")
        return hip_code, python_code
    elif src_file_path.endswith(".md"):
        hip_code = extract_by_line_marker(src_file_path, "```cpp", "```")
        if hip_code == "":
            hip_code = extract_by_line_marker(src_file_path, "```c++", "```")
        python_code = extract_by_line_marker(src_file_path, "```python", "```")
        return hip_code, python_code
    else:
        raise Exception(f"{src_file_path} not recognized.")

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

    hip_code, python_code = extract_sources(src_file_path)

    if args.verbose:
        print(f"=============== python_code:")
        print(python_code)

    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=not args.verbose) as tmp_cpp:
        tmp_cpp.write(hip_code.encode())
        tmp_cpp.flush()
        if args.verbose:
            print(f"=============== hip_code: {tmp_cpp.name}")
        hip = module(tmp_cpp.name, "-fbracket-depth=20480")
        exec(python_code, {"hip":hip})
