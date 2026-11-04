import os
import subprocess

DARKNET_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "darknet"
)


def check_CUDA_installation(required_CUDA_version=None):
    """
    Check whether GPU inference (darknet) can run. On Windows, this checks that
    the CUDA and cuDNN DLLs the bundled darknet.dll links against can be found,
    in any installed CUDA toolkit (CUDA_PATH, CUDA_PATH_V11_2, ...) or on the
    PATH, so several CUDA versions can be installed side by side.

    :param required_CUDA_version: not used anymore, the required CUDA version
        follows from darknet.dll (currently any CUDA 11.x with cuDNN 8.x)
    :return: True if GPU inference is available
    """
    print("-------------------------------------------------------")
    nvcc = _nvcc_version()
    if nvcc:
        print("INFO: CUDA toolkit on the PATH:", nvcc)

    if os.name == "nt":
        try:
            from ..darknet import cuda_dlls
        except ImportError:  # run as a script
            import sys

            sys.path.insert(0, DARKNET_FOLDER)
            import cuda_dlls

        dll = os.path.join(DARKNET_FOLDER, "darknet.dll")
        found, missing = cuda_dlls.find_dependencies(dll)
        for name, folder in found.items():
            print("INFO: Found", name, "in", folder)
        if missing:
            print("\nWARNING:", cuda_dlls.describe(missing, dll))
        else:
            print("\nINFO: Found all DLLs needed for GPU inference.")
        available = not missing
    else:
        library = os.path.join(DARKNET_FOLDER, "libdarknet.so")
        available = os.path.isfile(library)
        if not available:
            print(
                "\nINFO: GPU inference on this OS needs darknet built with GPU=1 "
                "and LIBSO=1, with libdarknet.so copied to",
                DARKNET_FOLDER,
                "\n      CPU inference works without it.",
            )

    print("-------------------------------------------------------\n")
    return available


def _nvcc_version():
    """
    Version of the CUDA compiler on the PATH (informational, darknet does not
    need it), e.g. "11.2", or None
    """
    try:
        output = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if "release " not in output:
        return None
    return output.split("release ")[-1].split(",")[0]


if __name__ == "__main__":
    check_CUDA_installation()
