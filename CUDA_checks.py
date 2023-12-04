import subprocess


def check_CUDA_installation(required_CUDA_version="11.2"):
    """
    CHeck whether the required version of CUDA is installed.
    :param required_CUDA_version: OmniTrax required CUDA version
    :return: correct_version : bool, True if correct / False if incorrect
    """
    print("-------------------------------------------------------")
    print("\nINFO: Required CUDA version:", required_CUDA_version)

    getVersion = subprocess.Popen("nvcc --version", shell=True, stdout=subprocess.PIPE).stdout
    found_CUDA_version = getVersion.read().decode().split("release ")[-1].split(",")[0]

    print("INFO: Found CUDA version:   ", found_CUDA_version)

    if found_CUDA_version == required_CUDA_version:
        print("\nINFO: Matching CUDA version detected! Enabled GPU processing.")
        print("      If tracking fails, double check your cudNN 8.1.0 installation.")
        correct_version = True
    else:
        print("\nWARNING: Incompatible CUDA version found! OmniTrax requires CUDA", required_CUDA_version,
              "\n         In case you have multiple installations, ensure your PATH variable points to the correct version.",
              "\n         For more information on CUDA version matching refer to: "
              "\n         https://github.com/FabianPlum/OmniTrax/blob/main/docs/CUDA_installation_guide.md"
              "\n         You will only be able to use the (terribly slow) CPU version of OmniTrax")
        correct_version = False

    print("-------------------------------------------------------\n")
    return correct_version


if __name__ == '__main__':
    check_CUDA_installation(required_CUDA_version="11.2")
