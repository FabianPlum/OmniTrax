"""
Find the CUDA, cuDNN and other DLLs that darknet.dll (GPU inference) needs,
and make them loadable.

Since Python 3.8, Windows looks for the dependencies of a DLL only in the DLL's
own folder, the application and system folders, and folders added with
os.add_dll_directory(), not on the PATH. The CUDA and cuDNN installers put
their DLLs into their own folders, so these folders are added here before
darknet.dll is loaded.

The required DLLs are read from darknet.dll itself, so a darknet.dll built for
a different CUDA version needs no changes here: the current build needs any
CUDA 11.x toolkit with cuDNN 8.x.
"""

import glob
import os
import re
import struct

# always present on Windows (API sets are virtual and resolved by the loader)
_WINDOWS_DLLS = re.compile(
    r"^(api-ms-win-.*|ext-ms-.*|kernel32|user32|gdi32|advapi32|shell32|ole32|"
    r"oleaut32|ws2_32|wsock32|ntdll|bcrypt|crypt32|secur32|setupapi|version|"
    r"winmm|psapi|dbghelp|shlwapi|comdlg32|comctl32|imm32|iphlpapi|userenv)\.dll$",
    re.IGNORECASE,
)
_CUDA_LIBRARY = re.compile(
    r"^(cudart|cublas|cublaslt|curand|cufft|cusolver|cusparse|nvrtc)64_\d+\.dll$",
    re.IGNORECASE,
)
_CUDNN_LIBRARY = re.compile(r"^cudnn\w*64_(\d+)\.dll$", re.IGNORECASE)
_VC_RUNTIME = re.compile(
    r"^(vcruntime140(_1)?|msvcp140|vcomp140|concrt140)\.dll$", re.I
)
# shipped next to darknet.dll
_BUNDLED_DLLS = ("pthreadvc3.dll",)


def imported_dlls(path):
    """
    Names of the DLLs that a Windows DLL or EXE imports, read from its PE
    import table (works on any OS).
    """
    with open(path, "rb") as f:
        data = f.read()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        raise ValueError(f"{path} is not a Windows DLL")
    (n_sections,) = struct.unpack_from("<H", data, pe + 6)
    (optional_size,) = struct.unpack_from("<H", data, pe + 20)
    optional = pe + 24
    (magic,) = struct.unpack_from("<H", data, optional)
    # data directory 1 is the import table, 64 bit (PE32+) headers are longer
    directories = optional + (112 if magic == 0x20B else 96)
    import_rva, _ = struct.unpack_from("<II", data, directories + 8)

    sections = []
    for i in range(n_sections):
        size, address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, optional + optional_size + 40 * i + 8
        )
        sections.append((address, max(size, raw_size), raw_offset))

    def file_offset(rva):
        for address, size, raw_offset in sections:
            if address <= rva < address + size:
                return rva - address + raw_offset
        raise ValueError(f"{path} has an invalid import table")

    names = []
    descriptor = file_offset(import_rva)
    while True:
        fields = struct.unpack_from("<5I", data, descriptor)
        if not any(fields):
            return names
        start = file_offset(fields[3])
        names.append(data[start : data.index(b"\0", start)].decode("ascii"))
        descriptor += 20


def search_folders(dll_path):
    """
    Folders that may hold darknet.dll's dependencies: its own folder, the CUDA
    toolkits (CUDA_PATH, CUDA_PATH_V11_2, ...), cuDNN installations and the PATH.
    """
    folders = [os.path.dirname(os.path.abspath(dll_path))]
    for key in sorted(os.environ):
        if key.upper().startswith("CUDA_PATH"):
            folders.append(os.path.join(os.environ[key], "bin"))
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    cudnn = os.path.join(program_files, "NVIDIA", "CUDNN")
    folders += sorted(glob.glob(os.path.join(cudnn, "v*", "bin")))
    folders += sorted(glob.glob(os.path.join(cudnn, "v*", "bin", "*")))
    folders += os.environ.get("PATH", "").split(os.pathsep)

    unique = []
    for folder in folders:
        folder = folder.strip().strip('"')
        if folder and os.path.isdir(folder):
            key = os.path.normcase(os.path.abspath(folder))
            if key not in [os.path.normcase(os.path.abspath(u)) for u in unique]:
                unique.append(folder)
    return unique


def find_dependencies(dll_path, folders=None):
    """
    :return: (found, missing): found maps each dependency found in one of the
             search folders to that folder, missing lists the dependencies that
             can be found neither there nor by Windows itself
    """
    if folders is None:
        folders = search_folders(dll_path)
    found, missing = {}, []
    for name in imported_dlls(dll_path):
        if _WINDOWS_DLLS.match(name):
            continue
        folder = next(
            (f for f in folders if os.path.isfile(os.path.join(f, name))), None
        )
        if folder is not None:
            found[name] = folder
        elif not _loadable_by_windows(name):
            missing.append(name)
    return found, missing


def prepare(dll_path):
    """
    Make the dependencies of dll_path that are found loadable, by adding their
    folders to the DLL search path.

    :return: names of the dependencies that are missing
    """
    found, missing = find_dependencies(dll_path)
    for folder in dict.fromkeys(found.values()):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.abspath(folder))
        else:
            # Python < 3.8 (Blender 2.92) searches the PATH
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    return missing


def describe(missing, dll_path):
    """
    Explain what to install for each missing dependency of dll_path.
    """
    if not missing:
        return ""
    imports = imported_dlls(dll_path)
    cuda = _cuda_version(imports)
    hints = {}
    for name in missing:
        cudnn = _CUDNN_LIBRARY.match(name)
        if _CUDA_LIBRARY.match(name):
            hint = f"install a CUDA {cuda}.x toolkit" if cuda else "install CUDA"
        elif cudnn:
            hint = f"install cuDNN {cudnn.group(1)}.x" + (
                f" for CUDA {cuda}" if cuda else ""
            )
        elif name.lower() == "nvcuda.dll":
            hint = "install the NVIDIA graphics driver"
        elif _VC_RUNTIME.match(name):
            hint = "install the Microsoft Visual C++ Redistributable (x64)"
        elif name.lower() in _BUNDLED_DLLS:
            hint = "part of OmniTrax, reinstall the add-on"
        else:
            hint = "not found"
        hints.setdefault(hint, []).append(name)
    lines = ["GPU inference needs these DLLs, which were not found:"]
    lines += [f"  {', '.join(names)}: {hint}" for hint, names in hints.items()]
    lines.append(
        "See docs/CUDA_installation_guide.md, or select the CPU as compute device."
    )
    return "\n".join(lines)


def _cuda_version(imports):
    """
    CUDA major version darknet.dll was built with, from its CUDA runtime import,
    e.g. cudart64_110.dll -> 11, cudart64_12.dll -> 12
    """
    for name in imports:
        match = re.match(r"^cudart64_(\d+)\.dll$", name, re.IGNORECASE)
        if match:
            digits = match.group(1)
            return int(digits[:-1]) if len(digits) == 3 else int(digits)
    return None


def _loadable_by_windows(name):
    """
    Whether Windows finds a DLL on its own (application and system folders,
    already loaded DLLs), e.g. nvcuda.dll from the graphics driver or the
    Visual C++ runtime next to blender.exe.
    """
    if os.name != "nt":
        return False
    import ctypes

    try:
        ctypes.WinDLL(name)
        return True
    except OSError:
        return False
