import subprocess
import sys
from pathlib import Path
import os
import platform
import addon_utils
from .CUDA_checks import check_CUDA_installation

script_file = os.path.realpath(__file__)
directory = os.path.dirname(script_file)

print("\nINFO: Checking requirements for omni_trax addon...\n")

required_CUDA_version = "11.2"
CUDA_match = check_CUDA_installation(required_CUDA_version=required_CUDA_version)

# only run the following checks upon initial installation
# setup_state.txt keeps track of the installation state

setup_complete = False

try:
    with open(os.path.join(directory, "utils", "setup_state.txt")) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip("\n")
            line_elems = line.split("=")
            if line_elems[0] == "setup_complete":
                if line_elems[1] == "True":
                    setup_complete = True
except FileNotFoundError:
    setup_complete = False

if not setup_complete:
    first_pkg = True
    setup_state_f_contents = []
    # add line on CUDA version matching
    # n.b. - this value is only stored for debugging purposes and will not be read at runtime
    if CUDA_match:
        setup_state_f_contents.append("Matching CUDA version=True")
    else:
        setup_state_f_contents.append("Matching CUDA version=False")

    # get path of blender internal python executable
    py_exec = str(sys.executable)

    # install packages (if they are not already installed)
    required_libraries = {
        "scipy": "scipy",
        "pandas": "pandas",
        "PyYAML": "PyYAML",
        "matplotlib": "matplotlib",
        "opencv-python": "opencv-python",
        "scikit-learn": "scikit-learn",
        "deeplabcut-live": "deeplabcut-live",
        "opencv-contrib-python": "opencv-contrib-python",
    }

    try:
        reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    except:
        subprocess.call([py_exec, "-m", "ensurepip", "--user"])

    try:
        reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    except:
        reqs = subprocess.check_output([sys.executable, "-m", "pip3", "freeze"])

    installed_packages = [r.decode().split("==")[0] for r in reqs.split()]

    print("\nInstalled packages:", installed_packages, "\n")

    # New in 1.1.0:
    # Because Numpy 2.0 messes with all the packages, we instead perform all installations in a single step so pip can actually
    # solve things correctly and we don't need to define each package version explicitly.

    # Check which packages need to be installed
    packages_to_install = []
    for library in required_libraries:
        if library in installed_packages:
            print(f"{library!r} already installed!")
            setup_state_f_contents.append(library + "=True")
        else:
            print(f"{library!r} not found! Will be installed...\n")
            packages_to_install.append(required_libraries[library])
            
    # Install missing packages if any
    if packages_to_install:
        # ensure pip is installed and updated
        subprocess.call([py_exec, "-m", "ensurepip", "--user"])
        subprocess.call([py_exec, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Construct install command with numpy constraint
        install_cmd = [py_exec, "-m", "pip", "install", "numpy<2.0"]
        install_cmd.extend(packages_to_install)
        
        if platform.system() != "Linux":
            install_cmd.insert(4, f"--target={py_exec[:-14]}lib")
            
        # Install all missing packages in one command
        subprocess.call(install_cmd)
        
        # Mark newly installed packages as complete
        for library in required_libraries:
            if library not in installed_packages:
                setup_state_f_contents.append(library + "=True")

    # We need to suppress loading tkinter as the package is incompatible with (some versions of) Blender python
    # We therefore overwrite the "display.py" file within lib/dlclive

    try:
        OVERWRITE_DLC_DISPLAY = False
        if platform.system() != "Linux":
            dlclive_display = Path.joinpath(
                Path(py_exec[:-14]), "lib", "dlclive", "display.py"
            )
        else:
            dlclive_display = Path.joinpath(
                Path(py_exec[:-15]),
                "lib",
                "python3.7",
                "site-packages",
                "dlclive",
                "display.py",
            )
            print(dlclive_display)

        orig_tk_import = "from tkinter import Tk, Label\n"
        orig_PIL_import = "from PIL import Image, ImageTk, ImageDraw\n"

        updated_tk_import = "# " + orig_tk_import
        updated_PIL_import = "# " + orig_PIL_import

        dlclive_display_file = open(str(dlclive_display), "r")
        updated_file_content = ""

        for line in dlclive_display_file:
            if line == orig_tk_import:
                line = updated_tk_import
                OVERWRITE_DLC_DISPLAY = True
            if line == orig_PIL_import:
                line = updated_PIL_import
                OVERWRITE_DLC_DISPLAY = True

            updated_file_content += line

        dlclive_display_file.close()

        if OVERWRITE_DLC_DISPLAY:
            dlclive_display_file = open(str(dlclive_display), "w")
            dlclive_display_file.write(updated_file_content)
            print(
                "\nINFO: /dlclive/display.py has been updated to suppress tkinter import!!!"
            )
            dlclive_display_file.close()

        setup_state_f_contents.append("overwritten_orig_dlclive=True")
    except FileNotFoundError:
        print("INFO: Did not suppress DLC imports - indicating unit testing")

    if platform.system() != "Linux":
        setup_state_f_contents.append("setup_complete=True")

    else:
        # clone and make darknet (for cpu support)
        print(
            "\n\n----------------------------------------------------------------------------\n"
        )
        print("INFO: Linux environment found - building darknet from source...")
        print(
            "INFO: for GPU, OPENMP, AVX, or OPENCV support, edit Makefile and make again!"
        )
        print(
            "\n----------------------------------------------------------------------------\n\n"
        )

        try:
            for mod in addon_utils.modules():
                addon_name = mod.bl_info.get("name")
                if addon_name == "omni_trax":
                    addon_loc = mod.__file__[:-12]

            print("INFO: removing current darknet folder...")
            # remove pre-built Windows darknet
            subprocess.call(
                ["rm", "-r", "-f", Path.joinpath(Path(addon_loc), "../../darknet")]
            )
            # clone darknet from source
            subprocess.call(["git", "clone", "https://github.com/AlexeyAB/darknet"])
            # (if required) move to correct location
            subprocess.call(["mv", "darknet", Path(addon_loc)])
            # update makefile to build .so

            print("INFO: Updating Makefile...")
            darknet_makefile = open(
                str(Path.joinpath(Path(addon_loc), "../../darknet", "Makefile")), "r"
            )
            updated_file_content = ""

            for line in darknet_makefile:
                if line == "LIBSO=0\n":
                    line = "LIBSO=1\n"

                updated_file_content += line

            darknet_makefile.close()

            darknet_makefile = open(
                str(Path.joinpath(Path(addon_loc), "../../darknet", "Makefile")), "w"
            )
            darknet_makefile.write(updated_file_content)
            print("\nINFO: /darknet/Makefile has been updated.")
            darknet_makefile.close()

            # make darknet
            print("INFO: Making darknet...")
            subprocess.call(
                ["make", "-C", Path.joinpath(Path(addon_loc), "../../darknet")]
            )
            setup_state_f_contents.append("compiled_darknet=True")
            setup_state_f_contents.append("setup_complete=True")

        except Exception as e:
            print(e)
            print("WARNING: Failed to rebuild darknet!")
            setup_state_f_contents.append("compiled_darknet=False")
            setup_state_f_contents.append("setup_complete=False")

    with open(os.path.join(directory, "setup_state.txt"), "w") as f:
        for line in setup_state_f_contents:
            f.write(line)
            f.write("\n")

    print(
        "\nINFO: Cool, all looks good here! \nINFO: Uncheck OmniTrax in your addons to remove checks at program start!"
    )
else:
    print("\nINFO: successfully loaded OmniTrax")
