import subprocess
import sys
from pathlib import Path

print("\nINFO: Checking requirements for OmniTrax addon...\n")

# get path of blender internal python executable
py_exec = str(sys.executable)

# install packages (if they are not already installed)
required_libraries = {"scipy": "scipy",
                      "pandas": "pandas",
                      "matplotlib": "matplotlib",
                      "opencv-python": "opencv-python",
                      "scikit-learn": "scikit-learn",
                      "tensorflow": "tensorflow==2.7.0",
                      "deeplabcut-live": "deeplabcut-live"}

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

print("\nInstalled packages:", installed_packages, "\n")

first_check = True

for library in required_libraries:
    if library in installed_packages:
        print(f"{library!r} already installed!")
    else:
        print(f"{library!r} not found! Installing package...\n")
        if first_check:
            # ensure pip is installed
            subprocess.call([py_exec, "-m", "ensurepip", "--user"])
            # update pip
            subprocess.call([py_exec, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.call(
            [py_exec, "-m", "pip", "install", f"--target={py_exec[:-14]}" + "lib", required_libraries[library]])

# Okay, this next part is hacky AF, but, for now, it seems like the only way to load DLC-Live...
# We need to suppress loading tkinter... So let's overwrite the "display.py" file within DLC-live.

# Cool. Here goes nothing

OVERWRITE_DLC_DISPLAY = False
dlclive_display = Path.joinpath(Path(py_exec[:-14]), "lib", "dlclive", "display.py")

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
    print("\nINFO: /dlclive/display.py has been updated to suppress tkinter import!!!")
    dlclive_display_file.close()

print("\nINFO: Cool, all looks good here! \nINFO: Uncheck OmniTrax in your addons to remove checks at program start!")
