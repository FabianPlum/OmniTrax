import subprocess
import sys

py_exec = str(sys.executable)
# ensure pip is installed
subprocess.call([py_exec, "-m", "ensurepip", "--user"])
# update pip
subprocess.call([py_exec, "-m", "pip", "install", "--upgrade", "pip"])
# install packages (if they are not already installed)


required_libraries = {"scipy": "scipy",
                      "pandas": "pandas",
                      "matplotlib": "matplotlib",
                      "cv2": "opencv-python",
                      "sklearn": "scikit-learn",
                      "tensorflow": "tensorflow==2.7.0"}

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

print("\nInstalled packages:", installed_packages, "\n")

for library in required_libraries:
    if library in sys.modules:
        print(f"{library!r} already installed!")
    else:
        print(f"{library!r} not found! Installing package...")
        subprocess.call(
            [py_exec, "-m", "pip", "install", f"--target={py_exec[:-14]}" + "lib", required_libraries[library]])

"""
try:
    import scipy
except ImportError:
    subprocess.call(
        [py_exec, "-m", "pip", "install", f"--target={py_exec[:-14]}" + "lib", "scipy"])

try:
    import pandas
except ImportError:
    subprocess.call(
        [py_exec, "-m", "pip", "install", f"--target={py_exec[:-14]}" + "lib", "pandas"])

try:
    import matplotlib
except ImportError:
    subprocess.call(
        [py_exec, "-m", "pip", "install", f"--target={py_exec[:-14]}" + "lib", "matplotlib"])

try:
    import cv2
except ImportError:
    subprocess.call(
        [py_exec, "-m", "pip", "install", f"--target={py_exec[:-14]}" + "lib", "opencv-python"])

try:
    import sklearn
except ImportError:
    subprocess.call(
        [py_exec, "-m", "pip", "install", f"--target={py_exec[:-14]}" + "lib", "scikit-learn"])

try:
    import tensorflow as tf
except ImportError:
    subprocess.call(
        [py_exec, "-m", "pip", "install", f"--target={py_exec[:-14]}" + "lib", "tensorflow==2.7.0"])
"""
