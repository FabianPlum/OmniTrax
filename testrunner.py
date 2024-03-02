import glob
import subprocess
import sys

"""
when running this locally use:

python testrunner.py "PATH/TO/YOUR/BLENDER/EXECUTABLE"

so, e.g., under Windows it will likely be

python testrunner.py "C:\Program Files\Blender Foundation\Blender 3.3\blender"

"""

blenderExecutable = "blender"

# testrunner structure built based on https://github.com/dfki-ric/phobos
# allow override of blender executable (important for TravisCI!)
if len(sys.argv) > 1:
	blenderExecutable = sys.argv[1]

# run all tests before aborting build
testfailed = False
failed_tests = []

# iterate over each *.test.py file in the "tests" directory
for file in glob.glob("tests/*.test.py"):
	print("#" * 100)
	print("Running {} tests...".format(file))
	print("#" * 100)
	code = subprocess.call(
		[
			blenderExecutable,
			"--addons",
			"omni_trax",
			"--factory-startup",
			"-noaudio",
			"--background",
			"--python",
			file,
			"--python-exit-code",
			"1",
		]
	)
	print("#" * 100)
	print("Exited with: ", code)
	print("#" * 100 + "\n\n\n")
	if code:
		testfailed = True
		failed_tests.append(file)

for test in failed_tests:
	print("#" * 100, "\n")
	print("FAILED: {}".format(test))
	print("See stderr above for details!".format())
	print("\n", "#" * 100 + "\n\n\n")

if testfailed:
	sys.exit(1)
else:
	print("#" * 100)
	print("--- ALL TESTS PASSED ---")
	print("#" * 100 + "\n")
