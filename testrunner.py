import glob
import subprocess
import sys

blenderExecutable = 'blender'

# allow override of blender executable (important for TravisCI!)
if len(sys.argv) > 1:
    blenderExecutable = sys.argv[1]

# run all tests before aborting build
testfailed = False

# iterate over each *.test.py file in the "tests" directory
for file in glob.glob('tests/*.test.py'):
    print('#' * 100)
    print('Running {} tests...'.format(file))
    print('#' * 100)
    code = subprocess.call([blenderExecutable, '--addons', 'omni_trax', '--factory-startup',
                            '-noaudio', '--background', '--python', file,
                            '--python-exit-code', '1'])
    print('#' * 100)
    print("Exited with: ", code)
    print('#' * 100 + '\n\n\n')
    if code:
        testfailed = True

if testfailed:
    sys.exit(1)
