import glob
import subprocess
import sys

blenderExecutable = 'blender'

# allow override of blender executable (important for CI!)
if len(sys.argv) > 1:
    blenderExecutable = sys.argv[1]

# run all tests before aborting build
testfailed = False

# iterate over each *.test.blend file in the "tests" directory
# and open up blender with the .test.blend file and the corresponding .test.py python script
for file in glob.glob('tests/*.test.py'):
    print('#' * 100)
    print('Running {} tests...'.format(file))
    print('#' * 100)
    code = subprocess.call([blenderExecutable, '--factory-startup',
                            '-noaudio', '--python', file,
                            '--python-exit-code', '1'])
    print('#' * 100)
    print("Exited with: ", code)
    print('#' * 100 + '\n\n\n')
    if code:
        testfailed = True

if testfailed:
    sys.exit(1)
