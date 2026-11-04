import os
import shutil
import sys
import tempfile
import unittest

try:
    from inspect import getsourcefile

    project_dir = os.path.dirname(os.path.dirname(getsourcefile(lambda: 0)))
    try:
        from omni_trax.utils.darknet import cuda_dlls
    except ImportError:
        sys.path.append(project_dir)
        from utils.darknet import cuda_dlls

    DARKNET_DLL = os.path.join(project_dir, "utils", "darknet", "darknet.dll")
    CUDA_DLLS = [
        "cudnn64_8.dll",
        "curand64_10.dll",
        "cublas64_11.dll",
        "cudart64_110.dll",
    ]
    OTHER_DLLS = [
        "nvcuda.dll",
        "MSVCP140.dll",
        "VCOMP140.DLL",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
    ]

    def touch(folder, names):
        os.makedirs(folder, exist_ok=True)
        for name in names:
            open(os.path.join(folder, name), "wb").close()

    class TestCudaDlls(unittest.TestCase):
        def test_imported_dlls(self):
            imports = cuda_dlls.imported_dlls(DARKNET_DLL)
            for name in CUDA_DLLS + ["pthreadVC3.dll", "KERNEL32.dll"]:
                self.assertIn(name, imports)
            self.assertEqual(cuda_dlls._cuda_version(imports), 11)
            self.assertEqual(cuda_dlls._cuda_version(["cudart64_12.dll"]), 12)
            with self.assertRaises(ValueError):
                cuda_dlls.imported_dlls(os.path.join(project_dir, "README.md"))

        def test_find_dependencies(self):
            tmp = tempfile.mkdtemp()
            try:
                cuda = os.path.join(tmp, "CUDA", "v11.8", "bin")
                other = os.path.join(tmp, "other")
                touch(cuda, CUDA_DLLS)
                touch(other, OTHER_DLLS)
                found, missing = cuda_dlls.find_dependencies(
                    DARKNET_DLL, folders=[os.path.dirname(DARKNET_DLL), cuda, other]
                )
                self.assertEqual(missing, [])
                self.assertEqual(found["cudart64_110.dll"], cuda)
                self.assertEqual(found["pthreadVC3.dll"], os.path.dirname(DARKNET_DLL))

                os.remove(os.path.join(cuda, "cudnn64_8.dll"))
                os.remove(os.path.join(cuda, "cudart64_110.dll"))
                found, missing = cuda_dlls.find_dependencies(
                    DARKNET_DLL, folders=[os.path.dirname(DARKNET_DLL), cuda, other]
                )
                self.assertEqual(sorted(missing), ["cudart64_110.dll", "cudnn64_8.dll"])
                message = cuda_dlls.describe(missing, DARKNET_DLL)
                self.assertIn("cudnn64_8.dll: install cuDNN 8.x for CUDA 11", message)
                self.assertIn("cudart64_110.dll: install a CUDA 11.x toolkit", message)
            finally:
                shutil.rmtree(tmp)

        def test_search_folders(self):
            tmp = tempfile.mkdtemp()
            saved = dict(os.environ)
            try:
                old, new = os.path.join(tmp, "v11.2"), os.path.join(tmp, "v12.8")
                touch(os.path.join(old, "bin"), [])
                touch(os.path.join(new, "bin"), [])
                os.environ["CUDA_PATH"] = new
                os.environ["CUDA_PATH_V11_2"] = old
                os.environ["CUDA_PATH_V12_8"] = new
                os.environ["PATH"] = os.pathsep.join(
                    [os.path.join(new, "bin"), os.path.join(tmp, "missing")]
                )
                folders = cuda_dlls.search_folders(DARKNET_DLL)
                self.assertEqual(folders[0], os.path.dirname(DARKNET_DLL))
                # every CUDA installation, each once, not only the one on the PATH
                self.assertIn(os.path.join(old, "bin"), folders)
                self.assertEqual(
                    [f for f in folders if f.startswith(tmp)],
                    [os.path.join(new, "bin"), os.path.join(old, "bin")],
                )
            finally:
                os.environ.clear()
                os.environ.update(saved)
                shutil.rmtree(tmp)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCudaDlls)
    success = unittest.TextTestRunner().run(suite)

    if success.errors or success.failures:
        raise Exception

except Exception as e:
    print(e)
    sys.exit(1)
