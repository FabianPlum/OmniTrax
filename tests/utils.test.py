import sys
import types
import unittest
import os

try:
    try:
        from omni_trax.utils.omni_trax_utils import (
            scale_detections,
            convertBack,
            setInferenceDevive,
        )
    except:
        from inspect import getsourcefile

        project_dir = os.path.dirname(os.path.dirname(getsourcefile(lambda: 0)))
        sys.path.append(project_dir)
        from utils.omni_trax_utils import (
            scale_detections,
            convertBack,
            setInferenceDevive,
        )

    class TestUtils(unittest.TestCase):
        def test_scale_detections(self):
            values = [100, 200, 320, 320, 1920, 1080]
            target = [600, 405]

            self.assertEqual(
                scale_detections(
                    x=values[0],
                    y=values[1],
                    network_w=values[2],
                    network_h=values[3],
                    output_w=values[4],
                    output_h=values[5],
                ),
                target,
            )

        def test_convert_back(self):
            values = [100, 200, 1920, 1080]
            target = (-860, -340, 1060, 740)

            self.assertEqual(
                convertBack(x=values[0], y=values[1], w=values[2], h=values[3]), target
            )

        def test_set_inference_device(self):
            # TensorFlow sees only the selected GPU, or no GPU at all for the CPU
            # and for unknown devices (e.g. one saved on another machine)
            calls = []

            class Config:
                def list_physical_devices(self, device_type=None):
                    return ["GPU:0", "GPU:1"]

                def set_visible_devices(self, devices, device_type=None):
                    calls.append((devices, device_type))

                def list_logical_devices(self):
                    return []

            fake_tf = types.ModuleType("tensorflow")
            fake_tf.config = Config()
            saved = sys.modules.get("tensorflow")
            sys.modules["tensorflow"] = fake_tf
            try:
                for device in ["GPU_1", "CPU_0", "", "GPU_x"]:
                    setInferenceDevive(device)
            finally:
                if saved is None:
                    del sys.modules["tensorflow"]
                else:
                    sys.modules["tensorflow"] = saved
            self.assertEqual(
                calls,
                [("GPU:1", "GPU"), ([], "GPU"), ([], "GPU"), ([], "GPU")],
            )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestUtils)
    success = unittest.TextTestRunner().run(suite)

    if success.errors or success.failures:
        raise Exception

except Exception as e:
    print(e)
    sys.exit(1)
