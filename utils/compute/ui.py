import os
import subprocess

import bpy
from bpy.props import EnumProperty as EnumProperty

# number of the CPU in the compute_device enum. The numbers are saved in .blend
# files, GPU_n keeps number n, as when GPUs were the only devices offered.
CPU_DEVICE_NUMBER = 1000

DARKNET_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "darknet"
)


def gpu_count():
    """
    Number of GPUs available for GPU inference (darknet).

    On Windows, these are the GPUs TensorFlow finds. Elsewhere, GPU inference
    needs a self-built libdarknet.so, and the GPUs are counted with nvidia-smi:
    TensorFlow is not imported there, as it crashes some Linux builds of Blender.
    """
    if os.name == "nt":
        import tensorflow as tf

        physical_devices = tf.config.list_physical_devices()
        print("INFO: Found computational devices:\n", physical_devices)
        return len([d for d in physical_devices if d.device_type == "GPU"])

    if not os.path.isfile(os.path.join(DARKNET_FOLDER, "libdarknet.so")):
        return 0
    output = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30
    ).stdout
    return sum(line.startswith("GPU ") for line in output.splitlines())


class OMNITRAX_PT_ComputePanel(bpy.types.Panel):
    """
    Select (CUDA) computation device to run inference
    """

    bl_label = "Computational Device"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "TOOLS"
    bl_category = "OmniTrax"

    # GPU inference (darknet) needs a CUDA-capable GPU, CPU inference (OpenCV) is
    # always available
    devices = []
    try:
        for g in range(gpu_count()):
            devices.append(
                (
                    "GPU_" + str(g),
                    "GPU_" + str(g),
                    "Use GPU for inference (requires CUDA)",
                    g,
                )
            )
    except Exception as e:
        print("WARNING: Could not list GPUs, only CPU inference is available:", e)
    devices.append(
        ("CPU_0", "CPU", "Use CPU for inference (no CUDA required)", CPU_DEVICE_NUMBER)
    )

    bpy.types.Scene.compute_device = EnumProperty(
        name="",
        description="Select inference device",
        items=devices,
        # prefer the (last) GPU if there is one, as before
        default=devices[-2][0] if len(devices) > 1 else "CPU_0",
    )

    def draw(self, context):
        layout = self.layout

        col = layout.column(align=True)
        col.label(text="Select computational device")
        col.separator()
        row = col.row(align=True)
        row.prop(context.scene, "compute_device")


def register():
    bpy.utils.register_class(OMNITRAX_PT_ComputePanel)


def unregister():
    bpy.utils.unregister_class(OMNITRAX_PT_ComputePanel)
