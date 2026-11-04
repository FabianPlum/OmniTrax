import bpy
from bpy.props import EnumProperty as EnumProperty

try:
    import tensorflow as tf
except ImportError:
    # TensorFlow lists the GPUs and runs pose estimation, CPU detection works without
    tf = None

# number of the CPU in the compute_device enum. The numbers are saved in .blend
# files, GPU_n keeps number n, as when GPUs were the only devices offered.
CPU_DEVICE_NUMBER = 1000


class OMNITRAX_PT_ComputePanel(bpy.types.Panel):
    """
    Select (CUDA) computation device to run inference
    """

    bl_label = "Computational Device"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "TOOLS"
    bl_category = "OmniTrax"

    # GPU inference (darknet) needs a CUDA-capable GPU that TensorFlow can see,
    # CPU inference (OpenCV) is always available
    devices = []
    try:
        if tf is None:
            raise ImportError("TensorFlow is not installed")
        physical_devices = tf.config.list_physical_devices()
        print("INFO: Found computational devices:\n", physical_devices)

        gpus = [device for device in physical_devices if device.device_type == "GPU"]
        for g in range(len(gpus)):
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
