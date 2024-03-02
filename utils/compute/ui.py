import bpy
import tensorflow as tf
from bpy.props import EnumProperty as EnumProperty


class OMNITRAX_PT_ComputePanel(bpy.types.Panel):
    """
    Select (CUDA) computation device to run inference
    """

    bl_label = "Computational Device"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "TOOLS"
    bl_category = "OmniTrax"

    try:
        physical_devices = tf.config.list_physical_devices()
        print("INFO: Found computational devices:\n", physical_devices)

        devices = []
        for d, device in enumerate(physical_devices):
            if device.device_type == "GPU":
                devices.append(
                    (
                        "GPU_" + str(d - 1),
                        "GPU_" + str(d - 1),
                        "Use GPU for inference (requires CUDA)",
                    )
                )
            else:
                if bpy.app.version_string == "2.92.0":
                    devices.append(("CPU_" + str(d), "CPU", "Use CPU for inference"))
    except:
        print("WARNING: Did not find suitable inference devices.")

    bpy.types.Scene.compute_device = EnumProperty(
        name="",
        description="Select inference device",
        items=devices,
        default=devices[-1][0],
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
