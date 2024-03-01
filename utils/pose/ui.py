import bpy
from bpy.props import BoolProperty as BoolProperty
from bpy.props import StringProperty as StringProperty
from bpy.props import IntProperty as IntProperty
from bpy.props import FloatProperty as FloatProperty


class OMNITRAX_PT_PoseEstimationPanel(bpy.types.Panel):
    bl_label = "Pose Estimation (DLC)"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "TOOLS"
    bl_category = "OmniTrax"

    bpy.types.Scene.pose_network_path = StringProperty(
        name="DLC Network path",
        description="Path to your trained and exported DLC network," +
                    "where your pose_cfg.yaml and snapshot files are stored",
        default="",
        subtype="FILE_PATH")
    bpy.types.Scene.pose_enforce_constant_size = BoolProperty(
        name="constant (input) detection sizes",
        description="Check to enforce constant detection sizes. This affects the following POSE inference," +
                    "regarding the used regions of interest.",
        default=False)
    bpy.types.Scene.pose_constant_size = IntProperty(
        name="Pose (input) frame size (px)",
        description="Constant detection size in pixels." +
                    "All detections will be rescaled and padded, if necessary for pose estimation",
        default=300)
    bpy.types.Scene.pose_pcutoff = FloatProperty(
        name="pcutoff (minimum key point confidence)",
        description="Predicted key points with a confidence below this threshold" +
                    "will be discarded during pose estimation",
        default=0.5)
    bpy.types.Scene.pose_plot_skeleton = BoolProperty(
        name="Plot skeleton",
        description="Plot the pre-defined skeleton based on the detected landmarks",
        default=False)
    bpy.types.Scene.pose_point_size = IntProperty(
        name="Key point marker size",
        description="(visualisation) Size of marker points on pose estimation",
        default=3)
    bpy.types.Scene.pose_skeleton_bone_width = IntProperty(
        name="Skeleton line thickness",
        description="(visualisation) Line width of skeleton bones in pixels",
        default=2)
    bpy.types.Scene.pose_save_video = BoolProperty(
        name="Export pose estimation video",
        description="Write the cropped video with tracked overlay to the location of the input video",
        default=False)
    bpy.types.Scene.pose_export_pose = BoolProperty(
        name="Export pose estimation data",
        description="Write estimated pose data to disk, including landmark locations in pixel space and joint angles.",
        default=False)
    bpy.types.Scene.pose_show_labels = BoolProperty(
        name="Display label names",
        description="Display label names as an overlay of the pose estimation",
        default=False)

    def draw(self, context):
        layout = self.layout

        col = layout.column(align=True)
        col.label(text="DLC network path: [path to the directory of your trained network]")
        col.prop(context.scene, "pose_network_path", text="")
        col.separator()

        col.prop(context.scene, "pose_enforce_constant_size")
        col.prop(context.scene, "pose_constant_size")
        col.prop(context.scene, "pose_pcutoff")
        col.separator()

        col.label(text="Visualisation:")
        col.prop(context.scene, "pose_plot_skeleton")
        col.prop(context.scene, "pose_point_size")
        col.prop(context.scene, "pose_skeleton_bone_width")
        col.prop(context.scene, "pose_show_labels")
        col.separator()

        col.label(text="Run Pose Estimation")
        col.prop(context.scene, "pose_save_video")
        col.prop(context.scene, "pose_export_pose")
        col.operator("scene.pose_estimation_run", text="ESTIMATE POSES").fullframe = False
        col.operator("scene.pose_estimation_run", text="ESTIMATE POSES [full frame]").fullframe = True


def register():
    bpy.utils.register_class(OMNITRAX_PT_PoseEstimationPanel)


def unregister():
    bpy.utils.unregister_class(OMNITRAX_PT_PoseEstimationPanel)
