import bpy
from bpy.props import BoolProperty as BoolProperty
from bpy.props import StringProperty as StringProperty
from bpy.props import IntProperty as IntProperty
from bpy.props import FloatProperty as FloatProperty
from bpy.props import EnumProperty as EnumProperty


class OMNITRAX_PT_DetectionPanel(bpy.types.Panel):
	bl_label = "Detection (YOLO)"
	bl_space_type = "CLIP_EDITOR"
	bl_region_type = "TOOLS"
	bl_category = "OmniTrax"

	bpy.types.Scene.detection_config_path = StringProperty(
		name="Config path",
		description="Path to your YOLO network config file",
		default="",  # bpy.data.filepath.rpartition("\\")[0],
		subtype="FILE_PATH",
	)
	bpy.types.Scene.detection_weights_path = StringProperty(
		name="Weights path",
		description="Path to your YOLO network weights file",
		default="",  # bpy.data.filepath.rpartition("\\")[0],
		subtype="FILE_PATH",
	)
	bpy.types.Scene.detection_data_path = StringProperty(
		name="Data path",
		description="Path to your YOLO network data file",
		default="",  # bpy.data.filepath.rpartition("\\")[0],
		subtype="FILE_PATH",
	)
	bpy.types.Scene.detection_names_path = StringProperty(
		name="Names path",
		description="Path to your YOLO network names file",
		default="",  # bpy.data.filepath.rpartition("\\")[0],
		subtype="FILE_PATH",
	)
	bpy.types.Scene.detection_enforce_constant_size = BoolProperty(
		name="constant detection sizes",
		description="Check to enforce constant detection sizes. This DOES NOT affect the actual inference, only the resulting regions of interest.",
		default=True,
	)
	bpy.types.Scene.detection_constant_size = IntProperty(
		name="constant detection sizes (px)",
		description="Constant detection size in pixels. All detections will be rescaled to this value in both width and height.",
		default=60,
	)
	bpy.types.Scene.detection_min_size = IntProperty(
		name="minimum detection sizes (px)",
		description="If the width or height of a detection is below this threshold, it will be discarded. This can be useful to decrease noise. Keep the value at 0 if this is not needed.",
		default=15,
	)
	bpy.types.Scene.detection_network_width = FloatProperty(
		name="Network width",
		description="YOLO network input width. Larger network inputs allow for smaller detected subjects but will increase inference time and memory usage. [MUST BE MULTIPLE OF 32]",
		default=480,
		step=3200,  # step size given in 1/100
		min=32,
		precision=0,
	)  # set to 0 emulate integer property (which currently does not support steps)
	bpy.types.Scene.detection_network_height = FloatProperty(
		name="Network height",
		description="YOLO network input height. Larger network inputs allow for smaller detected subjects but will increase inference time and memory usage. [MUST BE MULTIPLE OF 32]",
		default=480,
		step=3200,  # step size given in 1/100
		min=32,
		precision=0,
	)  # set to 0 emulate integer property (which currently does not support steps)
	bpy.types.Scene.detection_activation_threshold = FloatProperty(
		name="Confidence threshold",
		description="Detection confidence threshold. A higher confidence leads to fewer false positives but more missed detections.",
		default=0.5,
	)
	bpy.types.Scene.detection_nms = FloatProperty(
		name="Non-maximum suppression",
		description="Non-maximum suppression (NMS) refers to the maximum overlap allowed between proposed bounding boxes. E.g., a value of 0.45 corresponds to a maximum overlap of 45% between two compared bounding boxes to be retained simultaneously. In case the overlap is larger, the box with the lower objectness score or classification confidence will be 'suppressed', thus, only the highest confidence prediction is returned.",
		default=0.45,
	)

	def draw(self, context):
		layout = self.layout

		col = layout.column(align=True)
		col.label(text="Config path:")
		col.prop(context.scene, "detection_config_path", text="")
		col.separator()
		col.label(text="Weights path:")
		col.prop(context.scene, "detection_weights_path", text="")
		col.separator()
		col.label(text="Data path:")
		col.prop(context.scene, "detection_data_path", text="")
		col.separator()
		col.label(text="Name path:")
		col.prop(context.scene, "detection_names_path", text="")
		col.separator()

		col.label(text="Network settings")
		col.prop(context.scene, "detection_activation_threshold")
		col.prop(context.scene, "detection_nms")
		col.prop(context.scene, "detection_network_width")
		col.prop(context.scene, "detection_network_height")
		col.separator()

		col.label(text="Processing settings")
		col.prop(context.scene, "detection_enforce_constant_size")
		col.prop(context.scene, "detection_constant_size")
		col.prop(context.scene, "detection_min_size")
		col.separator()

		col.prop(context.scene, "frame_start")
		col.prop(context.scene, "frame_end")
		col.separator()


class OMNITRAX_PT_TrackingPanel(bpy.types.Panel):
	bl_label = "Tracking"
	bl_space_type = "CLIP_EDITOR"
	bl_region_type = "TOOLS"
	bl_category = "OmniTrax"

	# Buffer and Recover settings
	bpy.types.Scene.tracking_dist_thresh = IntProperty(
		name="Distance threshold",
		description="Maximum distance (px) between two detections that can be associated with the same track",
		default=100,
	)
	bpy.types.Scene.tracking_max_frames_to_skip = IntProperty(
		name="Frames skipped before termination",
		description="Maximum number of frames between two detections to be associated with the same track. This is to handle simple buffer and recover tracking as well as termination of tracks of subjects that leave the scene",
		default=10,
	)
	bpy.types.Scene.tracking_min_track_length = IntProperty(
		name="Minimum track length",
		description="Minimum number of tracked frames. If a track has fewer frames, no marker will be created for it and it will not be used for any analysis",
		default=25,
	)

	# Track Display settings
	bpy.types.Scene.tracking_max_trace_length = IntProperty(
		name="Maximum trace length",
		description="Maximum distance number of frames included in displayed trace",
		default=50,
	)
	bpy.types.Scene.tracking_trackIdCount = IntProperty(
		name="Initial tracks",
		description="Number of anticipated individuals in the first frame. Leave at 0, unless to help with tracking expected constant numbers of individuals",
		default=0,
	)

	# Kalman Filter settings
	bpy.types.Scene.tracker_use_KF = BoolProperty(
		name="Use Kalman Filter",
		description="Enables using Kalman filter based motion tracking. Ants don't seem to care about definable motion models, so I won't guarantee this improves the tracking results.",
		default=False,
	)
	bpy.types.Scene.tracker_std_acc = FloatProperty(
		name="process noise magnitude",
		description="The inherent noise of the detection process. Higher values lead to stronger corrections of the resulting track and predicted next position",
		default=5,
	)
	bpy.types.Scene.tracker_x_std_meas = FloatProperty(
		name="std in x-direction",
		description="standard deviation of the measurement in x-direction",
		default=0.1,
	)
	bpy.types.Scene.tracker_y_std_meas = FloatProperty(
		name="std in y-direction",
		description="standard deviation of the measurement in y-direction",
		default=0.1,
	)
	bpy.types.Scene.tracker_save_video = BoolProperty(
		name="Export tracked video",
		description="Write the video with tracked overlay to the location of the input video",
		default=False,
	)

	def draw(self, context):
		layout = self.layout

		col = layout.column(align=True)
		col.label(text="Buffer and Recover settings")
		col.prop(context.scene, "tracking_dist_thresh")
		col.prop(context.scene, "tracking_max_frames_to_skip")
		col.prop(context.scene, "tracking_min_track_length")
		col.label(text="Track Display settings")
		col.prop(context.scene, "tracking_max_trace_length")
		col.prop(context.scene, "tracking_trackIdCount")

		col.label(text="Kalman Filter settings")
		col.prop(context.scene, "tracker_use_KF")
		col.prop(context.scene, "tracker_std_acc")
		col.prop(context.scene, "tracker_x_std_meas")
		col.prop(context.scene, "tracker_y_std_meas")

		col.separator()

		col.label(text="Run tracking")
		col.prop(context.scene, "tracker_save_video")
		col.operator("scene.detection_run", text="TRACK").restart_tracking = False
		col.operator(
			"scene.detection_run", text="RESTART Tracking"
		).restart_tracking = True


class EXPORT_PT_ManualTrackingPanel(bpy.types.Panel):
	bl_space_type = "CLIP_EDITOR"
	bl_region_type = "TOOLS"
	bl_label = "Manual Tracking"
	bl_category = "OmniTrax"
	bl_options = {"DEFAULT_CLOSED"}

	def draw(self, context):
		layout = self.layout

		col = layout.column(align=True)
		row = col.row(align=True)
		row.label(text="Marker:")
		row.operator("clip.add_marker_at_click", text="Add")
		row.operator("clip.delete_track", text="Delete")
		col.separator()
		sc = context.space_data
		clip = sc.clip

		row = layout.row(align=True)
		row.label(text="Track:")

		props = row.operator("clip.track_markers", text="", icon="FRAME_PREV")
		props.backwards = True
		props.sequence = False
		props = row.operator("clip.track_markers", text="", icon="PLAY_REVERSE")
		props.backwards = True
		props.sequence = True
		props = row.operator("clip.track_markers", text="", icon="PLAY")
		props.backwards = False
		props.sequence = True
		props = row.operator("clip.track_markers", text="", icon="FRAME_NEXT")
		props.backwards = False
		props.sequence = False

		col = layout.column(align=True)
		row = col.row(align=True)
		row.label(text="Clear:")
		row.scale_x = 2.0

		props = row.operator("clip.clear_track_path", text="", icon="BACK")
		props.action = "UPTO"

		props = row.operator("clip.clear_track_path", text="", icon="FORWARD")
		props.action = "REMAINED"

		col = layout.column()
		row = col.row(align=True)
		row.label(text="Refine:")
		row.scale_x = 2.0

		props = row.operator("clip.refine_markers", text="", icon="LOOP_BACK")
		props.backwards = True

		props = row.operator("clip.refine_markers", text="", icon="LOOP_FORWARDS")
		props.backwards = False

		col = layout.column(align=True)
		row = col.row(align=True)
		row.label(text="Merge:")
		row.operator("clip.join_tracks", text="Join Tracks")


class EXPORT_PT_DataPanel(bpy.types.Panel):
	bl_label = "Track Export"
	bl_space_type = "CLIP_EDITOR"
	bl_region_type = "TOOLS"
	bl_category = "OmniTrax"
	bl_options = {"DEFAULT_CLOSED"}

	bpy.types.Scene.exp_path = StringProperty(
		name="Export Path",
		description="Path where data will be exported to",
		default="\\export",
		subtype="DIR_PATH",
	)
	bpy.types.Scene.exp_subdirs = BoolProperty(
		name="Export Subdirectories",
		description="Markers will be exported to subdirectories",
		default=False,
	)
	bpy.types.Scene.exp_logfile = BoolProperty(
		name="Write Logfile",
		description="Write logfile into export folder",
		default=False,
	)

	def draw(self, context):
		layout = self.layout

		col = layout.column(align=True)
		col.label(text="Export Path:")
		col.prop(context.scene, "exp_path", text="")
		col.separator()
		col.prop(context.scene, "exp_subdirs")
		col.prop(context.scene, "exp_logfile")
		col.separator()
		col.prop(context.scene, "frame_start")
		col.prop(context.scene, "frame_end")
		col.separator()
		col.label(text="Export:")
		row = col.row(align=True)
		row.operator("scene.export_marker", text="Selected").selected_only = True
		row.operator("scene.export_marker", text="All")


class EXPORT_PT_AdvancedSampleExportPanel(bpy.types.Panel):
	bl_label = "Advanced Sample Export"
	bl_space_type = "CLIP_EDITOR"
	bl_region_type = "TOOLS"
	bl_category = "OmniTrax"
	bl_options = {"DEFAULT_CLOSED"}

	# Input dimensions
	bpy.types.Scene.exp_ase_fixed_input_bounding_box_size = BoolProperty(
		name="Fixed input bounding box size",
		description="Use constant sized bounding boxes, overwriting the original marker shape and dimensions",
		default=True,
	)
	bpy.types.Scene.exp_ase_input_x = IntProperty(
		name="X (px)",
		description="Constant sized bounding box X dimension in pixels.",
		default=128,
	)
	bpy.types.Scene.exp_ase_input_y = IntProperty(
		name="Y (px)",
		description="Constant sized bounding box Y dimension in pixels.",
		default=128,
	)

	# Output dimensions
	bpy.types.Scene.exp_ase_fixed_output_bounding_box_size = BoolProperty(
		name="Fixed output patch size",
		description="Fixed dimensions of exported samples. Unless 'padding' is enabled, this option may change the "
		"aspect ratio of the extracted samples.",
		default=True,
	)
	bpy.types.Scene.exp_ase_output_x = IntProperty(
		name="X (px)", description="X dimension of exported samples.", default=128
	)
	bpy.types.Scene.exp_ase_output_y = IntProperty(
		name="Y (px)", description="Y dimension of exported samples.", default=128
	)

	# Optional settings
	bpy.types.Scene.exp_ase_export_every_nth_frame = IntProperty(
		name="Export every nth frame",
		description="Export only every nth frame, skipping intermediate frames",
		default=1,
		min=1,
	)

	bpy.types.Scene.exp_ase_padding = BoolProperty(
		name="Use padding",
		description="Use padding for images to preserve the original aspect ratio and produce mxm square patches",
		default=True,
	)

	bpy.types.Scene.exp_ase_grayscale = BoolProperty(
		name="Convert to grayscale",
		description="Export image samples in black and white with image dimensions as (X,Y,1)",
		default=False,
	)

	sample_formats = [
		(".jpg", ".jpg", "Use JPG as the sample output format"),
		(".png", ".png", "Use PNG as the sample output format"),
	]

	bpy.types.Scene.exp_ase_sample_format = EnumProperty(
		name="Sample format",
		description="Set the sample output format",
		items=sample_formats,
		default=sample_formats[0][0],
	)

	bpy.types.Scene.exp_ase_path = StringProperty(
		name="Sample Export Path",
		description="Path to which all samples will be exported",
		default="C:\\",
		subtype="DIR_PATH",
	)

	def draw(self, context):
		layout = self.layout

		col = layout.column(align=True)
		col.label(text="Export image samples from tracks")
		col.separator()

		col.label(text="Input settings")
		col.prop(context.scene, "exp_ase_fixed_input_bounding_box_size")
		row = col.row(align=True)
		row.label(text="Input dimensions")
		row.prop(context.scene, "exp_ase_input_x")
		row.prop(context.scene, "exp_ase_input_y")
		col.separator()

		col.label(text="Output settings")
		col.prop(context.scene, "exp_ase_fixed_output_bounding_box_size")
		row = col.row(align=True)
		row.label(text="Output dimensions")
		row.prop(context.scene, "exp_ase_output_x")
		row.prop(context.scene, "exp_ase_output_y")
		col.separator()

		col.label(text="Optional settings")
		col.prop(context.scene, "exp_ase_export_every_nth_frame")
		col.prop(context.scene, "exp_ase_sample_format")
		col.prop(context.scene, "exp_ase_padding")
		col.prop(context.scene, "exp_ase_grayscale")
		col.separator()

		col.label(text="Sample export path:")
		col.prop(context.scene, "exp_ase_path", text="")
		col.operator("scene.advanced_sample_export", text="Export samples")


def register():
	bpy.utils.register_class(OMNITRAX_PT_DetectionPanel)
	bpy.utils.register_class(OMNITRAX_PT_TrackingPanel)
	bpy.utils.register_class(EXPORT_PT_ManualTrackingPanel)
	bpy.utils.register_class(EXPORT_PT_DataPanel)
	bpy.utils.register_class(EXPORT_PT_AdvancedSampleExportPanel)


def unregister():
	bpy.utils.unregister_class(OMNITRAX_PT_DetectionPanel)
	bpy.utils.unregister_class(OMNITRAX_PT_TrackingPanel)
	bpy.utils.unregister_class(EXPORT_PT_ManualTrackingPanel)
	bpy.utils.unregister_class(EXPORT_PT_DataPanel)
	bpy.utils.unregister_class(EXPORT_PT_AdvancedSampleExportPanel)
