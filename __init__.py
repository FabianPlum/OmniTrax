import bpy
from bpy.props import BoolProperty as BoolProperty
from bpy.props import StringProperty as StringProperty
from bpy.props import IntProperty as IntProperty
from bpy.props import FloatProperty as FloatProperty

# check for installed packages and if they are missing, install them now
from omni_trax import check_packages

import numpy as np
import cv2
import csv
import os
import math
import time
import random
import sys
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from operator import itemgetter
from ctypes import *

# load darknet with compiled DLLs for windows from respective path
from omni_trax.build.darknet.x64 import darknet

os.add_dll_directory("${omni_trax/build/darknet/x64}")
# now for the file management fucntions
from omni_trax.OmniTrax_utils import import_tracks, display_video, get_exact_frame, extractPatches, display_patches, \
    sortByDistance

# kalman imports
import copy
from omni_trax.tracker import Tracker

bl_info = {
    "name": "OmniTrax",
    "author": "Fabian Plum",
    "description": "Using some deep learning to do fancy multi-animal tracking",
    "blender": (2, 92, 0),
    "version": (0, 0, 1),
    "location": "",
    "warning": "THIS ADDON WILL INSTALL 3RD PARTY LIBRARIES AND CAUSE SEVERE HEADACHES",
    "category": "motion capture"
}


class OMNITRAX_OT_DetectionOperator(bpy.types.Operator):
    """Run the detection based tracking pipeline according to the above defined parameters"""
    bl_idname = "scene.detection_run"
    bl_label = "Run Detection"

    def execute(self, context):
        """
        Tracker settings
        """
        # Variables initialization
        track_colors = {}
        np.random.seed(0)

        """
        Detector Settings
        """

        global netMain
        global metaMain
        global altNames
        global track_classes

        if "netMain" in globals():
            print("Initialised network found!")
        else:
            print("Initialising network...")
            netMain = None
            metaMain = None
            altNames = None

        video = bpy.path.abspath(bpy.context.edit_movieclip.filepath)

        # enter the number of annotated frames:
        tracked_frames = context.scene.frame_end

        # now we can load the captured video file and display it
        cap = cv2.VideoCapture(video)

        # get the fps of the clip and set the environment accordingly
        fps = cap.get(cv2.CAP_PROP_FPS)
        bpy.context.scene.render.fps = fps
        bpy.context.scene.render.fps_base = fps

        # Create Object Tracker
        tracker = Tracker(dist_thresh=context.scene.tracking_dist_thresh,
                          max_frames_to_skip=context.scene.tracking_max_frames_to_skip,
                          max_trace_length=context.scene.tracking_max_trace_length,
                          trackIdCount=context.scene.tracking_trackIdCount,
                          use_kf=context.scene.tracker_use_KF,
                          std_acc=context.scene.tracker_std_acc,
                          x_std_meas=context.scene.tracker_x_std_meas,
                          y_std_meas=context.scene.tracker_y_std_meas,
                          dt=1 / fps)

        # and produce an output file    
        video_output = bpy.path.abspath(bpy.context.edit_movieclip.filepath)[:-4] + "_online_tracking.avi"
        video_out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                    (int(cap.get(3)), int(cap.get(4))))

        # check the number of frames of the imported video file
        numFramesMax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("The imported clip:", video, "has a total of", numFramesMax, "frames.\n")

        # load configuration and weights (synthetic)
        yolo_cfg = bpy.path.abspath(context.scene.detection_config_path)
        yolo_weights = bpy.path.abspath(context.scene.detection_weights_path)
        yolo_data = bpy.path.abspath(context.scene.detection_data_path)
        yolo_names = bpy.path.abspath(context.scene.detection_names_path)

        # read obj.names file to create custom colours for each class
        class_names = []
        with open(yolo_names, "r") as yn:
            for line in yn:
                class_names.append(line.strip())

        class_colours = {}  # from green to red, low class to high class (light to heavy)
        class_id = {}
        if len(class_names) == 1:
            class_colours[class_names[0]] = [120, 0, 120]
            class_id[class_names[0]] = 0
        else:
            for c, cn in enumerate(class_names):
                class_colours[cn] = [int((255 / len(class_names)) * (c + 1)), int(255 - (255 / len(class_names)) * c),
                                     0]
                class_id[cn] = c

        # overwrite network dimensions
        # TODO

        configPath = yolo_cfg
        weightPath = yolo_weights
        metaPath = yolo_data

        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath) + "`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath) + "`")
        if netMain is None:
            netMain = darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if metaMain is None:
            metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass
        # cap = cv2.VideoCapture(0)

        print("Starting the YOLO loop...")

        # Create an image we reuse for each detect
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # darknet_image = darknet.make_image(darknet.network_width(netMain),
        #                                darknet.network_height(netMain),3)
        darknet_image = darknet.make_image(video_width, video_height, 3)

        all_detection_centres = []

        frame_counter = 0

        if context.scene.frame_start == 0:
            context.scene.frame_start = 1

        cap.set(1, context.scene.frame_start - 1)

        # build array for all tracks and classes     
        track_classes = {}

        while True:
            if frame_counter == context.scene.frame_end + 1 - context.scene.frame_start:
                break

            prev_time = time.time()
            ret, frame_read = cap.read()
            clip = context.edit_movieclip
            clip_width = clip.size[0]
            clip_height = clip.size[1]
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (video_width,
                                        video_height),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            # thresh : detection threshold -> lower = more sensitive (higher recall)
            # nms : non maximum suppression -> higher = allow for closer proximity between detections
            detections = darknet.detect_image(netMain, metaMain, darknet_image,
                                              thresh=bpy.context.scene.detection_activation_threshold,
                                              nms=bpy.context.scene.detection_nms)
            print("Frame:", frame_counter + 1)
            viable_detections = []
            centres = []
            predicted_classes = []

            bpy.context.scene.frame_current = frame_counter + 1

            for detection in detections:
                if detection[2][2] >= bpy.context.scene.detection_min_size and detection[2][
                    3] >= bpy.context.scene.detection_min_size:
                    predicted_classes.append(str(detection[0]).split("'")[1])

                    # we need to scale the detections to the original imagesize, as they are downsampled above
                    scaled_xy = scale_detections(x=detection[2][0], y=detection[2][1],
                                                 network_w=darknet.network_width(netMain),
                                                 network_h=darknet.network_height(netMain),
                                                 output_w=frame_rgb.shape[1], output_h=frame_rgb.shape[0])
                    viable_detections.append(scaled_xy)

                    # kalman stuff here
                    centres.append(np.round(np.array([[detection[2][0]], [detection[2][1]]])))

            all_detection_centres.append(viable_detections)

            if bpy.context.scene.detection_enforce_constant_size:
                image = cvDrawBoxes(detections, frame_resized, min_size=bpy.context.scene.detection_min_size,
                                    constant_size=bpy.context.scene.detection_constant_size,
                                    class_colours=class_colours)
            else:
                image = cvDrawBoxes(detections, frame_resized, min_size=bpy.context.scene.detection_min_size,
                                    class_colours=class_colours)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(1/(time.time()-prev_time))

            # before we show stuff, let's add some tracking fun
            # SO, if animals are detected then track them

            if (len(centres) > 0):

                # Track object using Kalman Filter
                tracker.Update(centres, predicted_classes=predicted_classes)

                # For identified object tracks draw tracking line
                # Use various colors to indicate different track_id
                for i in range(len(tracker.tracks)):
                    if (len(tracker.tracks[i].trace) > 1):
                        mname = "track_" + str(tracker.tracks[i].track_id)

                        # record the predicted class at each increment for every track
                        if mname in track_classes:
                            track_classes[mname][bpy.context.scene.frame_current] = tracker.tracks[i].predicted_class[
                                -1]
                        else:
                            track_classes[mname] = {
                                bpy.context.scene.frame_current: tracker.tracks[i].predicted_class[-1]}

                        if mname not in track_colors:
                            track_colors[mname] = np.random.randint(low=100, high=255, size=3).tolist()

                        # draw direction of movement onto footage
                        x_t, y_t = tracker.tracks[i].trace[-1]
                        tracker_velocity = 5 * (tracker.tracks[i].trace[-1] - tracker.tracks[i].trace[-2])
                        x_t_future, y_t_future = tracker.tracks[i].trace[-1] + tracker_velocity
                        cv2.arrowedLine(image, (int(x_t), int(y_t)), (int(x_t_future), int(y_t_future)),
                                        (np.array(track_colors[mname]) - np.array([70, 70, 70])).tolist(), 3,
                                        tipLength=0.75)

                        for j in range(len(tracker.tracks[i].trace) - 1):
                            hind_sight_frame = bpy.context.scene.frame_current - len(tracker.tracks[i].trace) + j
                            # Draw trace line on preview                         
                            x1 = tracker.tracks[i].trace[j][0][0]
                            y1 = tracker.tracks[i].trace[j][1][0]
                            x2 = tracker.tracks[i].trace[j + 1][0][0]
                            y2 = tracker.tracks[i].trace[j + 1][1][0]
                            if mname not in track_colors:
                                track_colors[mname] = np.random.randint(low=100, high=255, size=3).tolist()
                            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                                     track_colors[mname], 2)

                            if x1 != 0 and x2 != 0 and len(
                                    tracker.tracks[i].trace) > context.scene.tracking_max_frames_to_skip:
                                # write track to clip markers
                                if mname in clip.tracking.objects[0].tracks:
                                    # if the corresponding marker exists, update it's position on the current frame
                                    # remember to invert the y axis, because things are never easy or correct by default
                                    clip.tracking.objects[0].tracks[mname].markers.insert_frame(hind_sight_frame,
                                                                                                co=(x2 / clip_width,
                                                                                                    1 - y2 / clip_height))
                                else:
                                    # add new tracks to the set of markers
                                    bpy.ops.clip.add_marker(location=(x2 / clip_width, 1 - y2 / clip_height))
                                    clip.tracking.tracks.active.name = mname

                        cv2.putText(image,
                                    mname,
                                    (int(x1) - int(bpy.context.scene.detection_constant_size / 2),
                                     int(y1) - bpy.context.scene.detection_constant_size), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    track_colors[mname], 2)

            video_out.write(image)
            cv2.imshow('Detections on video', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_counter += 1

        cv2.destroyAllWindows()
        # always reset frame from capture at the end to avoid incorrect skips during access
        cap.set(1, context.scene.frame_start - 1)
        cap.release()
        video_out.release()

        return {"FINISHED"}


class EXPORT_OT_Operator(bpy.types.Operator):
    """Export the motion tracking data according to the settings."""
    bl_idname = "scene.export_marker"
    bl_label = "Export Tracking Markers"

    selected_only: BoolProperty(
        name="Selected Only",
        description="Export selected markers only",
        default=False)

    def execute(self, context):
        log = context.scene.exp_logfile
        path = bpy.path.abspath(context.scene.exp_path)
        subdirs = context.scene.exp_subdirs
        f_start = context.scene.frame_start
        f_end = context.scene.frame_end

        if not os.path.exists(path): os.makedirs(path)

        import time
        time_start = time.time()

        if log:
            log_file = open(path + "log.txt", "w")
            log_file.write("Starting Export\n")
            log_file.write("Export path: {0}\n".format(path))
            log_file.write("Exporting from scene {0}\n".format(context.scene.name))
            log_file.write("Exporting from frame {0} to {1}\n".format(f_start, f_end))
            log_file.write("-----------------------------------------------------------\n\n")

        movieclips = []
        if self.selected_only:
            sc = context.space_data
            movieclips.append(sc.clip)
        else:
            movieclips = bpy.data.movieclips

        for clip in movieclips:
            x_size = clip.size[0]
            y_size = clip.size[1]

            if log:
                log_file.write("Starting movieclip {0} ({1} x {2} pixels)\n".format(clip.name, x_size, y_size))

            if self.selected_only:
                tracks = [track for track in clip.tracking.tracks if track.select]
            else:
                tracks = clip.tracking.tracks

            for track in tracks:
                # get the predicted classes for each track at every frame
                single_track_classes = track_classes[track.name]

                if log:
                    log_file.write("  Track {0} started ...\n".format(track.name))

                if not subdirs:
                    export_file = open(path + "{0}_{1}.csv".format(clip.name.split(".")[0], track.name), 'w')
                else:
                    subpath = path + "\\{0}\\".format(clip.name[:-3])
                    try:
                        if not os.path.exists(subpath):
                            os.makedirs(subpath)
                    except:
                        log_file.write(str(subpath), "already exists.")
                    export_file = open(subpath + "{0}_{1}.csv".format(clip.name.split(".")[0], track.name), 'w')

                export_file.write("frame;x;y;class\n")
                success = True
                i = f_start
                while i <= f_end:
                    if i in single_track_classes:
                        class_pred = single_track_classes[i]
                    else:
                        class_pred = ""
                    marker = track.markers.find_frame(i)
                    if marker:
                        marker_x = round(marker.co.x * x_size)
                        marker_y = round(marker.co.y * y_size)
                        export_file.write("{0};{1};{2};{3}\n".format(i, marker_x, marker_y, class_pred))
                    else:
                        if log:
                            log_file.write("    Missing marker at frame {0}.\n".format(i))
                        success = False
                    i += 1

                export_file.close()
                if log:
                    log_file.write("  Finished Track {0} {1}...\n".format(track.name,
                                                                          "successfully" if success else "with errors"))

            if log:
                log_file.write("Finished movieclip {0} in {1:.4f} s\n\n".format(clip.name, time.time() - time_start))

        if log:
            log_file.write("-----------------------------------------------------------\n")
            log_file.write("Export finished ({0:.4f} s)".format(time.time() - time_start))
            log_file.close()

        self.report({'INFO'}, "Export done ({0:.4f} s)".format(time.time() - time_start))

        if len(movieclips) == 0:
            self.report({'INFO'}, "No clip opened...")

        return {"FINISHED"}


class OMNITRAX_PT_DetectionPanel(bpy.types.Panel):
    bl_label = "Detection (YOLO)"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "OmniTrax"

    bpy.types.Scene.detection_config_path = StringProperty(
        name="Config path",
        description="Path to your YOLO network config file",
        default="",  # bpy.data.filepath.rpartition("\\")[0],
        subtype='FILE_PATH')
    bpy.types.Scene.detection_weights_path = StringProperty(
        name="Weights path",
        description="Path to your YOLO network weights file",
        default="",  # bpy.data.filepath.rpartition("\\")[0],
        subtype='FILE_PATH')
    bpy.types.Scene.detection_data_path = StringProperty(
        name="Data path",
        description="Path to your YOLO network data file",
        default="",  # bpy.data.filepath.rpartition("\\")[0],
        subtype='FILE_PATH')
    bpy.types.Scene.detection_names_path = StringProperty(
        name="Names path",
        description="Path to your YOLO network names file",
        default="",  # bpy.data.filepath.rpartition("\\")[0],
        subtype='FILE_PATH')
    bpy.types.Scene.detection_enforce_constant_size = BoolProperty(
        name="constant detection sizes",
        description="Check to enforce constant detection sizes. This DOES NOT affect the actual inference, only the resulting regions of interest.",
        default=False)
    bpy.types.Scene.detection_constant_size = IntProperty(
        name="constant detection sizes (px)",
        description="Constant detection size in pixels. All detections will be rescaled to this value in both width and height.",
        default=40)
    bpy.types.Scene.detection_min_size = IntProperty(
        name="minimum detection sizes (px)",
        description="If the width or height of a detection is below this threshold, it will be discarded. This can be useful to decrease noise. Keep the value at 0 if this is not needed.",
        default=15)
    bpy.types.Scene.detection_activation_threshold = FloatProperty(
        name="Confidence threshold",
        description="Detection confidence threshold. A higher confidence leads to fewer false positives but more missed detections.",
        default=0.25)
    bpy.types.Scene.detection_nms = FloatProperty(
        name="Non-maximum suppression",
        description="Non-maximum suppression (NMS) refers to the maximum overlap allowed between proposed bounding boxes. E.g., a value of 0.45 corresponds to a maximum overlap of 45% between two compared bounding boxes to be retained simultaneously. In case the overlap is larger, the box with the lower objectness score or classification confidence will be 'suppressed', thus, only the highest confidence prediction is returned.",
        default=0.45)
    bpy.types.Scene.detection_network_size = IntProperty(
        name="Detection network size",
        description="Height and Width of the loaded detection network (MUST be a multiple of 32). Larger network sizes allow for smaller invididuals to be detected at the cost of inference speed.",
        default=640)

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
        col.prop(context.scene, "detection_network_size")
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
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "OmniTrax"

    # keep parameters simple for now
    # dist_thresh=100, max_frames_to_skip=10, max_trace_length=50, trackIdCount=0

    # Buffer and Recover settings
    bpy.types.Scene.tracking_dist_thresh = IntProperty(
        name="Distance threshold",
        description="Maximum distance (px) between two detections that can be associated with the same track",
        default=100)
    bpy.types.Scene.tracking_max_frames_to_skip = IntProperty(
        name="Frames skipped before termination",
        description="Maximum number of frames between two detections to be associated with the same track. This is to handle simple buffer and recover tracking as well as termination of tracks of subjects that leave the scene",
        default=10)

    # Track Display settings    
    bpy.types.Scene.tracking_max_trace_length = IntProperty(
        name="Maximum trace length",
        description="Maximum distance number of frames included in displayed trace",
        default=50)
    bpy.types.Scene.tracking_trackIdCount = IntProperty(
        name="Initial tracks",
        description="Number of anticipated individuals in the first frame. Leave at 0, unless to help with tracking expected constant numbers of individuals",
        default=0)

    # Kalman Filter settings    
    bpy.types.Scene.tracker_use_KF = BoolProperty(
        name="Use Kalman Filter",
        description="Enables using Kalman filter based motion tracking. Ants don't seem to care about definable motion models, so I won't guarantee this improves the tracking results.",
        default=False)
    bpy.types.Scene.tracker_std_acc = FloatProperty(
        name="process noise magnitude",
        description="The inherent noise of the detection process. Higher values lead to stronger corrections of the resulting track and predicted next position",
        default=5)
    bpy.types.Scene.tracker_x_std_meas = FloatProperty(
        name="std in x-direction",
        description="standard deviation of the measurement in x-direction",
        default=0.1)
    bpy.types.Scene.tracker_y_std_meas = FloatProperty(
        name="std in y-direction",
        description="standard deviation of the measurement in y-direction",
        default=0.1)

    def draw(self, context):
        layout = self.layout

        col = layout.column(align=True)
        col.label(text="Buffer and Recover settings")
        col.prop(context.scene, "tracking_dist_thresh")
        col.prop(context.scene, "tracking_max_frames_to_skip")
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
        col.operator("scene.detection_run", text="TRACK")


class EXPORT_PT_TrackingPanel(bpy.types.Panel):
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_label = "Manual Tracking"
    bl_category = "OmniTrax"

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

        props = row.operator("clip.track_markers", text="", icon='FRAME_PREV')
        props.backwards = True
        props.sequence = False
        props = row.operator("clip.track_markers", text="",
                             icon='PLAY_REVERSE')
        props.backwards = True
        props.sequence = True
        props = row.operator("clip.track_markers", text="", icon='PLAY')
        props.backwards = False
        props.sequence = True
        props = row.operator("clip.track_markers", text="", icon='FRAME_NEXT')
        props.backwards = False
        props.sequence = False

        col = layout.column(align=True)
        row = col.row(align=True)
        row.label(text="Clear:")
        row.scale_x = 2.0

        props = row.operator("clip.clear_track_path", text="", icon='BACK')
        props.action = 'UPTO'

        props = row.operator("clip.clear_track_path", text="", icon='FORWARD')
        props.action = 'REMAINED'

        col = layout.column()
        row = col.row(align=True)
        row.label(text="Refine:")
        row.scale_x = 2.0

        props = row.operator("clip.refine_markers", text="", icon='LOOP_BACK')
        props.backwards = True

        props = row.operator("clip.refine_markers", text="", icon='LOOP_FORWARDS')
        props.backwards = False

        col = layout.column(align=True)
        row = col.row(align=True)
        row.label(text="Merge:")
        row.operator("clip.join_tracks", text="Join Tracks")


class EXPORT_PT_DataPanel(bpy.types.Panel):
    bl_label = "Export"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "OmniTrax"

    # In andere Klasse kopieren
    bl_options = {'DEFAULT_CLOSED'}

    bpy.types.Scene.exp_path = StringProperty(
        name="Export Path",
        description="Path where data will be exported to",
        default="\\export",
        subtype='DIR_PATH')
    bpy.types.Scene.exp_subdirs = BoolProperty(
        name="Export Subdirectories",
        description="Markers will be exported to subdirectories",
        default=False)
    bpy.types.Scene.exp_logfile = BoolProperty(
        name="Write Logfile",
        description="Write logfile into export folder",
        default=False)

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


### Various Detection Processing functions ###

def scale_detections(x, y, network_w, network_h, output_w, output_h):
    scaled_x = x * (output_w / network_w)
    scaled_y = (network_h - y) * (output_h / network_h)  # y is inverted
    return [scaled_x, scaled_y]


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, min_size=20, constant_size=False, class_colours=None):
    for detection in detections:

        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]

        if w >= min_size and h >= min_size:

            if constant_size:
                w, h = constant_size, constant_size

            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cl_colour = class_colours[str(detection[0]).split("'")[1]]
            cv2.rectangle(img, pt1, pt2, (cl_colour[0], cl_colour[1], cl_colour[2]), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        cl_colour, 1)
    return img


def nonMaximumSupression(detections):
    """
    :param detections: detections returned from darknet
    :return: only detection of highest confidence. Return None, if no individual was detected
    """
    if len(detections) != 0:
        det_sorted = sorted(detections, key=itemgetter(2))
        max_conf_detection = det_sorted[0][0]
    else:
        max_conf_detection = 'No Detect'
    return max_conf_detection


### (un)register module ###
def register():
    bpy.utils.register_class(OMNITRAX_OT_DetectionOperator)
    bpy.utils.register_class(OMNITRAX_PT_DetectionPanel)
    bpy.utils.register_class(OMNITRAX_PT_TrackingPanel)
    bpy.utils.register_class(EXPORT_OT_Operator)
    bpy.utils.register_class(EXPORT_PT_TrackingPanel)
    bpy.utils.register_class(EXPORT_PT_DataPanel)


def unregister():
    bpy.utils.unregister_class(OMNITRAX_OT_DetectionOperator)
    bpy.utils.unregister_class(OMNITRAX_PT_DetectionPanel)
    bpy.utils.unregister_class(OMNITRAX_PT_TrackingPanel)
    bpy.utils.unregister_class(EXPORT_OT_Operator)
    bpy.utils.unregister_class(EXPORT_PT_TrackingPanel)
    bpy.utils.unregister_class(EXPORT_PT_DataPanel)


if __name__ == "__main__":
    register()
