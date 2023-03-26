import bpy
from bpy.props import BoolProperty as BoolProperty
from bpy.props import StringProperty as StringProperty
from bpy.props import IntProperty as IntProperty
from bpy.props import FloatProperty as FloatProperty
from bpy.props import EnumProperty as EnumProperty

# check for installed packages and if they are missing, install them now
# after the initial installation, dependency checks will be disabled
# in setup_state.txt set setup_completed to "False" to trigger a system check / reinstall when enabling omni_trax
from omni_trax import check_packages

import numpy as np
import cv2
import os
import time
import yaml
import tensorflow as tf

from omni_trax.omni_trax_utils import scale_detections, convertBack, cvDrawBoxes, setInferenceDevive

# kalman imports
from omni_trax.tracker import Tracker

# YOLO handler
from omni_trax.YOLOader import YOLOader

bl_info = {
    "name": "omni_trax",
    "author": "Fabian Plum",
    "description": "Deep learning-based multi animal tracker",
    "blender": (3, 3, 0),
    "version": (0, 2, 1),
    "location": "",
    "warning": "RUN IN ADMINISTRATOR MODE DURING INSTALLATION!",
    "category": "motion capture"
}


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
        print("Found computational devices:\n", physical_devices)

        devices = []
        for d, device in enumerate(physical_devices):
            if device.device_type == "GPU":
                devices.append(("GPU_" + str(d - 1), "GPU_" + str(d - 1), "Use GPU for inference (requires CUDA)"))
            else:
                devices.append(("CPU_" + str(d), "CPU", "Use CPU for inference"))
    except:
        print("INFO: Did not find suitable inference devices.")

    bpy.types.Scene.compute_device = EnumProperty(
        name="",
        description="Select inference device",
        items=devices,
        default=devices[-1][0])

    def draw(self, context):
        layout = self.layout

        col = layout.column(align=True)
        col.label(text="Select computational device")
        col.separator()
        row = col.row(align=True)
        row.prop(context.scene, "compute_device")


class OMNITRAX_OT_DetectionOperator(bpy.types.Operator):
    """
    Run the (yolo) detection based buffer-and-recover tracking pipeline
    RESTART Track: Begin tracking from defined Start-Frame, overwrite prior tracks if present
    TRACK: Resume tracking from previous state
    """
    bl_idname = "scene.detection_run"
    bl_label = "Run Detection"

    restart_tracking: BoolProperty(
        name="Restart Tracking",
        description="Re-initialises tracker with new settings. WARNING: Current identities will NOT be retained!",
        default=False)

    def execute(self, context):
        """
        Check which compute device is selected and set it as active
        """
        setInferenceDevive(context.scene.compute_device)

        # load darknet with compiled DLLs for windows for either GPU or CPU inference from respective path
        if context.scene.compute_device.split("_")[0] == "GPU":
            from omni_trax.darknet import darknet as darknet
            darknet.set_compute_device(int(context.scene.compute_device.split("_")[1]))
        else:
            from omni_trax.darknet import darknet_cpu as darknet

        """
        Tracker settings
        """
        # Variables initialization
        track_colors = {}
        np.random.seed(0)

        """
        Detector Settings
        """

        global network
        global altNames
        global class_names
        global class_colours
        global track_classes
        global tracks_dict
        global tracker_KF

        if "network" in globals():
            print("\nINFO: Initialised darknet network found!\n")
        else:
            print("\nINFO: Initialising darkent network...\n")
            network = None
            altNames = None

        clip = context.edit_movieclip
        video = bpy.path.abspath(clip.filepath)

        # enter the number of annotated frames:
        tracked_frames = context.scene.frame_end

        # now we can load the captured video file and display it
        cap = cv2.VideoCapture(video)

        # get the fps of the clip and set the environment accordingly
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # if defined, extract the mask and apply it to the imported footage
        try:
            masks = []

            for m, mask in enumerate(bpy.data.masks["Mask"].layers["MaskLayer"].splines):
                print(mask)
                mask_individual = []
                for point in bpy.data.masks["Mask"].layers["MaskLayer"].splines[m].points:
                    print(point.co[0], point.co[1])
                    mask_individual.append([point.co[0], point.co[1]])
                masks.append(mask_individual)

            print("\n", masks)

            img_mask = np.zeros((video_height, video_width, 1), np.uint8)
            for mask in masks:
                contours = np.array(
                    [[max(min(int(max((video_height - video_width), 0) / 2 +
                                  x * video_width), video_width), 0),
                      max(min(int(max((video_width - video_height), 0) / 2 +
                                  video_height - y * video_height * (video_width / video_height)), video_height), 0)]
                     for [x, y] in mask])
                print(contours)
                cv2.fillPoly(img_mask, pts=[contours], color=(1, 1, 1))

            # binarise mask
            ret_mask, img_mask = cv2.threshold(img_mask, 0, 1, cv2.THRESH_BINARY)

            # DEBUG
            # cv2.imshow("image mask", img_mask)
            # cv2.waitKey(0)

        except Exception as e:
            print(e)
            print("No mask found!")
            img_mask = None

            return {"FINISHED"}

        try:
            bpy.context.scene.render.fps = fps
            bpy.context.scene.render.fps_base = fps
        except TypeError:
            bpy.context.scene.render.fps = int(fps)
            bpy.context.scene.render.fps_base = int(fps)

        # Create Object Tracker
        tracker_KF = Tracker(dist_thresh=context.scene.tracking_dist_thresh,
                             max_frames_to_skip=context.scene.tracking_max_frames_to_skip,
                             max_trace_length=context.scene.tracking_max_trace_length,
                             trackIdCount=context.scene.tracking_trackIdCount,
                             use_kf=context.scene.tracker_use_KF,
                             std_acc=context.scene.tracker_std_acc,
                             x_std_meas=context.scene.tracker_x_std_meas,
                             y_std_meas=context.scene.tracker_y_std_meas,
                             dt=1 / fps)

        if self.restart_tracking:
            tracker_continue = False
        else:
            # remove previous tracks in tracker_KF and initialise from current state
            tracker_KF.clear_tracks()
            tracker_continue = True
            # remove markers from frames past current frame, then get the current state
            # check for the latest ID and begin counting new tracks from there
            latest_id = 0
            for mname in clip.tracking.objects[0].tracks:
                mname.select = True
                marker = mname.markers.find_frame(bpy.context.scene.frame_current)
                track_id_temp = mname.name.split("_")[-1]
                latest_id = max(latest_id, int(track_id_temp))
                try:
                    marker_x = float(marker.co.x * video_width)
                    marker_y = float(marker.co.y * video_height)
                    print(mname.name, ": ", marker_x, marker_y)
                    bpy.ops.clip.clear_track_path(action="REMAINED")
                    mname.select = False

                    # initialise new track from marker state
                    tracker_KF.initialise_from_prior_state(prior_state=[track_id_temp,
                                                                        marker_x,
                                                                        video_height - marker_y,
                                                                        "",
                                                                        [0, 0, 0, 0]])
                except AttributeError:
                    print(mname.name, "not present at current frame!")

            # the latest track (the one with the highest ID) informs the tracker from where to continue counting
            print("Beginning counting from ID", latest_id)
            tracker_KF.set_trackIdCount(latest_id)

        print("INITIALISED TRACKER!")

        # and produce an output file
        if context.scene.tracker_save_video:
            video_output = bpy.path.abspath(bpy.context.edit_movieclip.filepath)[:-4] + "_online_tracking.avi"
            video_out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps,
                                        (int(cap.get(3)), int(cap.get(4))))

        # check the number of frames of the imported video file
        numFramesMax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("The imported clip:", video, "has a total of", numFramesMax, "frames.\n")

        # load configuration and weights (synthetic)
        yolo_cfg = bpy.path.abspath(context.scene.detection_config_path)
        yolo_weights = bpy.path.abspath(context.scene.detection_weights_path)
        yolo_data = bpy.path.abspath(context.scene.detection_data_path)
        yolo_names = bpy.path.abspath(context.scene.detection_names_path)

        yolo_paths = YOLOader(cfg=yolo_cfg,
                              weights=yolo_weights,
                              data=yolo_data,
                              names=yolo_names)

        yolo_paths.update_cfg(nw_width=bpy.context.scene.detection_network_width,
                              nw_height=bpy.context.scene.detection_network_height)

        if yolo_paths.names is None:
            yolo_paths.create_names()

        if yolo_paths.data is None:
            yolo_paths.create_data()

        context.scene.detection_config_path = yolo_paths.cfg
        context.scene.detection_data_path = yolo_paths.data
        context.scene.detection_names_path = yolo_paths.names

        # read obj.names file to create custom colours for each class

        class_names = []
        with open(yolo_paths.names, "r") as yn:
            for line in yn:
                class_names.append(line.strip())

        class_colours = {}  # from green to red, low class to high class (light to heavy)
        class_id = {}
        if len(class_names) == 1:
            class_colours[class_names[0]] = [20, 150, 20]
            class_id[class_names[0]] = 0
        else:
            for c, cn in enumerate(class_names):
                class_colours[cn] = [int((255 / len(class_names)) * (c + 1)), int(255 - (255 / len(class_names)) * c),
                                     0]
                class_id[cn] = c

        if not os.path.exists(yolo_paths.cfg):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(yolo_paths.cfg) + "`")
        if not os.path.exists(yolo_paths.weights):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(yolo_paths.weights) + "`")
        if not os.path.exists(yolo_paths.data):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(yolo_paths.data) + "`")
        if network is None:
            network, class_names, _ = darknet.load_network(yolo_paths.cfg,
                                                           yolo_paths.data,
                                                           yolo_paths.weights,
                                                           batch_size=1)
        if altNames is None:
            try:
                with open(yolo_paths.data) as metaFH:
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

        print("Starting the YOLO loop...")

        # Create an image we reuse for each detect
        darknet_image = darknet.make_image(video_width, video_height, 3)

        all_detection_centres = []
        all_track_results = []  # holds all temporary tracks and writes them to scene markers

        frame_counter = 0

        if context.scene.frame_start == 0:
            context.scene.frame_start = 1

        if tracker_continue:
            # track from current frame when continuing, otherwise begin at the start frame
            cap.set(1, context.scene.frame_current)
        else:
            cap.set(1, context.scene.frame_start - 1)

        # build array for all tracks and classes
        track_classes = {}

        ROI_size = int(context.scene.detection_constant_size / 2)

        executed_from_frame = bpy.context.scene.frame_current

        while True:
            if tracker_continue:
                if frame_counter == context.scene.frame_end + 3 - executed_from_frame:
                    bpy.context.scene.frame_current = context.scene.frame_end
                    break
            elif frame_counter == context.scene.frame_end + 3 - context.scene.frame_start:
                bpy.context.scene.frame_current = context.scene.frame_end
                break

            prev_time = time.time()
            ret, frame_read = cap.read()
            if not ret:
                break

            clip_width = frame_read.shape[1]
            clip_height = frame_read.shape[0]
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

            # apply mask, if defined
            if img_mask is not None:
                frame_rgb = cv2.bitwise_and(frame_rgb, frame_rgb, mask=img_mask)

            frame_resized = cv2.resize(frame_rgb,
                                       (video_width,
                                        video_height),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            # thresh : detection threshold -> lower = more sensitive (higher recall)
            # nms : non maximum suppression -> higher = allow for closer proximity between detections
            detections = darknet.detect_image(network, class_names, darknet_image,
                                              thresh=bpy.context.scene.detection_activation_threshold,
                                              nms=bpy.context.scene.detection_nms)
            print("Frame:", frame_counter + 1)
            viable_detections = []
            centres = []
            bounding_boxes = []
            predicted_classes = []

            if tracker_continue:
                bpy.context.scene.frame_current = executed_from_frame + frame_counter + 1
            else:
                bpy.context.scene.frame_current = bpy.context.scene.frame_start + frame_counter

            for label, confidence, bbox in detections:
                if bbox[2] >= bpy.context.scene.detection_min_size and \
                        bbox[3] >= bpy.context.scene.detection_min_size:
                    predicted_classes.append(label)
                    # we need to scale the detections to the original imagesize, as they are downsampled above
                    scaled_xy = scale_detections(x=bbox[0], y=bbox[1],
                                                 network_w=darknet.network_width(network),
                                                 network_h=darknet.network_height(network),
                                                 output_w=frame_rgb.shape[1], output_h=frame_rgb.shape[0])
                    viable_detections.append(scaled_xy)

                    # kalman stuff here
                    centres.append(np.round(np.array([[bbox[0]], [bbox[1]]])))

                    # add bounding box info by associating it with corresponding matched centres / tracks
                    bounding_boxes.append([bbox[0] - bbox[2] / 2,
                                           bbox[0] + bbox[2] / 2,
                                           bbox[1] - bbox[3] / 2,
                                           bbox[1] + bbox[3] / 2])

            all_detection_centres.append(viable_detections)

            if bpy.context.scene.detection_enforce_constant_size:
                image = cvDrawBoxes(detections, frame_resized, min_size=bpy.context.scene.detection_min_size,
                                    constant_size=bpy.context.scene.detection_constant_size,
                                    class_colours=class_colours)
            else:
                image = cvDrawBoxes(detections, frame_resized, min_size=bpy.context.scene.detection_min_size,
                                    class_colours=class_colours)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # before we show stuff, let's add some tracking fun
            # SO, if animals are detected then track them

            if len(centres) > -1:

                # Track object using Kalman Filter
                tracker_KF.Update(centres,
                                  predicted_classes=predicted_classes,
                                  bounding_boxes=bounding_boxes)

                # For identified object tracks draw tracking line
                # Use various colors to indicate different track_id
                for i in range(len(tracker_KF.tracks)):
                    if len(tracker_KF.tracks[i].trace) > 1:
                        mname = "track_" + str(tracker_KF.tracks[i].track_id)

                        # record the predicted class at each increment for every track
                        if mname in track_classes:
                            track_classes[mname][bpy.context.scene.frame_current] = \
                                tracker_KF.tracks[i].predicted_class[
                                    -1]
                        else:
                            track_classes[mname] = {
                                bpy.context.scene.frame_current: tracker_KF.tracks[i].predicted_class[-1]}

                        if mname not in track_colors:
                            track_colors[mname] = np.random.randint(low=100, high=255, size=3).tolist()

                        # draw direction of movement onto footage
                        x_t, y_t = tracker_KF.tracks[i].trace[-1]
                        tracker_KF_velocity = 5 * (tracker_KF.tracks[i].trace[-1] - tracker_KF.tracks[i].trace[-2])
                        x_t_future, y_t_future = tracker_KF.tracks[i].trace[-1] + tracker_KF_velocity * 0.1
                        cv2.arrowedLine(image, (int(x_t), int(y_t)), (int(x_t_future), int(y_t_future)),
                                        (np.array(track_colors[mname]) - np.array([70, 70, 70])).tolist(), 3,
                                        tipLength=0.75)

                        for j in range(len(tracker_KF.tracks[i].trace) - 1):
                            hind_sight_frame = bpy.context.scene.frame_current - len(tracker_KF.tracks[i].trace) + j
                            # Draw trace line on preview
                            x1 = tracker_KF.tracks[i].trace[j][0][0]
                            y1 = tracker_KF.tracks[i].trace[j][1][0]
                            x2 = tracker_KF.tracks[i].trace[j + 1][0][0]
                            y2 = tracker_KF.tracks[i].trace[j + 1][1][0]
                            if mname not in track_colors:
                                track_colors[mname] = np.random.randint(low=100, high=255, size=3).tolist()
                            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                                     track_colors[mname], 2)

                        if x2 != 0 and y2 != 0 and x2 <= clip_width and y2 <= clip_height:
                            if mname not in clip.tracking.objects[0].tracks:
                                # add new tracks to the set of markers
                                bpy.ops.clip.add_marker(location=(x2 / clip_width, 1 - y2 / clip_height))
                                clip.tracking.tracks.active.name = mname
                                # return {"FINISHED"}

                            # write track to clip markers
                            # update marker positions on the current frame
                            # remember to invert the y axis, because things are never easy or correct by default
                            # constrain x1 and
                            clip.tracking.objects[0].tracks[mname].markers.insert_frame(hind_sight_frame,
                                                                                        co=(x2 / clip_width,
                                                                                            1 - y2 / clip_height))
                            if context.scene.detection_enforce_constant_size:
                                clip.tracking.objects[0].tracks[mname].markers.find_frame(
                                    hind_sight_frame).pattern_corners = (
                                    (ROI_size / clip_width, ROI_size / clip_height),
                                    (ROI_size / clip_width, -ROI_size / clip_height),
                                    (-ROI_size / clip_width, -ROI_size / clip_height),
                                    (-ROI_size / clip_width, ROI_size / clip_height))
                            else:
                                clip.tracking.objects[0].tracks[mname].markers.find_frame(
                                    hind_sight_frame).pattern_corners = (
                                    ((tracker_KF.tracks[i].bbox_trace[-1][0] - x2) / clip_width,
                                     (tracker_KF.tracks[i].bbox_trace[-1][3] - y2) / clip_height),
                                    ((tracker_KF.tracks[i].bbox_trace[-1][1] - x2) / clip_width,
                                     (tracker_KF.tracks[i].bbox_trace[-1][3] - y2) / clip_height),
                                    ((tracker_KF.tracks[i].bbox_trace[-1][1] - x2) / clip_width,
                                     (tracker_KF.tracks[i].bbox_trace[-1][2] - y2) / clip_height),
                                    ((tracker_KF.tracks[i].bbox_trace[-1][0] - x2) / clip_width,
                                     (tracker_KF.tracks[i].bbox_trace[-1][2] - y2) / clip_height))

                        cv2.putText(image,
                                    mname,
                                    (int(x2) - int(bpy.context.scene.detection_constant_size / 2),
                                     int(y2) - bpy.context.scene.detection_constant_size), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    track_colors[mname], 2)
            if context.scene.tracker_save_video:
                video_out.write(image)
            cv2.imshow("Live tracking", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                bpy.context.scene.frame_current -= 2
                break

            frame_counter += 1

        cv2.destroyAllWindows()
        # always reset frame from capture at the end to avoid incorrect skips during access
        cap.set(1, context.scene.frame_start - 1)
        cap.release()
        if context.scene.tracker_save_video:
            video_out.release()

        print("Cleaning short tracks...")
        for mname in clip.tracking.objects[0].tracks:
            mname.select = False
            if len(mname.markers) < context.scene.tracking_min_track_length:
                mname.select = True
                print("removed", mname.name, "of length", len(mname.markers))

        bpy.ops.clip.delete_track()

        # remove markers from initial frame, if tracking was stopped early
        if executed_from_frame > bpy.context.scene.frame_current:
            for mname in clip.tracking.objects[0].tracks:
                mname.select = True
                bpy.ops.clip.delete_marker()
                mname.select = False

        return {"FINISHED"}


class OMNITRAX_OT_PoseEstimationOperator(bpy.types.Operator):
    """
    Run Pose estimation on the tracked animals
    ESTIMATE POSES: Run pose estimation on tracked ROIs (defaulting to full frame, when no tracks are present)
    ESTIMATE POSES [full frame]: Run (single subject) pose estimation on full resolution original video footage
    """
    bl_idname = "scene.pose_estimation_run"
    bl_label = "Run Pose Estimation (using tracks as inputs)"

    fullframe: BoolProperty(
        name="Inference on full frame",
        description="Run pose estimation inference on entire frame instead of individual tracks.",
        default=False)

    def execute(self, context):
        print("\nRUNNING POSE ESTIMATION\n")

        # check if any tracks exist for the currently selected clip. If not, run full frame pose estimation
        if len(context.edit_movieclip.tracking.objects[0].tracks) == 0:
            print("INFO: Found no tracks for current clip.")
            self.fullframe = True

        if self.fullframe:
            print("INFO: Running inference on full frame!")

        global dlc_proc
        global dlc_live
        global network_initialised
        global pose_cfg
        global pose_joint_header
        global pose_joint_names

        if "dlc_proc" not in globals():
            from dlclive import DLCLive, Processor
            try:
                model_path = bpy.path.abspath(context.scene.pose_network_path)
                dlc_proc = Processor()
                print("Loading DLC Network from", model_path)
                dlc_live = DLCLive(model_path, processor=dlc_proc, pcutoff=context.scene.pose_pcutoff)

                # create a list of join names from those defined in the pose_cfg.yaml file
                dlc_pose_cfg = os.path.join(model_path, "pose_cfg.yaml")
                with open(dlc_pose_cfg, "r") as stream:
                    pose_cfg_yaml = yaml.safe_load(stream)

                pose_joint_names = pose_cfg_yaml["all_joints_names"]
                pose_joint_header = ",".join(str(e) + "_x," +
                                             str(e) + "_y," +
                                             str(e) + "_prob" for e in pose_joint_names)

            except:
                print("Failed to load trained network... Check your model path!")
                # remove dlc processor to attempt to re-initialise the network after user applies corrections
                del dlc_proc
                return {"FINISHED"}
        else:
            print("Initialised DLC Network found!")

        # next, try to load skeleton relationships from the config file
        try:
            model_path = bpy.path.abspath(context.scene.pose_network_path)
            dlc_config_path = os.path.join(model_path, "config.yaml")
            with open(dlc_config_path, "r") as stream:
                config_yaml = yaml.safe_load(stream)
                print("skeleton configuration:\n", config_yaml["skeleton"])

                # now, match the skeleton elements to their IDs to draw them as overlays
                skeleton = []
                try:
                    for bone in config_yaml["skeleton"]:
                        skeleton.append([pose_joint_names.index(bone[0]),
                                         pose_joint_names.index(bone[1])])

                    print("skeleton links:\n", skeleton)
                except ValueError:
                    print("Your config skeleton and pose joint names do not match!"
                          "\n could not create overlay skeleton!")
                    skeleton = []

        except FileNotFoundError:
            print("No config.yaml file found!\n"
                  "Place your config file in the exported model folder to overlay the skeleton!\n")
            skeleton = []

        try:
            clip = context.edit_movieclip
            clip_path = bpy.path.abspath(bpy.context.edit_movieclip.filepath)

            clip_width = clip.size[0]
            clip_height = clip.size[1]
        except:
            print("You need to load and track a video, before running pose estimation!\n")
            return {"FINISHED"}

        first_frame = context.scene.frame_start
        last_frames = context.scene.frame_end

        # now we can load the captured video file and display it
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if "network_initialised" not in globals():
            network_initialised = False

        ROI_size = int(context.scene.pose_constant_size / 2)

        print("Using", clip_path, "for pose estimation...")

        if self.fullframe:
            # run full frame pose-inference
            if context.scene.pose_save_video:
                video_output = bpy.path.abspath(bpy.context.edit_movieclip.filepath)[
                               :-4] + "_POSE_fullframe.mp4"
                video_out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                            (int(video_width),
                                             int(video_height)))
            track_pose = {}

            for frame_id in range(first_frame, last_frames):
                try:
                    cap.set(1, frame_id)
                    ret, frame_temp = cap.read()

                    if ret:
                        dlc_input_img = frame_temp

                        # initialise network (if it has not been initialised yet)
                        if not network_initialised:
                            dlc_live.init_inference(dlc_input_img)
                            network_initialised = True

                        # estimate pose in cropped frame
                        pose = dlc_live.get_pose(dlc_input_img)
                        thresh = context.scene.pose_pcutoff

                        track_pose[str(frame_id)] = pose.flatten()

                        for p, point in enumerate(pose):
                            if point[2] >= thresh:
                                dlc_input_img = cv2.circle(dlc_input_img, (int(point[0]), int(point[1])),
                                                           context.scene.pose_point_size,
                                                           (int(255 * p / len(pose_joint_names)),
                                                            int(255 - 255 * p / len(pose_joint_names)), 200),
                                                           -1)

                                if context.scene.pose_show_labels:
                                    dlc_input_img = cv2.putText(dlc_input_img, pose_joint_names[p],
                                                                (int(point[0]), int(point[1])),
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                1,
                                                                (int(255 * p / len(pose_joint_names)),
                                                                 int(255 - 255 * p / len(pose_joint_names)),
                                                                 200),
                                                                1)

                        for b, bone in enumerate(skeleton):
                            if pose[bone[0]][2] >= thresh and pose[bone[1]][2] >= thresh:
                                if context.scene.pose_plot_skeleton:
                                    dlc_input_img = cv2.line(dlc_input_img,
                                                             (int(pose[bone[0]][0]), int(pose[bone[0]][1])),
                                                             (int(pose[bone[1]][0]), int(pose[bone[1]][1])),
                                                             (120, 220, 120),
                                                             context.scene.pose_skeleton_bone_width)

                        cv2.imshow("DLC Pose Estimation", dlc_input_img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        if context.scene.pose_save_video:
                            video_out.write(dlc_input_img)

                except Exception as e:
                    print(e)
                    track_pose[str(frame_id)] = np.array([0])

            if context.scene.pose_export_pose:
                pose_output_file = open(bpy.path.abspath(bpy.context.edit_movieclip.filepath)[
                                        :-4] + "_POSE_fullframe.csv", "w")
                # write header line
                pose_output_file.write("frame," + pose_joint_header + "\n")
                for key, value in track_pose.items():
                    line = key + "," + ",".join(str(e) for e in value.flatten())
                    pose_output_file.write(line + "\n")
                pose_output_file.close()

        else:
            # run per-track pose-inference
            for track in clip.tracking.objects[0].tracks:

                if context.scene.pose_save_video:
                    video_output = bpy.path.abspath(bpy.context.edit_movieclip.filepath)[
                                   :-4] + "_POSE_" + track.name + ".mp4"
                    video_out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                (int(context.scene.pose_constant_size),
                                                 int(context.scene.pose_constant_size)))

                # keeps track of the pose info at each frame for each track
                # frame_id : [[joint_a_X, joint_a_Y, joint_a_Confidence],
                #             [joint_b_X, joint_b_Y, joint_b_Confidence],  ...]
                track_pose = {}

                for frame_id in range(first_frame, last_frames):

                    marker = track.markers.find_frame(frame_id)
                    try:
                        if marker:
                            marker_x = round(marker.co.x * clip_width)
                            marker_y = round(marker.co.y * clip_height)
                            print("Frame:", frame_id, " : ",
                                  "X", marker_x, ",",
                                  "Y", marker_y)

                            cap.set(1, frame_id)
                            ret, frame_temp = cap.read()
                            if ret:
                                # first, create an empty image object to be filled with the ROI
                                # this is important in case the detection lies close to the edge
                                # where the ROI would go outside the image
                                dlc_input_img = np.zeros([ROI_size * 2, ROI_size * 2, 3], dtype=np.uint8)
                                dlc_input_img.fill(0)  # fill with zeros

                                if context.scene.pose_enforce_constant_size:
                                    true_min_x = marker_x - ROI_size
                                    true_max_x = marker_x + ROI_size
                                    true_min_y = clip_height - marker_y - ROI_size
                                    true_max_y = clip_height - marker_y + ROI_size

                                    min_x = max([0, true_min_x])
                                    max_x = min([clip.size[0], true_max_x])
                                    min_y = max([0, true_min_y])
                                    max_y = min([clip.size[1], true_max_y])
                                    # crop frame to detection and rescale
                                    frame_cropped = frame_temp[min_y:max_y, min_x:max_x]

                                    # place the cropped frame in the previously created empty image
                                    x_min_offset = max([0, - true_min_x])
                                    x_max_offset = min([ROI_size * 2, ROI_size * 2 - (true_max_x - clip.size[0])])
                                    y_min_offset = max([0, - true_min_y])
                                    y_max_offset = min([ROI_size * 2, ROI_size * 2 - (true_max_y - clip.size[1])])

                                    print("Cropped image ROI:", x_min_offset, x_max_offset, y_min_offset, y_max_offset)
                                    dlc_input_img[y_min_offset:y_max_offset, x_min_offset:x_max_offset] = frame_cropped
                                else:
                                    bbox = marker.pattern_corners
                                    true_min_x = marker_x + int(bbox[0][0] * clip_width)
                                    true_max_x = marker_x - int(bbox[0][0] * clip_width)
                                    true_min_y = clip_height - marker_y - int(bbox[0][1] * clip_height)
                                    true_max_y = clip_height - marker_y + int(bbox[0][1] * clip_height)
                                    true_width = true_max_x - true_min_x
                                    true_height = true_max_y - true_min_y

                                    if true_height < 0:  # flip y axis, if required
                                        true_min_y, true_max_y = true_max_y, true_min_y
                                        true_height = -true_height

                                    if true_width < 0:  # flip x axis, if required
                                        true_min_x, true_max_x = true_max_x, true_min_x
                                        true_width = -true_width

                                    print("Cropped image ROI:",
                                          true_min_x, true_max_x,
                                          true_min_y, true_max_y, "\n Detection h/w:",
                                          true_height, true_width)

                                    # resize image and maintain aspect ratio to the specified ROI
                                    if true_width >= true_height:
                                        rescale_width = int(ROI_size * 2)
                                        rescale_height = int((true_height / true_width) * ROI_size * 2)
                                        border_height = max([int((rescale_width - rescale_height) / 2), 0])
                                        print(rescale_width, rescale_height, border_height)
                                        frame_cropped = cv2.resize(frame_temp[true_min_y:true_max_y,
                                                                   true_min_x:true_max_x],
                                                                   (rescale_width, rescale_height))

                                        dlc_input_img[border_height:rescale_height + border_height, :] = frame_cropped
                                    else:
                                        rescale_width = int((true_width / true_height) * ROI_size * 2)
                                        rescale_height = int(ROI_size * 2)
                                        border_width = max([int(abs((rescale_height - rescale_width)) / 2), 0])
                                        frame_cropped = cv2.resize(frame_temp[true_min_y:true_max_y,
                                                                   true_min_x:true_max_x],
                                                                   (rescale_width, rescale_height))

                                        dlc_input_img[:, border_width:rescale_width + border_width] = frame_cropped

                                # initialise network (if it has not been initialised yet)
                                if not network_initialised:
                                    dlc_live.init_inference(dlc_input_img)
                                    network_initialised = True

                                # estimate pose in cropped frame
                                pose = dlc_live.get_pose(dlc_input_img)
                                thresh = context.scene.pose_pcutoff

                                track_pose[str(frame_id)] = pose.flatten()

                                for p, point in enumerate(pose):
                                    if point[2] >= thresh:
                                        dlc_input_img = cv2.circle(dlc_input_img, (int(point[0]), int(point[1])),
                                                                   context.scene.pose_point_size,
                                                                   (int(255 * p / len(pose_joint_names)),
                                                                    int(255 - 255 * p / len(pose_joint_names)), 200),
                                                                   -1)

                                        if context.scene.pose_show_labels:
                                            dlc_input_img = cv2.putText(dlc_input_img, pose_joint_names[p],
                                                                        (int(point[0]), int(point[1])),
                                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                                        1,
                                                                        (int(255 * p / len(pose_joint_names)),
                                                                         int(255 - 255 * p / len(pose_joint_names)),
                                                                         200),
                                                                        1)

                                # TODO add robust joint angle calculation (for various species)
                                """
                                joint_angles = np.empty(42)
                                joint_angles_conf = np.empty(42)  # report confidence on each joint angle
                                main_body_axis = [pose[0][0] - pose[6][0], pose[0][1] - pose[6][1]]  # b_t to b_a
                                unit_vector_body_axis = main_body_axis / np.linalg.norm(main_body_axis)
                                for b, bone in enumerate(skeleton):
                                    if pose[bone[0]][2] >= thresh and pose[bone[1]][2] >= thresh:
                                        if context.scene.pose_export_pose:
                                            # save angles between keypoints
                                            if b < 42:
                                                bone_vector = [pose[bone[0]][0] - pose[bone[1]][0],
                                                               pose[bone[0]][1] - pose[bone[1]][1]]
                                                unit_vector_bone_vector = bone_vector / np.linalg.norm(bone_vector)
                                                dot_product = np.dot(unit_vector_body_axis, unit_vector_bone_vector)
                                                joint_angles[b] = np.arccos(np.clip(dot_product, -1.0, 1.0))
                                                joint_angles_conf[b] = pose[bone[0]][2] + pose[bone[1]][2]
    
                                        if context.scene.pose_plot_skeleton:
                                            dlc_input_img = cv2.line(dlc_input_img,
                                                                     (int(pose[bone[0]][0]), int(pose[bone[0]][1])),
                                                                     (int(pose[bone[1]][0]), int(pose[bone[1]][1])),
                                                                     (120, 220, 120),
                                                                     context.scene.pose_skeleton_bone_width)
    
                                # now get the angle of each leg by taking the median angle from each associated joint
                                leg_angles = np.array([np.average(joint_angles[1:3], weights=joint_angles_conf[1:3]),
                                                       np.average(joint_angles[8:10], weights=joint_angles_conf[8:10]),
                                                       np.average(joint_angles[15:17], weights=joint_angles_conf[15:17]),
                                                       np.average(joint_angles[22:24], weights=joint_angles_conf[22:24]),
                                                       np.average(joint_angles[29:31], weights=joint_angles_conf[29:31]),
                                                       np.average(joint_angles[36:38], weights=joint_angles_conf[36:38])])
    
                                track_pose[str(frame_id)] = np.concatenate((track_pose[str(frame_id)], leg_angles))
                                """
                                for b, bone in enumerate(skeleton):
                                    if pose[bone[0]][2] >= thresh and pose[bone[1]][2] >= thresh:
                                        if context.scene.pose_plot_skeleton:
                                            dlc_input_img = cv2.line(dlc_input_img,
                                                                     (int(pose[bone[0]][0]), int(pose[bone[0]][1])),
                                                                     (int(pose[bone[1]][0]), int(pose[bone[1]][1])),
                                                                     (120, 220, 120),
                                                                     context.scene.pose_skeleton_bone_width)

                                cv2.imshow("DLC Pose Estimation", dlc_input_img)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    break
                                if context.scene.pose_save_video:
                                    video_out.write(dlc_input_img)


                    except Exception as e:
                        print(e)
                        track_pose[str(frame_id)] = np.array([0])

                if context.scene.pose_export_pose:
                    pose_output_file = open(bpy.path.abspath(bpy.context.edit_movieclip.filepath)[
                                            :-4] + "_POSE_" + track.name + ".csv", "w")
                    # write header line
                    # pose_output_file.write("frame," + pose_joint_header + ",r1_deg,r2_deg,r3_deg,l1_deg,l2_deg,l3_deg\n")
                    # TODO add robust joint angle calculation (see above)
                    pose_output_file.write("frame," + pose_joint_header + "\n")
                    for key, value in track_pose.items():
                        line = key + "," + ",".join(str(e) for e in value.flatten())
                        pose_output_file.write(line + "\n")
                    pose_output_file.close()

                print("\n")

        cv2.destroyAllWindows()

        # always reset frame from capture at the end to avoid incorrect skips during access
        cap.set(1, context.scene.frame_start - 1)
        cap.release()
        if context.scene.pose_save_video:
            video_out.release()
        print("Read all frames")

        return {"FINISHED"}


class EXPORT_OT_Operator(bpy.types.Operator):
    """
    Export the motion tracking data according to the settings.
    """
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
                try:
                    single_track_classes = track_classes[track.name]
                except NameError:
                    single_track_classes = "default"
                except KeyError:
                    single_track_classes = "default"

                if log:
                    log_file.write("  Track {0} started ...\n".format(track.name))

                if not subdirs:
                    export_file = open(path + "{0}_{1}.csv".format(clip.name.split(".")[0], track.name), "w")
                else:
                    subpath = path + "\\{0}\\".format(clip.name[:-3])
                    try:
                        if not os.path.exists(subpath):
                            os.makedirs(subpath)
                    except:
                        log_file.write(str(subpath), "already exists.")
                    export_file = open(subpath + "{0}_{1}.csv".format(clip.name.split(".")[0], track.name), "w")

                export_file.write("frame;x;y;class\n")
                success = True
                i = f_start
                while i <= f_end:
                    try:
                        if i in single_track_classes:
                            class_pred = single_track_classes[i]
                        else:
                            class_pred = ""
                    except TypeError:
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

        self.report({"INFO"}, "Export done ({0:.4f} s)".format(time.time() - time_start))

        if len(movieclips) == 0:
            self.report({"INFO"}, "No clip opened...")

        return {"FINISHED"}


class EXPORT_OT_AdvancedSampleExportOperator(bpy.types.Operator):
    """
    Run Pose estimation on the tracked animals
    ESTIMATE POSES: Run pose estimation on tracked ROIs (defaulting to full frame, when no tracks are present)
    ESTIMATE POSES [full frame]: Run (single subject) pose estimation on full resolution original video footage
    """
    bl_idname = "scene.advanced_sample_export"
    bl_label = "Export track patches as individual image samples"

    def execute(self, context):
        print("\nRUNNING ADVANCED SAMPLE EXTRACTION\n")

        try:
            clip = context.edit_movieclip
            clip_path = bpy.path.abspath(bpy.context.edit_movieclip.filepath)

            clip_width = clip.size[0]
            clip_height = clip.size[1]
        except:
            print("You need to load and track a video, before extracting any samples!\n")
            return {"FINISHED"}

        first_frame = context.scene.frame_start
        last_frames = context.scene.frame_end

        # now we can load the captured video file and display it
        cap = cv2.VideoCapture(clip_path)

        print("Extracting samples from", clip_path, "...")

        # export all tracked frames from all tracks
        for track in clip.tracking.objects[0].tracks:

            for frame_id in range(first_frame, last_frames):

                # skip all but nth frames
                if frame_id % context.scene.exp_ase_export_every_nth_frame != 0:
                    continue

                marker = track.markers.find_frame(frame_id)
                try:
                    if marker:
                        marker_x = round(marker.co.x * clip_width)
                        marker_y = round(marker.co.y * clip_height)
                        print("Frame:", frame_id, " : ",
                              "X", marker_x, ",",
                              "Y", marker_y)

                        cap.set(1, frame_id)
                        ret, frame_temp = cap.read()
                        if ret:
                            # first, create an empty image object to be filled with the ROI
                            # this is important in case the detection lies close to the edge
                            # where the ROI would go outside the image
                            track_input_img = np.zeros([context.scene.exp_ase_input_y,
                                                        context.scene.exp_ase_input_x, 3], dtype=np.uint8)

                            if context.scene.exp_ase_fixed_input_bounding_box_size:
                                true_min_x = marker_x - int(context.scene.exp_ase_input_x / 2)
                                true_max_x = marker_x + int(context.scene.exp_ase_input_x / 2)
                                true_min_y = clip_height - marker_y - int(context.scene.exp_ase_input_y / 2)
                                true_max_y = clip_height - marker_y + int(context.scene.exp_ase_input_y / 2)

                                min_x = max([0, true_min_x])
                                max_x = min([clip.size[0], true_max_x])
                                min_y = max([0, true_min_y])
                                max_y = min([clip.size[1], true_max_y])
                                # crop frame to detection and rescale
                                frame_cropped = frame_temp[min_y:max_y, min_x:max_x]

                                # place the cropped frame in the previously created empty image
                                x_min_offset = max([0, - true_min_x])
                                x_max_offset = min([context.scene.exp_ase_input_x,
                                                    context.scene.exp_ase_input_x - (true_max_x - clip.size[0])])
                                y_min_offset = max([0, - true_min_y])
                                y_max_offset = min([context.scene.exp_ase_input_y,
                                                    context.scene.exp_ase_input_y - (true_max_y - clip.size[1])])

                                print("Cropped image ROI:", x_min_offset, x_max_offset, y_min_offset, y_max_offset)
                                track_input_img[y_min_offset:y_max_offset, x_min_offset:x_max_offset] = frame_cropped

                            else:
                                bbox = marker.pattern_corners
                                true_min_x = marker_x + int(bbox[0][0] * clip_width)
                                true_max_x = marker_x - int(bbox[0][0] * clip_width)
                                true_min_y = clip_height - marker_y - int(bbox[0][1] * clip_height)
                                true_max_y = clip_height - marker_y + int(bbox[0][1] * clip_height)
                                true_width = true_max_x - true_min_x
                                true_height = true_max_y - true_min_y

                                if true_height < 0:  # flip y axis, if required
                                    true_min_y, true_max_y = true_max_y, true_min_y
                                    true_height = -true_height

                                if true_width < 0:  # flip x axis, if required
                                    true_min_x, true_max_x = true_max_x, true_min_x
                                    true_width = -true_width

                                print("Cropped image ROI:",
                                      true_min_x, true_max_x,
                                      true_min_y, true_max_y, "\n Detection h/w:",
                                      true_height, true_width)

                                # resize image and maintain aspect ratio to the specified ROI
                                if context.scene.exp_ase_padding:
                                    if true_width >= true_height:
                                        border_height = max([int((true_width - true_height) / 2), 0])
                                        track_input_img = np.zeros([true_width, true_width, 3], dtype=np.uint8)

                                        track_input_img[border_height:true_height + border_height, :] = frame_temp[
                                                                                                        true_min_y:true_max_y,
                                                                                                        true_min_x:true_max_x]
                                    else:
                                        border_width = max([int(abs((true_height - true_width)) / 2), 0])
                                        track_input_img = np.zeros([true_width, true_width, 3], dtype=np.uint8)

                                        track_input_img[:, border_width:true_width + border_width] = frame_temp[
                                                                                                     true_min_y:true_max_y,
                                                                                                     true_min_x:true_max_x]

                                else:
                                    track_input_img = frame_temp[true_min_y:true_max_y, true_min_x:true_max_x]

                        if context.scene.exp_ase_fixed_output_bounding_box_size:
                            track_input_img = cv2.resize(track_input_img,
                                                         [context.scene.exp_ase_output_x,
                                                          context.scene.exp_ase_output_y])

                        if context.scene.exp_ase_grayscale:
                            track_input_img = cv2.cvtColor(track_input_img, cv2.COLOR_BGR2GRAY)

                        # save out image, using the following convention:
                        # clip-name_frame_track.format
                        out_path = str(os.path.join(os.path.abspath(context.scene.exp_ase_path),
                                                    os.path.basename(
                                                        bpy.context.edit_movieclip.filepath[:-4]))) + "_" + str(
                            frame_id) + "_" + track.name + context.scene.exp_ase_sample_format
                        print(out_path)

                        # now, write out the final patch to the desired location
                        cv2.imwrite(out_path, track_input_img, )

                except Exception as e:
                    print(e)

            print("\n")

        cv2.destroyAllWindows()

        # always reset frame from capture at the end to avoid incorrect skips during access
        cap.set(1, context.scene.frame_start - 1)
        cap.release()
        print("Read all frames")

        return {"FINISHED"}


class OMNITRAX_PT_DetectionPanel(bpy.types.Panel):
    bl_label = "Detection (YOLO)"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "TOOLS"
    bl_category = "OmniTrax"

    bpy.types.Scene.detection_config_path = StringProperty(
        name="Config path",
        description="Path to your YOLO network config file",
        default="",  # bpy.data.filepath.rpartition("\\")[0],
        subtype="FILE_PATH")
    bpy.types.Scene.detection_weights_path = StringProperty(
        name="Weights path",
        description="Path to your YOLO network weights file",
        default="",  # bpy.data.filepath.rpartition("\\")[0],
        subtype="FILE_PATH")
    bpy.types.Scene.detection_data_path = StringProperty(
        name="Data path",
        description="Path to your YOLO network data file",
        default="",  # bpy.data.filepath.rpartition("\\")[0],
        subtype="FILE_PATH")
    bpy.types.Scene.detection_names_path = StringProperty(
        name="Names path",
        description="Path to your YOLO network names file",
        default="",  # bpy.data.filepath.rpartition("\\")[0],
        subtype="FILE_PATH")
    bpy.types.Scene.detection_enforce_constant_size = BoolProperty(
        name="constant detection sizes",
        description="Check to enforce constant detection sizes. This DOES NOT affect the actual inference, only the resulting regions of interest.",
        default=True)
    bpy.types.Scene.detection_constant_size = IntProperty(
        name="constant detection sizes (px)",
        description="Constant detection size in pixels. All detections will be rescaled to this value in both width and height.",
        default=60)
    bpy.types.Scene.detection_min_size = IntProperty(
        name="minimum detection sizes (px)",
        description="If the width or height of a detection is below this threshold, it will be discarded. This can be useful to decrease noise. Keep the value at 0 if this is not needed.",
        default=15)
    bpy.types.Scene.detection_network_width = FloatProperty(
        name="Network width",
        description="YOLO network input width. Larger network inputs allow for smaller detected subjects but will increase inference time and memory usage. [MUST BE MULTIPLE OF 32]",
        default=480,
        step=3200,  # step size given in 1/100
        min=32,
        precision=0)  # set to 0 emulate integer property (which currently does not support steps)
    bpy.types.Scene.detection_network_height = FloatProperty(
        name="Network height",
        description="YOLO network input height. Larger network inputs allow for smaller detected subjects but will increase inference time and memory usage. [MUST BE MULTIPLE OF 32]",
        default=480,
        step=3200,  # step size given in 1/100
        min=32,
        precision=0)  # set to 0 emulate integer property (which currently does not support steps)
    bpy.types.Scene.detection_activation_threshold = FloatProperty(
        name="Confidence threshold",
        description="Detection confidence threshold. A higher confidence leads to fewer false positives but more missed detections.",
        default=0.5)
    bpy.types.Scene.detection_nms = FloatProperty(
        name="Non-maximum suppression",
        description="Non-maximum suppression (NMS) refers to the maximum overlap allowed between proposed bounding boxes. E.g., a value of 0.45 corresponds to a maximum overlap of 45% between two compared bounding boxes to be retained simultaneously. In case the overlap is larger, the box with the lower objectness score or classification confidence will be 'suppressed', thus, only the highest confidence prediction is returned.",
        default=0.45)

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
        default=100)
    bpy.types.Scene.tracking_max_frames_to_skip = IntProperty(
        name="Frames skipped before termination",
        description="Maximum number of frames between two detections to be associated with the same track. This is to handle simple buffer and recover tracking as well as termination of tracks of subjects that leave the scene",
        default=10)
    bpy.types.Scene.tracking_min_track_length = IntProperty(
        name="Minimum track length",
        description="Minimum number of tracked frames. If a track has fewer frames, no marker will be created for it and it will not be used for any analysis",
        default=25)

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
    bpy.types.Scene.tracker_save_video = BoolProperty(
        name="Export tracked video",
        description="Write the video with tracked overlay to the location of the input video",
        default=False)

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
        col.operator("scene.detection_run", text="RESTART Tracking").restart_tracking = True


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
        col.label(text="DLC network path:")
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


class EXPORT_PT_TrackingPanel(bpy.types.Panel):
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
        props = row.operator("clip.track_markers", text="",
                             icon="PLAY_REVERSE")
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
    bl_label = "Export"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "TOOLS"
    bl_category = "OmniTrax"
    bl_options = {"DEFAULT_CLOSED"}

    bpy.types.Scene.exp_path = StringProperty(
        name="Export Path",
        description="Path where data will be exported to",
        default="\\export",
        subtype="DIR_PATH")
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
        default=True)
    bpy.types.Scene.exp_ase_input_x = IntProperty(
        name="X (px)",
        description="Constant sized bounding box X dimension in pixels.",
        default=128)
    bpy.types.Scene.exp_ase_input_y = IntProperty(
        name="Y (px)",
        description="Constant sized bounding box Y dimension in pixels.",
        default=128)

    # Output dimensions
    bpy.types.Scene.exp_ase_fixed_output_bounding_box_size = BoolProperty(
        name="Fixed output patch size",
        description="Fixed dimensions of exported samples. Unless 'padding' is enabled, this option may change the "
                    "aspect ratio of the extracted samples.",
        default=True)
    bpy.types.Scene.exp_ase_output_x = IntProperty(
        name="X (px)",
        description="X dimension of exported samples.",
        default=128)
    bpy.types.Scene.exp_ase_output_y = IntProperty(
        name="Y (px)",
        description="Y dimension of exported samples.",
        default=128)

    # Optional settings
    bpy.types.Scene.exp_ase_export_every_nth_frame = IntProperty(
        name="Export every nth frame",
        description="Export only every nth frame, skipping intermediate frames",
        default=1,
        min=1)

    bpy.types.Scene.exp_ase_padding = BoolProperty(
        name="Use padding",
        description="Use padding for images to preserve the original aspect ratio and produce mxm square patches",
        default=True)

    sample_formats = [(".jpg", ".jpg", "Use JPG as the sample output format"),
                      (".png", ".png", "Use PNG as the sample output format")]

    bpy.types.Scene.exp_ase_grayscale = BoolProperty(
        name="Convert to grayscale",
        description="Export image samples in black and white with image dimensions as (X,Y,1)",
        default=False)

    bpy.types.Scene.exp_ase_sample_format = EnumProperty(
        name="Sample format",
        description="Set the sample output format",
        items=sample_formats,
        default=sample_formats[0][0])

    bpy.types.Scene.exp_ase_path = StringProperty(
        name="Sample Export Path",
        description="Path to which all samples will be exported to",
        default="C:\\",
        subtype="DIR_PATH")

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


### (un)register module ###
def register():
    bpy.utils.register_class(OMNITRAX_PT_ComputePanel)

    bpy.utils.register_class(OMNITRAX_OT_DetectionOperator)
    bpy.utils.register_class(OMNITRAX_PT_DetectionPanel)
    bpy.utils.register_class(OMNITRAX_PT_TrackingPanel)

    bpy.utils.register_class(EXPORT_OT_Operator)
    bpy.utils.register_class(EXPORT_PT_TrackingPanel)

    bpy.utils.register_class(EXPORT_PT_DataPanel)

    bpy.utils.register_class(OMNITRAX_OT_PoseEstimationOperator)
    bpy.utils.register_class(OMNITRAX_PT_PoseEstimationPanel)

    bpy.utils.register_class(EXPORT_OT_AdvancedSampleExportOperator)
    bpy.utils.register_class(EXPORT_PT_AdvancedSampleExportPanel)


def unregister():
    bpy.utils.unregister_class(OMNITRAX_PT_ComputePanel)

    bpy.utils.unregister_class(OMNITRAX_OT_DetectionOperator)
    bpy.utils.unregister_class(OMNITRAX_PT_DetectionPanel)
    bpy.utils.unregister_class(OMNITRAX_PT_TrackingPanel)

    bpy.utils.unregister_class(EXPORT_OT_Operator)
    bpy.utils.unregister_class(EXPORT_PT_TrackingPanel)

    bpy.utils.unregister_class(EXPORT_PT_DataPanel)

    bpy.utils.unregister_class(OMNITRAX_OT_PoseEstimationOperator)
    bpy.utils.unregister_class(OMNITRAX_PT_PoseEstimationPanel)

    bpy.utils.unregister_class(EXPORT_OT_AdvancedSampleExportOperator)
    bpy.utils.unregister_class(EXPORT_PT_AdvancedSampleExportPanel)
