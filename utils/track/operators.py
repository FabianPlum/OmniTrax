import bpy
from bpy.props import BoolProperty as BoolProperty

import numpy as np
import cv2
import os
import time
import platform

from ..omni_trax_utils import scale_detections, cvDrawBoxes, setInferenceDevive
from ..setup.CUDA_checks import check_CUDA_installation
from .tracker import Tracker
from ..YOLOader import YOLOader


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
        # start by checking all necessary paths are provided before attempting to do anything
        clip = context.edit_movieclip
        try:
            video = bpy.path.abspath(clip.filepath)
        except:
            self.report({'ERROR'}, 'Open a video to track from your drive, then click TRACK / RESTART Tracking again')
            return {'CANCELLED'}

        # check required YOLO paths
        yolo_cfg_path_temp = bpy.path.abspath(context.scene.detection_config_path)
        if not os.path.isfile(yolo_cfg_path_temp) or yolo_cfg_path_temp.split(".")[-1] != "cfg":
            self.report({'ERROR'}, 'Provide the .cfg file of a suitable YOLOv3 or V4 model')
            return {'CANCELLED'}

        yolo_weights_path_temp = bpy.path.abspath(context.scene.detection_weights_path)
        if not os.path.isfile(yolo_weights_path_temp) or yolo_weights_path_temp.split(".")[-1] != "weights":
            self.report({'ERROR'}, 'Provide the .weights file of a trained YOLOv3 or V4 model')
            return {'CANCELLED'}

        """
        Check which compute device is selected and set it as active
        """
        setInferenceDevive(context.scene.compute_device)

        # load darknet with compiled DLLs for windows for either GPU or CPU inference from respective path
        if context.scene.compute_device.split("_")[0] == "GPU":
            required_CUDA_version = "11.2"
            CUDA_match = check_CUDA_installation(required_CUDA_version=required_CUDA_version)

            if not CUDA_match:
                self.report({'ERROR'}, 'No matching CUDA version found! Refer to console for full error message')
                return {'CANCELLED'}

            from ..darknet import darknet as darknet
            darknet.set_compute_device(int(context.scene.compute_device.split("_")[1]))
        else:
            if platform.system() != "Linux":
                from ..darknet import darknet_cpu as darknet
            else:
                if bpy.app.version_string == "2.92.0":
                    from ..darknet import darknet as darknet
                else:
                    self.report({'ERROR'}, 'To use CPU inference, you need to use OmniTrax with Blender version 2.9.2!')
                    return {'CANCELLED'}

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
        else:
            # update the data file to ensure it points to the correct absolute location of the names file
            yolo_paths.update_data()

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
            try:
                network, class_names, _ = darknet.load_network(yolo_paths.cfg,
                                                               yolo_paths.data,
                                                               yolo_paths.weights,
                                                               batch_size=1)
            except Exception as e:  # work on python 3.x
                self.report({'ERROR'}, e)
                return {'CANCELLED'}

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
            except Exception as e:  # work on python 3.x
                self.report({'ERROR'}, e)
                return {'CANCELLED'}

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
            try:
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

            except Exception as e:  # work on python 3.x
                self.report({'ERROR'}, e)
                return {'CANCELLED'}

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


def register():
    bpy.utils.register_class(OMNITRAX_OT_DetectionOperator)
    bpy.utils.register_class(EXPORT_OT_Operator)
    bpy.utils.register_class(EXPORT_OT_AdvancedSampleExportOperator)


def unregister():
    bpy.utils.unregister_class(OMNITRAX_OT_DetectionOperator)
    bpy.utils.unregister_class(EXPORT_OT_Operator)
    bpy.utils.unregister_class(EXPORT_OT_AdvancedSampleExportOperator)
