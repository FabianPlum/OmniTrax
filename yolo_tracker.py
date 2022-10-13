try:
    from omni_trax.tracker import Tracker
except ModuleNotFoundError:
    from tracker import Tracker
import numpy as np
import cv2
import os
import time
import tensorflow as tf
from operator import itemgetter


def getInferenceDevices():
    physical_devices = tf.config.list_physical_devices()
    print("Found computational devices:\n", physical_devices)

    devices = []
    for d, device in enumerate(physical_devices):
        if device.device_type == "GPU":
            devices.append(("GPU_" + str(d - 1), "GPU_" + str(d - 1), "Use GPU for inference (requires CUDA)"))
        else:
            devices.append(("CPU_" + str(d), "CPU", "Use CPU for inference"))

    return devices


def setInferenceDevive(device):
    # disables all but the selected computational device (by setting them invisible)
    physical_devices = tf.config.list_physical_devices(device.split("_")[0])
    try:
        tf.config.set_visible_devices(physical_devices[int(device.split("_")[1])], device_type=device.split("_")[0])
        logical_devices = tf.config.list_logical_devices()
        # Logical device was not created for first GPU
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("WARNING: Unable to switch active compute device")
        pass

    print("Running inference on: ", logical_devices)


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
    for label, confidence, bbox in detections:

        x, y, w, h = bbox[0], \
                     bbox[1], \
                     bbox[2], \
                     bbox[3]

        if w >= min_size and h >= min_size:

            if constant_size:
                w, h = constant_size, constant_size

            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cl_colour = class_colours[label]
            cv2.rectangle(img, pt1, pt2, (cl_colour[0], cl_colour[1], cl_colour[2]), 1)
            cv2.putText(img, label + " [" + confidence + "]",
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


class YoloTracker:

    def __init__(self, net_cfg, net_weight, net_names, net_data, video_path,
                 detection_activation_threshold=0.5, detection_nms=0.45,
                 tracker_dist_thresh=100, tracker_max_frames_to_skip=0, tracker_max_trace_length=200,
                 tracker_track_id_count=0, prior_tracks=[], prior_classes=[],
                 tracker_use_kf=True, tracker_std_acc=5,
                 tracker_x_std_meas=0.25, tracker_y_std_meas=.25, dt=None,
                 frame_start=0, frame_end=-1, continue_tracking=False,
                 detection_min_size=50, detection_constant_size=100, detection_enforce_constant_size=False,
                 inference_device=None):

        """
        Check which compute device is selected and set it as active
        """
        """
        if inference_device is not None:
            self.devices = [inference_device]
        else:
            self.devices = getInferenceDevices()
        setInferenceDevive(self.devices[-1])

        # load darknet with compiled DLLs for windows for either GPU or CPU inference from respective path
        if self.devices[-1].split("_")[0] == "GPU":
            from darknet import darknet as darknet
        else:
            from darknet import darknet_cpu as darknet
        """
        from darknet import darknet as darknet
        ###

        # now we can load the captured video file and display it
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        # get the fps of the clip and set the environment accordingly
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if dt is None:
            self.dt = 1 / self.fps
        else:
            self.dt = dt

        if frame_end == -1:
            self.frame_end = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.frame_end = frame_end

        if frame_start == 0:
            self.frame_start = 0
        else:
            self.frame_start = frame_start

        """
        Tracker settings
        """
        # Variables initialization
        self.track_colors = {}
        np.random.seed(0)

        self.continue_tracking = continue_tracking

        self.tracker_KF = Tracker(dist_thresh=tracker_dist_thresh,
                                  max_frames_to_skip=tracker_max_frames_to_skip,
                                  max_trace_length=tracker_max_trace_length,
                                  trackIdCount=tracker_track_id_count,
                                  use_kf=tracker_use_kf,
                                  std_acc=tracker_std_acc,
                                  x_std_meas=tracker_x_std_meas,
                                  y_std_meas=tracker_y_std_meas,
                                  dt=self.dt)

        """
        Detector Settings
        """
        self.net_cfg = net_cfg
        self.net_weight = net_weight
        self.net_names = net_names
        self.net_data = net_data

        self.detection_activation_threshold = detection_activation_threshold
        self.detection_nms = detection_nms
        self.detection_min_size = detection_min_size
        self.detection_constant_size = detection_constant_size
        self.detection_enforce_constant_size = detection_enforce_constant_size

        print("\nINFO: Initialising darkent network...\n")

        # overwrite network dimensions
        # TODO

        if not os.path.exists(self.net_cfg):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(self.net_cfg) + "`")
        if not os.path.exists(self.net_weight):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(self.net_weight) + "`")
        if not os.path.exists(self.net_data):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(self.net_data) + "`")

    def run_inference(self, export_video=False, write_csv=False, write_h5=False, write_pkl=False):
        from darknet import darknet as darknet
        self.network, self.class_names, self.class_colours = darknet.load_network(self.net_cfg,
                                                                                  self.net_data,
                                                                                  self.net_weight,
                                                                                  batch_size=1)

        try:
            with open(self.net_data) as metaFH:
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

        # clear the tracker trace to ensure existing tracks are not overwritten
        if self.continue_tracking:
            for track in self.tracker_KF.tracks:
                track.trace = []

        # and produce an output file
        if export_video:
            video_output = self.video_path[:-4] + "_online_tracking.avi"
            video_out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps,
                                        (int(self.cap.get(3)), int(self.cap.get(4))))

        # check the number of frames of the imported video file
        numFramesMax = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("The imported clip:", self.video_path, "has a total of", numFramesMax, "frames.\n")

        print("Starting the YOLO loop...")

        # Create an image we reuse for each detect
        video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # darknet_image = darknet.make_image(darknet.network_width(netMain),
        #                                darknet.network_height(netMain),3)
        darknet_image = darknet.make_image(video_width, video_height, 3)

        all_detection_centres = []
        all_track_results = []  # holds all temporary tracks and writes them to scene markers

        frame_counter = 0

        self.cap.set(1, self.frame_start)

        # build array for all tracks and classes
        track_classes = {}

        ROI_size = int(self.detection_constant_size / 2)

        while True:
            if frame_counter == self.frame_end + 1 - self.frame_start:
                break

            prev_time = time.time()
            ret, frame_read = self.cap.read()
            clip_width = frame_read.shape[1]
            clip_height = frame_read.shape[0]
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (video_width,
                                        video_height),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            # thresh : detection threshold -> lower = more sensitive (higher recall)
            # nms : non maximum suppression -> higher = allow for closer proximity between detections
            detections = darknet.detect_image(self.network, self.class_names, darknet_image,
                                              thresh=self.detection_activation_threshold,
                                              nms=self.detection_nms)
            print("Frame:", frame_counter + 1)
            viable_detections = []
            centres = []
            bounding_boxes = []
            predicted_classes = []

            for label, confidence, bbox in detections:
                if bbox[2] >= self.detection_min_size and \
                        bbox[3] >= self.detection_min_size:
                    predicted_classes.append(label)
                    # we need to scale the detections to the original imagesize, as they are downsampled above
                    scaled_xy = scale_detections(x=bbox[0], y=bbox[1],
                                                 network_w=darknet.network_width(self.network),
                                                 network_h=darknet.network_height(self.network),
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

            if self.detection_enforce_constant_size:
                image = cvDrawBoxes(detections, frame_resized, min_size=self.detection_min_size,
                                    constant_size=self.detection_constant_size,
                                    class_colours=self.class_colours)
            else:
                image = cvDrawBoxes(detections, frame_resized, min_size=self.detection_min_size,
                                    class_colours=self.class_colours)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # before we show stuff, let's add some tracking fun
            # SO, if animals are detected then track them

            if len(centres) > 0:

                # Track object using Kalman Filter
                self.tracker_KF.Update(centres,
                                       predicted_classes=predicted_classes,
                                       bounding_boxes=bounding_boxes)
                # For identified object tracks draw tracking line
                # Use various colors to indicate different track_id
                for i in range(len(self.tracker_KF.tracks)):
                    if len(self.tracker_KF.tracks[i].trace) > 1:
                        mname = "track_" + str(self.tracker_KF.tracks[i].track_id)

                        # record the predicted class at each increment for every track
                        if mname in track_classes:
                            track_classes[mname][self.frame_start + frame_counter] = \
                                self.tracker_KF.tracks[i].predicted_class[-1]
                        else:
                            track_classes[mname] = {
                                self.frame_start + frame_counter: self.tracker_KF.tracks[i].predicted_class[-1]}

                        if mname not in self.track_colors:
                            self.track_colors[mname] = np.random.randint(low=100, high=255, size=3).tolist()

                        # draw direction of movement onto footage
                        x_t, y_t = self.tracker_KF.tracks[i].trace[-1]
                        tracker_KF_velocity = 5 * (
                                self.tracker_KF.tracks[i].trace[-1] - self.tracker_KF.tracks[i].trace[-2])
                        x_t_future, y_t_future = self.tracker_KF.tracks[i].trace[-1] + tracker_KF_velocity * 0.1
                        cv2.arrowedLine(image, (int(x_t), int(y_t)), (int(x_t_future), int(y_t_future)),
                                        (np.array(self.track_colors[mname]) - np.array([70, 70, 70])).tolist(), 3,
                                        tipLength=0.75)

                        for j in range(len(self.tracker_KF.tracks[i].trace) - 1):
                            hind_sight_frame = self.frame_start + frame_counter - len(
                                self.tracker_KF.tracks[i].trace) + j
                            # Draw trace line on preview
                            x1 = self.tracker_KF.tracks[i].trace[j][0][0]
                            y1 = self.tracker_KF.tracks[i].trace[j][1][0]
                            x2 = self.tracker_KF.tracks[i].trace[j + 1][0][0]
                            y2 = self.tracker_KF.tracks[i].trace[j + 1][1][0]
                            if mname not in self.track_colors:
                                self.track_colors[mname] = np.random.randint(low=100, high=255, size=3).tolist()
                            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                                     self.track_colors[mname], 2)

                        if x2 != 0 and y2 != 0 and x2 <= clip_width and y2 <= clip_height:
                            # TODO
                            # add filtered export options
                            """
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
                            """

                        cv2.putText(image,
                                    mname,
                                    (int(x2) - int(self.detection_constant_size / 2),
                                     int(y2) - self.detection_constant_size), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    self.track_colors[mname], 2)
            if export_video:
                video_out.write(image)
            cv2.imshow('Detections on video', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_counter += 1

        cv2.destroyAllWindows()
        # always reset frame from capture at the end to avoid incorrect skips during access
        self.cap.release()
        if export_video:
            video_out.release()

        return {"FINISHED"}


if __name__ == '__main__':
    path_to_yolo = "C:/Users/Legos/Documents/PhD/Blender/OmniTrax/trained_networks/atta_single_class/"

    YT = YoloTracker(net_cfg=path_to_yolo + "yolov4-big_and_small_ants.cfg",
                     net_weight=path_to_yolo + "yolov4-big_and_small_ants_1024px_refined_with_2048_px_27000.weights",
                     net_names=path_to_yolo + "obj.names",
                     net_data=path_to_yolo + "obj.data",
                     video_path="C:/Users/Legos/Desktop/yolov4/example_recordings/2019-06-28_19-24-05.mp4")

    YT.run_inference(export_video=True)
