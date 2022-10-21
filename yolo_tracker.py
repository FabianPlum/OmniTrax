try:
    from omni_trax.tracker import Tracker
except ModuleNotFoundError:
    from tracker import Tracker
import numpy as np
import os
import cv2
import time
import argparse
from operator import itemgetter

# TODO
# add option to switch networks through sub-processes after loading an initial model
# use a blender background process to ensure python is loaded with all dependencies (get full paths from bpy)
# ./blender.exe --background --python yolo_tracker.py -- args
# for advanced argument parsing refer to:
# https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
# the below example can be added to __init__.py once the implementation is completed

"""
class OMNITRAX_OT_DetectionOperator(bpy.types.Operator):
    #Run the detection based tracking pipeline according to the above defined parameters
    bl_idname = "scene.detection_run"
    bl_label = "Run Detection"

    restart_tracking: BoolProperty(
        name="Restart Tracking",
        description="Re-initialises tracker with new settings. WARNING: Current identities will NOT be retained!",
        default=False)

    def execute(self, context):
    
        # get blender python path
        py_exec = str(sys.executable)
        # get omni_trax working directory to access scripts
        script_file = os.path.realpath(__file__)
        wd = os.path.dirname(script_file)
        
        yolo_tracker_path = os.path.join(wd, "yolo_tracker.py")
        
        full_yolo_command = [py_exec, yolo_tracker_path]
        out = subprocess.run(full_yolo_command)

print(out)

return {"FINISHED"}
"""

np.random.seed(0)


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
                 tracker_dist_thresh=100, tracker_max_frames_to_skip=20, tracker_max_trace_length=200,
                 tracker_track_id_count=0, prior_tracks=[], prior_classes=[],
                 tracker_use_kf=True, tracker_std_acc=5,
                 tracker_x_std_meas=0.25, tracker_y_std_meas=0.25, dt=0,
                 frame_start=0, frame_end=-1, continue_tracking=False,
                 detection_min_size=50, detection_constant_size=100, detection_enforce_constant_size=False):

        # now we can load the captured video file and display it
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        # get the fps of the clip and set the environment accordingly
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if dt == 0:
            self.dt = 1.0 / self.fps
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

    def run_inference(self, export_video=False, write_csv=False, write_h5=False, write_pkl=False,
                      inference_device=None):

        """
        Check which compute device is selected and set it as active
        """

        if inference_device is not None:
            # load darknet with compiled DLLs for windows for either GPU or CPU inference from respective path
            if inference_device.split("_")[0] == "CPU":
                from darknet import darknet_cpu as darknet
        else:
            # use GPU inference by default
            from darknet import darknet as darknet
            darknet.set_compute_device(int(inference_device.split("_")[1]))

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
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # input video
    ap.add_argument("-v", "--video_path", required=True,
                    help="path to input video (use quotation marks for file paths with spaces!")
    ap.add_argument("-vs", "--frame_start", type=int, default=0,
                    help="First frame to analyze")
    ap.add_argument("-ve", "--frame_end", type=int, default=-1,
                    help="Last frame to analyze")

    # detector settings [required]
    ap.add_argument("-ncfg", "--net_cfg", required=True,
                    help="path to yolo (v3 or v4, trained darknet model) config file")
    ap.add_argument("-nw", "--net_weight", required=True,
                    help="path to yolo (v3 or v4, trained darknet model) weight file")
    ap.add_argument("-nn", "--net_names", required=True,
                    help="path to yolo (v3 or v4, trained darknet model) names file")
    ap.add_argument("-nd", "--net_data", required=True,
                    help="path to yolo (v3 or v4, trained darknet model) data file")

    # detector settings [optional]
    ap.add_argument("-detthresh", "--detection_activation_threshold", type=float, default=0.5,
                    help="confidence threshold for viable detections (higher > higher precision")
    ap.add_argument("-detnms", "--detection_nms", type=float, default=0.45,
                    help="non-maximum suppression overlap (higher > closer proximity, " +
                         "but may cause overlapping detections")
    ap.add_argument("-detmin", "--detection_min_size", type=int, default=25,
                    help="minimum element size to be considered a viable detection (for small objects)")
    ap.add_argument("-detconst", "--detection_enforce_constant_size", type=bool, default=False,
                    help="Use square, fixed size bounding boxes, instead of natively returned size and shape")
    ap.add_argument("-detconstsize", "--detection_constant_size", type=int, default=100,
                    help="Square bounding box side length when enforcing constant bounding box size")

    # compute device settings
    ap.add_argument("-device", "--inference_device", type=str, default=None,
                    help="Select compute device, e.g. CPU_0, GPU_0, GPU_1 ...")

    # tracker settings
    ap.add_argument("-trthresh", "--tracker_dist_thresh", type=float, default=100,
                    help="maximum (squared) pixel distance to associate a detection with a track")
    ap.add_argument("-trskip", "--tracker_max_frames_to_skip", type=int, default=25,
                    help="Attempt to recover tracker for ## frames, before terminating track")
    ap.add_argument("-trtrace", "--tracker_max_trace_length", type=int, default=100,
                    help="Number of simultaneously displayed tracking states (does not affect output)")
    ap.add_argument("-trcount", "--tracker_track_id_count", type=int, default=0,
                    help="Begin assigning IDs from this value onward (e.g. when combining multiple " +
                         "independently tracked segments")

    # tracker KF settings
    ap.add_argument("-trKF", "--tracker_use_kf", type=bool, default=True,
                    help="Use Kalman Filter for tracking")
    ap.add_argument("-trstd", "--tracker_std_acc", type=float, default=10,
                    help="Kalman Filter process noise magnitude")
    ap.add_argument("-trstdx", "--tracker_x_std_meas", type=float, default=0.25,
                    help="Kalman Filter standard deviation of the measurement in x-direction")
    ap.add_argument("-trstdy", "--tracker_y_std_meas", type=float, default=0.25,
                    help="Kalman Filter standard deviation of the measurement in y-direction")
    ap.add_argument("-trdt", "--dt", type=float, default=0,
                    help="Kalman Filter sampling time (time for 1 cycle). If not explicitly defined, dt will be "
                         + "1 / input_video_fps")

    # continue tracking settings
    ap.add_argument("-trc", "--continue_tracking", type=bool, default=False,
                    help="Continue tracking from prior state (requires prior_tracks input)")
    ap.add_argument('-trpt', '--prior_tracks', action='append',
                    help='Pass last state of prior tracks as -trpt mname -trpt x -trpt y -trpt mname -trpt x ...')
    ap.add_argument('-trpc', '--prior_classes', action='append',
                    help='Pass last state of prior classes as -trpc mname -trpc class -trpc mname -trpc class ')

    # output settings
    ap.add_argument("-wv", "--write_video", type=bool, default=False,
                    help="Export .avi video of live-tracking to input video folder")
    ap.add_argument("-wcsv", "--write_csv", type=bool, default=False,
                    help="Export .csv file of all final tracks")
    ap.add_argument("-wh5", "--write_h5", type=bool, default=False,
                    help="Export h5 file of all final tracks")
    ap.add_argument("-wpkl", "--write_pkl", type=bool, default=False,
                    help="Export .pkl file of all final tracks (used in Blender communication)")

    args = vars(ap.parse_args())

    YT = YoloTracker(net_cfg=args["net_cfg"], net_weight=args["net_weight"],
                     net_names=args["net_names"], net_data=args["net_data"], video_path=args["video_path"],
                     detection_activation_threshold=args["detection_activation_threshold"],
                     detection_nms=args["detection_nms"], tracker_dist_thresh=args["tracker_dist_thresh"],
                     tracker_max_frames_to_skip=args["tracker_max_frames_to_skip"],
                     tracker_max_trace_length=args["tracker_max_trace_length"],
                     tracker_track_id_count=args["tracker_track_id_count"],
                     continue_tracking=["continue_tracking"],
                     prior_tracks=args["prior_tracks"], prior_classes=args["prior_classes"],
                     tracker_use_kf=args["tracker_use_kf"], tracker_std_acc=args["tracker_std_acc"],
                     tracker_x_std_meas=args["tracker_x_std_meas"],
                     tracker_y_std_meas=args["tracker_y_std_meas"],
                     dt=args["dt"], frame_start=args["frame_start"],
                     frame_end=args["frame_end"],
                     detection_min_size=args["detection_min_size"],
                     detection_constant_size=args["detection_constant_size"],
                     detection_enforce_constant_size=args["detection_enforce_constant_size"])

    output = YT.run_inference(export_video=args["write_video"],
                              write_csv=args["write_csv"],
                              write_h5=args["write_h5"],
                              write_pkl=args["write_pkl"])

    print(output)
