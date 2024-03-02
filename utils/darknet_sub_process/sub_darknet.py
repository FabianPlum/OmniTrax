# load darknet with compiled DLLs for windows from respective path
import sys
import argparse
import pickle

from os import listdir
from os.path import join
from operator import itemgetter


def nonMaximumSupression(detections):
    """
    :param detections: detections returned from darknet
    :return: only detection of highest confidence. Return None, if no individual was detected
    """
    if len(detections) != 0:
        det_sorted = sorted(detections, key=itemgetter(2))
        max_conf_detection = det_sorted[0][0]
    else:
        max_conf_detection = "No Detect"
    return max_conf_detection


from ctypes import *
import os
import cv2


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, min_size=20, constant_size=False):
    for detection in detections:
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]

        if w >= min_size and h >= min_size:
            if constant_size:
                w, h = constant_size, constant_size

            xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (150, 0, 160), 1)
            cv2.putText(
                img,
                detection[0].decode() + " [" + str(round(detection[1] * 100, 2)) + "]",
                (pt1[0], pt1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                [150, 0, 160],
                2,
            )
    return img


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--configPath", required=True, type=str)
    ap.add_argument("-w", "--weightPath", required=True, type=str)
    ap.add_argument("-m", "--metaPath", required=True, type=str)
    ap.add_argument("-s", "--samplePath", required=True, type=str)
    ap.add_argument("-o", "--outputName", required=True, type=str)
    ap.add_argument("-min", "--min_size", default=10, required=False)
    ap.add_argument("-da", "--darknetFolder", required=True, type=str)
    ap.add_argument("-sd", "--showDetections", default=False, required=False, type=bool)

    args = vars(ap.parse_args())

    configPath = args["configPath"]
    weightPath = args["weightPath"]
    metaPath = args["metaPath"]
    sample_folder = args["samplePath"]
    outputName = args["outputName"]
    min_size = int(args["min_size"])
    darknetFolder = args["darknetFolder"]
    showDetections = args["showDetections"]

    sys.path.append(darknetFolder)
    import darknet

    sample_paths = [
        join(sample_folder, f)
        for f in listdir(sample_folder)
        if str(join(sample_folder, f)).split(".")[-1] == "JPG"
    ]

    netMain = None
    metaMain = None
    altNames = None

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(
            configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1
        )  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re

                match = re.search(
                    "names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE
                )
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
    darknet_image = darknet.make_image(
        darknet.network_width(netMain), darknet.network_height(netMain), 3
    )

    all_detections = []

    def scale_detections(x, y, network_w, network_h, output_w, output_h):
        scaled_x = x * (output_w / network_w)
        scaled_y = (network_h - y) * (output_h / network_h)  # y is inverted
        return [scaled_x, scaled_y]

    for s, sample in enumerate(sample_paths):
        frame_read = cv2.imread(sample)
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb,
            (darknet.network_width(netMain), darknet.network_height(netMain)),
            interpolation=cv2.INTER_LINEAR,
        )

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        # thresh : detection threshold -> lower = more sensitive
        # nms : non maximum suppression -> higher = allow for closer proximity between detections
        detections = darknet.detect_image(
            netMain, metaMain, darknet_image, thresh=0.25, nms=0.55
        )
        viable_detections = []

        for detection in detections:
            if detection[2][2] >= min_size and detection[2][3] >= min_size:
                viable_detections.append(detection)

        all_detections.append(viable_detections)

        print("Frame: {}".format(s))

        if showDetections:
            image = cvDrawBoxes(
                viable_detections, frame_resized, min_size=min_size
            )  # , constant_size=50)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow("Detections on video", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    with open(outputName + ".pkl", "wb") as fp:
        pickle.dump(all_detections, fp)
