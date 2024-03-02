import numpy as np
import pickle
import os
from os import listdir
from os.path import join
import time
import sys
from scipy.spatial import distance
from multiprocessing import Pool


def compare_points(gt, detection, max_dist=25):
    match = False
    px_distance = distance.euclidean(gt, detection)
    if px_distance <= max_dist:
        match = True
    return match, px_distance


def compare_frame(
    frame_gt, frame_detections, max_dist=0.05, network_shape=[None, None], confidence=0
):
    # strip away all sub threshold detections!
    frame_detections = [f for f in frame_detections if f[1] > confidence]

    matches_gt = np.ones(len(frame_gt))
    matches_det = np.ones(len(frame_detections))
    below_thresh = 0
    detection_distances = []

    # now strip all empty entries from the ground truth

    for i in range(len(matches_gt)):
        min_dist = max_dist
        for j in range(len(matches_det)):
            if network_shape[0] is not None:
                norm_frame_detection = [
                    frame_detections[j][2][0] / network_shape[0],
                    frame_detections[j][2][1] / network_shape[1],
                ]

            else:
                norm_frame_detection = frame_detections[j][2][0:2]

            match, px_dist = compare_points(
                gt=frame_gt[i][0:2], detection=norm_frame_detection, max_dist=max_dist
            )

            if match:
                matches_gt[i] = 0
                matches_det[j] = 0
                if px_dist < min_dist:
                    min_dist = px_dist

        if min_dist < max_dist:
            detection_distances.append(min_dist)

    missed_detections = int(np.sum(matches_gt))
    false_positives = int(np.sum(matches_det)) - below_thresh

    if len(detection_distances) == 0:
        mean_detection_distance = 0
    else:
        mean_detection_distance = np.mean(np.array(detection_distances))

    return len(frame_gt), missed_detections, false_positives, mean_detection_distance


def getThreads():
    """Returns the number of available threads on a posix/win based system"""
    if sys.platform == "win32":
        return int(os.environ["NUMBER_OF_PROCESSORS"])
    else:
        return int(os.popen("grep -c cores /proc/cpuinfo").read())


def process_detections(data):
    print("Running evaluation of ", data, "...")

    with open("ANNOTATIONS_ALL.pkl", "rb") as f:
        all_annotations = pickle.load(f)

    snapshots = [join(data, f) for f in listdir(data)]
    all_detections = []

    for snapshot in snapshots:
        with open(snapshot, "rb") as f:
            all_detections.append([snapshot, pickle.load(f)])

    print(
        "ran inference on {} frames, using {}".format(len(all_detections[-1][1]), data)
    )

    max_detection_distance_px = (
        0.1  # 0.1 = 10% away from centre to be considered a valid detection
    )
    thresh_list = [
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
    ]
    print("Computing AP scores for thresholds of {}".format(thresh_list))

    Results_mat = []

    # matrix shape: dataset(samples) , model x iteration x threshold

    for model in all_detections:
        print("\n", model[0])

        Results_mat.append([model[0]])

        for confidence in thresh_list:
            Results_mat[-1].append([confidence])

            print("\n running inference at {} confidence threshold".format(confidence))

            for unique_dataset in all_annotations:
                print("dataset:", unique_dataset[0])

                total_gt_detections = (
                    0  # number of total detections in the ground truth dataset
                )
                total_missed_detections = 0  # number of missed detections which are present in the groud truth dataset
                total_false_positives = 0  # number of incorrect detections that do not match any groud thruth tracks
                all_frame_detection_deviations = []  # list of mean deviations for correct detections

                for detection, annotation in zip(model[1], unique_dataset[1:]):
                    (
                        gt_detections,
                        missed_detections,
                        false_positives,
                        mean_detection_distance,
                    ) = compare_frame(
                        annotation,
                        detection,
                        max_detection_distance_px,
                        [800, 800],
                        confidence,
                    )

                    total_gt_detections += gt_detections
                    total_missed_detections += missed_detections
                    total_false_positives += false_positives
                    all_frame_detection_deviations.append(mean_detection_distance)

                mean_px_error = np.mean(all_frame_detection_deviations) * 100
                detection_accuracy = (
                    (
                        total_gt_detections
                        - total_missed_detections
                        - total_false_positives
                    )
                    / total_gt_detections
                ) * 100

                if total_gt_detections == total_missed_detections:
                    # the accuracy is zero if no objects are correctly detected
                    AP = 0
                else:
                    AP = (total_gt_detections - total_missed_detections) / (
                        total_gt_detections
                        - total_missed_detections
                        + total_false_positives
                    )
                    Recall = (
                        total_gt_detections - total_missed_detections
                    ) / total_gt_detections

                print("Total ground truth detections:", total_gt_detections)
                print(
                    "Total correct detections:",
                    total_gt_detections - total_missed_detections,
                )
                print("Total missed detections:", total_missed_detections)
                print("Total false positives:", total_false_positives)
                print("Average Precision:", round(AP, 3))
                print("Recall:", round(Recall, 3))
                print(
                    "Detection accuracy (GT - FP - MD) / GT):",
                    np.round(detection_accuracy, 1),
                    "%",
                )
                print(
                    "Mean relative deviation: {} %\n".format(np.round(mean_px_error, 3))
                )

                Results_mat[-1][-1].append(
                    [
                        unique_dataset[0],
                        total_gt_detections,
                        total_gt_detections - total_missed_detections,
                        total_missed_detections,
                        total_false_positives,
                        AP,
                        Recall,
                    ]
                )

    outputFolder = "I:\\BENCHMARK\\DARKNET_TRAIN\\EVALUATION\\RESULTS"
    output_results = join(outputFolder, str(os.path.basename(data)) + "_RESULTS.pkl")
    print(output_results)

    with open(output_results, "wb") as fp:
        pickle.dump(Results_mat, fp)

    print("--- %s seconds ---" % (time.time() - start_time))
    exit()


if __name__ == "__main__":
    # Data input and output
    modelFolder = "I:\\BENCHMARK\\DARKNET_TRAIN\\OUTPUT"
    outputFolder = "I:\\BENCHMARK\\DARKNET_TRAIN\\EVALUATION\\RESULTS"
    DEBUG = True

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    start_time = time.time()

    """
    
    AND NOW EVALUATE ALL OF THIS!
    
    """

    # get all model paths
    model_paths = [join(modelFolder, f) for f in listdir(modelFolder)]
    model_paths.sort()

    print("\nFound a total of {} trained models".format(len(model_paths)))

    # each model should be handled by a separate thread, regardless of the number of snapshots in there.
    # As soon as a thread is done with one model it should move on to the next.
    # Given this is just a simple numerical comparison, use as many threads as there are virtual cores

    # only use a fourth of the number of CPUs for export as hugin and enfuse utilise multi core processing in part

    with Pool(20) as p:
        p.map(process_detections, model_paths)
