import numpy as np
import pickle
import os
from os import listdir
from os.path import join
from pathlib import Path
import subprocess
import argparse
import time


def compare_points(gt, detection, max_dist=25):
    match = False
    px_distance = distance.euclidean(gt, detection)
    if px_distance <= max_dist:
        match = True
    return match, px_distance


def compare_frame(frame_gt, frame_detections, max_dist=0.05, network_shape=[None, None], confidence=0):
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
                norm_frame_detection = [frame_detections[j][2][0] / network_shape[0],
                                        frame_detections[j][2][1] / network_shape[1]]

            else:
                norm_frame_detection = frame_detections[j][2][0:2]

            match, px_dist = compare_points(gt=frame_gt[i][0:2],
                                            detection=norm_frame_detection,
                                            max_dist=max_dist)

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


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # Data input and output
    ap.add_argument("-md", "--modelFolder", required=True, type=str)
    ap.add_argument("-dt", "--dataFolder", required=True, type=str)
    ap.add_argument("-of", "--outputFolder", required=False, type=str,
                    default=Path(__file__).parent.resolve())
    ap.add_argument("-da", "--darknetFolder", required=True, type=str)

    # Darknet setup
    ap.add_argument("-c", "--configPath", required=True, type=str)
    ap.add_argument("-m", "--metaPath", required=True, type=str)
    ap.add_argument("-min", "--min_size", default=0.1, required=False, type=int)
    ap.add_argument("-so", "--showDetections", default="", required=False, type=bool)
    ap.add_argument("-GPU", "--GPU", default="0", required=False, type=str)

    args = vars(ap.parse_args())

    # set which GPU to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["GPU"]

    # Data input and output
    modelFolder = args["modelFolder"]
    dataFolder = args["dataFolder"]
    outputFolder = args["outputFolder"]
    darknetFolder = args["darknetFolder"]

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    start_time = time.time()

    print("Writing to output folder:", outputFolder, "\n")

    # Darknet setup
    configPath = args["configPath"]
    metaPath = args["metaPath"]
    min_size = int(args["min_size"])
    showDetections = args["showDetections"]

    model_folder = Path(args["modelFolder"])
    # get all weights files
    model_paths = [join(model_folder, f) for f in listdir(model_folder)
                   if
                   str(join(model_folder, f)).split(".")[-1] == "weights" and str(join(model_folder, f)).split(".")[0][
                       -1] == "0"]

    model_paths.sort()

    print("\nFound a total of {} trained models".format(len(model_paths)))

    sample_folder = Path(args["dataFolder"])
    # get test sample files
    sample_paths = [join(sample_folder, f) for f in listdir(sample_folder)
                    if str(join(sample_folder, f)).split(".")[-1] == "JPG"]

    print("\nFound a total of {} samples".format(len(sample_paths)))

    # read config file to determine network (input) shape
    network_shape = [None, None]
    f = open(configPath, 'r')
    Lines = f.readlines()
    # Strips the "\n" newline character
    for line in Lines:
        line_cleaned = line.strip()
        line_arr = line_cleaned.split("=")
        if line_arr[0] == "width":
            network_shape[0] = int(line_arr[1])
        if line_arr[0] == "height":
            network_shape[1] = int(line_arr[1])

        if network_shape[0] is not None and network_shape[1] is not None:
            break

    f.close()
    print("Network shape:", network_shape)

    # CALLING SUB_DARKNET.py TO PERFORM ALL DETECTIONS

    all_detections = []

    for weightPath in model_paths:
        output_name = str(os.path.join(outputFolder,
                                       os.path.basename(os.path.dirname(model_folder))
                                       + "_" + str(os.path.basename(weightPath)).split(".")[0]))
        print(output_name)

        if showDetections:
            subprocess.call(['python', 'sub_darknet.py',
                             "--darknetFolder", darknetFolder,
                             "--configPath", configPath,
                             "--weightPath", weightPath,
                             "--metaPath", metaPath,
                             "--samplePath", str(sample_folder),
                             "--outputName", output_name,
                             "--min_size", str(10),
                             "--showDetections", "True"])
        else:
            subprocess.call(['python', 'sub_darknet.py',
                             "--darknetFolder", darknetFolder,
                             "--configPath", configPath,
                             "--weightPath", weightPath,
                             "--metaPath", metaPath,
                             "--samplePath", str(sample_folder),
                             "--outputName", output_name,
                             "--min_size", str(10)])

        with open(os.path.join(outputFolder, output_name + ".pkl"), 'rb') as f:
            all_detections.append([output_name, pickle.load(f)])

        print("ran inference on {} frames, using {}".format(len(all_detections[-1][1]), output_name))

    # structure of ground truth data
    all_annotations = []
    unique_datasets = []

    print("Folder contains the following datasets:\n")

    for sample in sample_paths:
        annotation = sample.split(".")[0] + ".txt"
        dataset_name = "_".join(str(os.path.basename(sample)).split("_")[:-1])

        # get all unique datasets from input folder
        if dataset_name not in unique_datasets:
            unique_datasets.append(dataset_name)
            all_annotations.append([dataset_name])
            print(all_annotations[-1])

        f = open(annotation, 'r')
        Lines = f.readlines()

        bounding_boxes = []  # x,y,w,h
        # Strips the "\n" newline character
        for line in Lines:
            line_cleaned = line.strip()
            line_arr = line_cleaned.split(" ")[1:]
            bounding_boxes.append([float(f) for f in line_arr])

        all_annotations[-1].append(bounding_boxes)
        f.close()

    print("\nLoaded {} annotated samples in total\n".format(len(all_annotations)))
    # print(all_annotations,"\n")
    for sample in all_annotations[0][1][:3]:
        print(sample)  # first annotation of each sample [dataset , [x,y,w,h]]

    from scipy.spatial import distance

    max_detection_distance_px = 0.05  # 0.1 = 10% away from centre to be considered a valid detection
    # DEFAULT:
    # thresh_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    thresh_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                   0.95]
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

                total_gt_detections = 0  # number of total detections in the ground truth dataset
                total_missed_detections = 0  # number of missed detections which are present in the groud truth dataset
                total_false_positives = 0  # number of incorrect detections that do not match any groud thruth tracks
                all_frame_detection_deviations = []  # list of mean deviations for correct detections

                for detection, annotation in zip(model[1], unique_dataset[1:]):
                    gt_detections, missed_detections, false_positives, mean_detection_distance = compare_frame(
                        annotation,
                        detection,
                        max_detection_distance_px,
                        network_shape,
                        confidence)

                    total_gt_detections += gt_detections
                    total_missed_detections += missed_detections
                    total_false_positives += false_positives
                    all_frame_detection_deviations.append(mean_detection_distance)

                mean_px_error = np.mean(all_frame_detection_deviations) * 100
                detection_accuracy = ((
                                              total_gt_detections - total_missed_detections - total_false_positives) / total_gt_detections) * 100

                if total_gt_detections == total_missed_detections:
                    # the accuracy is zero if no objects are correctly detected
                    AP = 0
                else:
                    AP = (total_gt_detections - total_missed_detections) / (
                            total_gt_detections - total_missed_detections + total_false_positives)
                    Recall = (total_gt_detections - total_missed_detections) / total_gt_detections

                print("Total ground truth detections:", total_gt_detections)
                print("Total correct detections:", total_gt_detections - total_missed_detections)
                print("Total missed detections:", total_missed_detections)
                print("Total false positives:", total_false_positives)
                print("Average Precision:", round(AP, 3))
                print("Recall:", round(Recall, 3))
                print("Detection accuracy (GT - FP - MD) / GT):", np.round(detection_accuracy, 1), "%")
                print("Mean relative deviation: {} %\n".format(np.round(mean_px_error, 3)))

                Results_mat[-1][-1].append([unique_dataset[0],
                                            total_gt_detections,
                                            total_gt_detections - total_missed_detections,
                                            total_missed_detections,
                                            total_false_positives,
                                            AP,
                                            Recall])

    output_results = output_name.split("_")[0] + "_RESULTS.pkl"
    print(output_results)

    with open(output_results, 'wb') as fp:
        pickle.dump(Results_mat, fp)

    print("--- %s seconds ---" % (time.time() - start_time))
