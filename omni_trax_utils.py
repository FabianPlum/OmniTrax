import numpy as np
import cv2
import csv
import os
import math
import time
import tensorflow as tf


### Various Detection Processing functions ###

def scale_detections(x, y, network_w, network_h, output_w, output_h):
    """
    scaling detections from their original darknet output to the blender XY convention
    :param x: x coordinate (darknet pixel space)
    :param y: y coordinate (darknet pixel space)
    :param network_w: darknet network width
    :param network_h: darknet network height
    :param output_w: video width
    :param output_h: video height
    :return: array of scaled detections
    """
    scaled_x = x * (output_w / network_w)
    scaled_y = (network_h - y) * (output_h / network_h)  # y is inverted
    return [scaled_x, scaled_y]


def convertBack(x, y, w, h):
    """
    Get bounding box shape for OpenCV plotting functions from darknet detections
    :param x: x coordinate (darknet pixel space)
    :param y: y coordinate (darknet pixel space)
    :param w: width of darknet detection
    :param h: height of darknet detection
    :return: converted bounding box (for OpenCV display)
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, min_size=20, constant_size=False, class_colours=None):
    """
    draw detection bounding boxes on input image
    :param detections: darknet detection format (x,y,w,h)
    :param img: input (colour) image [H,W,C]
    :param min_size: minimum detection width and height for display
    :param constant_size: enforcing constant (square) bounding boxes at this pixel size
    :param class_colours: RBG colours for each class
    :return: image with superimposed detections
    """
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


def setInferenceDevive(device):
    """
    Set the (CUDA) inference device for the project
    :param device: e.g. "CPU_0" or "GPU_0"
    """
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


def import_tracks(path, numFrames, export=False):
    """
    Import all tracked paths (using blender motionExport.py) from specified folder and join them to a single array.
    Optionally, allows for export of created array containing all tracks into single .csv file
    :param path: location of exported .csv tracks
    :param numFrames: number of total analysed frames
    :param export: boolean, writes .csv file of all combined tracks if True
    :return: array of all imported tracks, row: frames, columns X / Y coordinates of individual track.
             The first column consists of the frame numbers for easier readability if exported as a single file.
    """
    print("importing tracks...")
    files = []
    tracks = np.empty([numFrames + 1, 1])  # create array for all tracks
    tracks[:, 0] = np.arange(start=1, stop=numFrames + 2, step=1, dtype=int)  # insert frame numbers

    imported = 0

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))

                # for each new track create two "zeros" columns
                # zeros are handled as nonexistent instances
                tracks = np.append(tracks, np.zeros([numFrames + 1, 2]), axis=1)

                with open(files[imported]) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=';')
                    line_count = 0

                    next(csv_reader, None)  # skip the headers

                    for row in csv_reader:
                        tracks[int(row[0]) - 1, imported * 2 + 1] = int(row[1])
                        tracks[int(row[0]) - 1, imported * 2 + 2] = int(row[2])
                        line_count += 1
                    print("imported", str(file), f' with {line_count} points.')

                imported += 1

    tracks = tracks.astype(int)
    if export:
        export_path = path + "_all_tracks.csv"
        np.savetxt(export_path, tracks, delimiter=",")

    print("\nSuccessfully combined the tracks of", imported, "individuals for training and display!")
    return tracks


def display_video(cap, tracks, show=(0, math.inf), scale=1.0, target_size=100):
    """
    Function displays imported footage with tracking results as overlay
    :param cap: Imported video file
    :param tracks: all imported tracks as a single array, created with import_tracks()
    :param show: tuple of desired displayed frames
    :param scale: single float to up- or downscale resolution of display
    """
    tracks = (scale * tracks).astype(int)  # rescale pixel values of tracks
    # frame counter
    frame_num = show[0]

    # define the size of each tracking rectangle
    target_size *= scale

    # get frame rate of imported footage
    fps = cap.get(cv2.CAP_PROP_FPS)

    # fix the seed for the same set of randomly assigned colours for each track
    np.random.seed(seed=0)
    colours = np.random.randint(low=0, high=255, size=((math.floor(((tracks.shape[1]) - 1) / 2)), 3))

    print("\nDisplaying tracked footage!\npress 'q' to end display")

    # skip to desired start frame
    # Property identifier of cv2.CV_CAP_PROP_POS_FRAMES is 1, thus the first entry is 1
    cap.set(1, show[0])

    # set font from info display on frame
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:  # run until no more frames are available
        time_prev = time.time()
        # return single frame (ret = boolean, frame = image)
        ret, frame = cap.read()
        if not ret:
            break

        # scale down the video
        new_height = int(np.shape(frame)[0] * scale)
        new_width = int(np.shape(frame)[1] * scale)
        frame = cv2.resize(frame, (new_width, new_height))

        # iterate through all columns and draw rectangles for all non 0 values
        for track in range(math.floor(((tracks.shape[1]) - 1) / 2)):
            if tracks[frame_num, track * 2 + 1] != 0:
                # the tracks are read as centres
                target_centre = np.asarray([tracks[frame_num, track * 2 + 1], tracks[frame_num, track * 2 + 2]])

                # invert y axis, to fit openCV convention ( lower left -> (x=0,y=0) )
                target_centre[1] = new_height - target_centre[1]
                # define the starting and ending point of the bounding box rectangle, defined by "target_size"
                px_start = target_centre - np.asarray([math.floor(target_size / 2), math.floor(target_size / 2)])
                px_end = target_centre + np.asarray([math.floor(target_size / 2), math.floor(target_size / 2)])
                # draw the defined rectangle of the track on top of the frame
                cv2.rectangle(frame, (px_start[0], px_start[1]), (px_end[0], px_end[1]),
                              (int(colours[track, 0]), int(colours[track, 1]), int(colours[track, 2])), 2)
                # write out track number of each active track
                cv2.putText(frame, "track: " + str(track),
                            (int(target_centre[0] - target_size / 2), int(target_centre[1] - target_size / 2 - 10)),
                            font, 0.3, (int(colours[track, 0]), int(colours[track, 1]), int(colours[track, 2])), 1,
                            cv2.LINE_AA)

        cv2.putText(frame, "frame: " + str(frame_num), (int(new_width / 2) - 100, 35),
                    font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('original frame', frame)

        if frame_num > show[1]:
            break

        # enforce constant frame rate during display
        time_to_process = (time.time() - time_prev)  # compute elapsed time to enforce constant frame rate (if possible)
        if time_to_process < 1 / fps:
            time.sleep((1 / fps) - time_to_process)

        # press q to quit, i.e. exit the display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

    cv2.destroyAllWindows()

    # always reset frame from capture at the end to avoid incorrect skips during access
    cap.set(1, 0)

    print("\nReached last frame of specified video or ended by user input.\n")


def get_exact_frame(frame_no, num_frames_max, file, display=False, num_frames=1):
    """
    extracts specific frame from video footage and displays it, if desired
    :param frame_no: integer value of desired frame
    :param num_frames_max: total number of frames within footage (could be extracted but to minimise the number of times
                           the function is executed passed directly into the function
    :param file: video (as cap from OpenCV) from which the frame(s) is/are to be extracted
    :param display: display frame(s), if desired
    :param num_frames: number of frames to be extracted (1 by default to only return a single frame) If greater than 1,
                      the PREVIOUS number of frames leading to the specified one will be extracted as well.
    :return: array of frames or single frame
    """
    if frame_no - num_frames < 0:
        num_frames = frame_no

    all_frames = []

    if frame_no > num_frames_max:
        print("ERROR: frame number exceeds total number of frames in footage!")
        return

    for i in range(1, num_frames + 1):
        # skip back specified number of frames to include previous frames if desired / possible
        frame_val = (frame_no - 1 + (i - num_frames))
        # set desired frame (Property identifier of cv2.CV_CAP_PROP_POS_FRAMES is 1, thus the first entry is 1
        file.set(1, frame_val)

        # Read the next frame from the video
        ret, frame = file.read()

        all_frames.append(frame)

        if display:
            # resize frame to fit screen
            new_height = int(np.shape(frame)[0] * 0.5)
            new_width = int(np.shape(frame)[1] * 0.5)
            frame = cv2.resize(frame, (new_width, new_height))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "frame: " + str(frame_no + (i - num_frames)), (int(new_width / 2) - 100, 35),
                        font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow("Selected frames: ", frame)
            # Set wait key
            cv2.waitKey()

    # always reset frame from capture at the end to avoid incorrect skips during access
    file.set(1, 0)

    return all_frames


def extractPatches(frame_no, frames, tracks, patch_size=128, BW=True):
    """
    extracts images patches for stacks and detections during TRAINING
    :param frame_no: desired frame of NEW detections
    :param frames: list of frames
    :param tracks: array of all labelled tracks imported
    :param patch_size: resolution (width / height in px) of extracted patches
    :param BW: Boolean, if True returns the patches in black and white
    :return: stacks of previous tracks and detections of new instances
    """
    stacks = []
    stacks_label = []
    stacks_pos = []

    detections = []
    detections_label = []
    detections_pos = []

    # convert images to black and white if required
    if BW:
        for img in range(len(frames)):
            frames[img] = cv2.cvtColor(frames[img], cv2.COLOR_BGR2GRAY)
        blank_image = np.zeros((patch_size, patch_size), np.uint8)
    else:
        # coloured images require 3 channels
        blank_image = np.zeros((patch_size, patch_size, 3), np.uint8)

    # by default no images should be blank, exception occurs at the beginning of the footage when there are only
    # detections and no initialized tracks yet.
    num_empty_img = 0

    # insert blank images to fill stacks which would otherwise be too small
    blank_stack = []

    if frame_no - len(frames) < 0:
        num_empty_img = len(frames) - frame_no
        for img in range(num_empty_img):
            blank_stack.append(blank_image)

    # iterate over all available tracks, step size of two, as X and Y are given
    for track in range(1, tracks.shape[1], 2):
        stack = []
        pos = []
        no_detection = 0
        # iterate over all imported frames
        for img in range(len(frames) - num_empty_img):
            if tracks[frame_no + (img - len(frames) + num_empty_img), track] != 0:
                # the tracks are read as centres
                target_centre = np.asarray([tracks[frame_no + (img - len(frames) + num_empty_img), track],
                                            tracks[frame_no + (img - len(frames) + num_empty_img), track + 1]])

                # invert y-axis, to fit OpenCV convention ( lower left -> (x=0,y=0) )
                target_centre[1] = frames[0].shape[0] - target_centre[1]
                # define the starting and ending point of the bounding box rectangle, defined by "target_size"
                px_start = target_centre - np.asarray([math.floor(patch_size / 2), math.floor(patch_size / 2)])
                px_end = target_centre + np.asarray([math.floor(patch_size / 2), math.floor(patch_size / 2)])
                # extract the defined rectangle of the track from the frame and save to the stack
                stack.append(frames[img][px_start[1]:px_end[1], px_start[0]:px_end[0]])

                # save the position of each patch within the stack
                pos.append(target_centre)
            else:
                # if not detection can be found insert a black image instead
                stack.append(blank_image)
                # in case of a blank image / no defined patch the position is set to 0,0
                pos.append((0, 0))
                no_detection += 1

        # only return stacks which are active (i.e. at least one detection)
        if no_detection != len(frames):
            # add stack label to identify the stack later on
            label = int(track / 2 + 0.5)
            # set the newest entry of the stack as a detection for training purposes, retaining the label
            if np.bitwise_xor(stack[-1], blank_image).any():
                detections.append(stack[-1])
                detections_label.append(label)
                detections_pos.append(pos[-1])

            # remove the last entry of the stack, to only have in represented as a new detection
            del stack[-1]
            del pos[-1]
            # only return stacks if they are not empty without the newest detection
            if no_detection + 1 != len(frames):
                stacks.append(stack)
                stacks_label.append(label)
                stacks_pos.append(pos)

    # convert all outputs to numpy arrays
    stacks = np.array(stacks)
    stacks_label = np.array(stacks_label)
    stacks_pos = np.array(stacks_pos)

    detections = np.array(detections)
    detections_label = np.array(detections_label)
    detections_pos = np.array(detections_pos)

    return stacks, stacks_label, stacks_pos, detections, detections_label, detections_pos


def sortByDistance(detections, stack_pos, detections_pos, labels, verbose=False):
    """
    Sort detections by their distance to track stacks
    :param detections: darknet detection format (x,y,w,h)
    :param stack_pos: x,y positions of stack
    :param detections_pos: x,y positions of detections
    :param labels: labels in the order of detections
    :param verbose: print sorting results
    :return: sorted detections, labels, and detection position from closest to furthest from the stack position
    """
    all_dist = []
    # find last valid position (non-zero element) in active stack
    valid_positions = [i for i, element in enumerate(stack_pos[:, 0]) if element != 0]
    last_valid_pos = stack_pos[valid_positions[-1]]

    for i in range(len(detections)):
        # compute distance between all detections and a given active stack
        all_dist.append(
            math.sqrt(
                ((last_valid_pos[0] - detections_pos[i, 0]) ** 2) + ((last_valid_pos[1] - detections_pos[i, 1]) ** 2)))

    # sort detections based on smallest distance to given stack (closest detection first)
    detections_sorted = [i for _, i in sorted(zip(all_dist, detections), key=lambda pair: pair[0])]
    labels_sorted = [i for _, i in sorted(zip(all_dist, labels), key=lambda pair: pair[0])]
    detections_pos_sorted = [i for _, i in sorted(zip(all_dist, detections_pos), key=lambda pair: pair[0])]
    all_dist.sort()

    if verbose:
        print("\nvalid positions:", valid_positions)
        print("last valid position:", last_valid_pos)
        print("input labels:", labels)
        print("Sorted by distance to stack:", labels_sorted)
        print("Ascending pixel distances:", all_dist)

    return detections_sorted, labels_sorted, detections_pos_sorted
