'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan (but he's done a kinda shitty job. So yeah, I'll put my own name here)
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
from omni_trax.kalman_filter_new import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount, predicted_class=None,
                 dt=0.033, u_x=0, u_y=0,
                 std_acc=5, y_std_meas=0.1, x_std_meas=0.1):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter(dt=dt, u_x=u_x, u_y=u_y,
                               std_acc=std_acc, y_std_meas=y_std_meas, x_std_meas=x_std_meas,
                               initial_state=prediction)  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path
        if predicted_class is not None:
            # we create a list of predicted classes for each frame, so when terminating the track,
            # we can perform a majority vote to determine the most likely class.
            # Additionally, at sufficient class resolution, the predicted class can be used as part of an extended cost
            # function when linking detections to existing tracks.
            self.predicted_class = [predicted_class]


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount, use_kf=False, dt=0.033, u_x=0, u_y=0,
                 std_acc=5, y_std_meas=0.1, x_std_meas=0.1):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount
        self.use_kf = use_kf
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.y_std_meas = y_std_meas
        self.x_std_meas = x_std_meas

    def Update(self, detections, predicted_classes=None):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for a long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector was found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount, predicted_class=predicted_classes[i],
                              dt=self.dt, u_x=self.u_x, u_y=self.u_y, std_acc=self.std_acc,
                              y_std_meas=self.y_std_meas, x_std_meas=self.x_std_meas)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using euclidean distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))  # Cost matrix

        for i in range(N):
            for j in range(len(detections)):
                diff = self.tracks[i].prediction[:2] - detections[j]
                distance = np.sqrt(diff[0][0] * diff[0][0] +
                                   diff[1][0] * diff[1][0])
                cost[i][j] = distance

        # add columns equal to the number of tracks, so that if a track cannot be assigned to
        # a detection, it is instead assigned to a placeholder instead to avoid forced incorrect matches.
        # This step also removes the need to filter for "unmatchable" tracks due to large distances
        cost = np.c_[cost, np.ones((N, N)) * self.dist_thresh]

        # Use hungarian algorithm to find likely matches, minimising cost
        assignment = []
        for _ in range(N):
            assignment.append(-1)

        row_ind, col_ind = linear_sum_assignment(cost)

        for i in range(len(col_ind)):
            # lowest cost along the diagonal
            assignment[i] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(N):
            if assignment[i] == -1 or assignment[i] >= M:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                print("cost to assign", i, "is =", cost[i][assignment[i]])
                assignment[i] = -1
                un_assigned_tracks.append(i)
                self.tracks[i].skipped_frames += 1
                print("Track", i, "has been invisible for", self.tracks[i].skipped_frames, "frames!")

        print("Unassigned tracks:", un_assigned_tracks, "\n")

        # If tracks are not detected for a long time, remove them
        del_tracks = []
        for i in range(N):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)

        if len(del_tracks) > 0:  # only when skipped frames exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    print("\n!!!! Deleted track:", self.tracks[id].track_id, "\n !!!!")
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("something fucky is up...")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(M):
            if i not in assignment:
                un_assigned_detects.append(i)

        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                if predicted_classes is not None:
                    track = Track(detections[un_assigned_detects[i]],
                                  self.trackIdCount, predicted_class=predicted_classes[un_assigned_detects[i]],
                                  dt=self.dt, u_x=self.u_x, u_y=self.u_y, std_acc=self.std_acc,
                                  y_std_meas=self.y_std_meas, x_std_meas=self.x_std_meas)
                else:
                    track = Track(detections[un_assigned_detects[i]],
                                  self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
                assignment.append(-1)
                print("Started new track:", self.tracks[-1].track_id)

        print("Number of detections M:   ", len(detections))
        print("Number of Tracks N:       ", len(self.tracks))
        print("Shape of cost matrix C: ", cost.shape)

        print("\nAssignment vector:        ", assignment, "\n")

        # Update KalmanFilter state, lastResults and tracks trace

        for i in range(len(self.tracks)):
            if i in del_tracks:
                continue
            if predicted_classes is not None:
                if i not in un_assigned_tracks:
                    self.tracks[i].predicted_class.append(predicted_classes[assignment[i]])
                else:
                    self.tracks[i].predicted_class.append("")

            if self.use_kf:
                # Use Kalman Filter for track predictions
                self.tracks[i].KF.predict()

                if assignment[i] != -1:
                    self.tracks[i].skipped_frames = 0
                    self.tracks[i].prediction = self.tracks[i].KF.update(
                        detections[assignment[i]], 1)
                else:
                    if len(self.tracks[i].trace) > 1:
                        self.tracks[i].prediction = self.tracks[i].KF.update(
                            np.array([[0], [0]]), 0)

                if len(self.tracks[i].trace) > self.max_trace_length:
                    for j in range(len(self.tracks[i].trace) -
                                   self.max_trace_length):
                        del self.tracks[i].trace[j]

                self.tracks[i].trace.append(self.tracks[i].prediction[:2])
                self.tracks[i].KF.lastResult = self.tracks[i].prediction

            else:
                # No Kalman Filtering, just pure matching

                # only update the the state of matched detections.
                # unmatched tracks will retain the same state as at t-1
                if assignment[i] != -1:
                    self.tracks[i].skipped_frames = 0
                    self.tracks[i].prediction = detections[assignment[i]]

                if len(self.tracks[i].trace) > self.max_trace_length:
                    for j in range(len(self.tracks[i].trace) -
                                   self.max_trace_length):
                        del self.tracks[i].trace[j]

                self.tracks[i].trace.append(self.tracks[i].prediction[:2])
