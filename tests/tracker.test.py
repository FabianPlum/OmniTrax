import sys
import unittest

try:
    from omni_trax.tracker import *


    class TestTracker(unittest.TestCase):

        def test_update_single(self):
            dist_thresh_1 = 50
            detections_1 = [np.array([[100], [200]]),
                            np.array([[300], [400]])]
            target_1 = [np.array([[100], [200]]),
                        np.array([[300], [400]])]
            bbox = [0, 0, 0, 0]

            tracker = Tracker(dist_thresh=dist_thresh_1,
                              max_frames_to_skip=2,
                              max_trace_length=10,
                              trackIdCount=0,
                              use_kf=False,
                              dt=1)

            tracker.Update(detections=detections_1,
                           predicted_classes=[0, 0],
                           bounding_boxes=[bbox, bbox])

            self.assertEqual(len(tracker.tracks), 2)
            self.assertIsNone(np.testing.assert_array_equal(tracker.tracks[0].trace[-1], target_1[0]))
            self.assertIsNone(np.testing.assert_array_equal(tracker.tracks[1].trace[-1], target_1[1]))

        def test_update_multi(self):
            dist_thresh_1 = 200
            dist_thresh_2 = 300
            detections_1 = [np.array([[100], [200]])]
            detections_2 = [np.array([[300], [400]]),
                            np.array([[500], [600]])]
            target_1 = [np.array([[100], [200]]),
                        np.array([[300], [400]]),
                        np.array([[500], [600]])]
            target_2 = [np.array([[300], [400]]),
                        np.array([[500], [600]])]
            bbox = [0, 0, 0, 0]

            tracker = Tracker(dist_thresh=dist_thresh_1,
                              max_frames_to_skip=2,
                              max_trace_length=10,
                              trackIdCount=0,
                              use_kf=False,
                              dt=1)

            tracker.Update(detections=detections_1,
                           predicted_classes=[0, 0],
                           bounding_boxes=[bbox, bbox])

            self.assertEqual(len(tracker.tracks), 1)

            tracker.Update(detections=detections_2,
                           predicted_classes=[0, 0],
                           bounding_boxes=[bbox, bbox])

            self.assertEqual(len(tracker.tracks), 3)
            self.assertIsNone(np.testing.assert_array_equal(tracker.tracks[0].trace[-1], target_1[0]))
            self.assertIsNone(np.testing.assert_array_equal(tracker.tracks[1].trace[-1], target_1[1]))
            self.assertIsNone(np.testing.assert_array_equal(tracker.tracks[2].trace[-1], target_1[2]))

            tracker = Tracker(dist_thresh=dist_thresh_2,
                              max_frames_to_skip=2,
                              max_trace_length=10,
                              trackIdCount=0,
                              use_kf=False,
                              dt=1)

            tracker.Update(detections=detections_1,
                           predicted_classes=[0, 0],
                           bounding_boxes=[bbox, bbox])

            self.assertEqual(len(tracker.tracks), 1)

            tracker.Update(detections=detections_2,
                           predicted_classes=[0, 0],
                           bounding_boxes=[bbox, bbox])

            self.assertEqual(len(tracker.tracks), 2)
            self.assertIsNone(np.testing.assert_array_equal(tracker.tracks[0].trace[-1], target_2[0]))
            self.assertIsNone(np.testing.assert_array_equal(tracker.tracks[1].trace[-1], target_2[1]))

        def test_update_KF(self):
            dist_thresh_1 = 600
            detections_1 = [np.array([[100], [200]]),
                            np.array([[300], [400]])]
            detections_2 = [np.array([[300], [400]]),
                            np.array([[500], [600]])]
            target_1 = [np.array([[295], [395]]),
                        np.array([[495], [595]])]
            bbox = [0, 0, 0, 0]

            tracker = Tracker(dist_thresh=dist_thresh_1,
                              max_frames_to_skip=2,
                              max_trace_length=10,
                              trackIdCount=0,
                              use_kf=True,
                              std_acc=10,
                              x_std_meas=1,
                              y_std_meas=1,
                              dt=1)

            tracker.Update(detections=detections_1,
                           predicted_classes=[0, 0],
                           bounding_boxes=[bbox, bbox])

            tracker.Update(detections=detections_2,
                           predicted_classes=[0, 0],
                           bounding_boxes=[bbox, bbox])

            self.assertEqual(len(tracker.tracks), 2)
            self.assertIsNone(np.testing.assert_array_equal(tracker.tracks[0].trace[-1], target_1[0]))
            self.assertIsNone(np.testing.assert_array_equal(tracker.tracks[1].trace[-1], target_1[1]))


    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTracker)
    success = unittest.TextTestRunner().run(suite)

    if success.errors or success.failures:
        raise Exception

except Exception:
    sys.exit(1)
