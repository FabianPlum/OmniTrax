import sys
import os
import unittest
import numpy as np

try:
    try:
        from omni_trax.utils.track.kalman_filter_new import KalmanFilter
    except:
        from inspect import getsourcefile

        project_dir = os.path.dirname(os.path.dirname(getsourcefile(lambda: 0)))
        sys.path.append(project_dir)
        from utils.track.kalman_filter_new import KalmanFilter

    class TestKalman(unittest.TestCase):
        def test_predict(self):
            values = np.matrix([[100], [200]])
            target = np.matrix([[150], [300]])
            KF = KalmanFilter(
                dt=1,
                u_x=100,
                u_y=200,
                std_acc=10,
                x_std_meas=0.25,
                y_std_meas=0.25,
                initial_state=values,
            )

            self.assertIsNone(np.testing.assert_array_equal(KF.predict(), target))

        def test_update(self):
            values_1 = [[100], [200]]
            values_2 = [[300], [400]]

            target_1 = [[150], [300]]
            target_2 = [[225], [350]]
            target_3 = [[250], [367]]

            KF = KalmanFilter(
                dt=1,
                u_x=100,
                u_y=200,
                std_acc=10,
                x_std_meas=0.25,
                y_std_meas=0.25,
                initial_state=values_1,
            )
            KF.predict()

            self.assertIsNone(
                np.testing.assert_array_equal(
                    KF.update(z=values_2, flag=False), target_1
                )
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    KF.update(z=values_2, flag=True), target_2
                )
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    KF.update(z=values_2, flag=True), target_3
                )
            )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestKalman)
    success = unittest.TextTestRunner().run(suite)

    if success.errors or success.failures:
        raise Exception

except Exception as e:
    print(e)
    sys.exit(1)
