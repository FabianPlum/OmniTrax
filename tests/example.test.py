import sys
import unittest

try:
    class TestMain(unittest.TestCase):

        def test_addValues(self):
            values = [1, 2, 3]
            sum = 0
            for val in values:
                sum += val
            target = 6

            self.assertEqual(sum, target)


    # we have to manually invoke the test runner here, as we cannot use the CLI
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMain)
    success = unittest.TextTestRunner().run(suite)

    if success.errors or success.failures:
        raise Exception

except Exception:
    sys.exit(1)
