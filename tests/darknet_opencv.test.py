import os
import shutil
import struct
import sys
import tempfile
import types
import unittest

import numpy as np

try:
    try:
        from omni_trax.utils.darknet import darknet_opencv as dk
    except ImportError:
        from inspect import getsourcefile

        project_dir = os.path.dirname(os.path.dirname(getsourcefile(lambda: 0)))
        sys.path.append(project_dir)
        from utils.darknet import darknet_opencv as dk

    def resize_reference(frame, w, h):
        """
        Line-by-line port of darknet's resize_image() (image.c), applied to a
        frame converted the way copy_image_from_bytes() does.
        """
        im = (frame.astype(np.float64) / 255.0).astype(np.float32)
        H, W, C = im.shape
        f32 = np.float32
        part = np.zeros((H, w, C), np.float32)
        w_scale = f32(W - 1) / f32(w - 1) if w > 1 else f32(0)
        h_scale = f32(H - 1) / f32(h - 1) if h > 1 else f32(0)
        for k in range(C):
            for r in range(H):
                for c in range(w):
                    if c == w - 1 or W == 1:
                        val = im[r, W - 1, k]
                    else:
                        sx = f32(c) * w_scale
                        ix = int(sx)
                        dx = sx - f32(ix)
                        val = (f32(1) - dx) * im[r, ix, k] + dx * im[r, ix + 1, k]
                    part[r, c, k] = val
        resized = np.zeros((h, w, C), np.float32)
        for k in range(C):
            for r in range(h):
                sy = f32(r) * h_scale
                iy = int(sy)
                dy = sy - f32(iy)
                for c in range(w):
                    resized[r, c, k] = (f32(1) - dy) * part[iy, c, k]
                if r == h - 1 or H == 1:
                    continue
                for c in range(w):
                    resized[r, c, k] += dy * part[iy + 1, c, k]
        return resized

    class FakeNet:
        """Stands in for cv2.dnn.Net, returning fixed detection layer outputs."""

        def __init__(self, outputs):
            self.outputs = outputs

        def setInput(self, blob):
            self.blob = blob

        def forward(self, names):
            return self.outputs

    def fake_network(rows):
        rows = np.array(rows, dtype=np.float32)
        return dk.DarknetNetwork(FakeNet([rows]), 32, 32, 3, ["yolo"])

    def blank_image(w, h):
        image = dk.make_image(w, h, 3)
        dk.copy_image_from_bytes(image, bytes(w * h * 3))
        return image

    class TestDarknetOpenCV(unittest.TestCase):
        def test_resize_matches_darknet(self):
            rng = np.random.default_rng(0)
            for (H, W), (h, w) in [
                ((7, 9), (4, 5)),  # downscale
                ((5, 4), (11, 9)),  # upscale
                ((6, 6), (6, 6)),  # same size
                ((1, 8), (3, 4)),  # single row
                ((8, 1), (4, 3)),  # single column
                ((17, 31), (8, 8)),  # anisotropic
                # the last row is sampled just below the last input row in
                # float32, Darknet then keeps only (1 - dy) * row H-2 (about 0)
                ((14, 6), (12, 5)),
                ((720, 4), (544, 4)),
            ]:
                frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
                expected = resize_reference(frame, w, h)
                result = dk._resize_like_darknet(frame, w, h)
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_array_equal(result, expected)
                if (H, h) in ((14, 12), (720, 544)):
                    self.assertLess(result[-1].max(), 1e-3)

        def test_copy_image_from_bytes(self):
            image = dk.make_image(4, 3, 3)
            np.testing.assert_array_equal(image.data, 0)
            frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            dk.copy_image_from_bytes(image, frame.tobytes())
            np.testing.assert_array_equal(image.data, frame)
            with self.assertRaises(ValueError):
                dk.copy_image_from_bytes(image, bytes(35))

        def test_cfg_for_opencv(self):
            cfg = (
                "# model\r\n[network]\r\nwidth=64\r\nthresh = 0.7\r\nwidth=32\r\n\r\n"
                "[ yolo ]\r\nclasses=1\r\nanchors = 10,13, 16,30 # small\r\n"
                "thresh = .5\r\nnms_threshold=0.4\r\n; comment\r\nnot a key\r\n"
                "[region]\r\nclasses=2\r\n[convolutional]\r\nthresh=0.3\r\n"
            )
            lines = dk._cfg_for_opencv(dk._parse_cfg(cfg)).splitlines()
            self.assertEqual(
                lines,
                [
                    "[net]",
                    "width=64",
                    "thresh=0.7",
                    "",
                    "[yolo]",
                    "classes=1",
                    "anchors=10,13, 16,30",
                    "thresh=0",
                    "nms_threshold=0",
                    "",
                    "[region]",
                    "classes=2",
                    "thresh=0",
                    "nms_threshold=0",
                    "",
                    "[convolutional]",
                    "thresh=0.3",
                ],
            )

        def test_check_opencv_support(self):
            sections = dk._parse_cfg("[net]\n[yolo]\nnew_coords=1 # scaled\n")
            original = dk.cv2
            try:
                dk.cv2 = types.SimpleNamespace(dnn=original.dnn, __version__="4.5.4")
                dk._check_opencv_support(sections, "x.cfg")
                dk.cv2 = types.SimpleNamespace(dnn=original.dnn, __version__="4.5.3")
                with self.assertRaises(RuntimeError):
                    dk._check_opencv_support(sections, "x.cfg")
                # OpenCV 5 has no Darknet importer
                dk.cv2 = types.SimpleNamespace(
                    dnn=types.SimpleNamespace(), __version__="5.0.0"
                )
                with self.assertRaises(RuntimeError):
                    dk._check_opencv_support(dk._parse_cfg("[net]\n"), "x.cfg")
            finally:
                dk.cv2 = original

        def test_network_input_size(self):
            sections = dk._parse_cfg(
                "# model\n[net]\nwidth = 416\nheight=320\nwidth=1\n[yolo]\n"
            )
            self.assertEqual(dk._network_input_size(sections, "x.cfg"), (416, 320, 3))
            with self.assertRaises(ValueError):
                dk._network_input_size(dk._parse_cfg("[yolo]\nwidth=1\n"), "x.cfg")
            with self.assertRaises(ValueError):
                dk._network_input_size(dk._parse_cfg("[net]\nheight=32\n"), "x.cfg")

        def test_load_class_names(self):
            tmp = tempfile.mkdtemp(prefix="omnitrax names ")
            try:
                with open(os.path.join(tmp, "obj names.txt"), "w", newline="") as f:
                    f.write("ant\r\nbig ant\r\nqueen\r\n")
                data = os.path.join(tmp, "obj.data")
                with open(data, "w") as f:
                    # relative to the .data file, path with a space, first entry wins
                    f.write("classes = 2\nnames = obj names.txt\nnames = other.names\n")
                self.assertEqual(dk._load_class_names(data), ["ant", "big ant"])
                with open(data, "w") as f:
                    f.write("labels=" + os.path.join(tmp, "obj names.txt") + "\n")
                # darknet's default is classes = 2
                self.assertEqual(dk._load_class_names(data), ["ant", "big ant"])
                with open(data, "w") as f:
                    f.write("classes = 4\nnames = obj names.txt\n")
                with self.assertRaises(ValueError):
                    dk._load_class_names(data)
                # inline comment after the number, and darknet's reading of the
                # path without whitespace
                with open(os.path.join(tmp, "nospace.names"), "w") as f:
                    f.write("worker\nsoldier\n")
                with open(data, "w") as f:
                    f.write("classes = 1 # one\nnames = no space.names\n")
                self.assertEqual(dk._load_class_names(data), ["worker"])
                with open(data, "w") as f:
                    f.write("classes = 1\nnames = missing.names\n")
                with self.assertRaises(FileNotFoundError):
                    dk._load_class_names(data)
                # Blender 2.92 writes .data files in the system's code page
                folder = os.path.join(tmp, "Bajić")
                os.makedirs(folder)
                with open(os.path.join(folder, "ants.names"), "w") as f:
                    f.write("ant\n")
                line = "classes = 1\nnames = " + os.path.join(folder, "ants.names")
                with open(data, "wb") as f:
                    f.write(line.encode("cp1250"))
                original = dk.locale.getpreferredencoding
                dk.locale.getpreferredencoding = lambda do_setlocale=True: "cp1250"
                try:
                    self.assertEqual(dk._load_class_names(data), ["ant"])
                finally:
                    dk.locale.getpreferredencoding = original
            finally:
                shutil.rmtree(tmp)

        def test_detection_postprocessing(self):
            # rows: x, y, w, h (relative), objectness, score class a, score class b
            rows = [
                [0.5, 0.5, 0.2, 0.2, 1, 0.875, 0],  # A
                [0.52, 0.5, 0.2, 0.2, 1, 0.75, 0],  # B overlaps A (IoU 0.82)
                [0.1, 0.1, 0.1, 0.1, 1, 0.625, 0],  # C
                [0.5, 0.5, 0.2, 0.2, 1, 0, 0.5],  # D same box as A, other class
                [0.9, 0.9, 0.1, 0.1, 1, 0.4, 0],  # E score == threshold
                [0.3, 0.7, 0.1, 0.1, 1, 0.9375, 0.9375],  # F two classes
                [0.7, 0.2, 0.05, 0.05, 1, 1.0, 0],  # G
            ]
            names = ["a", "b"]
            image = blank_image(200, 100)
            a, c, d = (100, 50, 40, 20), (20, 10, 20, 10), (100, 50, 40, 20)
            f, g = (60, 70, 20, 10), (140, 20, 10, 5)

            def check(result, expected):
                self.assertEqual(
                    [(label, s) for label, s, _ in result],
                    [(label, s) for label, s, _ in expected],
                )
                for (_, _, box), (_, _, expected_box) in zip(result, expected):
                    np.testing.assert_allclose(box, expected_box, atol=1e-4)
                    self.assertTrue(all(type(v) is float for v in box))

            # strict threshold, per class NMS, sorted by the confidence string
            check(
                dk.detect_image(fake_network(rows), names, image, thresh=0.4, nms=0.45),
                [
                    ("a", "100.0", g),
                    ("b", "50.0", d),
                    ("a", "62.5", c),
                    ("a", "87.5", a),
                    ("a", "93.75", f),
                    ("b", "93.75", f),
                ],
            )
            # nms = 0 disables suppression
            check(
                dk.detect_image(fake_network(rows), names, image, thresh=0.4, nms=0),
                [
                    ("a", "100.0", g),
                    ("b", "50.0", d),
                    ("a", "62.5", c),
                    ("a", "75.0", (104, 50, 40, 20)),
                    ("a", "87.5", a),
                    ("a", "93.75", f),
                    ("b", "93.75", f),
                ],
            )
            # B survives when the NMS threshold is above its IoU with A
            result = dk.detect_image(
                fake_network(rows), names, image, thresh=0.4, nms=0.9
            )
            self.assertIn("75.0", [s for _, s, _ in result])
            with self.assertRaises(ValueError):
                dk.detect_image(fake_network(rows), names + ["c"], image)

            # greedy NMS: A suppresses B, so B no longer suppresses C, although
            # IoU(B, C) = 0.6 (IoU(A, B) = 0.6, IoU(A, C) = 1/3)
            chain = [
                [0.5, 0.5, 0.2, 0.2, 1, 0.9],
                [0.55, 0.5, 0.2, 0.2, 1, 0.8],
                [0.6, 0.5, 0.2, 0.2, 1, 0.7],
            ]
            check(
                dk.detect_image(
                    fake_network(chain), ["a"], image, thresh=0.25, nms=0.45
                ),
                [("a", "70.0", (120, 50, 40, 20)), ("a", "90.0", (100, 50, 40, 20))],
            )
            # an IoU equal to the NMS threshold does not suppress
            a_and_c = fake_network([chain[0], chain[2]])
            self.assertEqual(
                len(dk.detect_image(a_and_c, ["a"], image, thresh=0.25, nms=1 / 3)), 2
            )
            self.assertEqual(
                len(dk.detect_image(a_and_c, ["a"], image, thresh=0.25, nms=0.33)), 1
            )

            # equal confidence strings keep Darknet's order: do_nms_sort leaves the
            # rows sorted by the score of the last class, without NMS they keep
            # the order of the network outputs
            pair = [
                [0.1, 0.5, 0.1, 0.1, 1, 0.7500001, 0.5],
                [0.9, 0.5, 0.1, 0.1, 1, 0.75, 0.6],
            ]
            for nms, expected in [
                (
                    0.45,
                    [
                        ("b", "50.0", 20),
                        ("b", "60.0", 180),
                        ("a", "75.0", 180),
                        ("a", "75.0", 20),
                    ],
                ),
                (
                    0,
                    [
                        ("b", "50.0", 20),
                        ("b", "60.0", 180),
                        ("a", "75.0", 20),
                        ("a", "75.0", 180),
                    ],
                ),
            ]:
                result = dk.detect_image(
                    fake_network(pair), names, image, thresh=0.25, nms=nms
                )
                self.assertEqual([(n, s, round(b[0])) for n, s, b in result], expected)

        def test_weights_layout(self):
            cfg = dk._parse_cfg(
                "[net]\nchannels=3\n"
                "[convolutional]\nbatch_normalize=1\nfilters=8\nsize=3\n"  # 0
                "[convolutional]\nfilters=4\nsize=1\ngroups=2\n"  # 1
                "[route]\nlayers=-1,-2\n"  # 2: 4 + 8 channels
                "[convolutional]\nbatch_normalize=1\nfilters=6\nsize=1\n"  # 3
                "[shortcut]\nfrom=-2\n[maxpool]\nsize=2\nstride=2\n[upsample]\n"
                "[route]\nlayers=0\ngroups=2\ngroup_id=1\n"  # 7: 8 / 2 channels
                "[convolutional]\nbatch_normalize=1\nfilters=2\nsize=1\n"  # 8
                "[yolo]\nclasses=1\n"
            )
            layer_sizes = [
                8 * 4 + 3 * 8 * 9,
                4 + (8 // 2) * 4,
                6 * 4 + 12 * 6,
                2 * 4 + 4 * 2,
            ]
            header = 20
            buffer = np.zeros(header + 4 * sum(layer_sizes), np.uint8)
            buffer[:12] = np.frombuffer(struct.pack("<iii", 0, 2, 0), np.uint8)
            start = [header + 4 * sum(layer_sizes[:i]) for i in range(4)]
            self.assertEqual(
                dk._weights_layout(cfg, buffer),
                (
                    [
                        (start[0] + 4 * 24, 8),
                        (start[2] + 4 * 18, 6),
                        (start[3] + 4 * 6, 2),
                    ],
                    buffer.size,
                ),
            )
            # weights format 0.1 has a 32 bit 'seen' counter
            old = buffer[4:].copy()
            old[:8] = np.frombuffer(struct.pack("<ii", 0, 1), np.uint8)
            self.assertEqual(dk._weights_layout(cfg, old)[1], buffer.size - 4)
            # a layer whose weights are not handled here
            self.assertIsNone(dk._weights_layout(cfg + [("[reorg]", {})], buffer))

            # the batch normalisation adjustment only changes the variances
            weights = np.random.default_rng(0).uniform(0.1, 1, buffer.size // 4)
            buffer = weights.astype("<f4").view(np.uint8).copy()
            buffer[:12] = np.frombuffer(struct.pack("<iii", 0, 2, 0), np.uint8)
            before = buffer.copy()
            slices, _ = dk._weights_layout(cfg, buffer)
            dk._match_darknet_batchnorm(buffer, slices)
            changed = np.zeros(buffer.size, bool)
            changed[before != buffer] = True
            expected = np.zeros(buffer.size, bool)
            for offset, count in slices:
                expected[offset : offset + 4 * count] = True
            self.assertFalse((changed & ~expected).any())
            delta = dk._DARKNET_BATCHNORM_EPS - dk._opencv_batchnorm_epsilon()
            for offset, count in slices:
                np.testing.assert_allclose(
                    buffer[offset : offset + 4 * count].view("<f4"),
                    before[offset : offset + 4 * count].view("<f4") + delta,
                    rtol=1e-6,
                )

        def test_opencv_batchnorm_epsilon(self):
            eps = dk._opencv_batchnorm_epsilon()
            self.assertTrue(0 < eps < 1e-4)

        def test_tiny_network_matches_darknet_maths(self):
            """
            Load a small generated YOLO model through load_network and compare
            OpenCV's outputs with Darknet's maths (batch normalisation with
            eps = 1e-5, YOLO box decoding, scores = objectness * class probability).
            """
            rng = np.random.default_rng(1)
            # small variances, where the batch normalisation epsilon matters, with
            # kernels and means scaled to match so that activations stay around 1
            var = np.array([1e-6, 1e-5, 1e-3, 0.5], np.float32)
            scale = np.array([0.002, 0.003, 0.03, 1.0])
            w1 = (rng.normal(0, 1, (4, 3)) * scale[:, None]).astype(np.float32)
            mean = (rng.normal(0, 0.1, 4) * scale).astype(np.float32)
            beta = rng.normal(0, 0.1, 4).astype(np.float32)
            gamma = rng.uniform(0.5, 1.5, 4).astype(np.float32)
            w2 = rng.normal(0, 0.5, (18, 4)).astype(np.float32)
            b2 = rng.normal(0, 0.5, 18).astype(np.float32)
            anchors = [(4, 6), (10, 8), (16, 20)]
            scale_x_y = 1.1

            cfg = (
                "[net]\nwidth=32\nheight=32\nchannels=3\n\n"
                "[convolutional]\nbatch_normalize=1\nfilters=4\nsize=1\nstride=1\n"
                "pad=0\nactivation=linear\n\n"
                "[convolutional]\nfilters=18\nsize=1\nstride=1\npad=0\n"
                "activation=linear\n\n"
                "[yolo]\nmask=0,1,2\nanchors=4,6, 10,8, 16,20\nclasses=1\nnum=3\n"
                "scale_x_y=1.1\n"
            )
            weights = (
                struct.pack("<iiiQ", 0, 2, 0, 0)
                + np.concatenate([beta, gamma, mean, var, w1.ravel(), b2, w2.ravel()])
                .astype("<f4")
                .tobytes()
            )

            # a folder name with spaces and non-ASCII characters
            tmp = tempfile.mkdtemp(prefix="omnitrax tiny éć ")
            try:
                for name, content in [
                    ("tiny.cfg", cfg.encode()),
                    ("tiny.weights", weights),
                    ("tiny.names", b"ant\n"),
                    ("tiny.data", b"classes = 1\nnames = tiny.names\n"),
                ]:
                    with open(os.path.join(tmp, name), "wb") as f:
                        f.write(content)
                network, names, colors = dk.load_network(
                    os.path.join(tmp, "tiny.cfg"),
                    os.path.join(tmp, "tiny.data"),
                    os.path.join(tmp, "tiny.weights"),
                    batch_size=1,
                )
                # an incomplete weights file is reported instead of loaded
                for size in (len(weights) - 4, 15, 0):
                    with open(os.path.join(tmp, "tiny.weights"), "wb") as f:
                        f.write(weights[:size])
                    with self.assertRaises(ValueError):
                        dk.load_network(
                            os.path.join(tmp, "tiny.cfg"),
                            os.path.join(tmp, "tiny.data"),
                            os.path.join(tmp, "tiny.weights"),
                        )
            finally:
                shutil.rmtree(tmp)
            self.assertEqual(names, ["ant"])
            self.assertEqual(list(colors), ["ant"])
            self.assertEqual(
                (dk.network_width(network), dk.network_height(network)), (32, 32)
            )

            W, H = 48, 40
            frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
            image = dk.make_image(W, H, 3)
            dk.copy_image_from_bytes(image, frame.tobytes())

            # Darknet's maths, in float64
            x = dk._resize_like_darknet(frame, 32, 32).astype(np.float64)
            x = x.transpose(2, 0, 1).reshape(3, -1)
            bn = (
                gamma[:, None] * (w1 @ x - mean[:, None]) / np.sqrt(var[:, None] + 1e-5)
            )
            out = (w2 @ (bn + beta[:, None]) + b2[:, None]).reshape(3, 6, 32, 32)

            def logistic(v):
                return 1 / (1 + np.exp(-v))

            rows, cols = np.mgrid[0:32, 0:32]
            expected_boxes, expected_scores = [], []
            for a, (aw, ah) in enumerate(anchors):
                t = out[a]
                bx = (cols + logistic(t[0]) * scale_x_y - 0.5 * (scale_x_y - 1)) / 32
                by = (rows + logistic(t[1]) * scale_x_y - 0.5 * (scale_x_y - 1)) / 32
                bw = np.exp(t[2]) * aw / 32
                bh = np.exp(t[3]) * ah / 32
                expected_boxes.append(np.stack([bx * W, by * H, bw * W, bh * H], -1))
                expected_scores.append(logistic(t[4]) * logistic(t[5]))
            # Darknet's order: grid cells row by row, anchors within a cell
            expected_boxes = np.stack(expected_boxes, 2).reshape(-1, 4)
            expected_scores = np.stack(expected_scores, 2).reshape(-1, 1)

            boxes, scores = dk._boxes_and_scores(dk._forward(network, image), 1, W, H)
            np.testing.assert_allclose(boxes, expected_boxes, rtol=1e-4, atol=1e-3)
            np.testing.assert_allclose(scores, expected_scores, rtol=1e-4, atol=1e-5)
            # low scores must not be dropped by OpenCV's own region layer threshold
            self.assertTrue((scores < 0.2).any() and (scores > 0).all())

            detections = dk.detect_image(network, names, image, thresh=0.25, nms=0.45)
            self.assertTrue(len(detections) > 0)
            for label, confidence, box in detections:
                self.assertEqual(label, "ant")
                self.assertGreater(float(confidence), 25)
                self.assertEqual(len(box), 4)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDarknetOpenCV)
    success = unittest.TextTestRunner().run(suite)

    if success.errors or success.failures:
        raise Exception

except Exception as e:
    print(e)
    sys.exit(1)
