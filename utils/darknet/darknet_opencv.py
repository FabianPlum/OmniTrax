"""
CPU inference backend for Darknet YOLO models, built on OpenCV's DNN module.

This module is a drop-in replacement for the parts of the ``darknet.py`` ctypes
wrapper that OmniTrax uses:

    load_network, make_image, copy_image_from_bytes, detect_image,
    network_width, network_height

It reads the same .cfg, .weights, .data and .names files through
``cv2.dnn.readNetFromDarknet``, so no compiled Darknet library (and none of its
DLL dependencies) is required, only the ``opencv-python`` package that OmniTrax
already installs.

Darknet's behaviour is reproduced on purpose, so that CPU results match the GPU
(Darknet) backend:

* batch normalisation uses Darknet's epsilon (OpenCV's Darknet importer uses a
  smaller one, which is compensated for in the loaded weights),
* frames are resized to the network size with Darknet's own bilinear
  ``resize_image`` (not ``cv2.resize``, which samples pixels differently),
* a class score (objectness * class probability) is kept only if it is strictly
  greater than ``thresh``,
* overlapping boxes are removed per class with Darknet's greedy ``do_nms_sort``
  (a box is suppressed if its IoU with a higher-scoring box exceeds ``nms``),
* detections are returned as ``(label, confidence, (x, y, w, h))`` with the
  confidence as a percentage string and the box centre / size in pixels of the
  image passed to ``detect_image``, in the same order Darknet returns them.
"""

import os
import random
import re
import struct

import cv2
import numpy as np

# darknet's copy_image_from_bytes() stores every byte as (float)byte / 255.
_BYTE_TO_FLOAT = (np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)

# cfg sections that become OpenCV "Region" layers, and the keys OpenCV reads from
# them that would filter detections before OmniTrax's own threshold and NMS run
_DETECTION_SECTIONS = ("[yolo]", "[region]")
_OPENCV_FILTER_KEYS = ("thresh", "nms_threshold")

# Darknet normalises with sqrt(variance + 1e-5) (fuse_conv_batchnorm in network.c),
# OpenCV's Darknet importer with sqrt(variance + 1e-6)
_DARKNET_BATCHNORM_EPS = 1e-5
_OPENCV_BATCHNORM_EPS_FALLBACK = 1e-6
_opencv_batchnorm_eps = None

# layers that pass the number of channels through and store no weights
_PASS_THROUGH_SECTIONS = (
    "[upsample]",
    "[yolo]",
    "[region]",
    "[dropout]",
    "[avgpool]",
    "[avg]",
    "[softmax]",
    "[soft]",
    "[cost]",
    "[activation]",
)


class DarknetImage:
    """
    Stand-in for Darknet's IMAGE struct: a single w x h frame with c channels,
    stored as an (h, w, c) uint8 array in the channel order the caller provides
    (OmniTrax passes RGB, as Darknet expects).
    """

    def __init__(self, w, h, c):
        self.w = int(w)
        self.h = int(h)
        self.c = int(c)
        # darknet's make_image() returns a zero-initialised (black) image
        self.data = np.zeros((self.h, self.w, self.c), dtype=np.uint8)


class DarknetNetwork:
    """
    A Darknet YOLO model loaded through cv2.dnn, plus the input size Darknet would
    use for it.
    """

    def __init__(self, net, width, height, channels, output_names):
        self.net = net
        self.width = width
        self.height = height
        self.channels = channels
        self.output_names = output_names


def network_width(net):
    return net.width


def network_height(net):
    return net.height


def class_colors(names):
    """
    Create a dict with one random BGR color for each class name
    """
    return {
        name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for name in names
    }


def load_network(config_file, data_file, weights, batch_size=1):
    """
    Load a Darknet YOLO model for CPU inference.

    :param config_file: path to the .cfg model file
    :param data_file: path to the .data file (provides the class names)
    :param weights: path to the .weights file
    :param batch_size: accepted for compatibility with darknet.py; frames are
                       always processed one at a time
    :return: network, class_names, class_colors
    """
    cfg_text = _read_text(config_file)
    sections = _parse_cfg(cfg_text)
    width, height, channels = _network_input_size(sections, config_file)
    _check_opencv_support(sections, config_file)

    # Files are read here and passed to OpenCV as buffers, so paths containing
    # spaces or non-ASCII characters work on every platform.
    cfg_buffer = np.frombuffer(_cfg_for_opencv(sections).encode("utf-8"), np.uint8)
    weights_buffer = np.fromfile(weights, dtype=np.uint8)
    layout = _weights_layout(sections, weights_buffer)
    if layout is None:
        print(
            "WARNING: Could not adjust the batch normalisation of",
            config_file,
            "to Darknet's. CPU detections may differ slightly from GPU detections.",
        )
    else:
        variance_slices, required_size = layout
        # OpenCV loads a truncated file without complaint, then detects nonsense
        if required_size > weights_buffer.size:
            raise ValueError(
                f"{weights} has {weights_buffer.size} bytes, but {config_file} "
                f"needs {required_size}. The file is incomplete or belongs to a "
                "different model."
            )
        _match_darknet_batchnorm(weights_buffer, variance_slices)
    try:
        net = cv2.dnn.readNetFromDarknet(cfg_buffer, weights_buffer)
    except cv2.error as e:
        raise RuntimeError(
            f"OpenCV could not load the YOLO model from {config_file} "
            f"and {weights}: {e}"
        ) from e
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # [yolo] and [region] sections become "Region" layers, listed in cfg order
    detection_layers = [
        name for name in net.getLayerNames() if net.getLayer(name).type == "Region"
    ]
    if not detection_layers:
        raise ValueError(f"{config_file} has no [yolo] layer")
    network = DarknetNetwork(
        net=net,
        width=width,
        height=height,
        channels=channels,
        output_names=detection_layers,
    )
    class_names = _load_class_names(data_file)
    return network, class_names, class_colors(class_names)


def make_image(w, h, c):
    """
    Create a reusable image buffer for frames of size w x h with c channels.
    """
    return DarknetImage(w, h, c)


def copy_image_from_bytes(image, data):
    """
    Copy an interleaved (row-major, h x w x c) uint8 frame into the image buffer,
    e.g. copy_image_from_bytes(image, frame_rgb.tobytes()).
    """
    frame = np.frombuffer(data, dtype=np.uint8)
    expected = image.w * image.h * image.c
    if frame.size != expected:
        raise ValueError(
            f"Frame has {frame.size} bytes, but the image buffer was created for "
            f"{image.w} x {image.h} x {image.c} = {expected} bytes. "
            "Create it with make_image(frame_width, frame_height, 3)."
        )
    image.data = frame.reshape(image.h, image.w, image.c)


def detect_image(network, class_names, image, thresh=0.5, hier_thresh=0.5, nms=0.45):
    """
    Run the detector on an image and return its detections, as darknet.py does.

    :param network: network returned by load_network
    :param class_names: class names returned by load_network
    :param image: image created by make_image and filled by copy_image_from_bytes
    :param thresh: detection threshold, a class score must be greater than this
    :param hier_thresh: unused, kept for compatibility (it only affects YOLOv2
                        hierarchical models in Darknet)
    :param nms: IoU above which overlapping detections of the same class are
                suppressed, 0 disables non-maximum suppression
    :return: list of (label, confidence, (x, y, w, h)), with confidence as a
             percentage string (e.g. "87.21") and the box centre and size in
             pixels of the image
    """
    num_classes = len(class_names)
    outputs = _forward(network, image)
    boxes, scores = _boxes_and_scores(outputs, num_classes, image.w, image.h)

    # A detection (box, class) is kept if its score is strictly greater than the
    # threshold. Comparisons happen in float32, as in Darknet.
    scores = np.where(scores > np.float32(thresh), scores, np.float32(0))

    if nms:
        # do_nms_sort() leaves the detections sorted by the score of the last
        # class (before that class is suppressed), which sets the output order.
        order = np.argsort(-scores[:, num_classes - 1], kind="stable")
        _suppress_overlaps(boxes, scores, np.float32(nms))
    else:
        order = np.arange(scores.shape[0])

    rows, cols = np.nonzero(scores[order] > 0)
    rows = order[rows]
    predictions = [
        (
            str(class_names[k]),
            str(round(float(scores[j, k]) * 100, 2)),
            (
                float(boxes[j, 0]),
                float(boxes[j, 1]),
                float(boxes[j, 2]),
                float(boxes[j, 3]),
            ),
        )
        for j, k in zip(rows.tolist(), cols.tolist())
    ]
    # darknet.py sorts by the confidence string
    return sorted(predictions, key=lambda x: x[1])


def _forward(network, image):
    """
    Resize the image to the network input size the way Darknet does and run the
    network. Returns the outputs of all detection layers.
    """
    if image.c < network.channels:
        raise ValueError(
            f"The network expects {network.channels} channels, "
            f"but the image has {image.c}"
        )
    frame = image.data[:, :, : network.channels]
    resized = _resize_like_darknet(frame, network.width, network.height)
    blob = np.ascontiguousarray(resized.transpose(2, 0, 1)[np.newaxis])
    network.net.setInput(blob)
    try:
        return network.net.forward(network.output_names)
    except cv2.error as e:
        if network.width % 32 or network.height % 32:
            raise ValueError(
                f"The network size {network.width} x {network.height} is not "
                "supported, width and height must be multiples of 32"
            ) from e
        raise


def _boxes_and_scores(outputs, num_classes, image_w, image_h):
    """
    Collect the rows of all detection layers.

    Each row of an OpenCV region layer is [x, y, w, h, objectness, scores...]
    with the box relative to the network input and every score already
    multiplied by the objectness, in the same order Darknet iterates over
    layers, grid cells and anchors.

    :return: boxes (N, 4) as centre x, centre y, width, height in image pixels,
             and scores (N, num_classes), both float32
    """
    rows = np.concatenate(
        [
            np.asarray(out, dtype=np.float32).reshape(-1, out.shape[-1])
            for out in outputs
        ]
    )
    network_classes = rows.shape[1] - 5
    if num_classes < 1 or num_classes > network_classes:
        raise ValueError(
            f"The .data/.names files list {num_classes} classes, but the network "
            f"detects {network_classes}. Check the 'classes' entry of the .data file."
        )
    # Darknet scales relative boxes to the image size in float32
    scale = np.array([image_w, image_h, image_w, image_h], dtype=np.float32)
    boxes = rows[:, :4] * scale
    scores = np.array(rows[:, 5 : 5 + num_classes], dtype=np.float32)
    return boxes, scores


def _suppress_overlaps(boxes, scores, nms_thresh):
    """
    Darknet's do_nms_sort(): for every class, visit detections from the highest
    to the lowest score and zero the score of every later detection whose IoU
    with a surviving one is greater than nms_thresh. Modifies scores in place.
    """
    half_w = boxes[:, 2] / np.float32(2)
    half_h = boxes[:, 3] / np.float32(2)
    left = boxes[:, 0] - half_w
    right = boxes[:, 0] + half_w
    top = boxes[:, 1] - half_h
    bottom = boxes[:, 1] + half_h
    area = boxes[:, 2] * boxes[:, 3]

    for k in range(scores.shape[1]):
        candidates = np.flatnonzero(scores[:, k] > 0)
        if candidates.size < 2:
            continue
        order = candidates[np.argsort(-scores[candidates, k], kind="stable")]
        alive = np.ones(order.size, dtype=bool)
        for i in range(order.size - 1):
            if not alive[i]:
                continue
            rest = order[i + 1 :]
            a = order[i]
            iw = np.minimum(right[a], right[rest]) - np.maximum(left[a], left[rest])
            ih = np.minimum(bottom[a], bottom[rest]) - np.maximum(top[a], top[rest])
            inter = np.where((iw < 0) | (ih < 0), np.float32(0), iw * ih)
            union = area[a] + area[rest] - inter
            valid = (inter != 0) & (union != 0)
            iou = np.zeros_like(inter)
            np.divide(inter, union, out=iou, where=valid)
            alive[i + 1 :] &= ~(iou > nms_thresh)
        scores[order[~alive], k] = 0


def _resize_like_darknet(frame, w, h):
    """
    Convert an (H, W, C) uint8 frame to float32 in [0, 1] and resize it to
    (h, w, C) exactly like Darknet's resize_image(): bilinear sampling at
    positions scaled by (H - 1) / (h - 1) and (W - 1) / (w - 1), first along
    rows, then along columns, in float32 arithmetic.
    """
    H, W = frame.shape[:2]
    if W == w and H == h:
        return _BYTE_TO_FLOAT[frame]

    one = np.float32(1)

    # horizontal pass: (H, W, C) -> (H, w, C)
    cols = np.arange(w)
    w_scale = np.float32(W - 1) / np.float32(max(w - 1, 1))
    sx = cols.astype(np.float32) * w_scale
    ix = np.minimum(sx.astype(np.int64), W - 1)
    dx = (sx - ix.astype(np.float32))[np.newaxis, :, np.newaxis]
    ix_next = np.minimum(ix + 1, W - 1)
    part = (one - dx) * _BYTE_TO_FLOAT[frame[:, ix]] + dx * _BYTE_TO_FLOAT[
        frame[:, ix_next]
    ]
    # the last column (and every column of a 1 pixel wide frame) is copied as is
    last = (cols == w - 1) | (W == 1)
    part[:, last] = _BYTE_TO_FLOAT[frame[:, W - 1 : W]]

    # vertical pass: (H, w, C) -> (h, w, C)
    rows = np.arange(h)
    h_scale = np.float32(H - 1) / np.float32(max(h - 1, 1))
    sy = rows.astype(np.float32) * h_scale
    iy = np.minimum(sy.astype(np.int64), H - 1)
    dy = (sy - iy.astype(np.float32))[:, np.newaxis, np.newaxis]
    resized = (one - dy) * part[iy]
    # Darknet skips the second sample for the last row (and for 1 pixel high
    # frames), keeping only (1 - dy) * part[iy] there
    blend = ~((rows == h - 1) | (H == 1))
    resized[blend] += dy[blend] * part[np.minimum(iy[blend] + 1, H - 1)]
    return resized


def _read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _parse_cfg(cfg_text):
    """
    Split a Darknet cfg into a list of (section, options), e.g.
    ("[convolutional]", {"filters": "32", ...}). As in Darknet, comments start
    with # or ;, whitespace in section names and keys is ignored, lines without
    '=' are skipped and the first value of a repeated key is used.
    """
    sections = []
    for line in cfg_text.splitlines():
        line = line.strip()
        if not line or line[0] in "#;":
            continue
        if line[0] == "[":
            sections.append(("".join(line.split()), {}))
        elif sections:
            key, sep, value = line.partition("=")
            if sep:
                sections[-1][1].setdefault("".join(key.split()), value.strip())
    return sections


def _network_input_size(sections, config_file):
    """
    Read width, height and channels from the [net] section of a Darknet cfg.
    """
    if not sections or sections[0][0] not in ("[net]", "[network]"):
        raise ValueError(
            f"{config_file} is not a Darknet cfg file: it must start with [net]"
        )
    options = sections[0][1]
    width = _leading_int(options.get("width", "0"))
    height = _leading_int(options.get("height", "0"))
    channels = _leading_int(options.get("channels", "3"))
    if width <= 0 or height <= 0 or channels <= 0:
        raise ValueError(
            f"The [net] section of {config_file} must define a positive width "
            "and height"
        )
    return width, height, channels


def _match_darknet_batchnorm(weights_buffer, variance_slices):
    """
    Make OpenCV's batch normalisation match Darknet's: OpenCV divides by
    sqrt(variance + 1e-6), Darknet by sqrt(variance + 1e-5). Adding the
    difference to the rolling variances stored in the .weights buffer makes
    both compute the same normalisation. Channels with a small variance would
    otherwise give noticeably different detections.

    :param weights_buffer: contents of the .weights file (uint8), modified in place
    :param variance_slices: location of the rolling variances, see _weights_layout
    """
    delta = _DARKNET_BATCHNORM_EPS - _opencv_batchnorm_epsilon()
    if abs(delta) < 1e-12:
        return
    for offset, count in variance_slices:
        variance = weights_buffer[offset : offset + 4 * count].view("<f4")
        variance[:] = (variance.astype(np.float64) + delta).astype("<f4")


def _weights_layout(sections, weights_buffer):
    """
    Walk the cfg the way Darknet's load_weights() reads a .weights file: a
    header, then for every convolutional layer its biases, (if batch-normalised)
    its scales, rolling means and rolling variances, and its kernels.

    :return: (variance_slices, size): the byte offset and number of float32
             values of the rolling variances of every batch-normalised layer, and
             the number of bytes the cfg needs. None if the cfg uses a layer
             whose weights are not handled here.
    """
    if weights_buffer.size < 16:
        return None
    major, minor, _ = struct.unpack_from("<iii", weights_buffer, 0)
    # the 'seen' counter is 64 bit from format version 0.2 on
    seen_size = 8 if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000 else 4
    size = 12 + seen_size

    def number(options, key, default):
        return _leading_int(options[key]) if key in options else default

    net_channels = number(sections[0][1], "channels", 3)
    channels = []  # output channels of every layer after [net]
    variance_slices = []
    try:
        for index, (section, options) in enumerate(sections[1:]):
            if number(options, "dontload", 0):
                return None
            previous = channels[-1] if channels else net_channels
            if section in ("[convolutional]", "[conv]"):
                if "share_index" in options or any(
                    number(options, key, 0)
                    for key in ("dontloadscales", "deform", "coordconv")
                ):
                    return None
                filters = number(options, "filters", 1)
                kernel = number(options, "size", 1)
                groups = number(options, "groups", 1)
                kernels = (previous // groups) * filters * kernel * kernel
                if number(options, "batch_normalize", 0):
                    variance_slices.append((size + 4 * 3 * filters, filters))
                    size += 4 * (4 * filters + kernels)
                else:
                    size += 4 * (filters + kernels)
                out = filters
            elif section == "[route]":
                inputs = []
                for value in options["layers"].split(","):
                    source = _leading_int(value)
                    inputs.append(channels[source if source >= 0 else index + source])
                out = sum(inputs) // number(options, "groups", 1)
            elif section == "[shortcut]":
                if options.get("weights_type", "none") != "none":
                    return None
                out = previous
            elif section in ("[maxpool]", "[max]"):
                if number(options, "maxpool_depth", 0):
                    return None
                out = previous
            elif section in _PASS_THROUGH_SECTIONS:
                out = previous
            else:
                return None
            channels.append(out)
    except (KeyError, IndexError, ValueError, ZeroDivisionError):
        return None
    return variance_slices, size


def _opencv_batchnorm_epsilon():
    """
    Measure the epsilon OpenCV's Darknet importer uses for batch normalisation
    (1e-6 in OpenCV 4.x), with a one-layer network whose output is 1 / sqrt(eps).
    """
    global _opencv_batchnorm_eps
    if _opencv_batchnorm_eps is None:
        cfg = (
            b"[net]\nwidth=1\nheight=1\nchannels=1\n\n"
            b"[convolutional]\nbatch_normalize=1\nfilters=1\nsize=1\nstride=1\n"
            b"pad=0\nactivation=linear\n"
        )
        # header (major, minor, revision, seen), then bias = 0, scale = 1,
        # rolling mean = 0, rolling variance = 0 and a single kernel weight = 1
        weights = (
            struct.pack("<iiiQ", 0, 2, 0, 0)
            + np.array([0, 1, 0, 0, 1], dtype="<f4").tobytes()
        )
        try:
            probe = cv2.dnn.readNetFromDarknet(
                np.frombuffer(cfg, np.uint8), np.frombuffer(weights, np.uint8)
            )
            probe.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            probe.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            probe.setInput(np.ones((1, 1, 1, 1), dtype=np.float32))
            output = float(probe.forward().ravel()[0])
            eps = 1.0 / (output * output)
        except (cv2.error, ZeroDivisionError, OverflowError, IndexError):
            eps = _OPENCV_BATCHNORM_EPS_FALLBACK
        if not 0 < eps < 1e-3:
            eps = _OPENCV_BATCHNORM_EPS_FALLBACK
        _opencv_batchnorm_eps = eps
    return _opencv_batchnorm_eps


def _check_opencv_support(sections, config_file):
    """
    Stop (or warn) for models that OpenCV cannot run the way Darknet does.
    """
    if not hasattr(cv2.dnn, "readNetFromDarknet"):
        raise RuntimeError(
            f"OpenCV {cv2.__version__} cannot read Darknet models (OpenCV 5 "
            "removed this), CPU inference needs opencv-python 4.x"
        )
    uses_new_coords = any(
        _leading_int(options.get("new_coords", "0"))
        for section, options in sections
        if section == "[yolo]"
    )
    version = tuple(int(v) for v in re.findall(r"\d+", cv2.__version__)[:3])
    if uses_new_coords and version < (4, 5, 4):
        raise RuntimeError(
            f"{config_file} uses new_coords=1 (scaled YOLOv4), which needs "
            f"opencv-python 4.5.4 or newer, but {cv2.__version__} is installed"
        )
    if any(section == "[region]" for section, _ in sections):
        print(
            "WARNING:",
            config_file,
            "is a YOLOv2 model. OpenCV removes overlapping YOLOv2 detections "
            "(IoU above 0.4) on its own, so CPU detections can differ from GPU "
            "detections.",
        )


def _cfg_for_opencv(sections):
    """
    Write a parsed cfg (see _parse_cfg) back out for OpenCV's cfg parser, which
    differs from Darknet's:

    * OpenCV's [yolo] layers drop every score below the layer's 'thresh' (0.2 if
      not set) and can run their own NMS ('nms_threshold'). Both are disabled,
      so that the threshold and NMS chosen in OmniTrax decide which detections
      are kept, as with Darknet. ([region] layers only read 'thresh'.)
    * OpenCV uses the last value of a repeated key, Darknet the first.
    * Darknet ignores what follows a '#' in a number list, OpenCV does not.
    * OpenCV only accepts a '[net]' header, Darknet also '[network]'.
    """
    lines = []
    for index, (section, options) in enumerate(sections):
        if index == 0 and section == "[network]":
            section = "[net]"
        lines.append(section)
        for key, value in options.items():
            if section in _DETECTION_SECTIONS and key in _OPENCV_FILTER_KEYS:
                continue
            lines.append(key + "=" + value.partition("#")[0].strip())
        if section in _DETECTION_SECTIONS:
            lines += ["thresh=0", "nms_threshold=0"]
        lines.append("")
    return "\n".join(lines)


def _leading_int(value):
    """
    Parse an integer the way C's atoi() does, e.g. "3 # classes" -> 3, "" -> 0
    """
    match = re.match(r"\s*[+-]?\d+", value)
    return int(match.group()) if match else 0


def _load_class_names(data_file):
    """
    Read the class names the way Darknet's get_metadata() does: the 'names' (or
    'labels') file listed in the .data file, limited to its 'classes' entry
    (2 if not set). Relative names paths are tried from the working directory
    first (like Darknet) and then from the folder of the .data file.
    """
    options = {}
    for line in _read_text(data_file).splitlines():
        line = line.strip()
        if not line or line[0] in "#;":
            continue
        key, sep, value = line.partition("=")
        if sep:
            # Darknet ignores all whitespace in keys
            options.setdefault("".join(key.split()), value.strip())

    names_value = options.get("names") or options.get("labels")
    if not names_value:
        raise ValueError(f"No 'names' entry found in {data_file}")
    names_path = _find_names_file(names_value, data_file)

    classes = _leading_int(options.get("classes", "2"))
    if classes < 1:
        raise ValueError(f"Invalid 'classes' entry in {data_file}")

    names = [line.rstrip("\r\n") for line in _read_text(names_path).splitlines()]
    if len(names) < classes:
        raise ValueError(
            f"{data_file} declares classes = {classes}, "
            f"but {names_path} only lists {len(names)} names"
        )
    return names[:classes]


def _find_names_file(names_value, data_file):
    """
    Locate the names file listed in a .data file. Darknet removes all whitespace
    from .data lines, so a path is tried both as written and without whitespace,
    each from the working directory and from the folder of the .data file.
    """
    data_folder = os.path.dirname(os.path.abspath(data_file))
    for path in dict.fromkeys([names_value, "".join(names_value.split())]):
        for candidate in (path, os.path.join(data_folder, path)):
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        f"The names file {names_value} listed in {data_file} does not exist"
    )
