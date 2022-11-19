<img src=../images/omnitrax_logo.svg#gh-dark-mode-only height="60">
<img src=../images/omnitrax_logo_light.svg#gh-light-mode-only height="60">

**OmniTrax - Tutorial : Pose-Estimation**
***

**OmniTrax** is a deep learning-based interactive multi-animal tracking and pose-estimation tool. It combines
Buffer-and-Recover tracking with
**Blender**'s internal **Motion Tracking** pipeline to streamline the annotation and analysis process of large video
files with
thousands of individuals. Integrating [DeepLabCut-Live](https://github.com/DeepLabCut/DeepLabCut-live) into this
pipeline makes
it possible to additionally run marker-less pose-estimation on arbitrary numbers of animals, leveraging the existing
[DLC-Model-Zoo](https://www.mackenziemathislab.org/dlc-modelzoo), as well as our own
custom [trained networks](trained_networks.md).

Alongside **OmniTrax** we offer a selection of [example video](example_footage.md) footage
and [trained networks](trained_networks.md).
To curate your own datasets, as well as train further custom networks, refer to the
official [YOLO](https://github.com/AlexeyAB/darknet)
as well as [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) documentation.

# Tutorial : Pose-Estimation

This tutorial is divided into two separate sections : **Single-animal pose-estimation**
& **Multi-animal pose-estimation**. **OmniTrax** is primarily tailored to performing analysis
of large groups of animals in discontinuous settings, utilising a two-step approach of (**1**) tracking all individuals and
(**2**) performing pose-estimation on each returned ROI (region of interest).

To benefit from **OmniTrax** maximally, both, a [trained detector](https://github.com/AlexeyAB/darknet) and
[pose-estimator](https://github.com/DeepLabCut/DeepLabCut), are required.
For the provided [trained networks](trained_networks.md) we minimise manual labelling effort by training these networks
on
[synthetically generated data](https://github.com/FabianPlum/FARTS)

![](../images/single_ant_1080p_POSE_track_0.gif) ![](../images/single_ant_1080p_POSE_track_0_skeleton.gif)

_Pose-estimation and skeleton overlay example (trained on [synthetic data](https://github.com/FabianPlum/FARTS))_

## Preparing a trained [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) model

When using your own [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) model, all you need to do is export the
trained
model and enter the resulting folder path as the **DLC network path** in the **OmniTrax** Panel.

Ensure the model has been trained with Tensorflow version 2.0 or higher to be compatible with
the respective version of deeplabcut-live.

Open an instance of python and run the following commands (with your DeepLabCut environment activated):

```bash
import deeplabcut
cfg_path = "path/to/your/config.yaml"
deeplabcut.export_model(cfg_path, iteration=None, shuffle=1, trainingsetindex=0, snapshotindex=None)
```

Enter the folder path to the exported model as the **DLC network path** in the **OmniTrax/Pose-Estimation (DLC)** panel.

_**NOTE** : Curating the required datasets and training a pose-estimator is beyond the scope of this introduction.
For further information, refer to the official documentation of [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)._

## System Console

We recommend keeping the Blender **System Console** open while using **OmniTrax** to monitor the tracking progression
and make spotting potential issues easier.

Simply cick on **Window** > **Toggle System Console** to open it in a separate window (repeat the process to close in
again)

## Single-animal pose-estimation

### Requirements

* trained [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) model for full-frame pose-estimation
* optionally: hand-annotated (or auto-tracked) ROI to run inference on a subsection of the video

### Used in this example
* **Video** : [VID_20220201_160304.mp4](https://drive.google.com/file/d/156t8r3ZHrkzC72jZapFl9OBFPqNIvIXg/view?usp=share_link)
* **DLC Network** : [ResNet101 (single) stick-insect [full-frame]](https://drive.google.com/drive/folders/1-DHkegHiTkWbO7YboXxDC5tU4Aa71-9z?usp=share_link)
   (trained on synthetic data, refined with real samples)

### Setup and Inference

To run pose-estimation inference of individual animals across the **full frame**, you only need to specify the path to 
your exported trained model, as you will not need to supply sub-ROIs. You can therefore disregard the options for 
**constant (input) detection sizes** as well as the **Pose (input) frame size (px)** which will default to the original
dimensions of the loaded video.

* **DLC network path** : Path to your trained and exported DLC network where your pose_cfg.yaml and snapshot files are stored.
  In order to enable plotting the defined skeleton as an overlay, simply include your original **config.yaml** file in the
  same folder. **OmniTrax** will read the skeleton configuration from the **config.yaml** file directly, so ensure that
  the naming conventions of **config.yaml** matches **pose_cfg.yaml**
* **Constant (input) detection sizes** : If enabled, enforces constant input ROIs, as defined by the **Pose (input) frame 
  size (px)detection sizes. If not enabled, the tracking marker bounding box will determine the input ROI. ***NOTE** : this
  option will have no effect in [full frame] mode.*
* **Pose (input) frame size (px)** : Constant detection size in pixels. All ROIs will be rescaled and padded, if 
  necessary. ***NOTE** : this option will have no effect in [full frame] mode.*
* **pcutoff (minimum key point confidence)** : Predicted key points with a confidence below this threshold will be 
  discarded during pose-estimation.
* **Visualisation** :
  * **Plot skeleton** : Plot the skeleton defined in **config.yaml** file based on the detected landmarks. In order to 
    enable plotting the defined skeleton as an overlay, simply include your original **config.yaml** file in the
    same folder. **OmniTrax** will read the skeleton configuration from the **config.yaml** file directly, so ensure 
    that the naming conventions of **config.yaml** matches **pose_cfg.yaml**
  * **Keypoint marker size** : Size of marker points (in pixels) displayed in pose-estimation preview
  * **Skeleton line thickness** : Line width of skeleton bones (in pixels) displayed in pose-estimation preview
  * **Display label names** : Display label names as an overlay in pose-estimation preview
* Run Pose-Estimation:
  * **Export pose-estimation video** : Save the (cropped) video with tracked overlay to the location of the input video. 
  * **Export pose-estimation data** : Write estimated pose data to disk landmark locations in (relative) pixel space.

When you have completed configuring the pose-estimation process click on **ESTIMATE POSES [full frame]**.

![](../images/VID_20220201_160304_50%25_POSE_fullframe.gif)

## Multi-animal pose-estimation

### Requirements

* trained [YOLO](https://github.com/AlexeyAB/darknet) model for automated buffer-and-recover tracking and producing ROIs
    * _alternatively_: hand-annotated (or blender-tracked) ROI(s) to run inference on subsection(s) of the video
* trained [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) model for (cropped frame) pose-estimation

### Used in this example
* **Video** : [1080p DSLR recording (**multi-animal pose-estimation**)](https://drive.google.com/file/d/1izoE7bLScQODYloV5B6bwzWtJ4jcqp1K/view?usp=sharing)
* **YOLO Network** : [single class ant detector (trained on synthetic data)](https://drive.google.com/drive/folders/1PSseMeClcYIe9dcYG-JaOD2CzYceiWdl?usp=sharing)
  (using the _yolov4-big_and_small_ants_320.cfg_ configuration given animals are large relative to the image resolution)
* **DLC Network** : [ResNet101 ant pose-estimator](https://drive.google.com/drive/folders/1FY3lAkAisOG_RIUBuaynz1OjBkzjH5LL?usp=sharing)
  (trained on mixed synthetic/real data, 10:1 ratio)

### Setup and Inference

To run pose-estimation inference of multiple animals, you will first need to track the footage to provide ROIs of all
relevant individuals. Refer to the [Tutorial : Tracking](tutorial-tracking.md) for an in-depth guide of how to set up
the automated buffer-and-recover tracker.

When using **Constant detection sizes**, make sure to keep them sufficiently large to include all body parts in the 
resulting ROIs you wish to consider in the pose-estimation step. Once you have finished tracking the video footage,
continue in the **Pose Estimation (DLC)** panel:

* **DLC network path** : Path to your trained and exported DLC network where your pose_cfg.yaml and snapshot files are stored.
  In order to enable plotting the defined skeleton as an overlay, simply include your original **config.yaml** file in the
  same folder. **OmniTrax** will read the skeleton configuration from the **config.yaml** file directly, so ensure that
  the naming conventions of **config.yaml** matches **pose_cfg.yaml**
* **Constant (input) detection sizes** : If enabled, enforces constant input ROIs, as defined by the **Pose (input) frame 
  size (px)detection sizes. If not enabled, the tracking marker bounding box will determine the input ROI. ***NOTE** : this
  option will have no effect in [full frame] mode.*
* **Pose (input) frame size (px)** : Constant detection size in pixels. All ROIs will be rescaled and padded, if 
  necessary.
* **pcutoff (minimum key point confidence)** : Predicted key points with a confidence below this threshold will be 
  discarded during pose-estimation.
* **Visualisation** :
  * **Plot skeleton** : Plot the skeleton defined in **config.yaml** file based on the detected landmarks. In order to 
    enable plotting the defined skeleton as an overlay, simply include your original **config.yaml** file in the
    same folder. **OmniTrax** will read the skeleton configuration from the **config.yaml** file directly, so ensure 
    that the naming conventions of **config.yaml** matches **pose_cfg.yaml**
  * **Keypoint marker size** : Size of marker points (in pixels) displayed in pose-estimation preview
  * **Skeleton line thickness** : Line width of skeleton bones (in pixels) displayed in pose-estimation preview
  * **Display label names** : Display label names as an overlay in pose-estimation preview
* Run Pose-Estimation:
  * **Export pose-estimation video** : Save the (cropped) video with tracked overlay to the location of the input video. 
  * **Export pose-estimation data** : Write estimated pose data to disk landmark locations in (relative) pixel space.

When you have completed configuring the pose-estimation process click on **ESTIMATE POSES** and **OmniTrax** will run
inference on each extracted ROI defined in the **tracking** step.

![](../images/multi_ants_online_tracking_&_pose_estimation.gif)

***

## License

© Fabian Plum, 2021
[MIT License](https://choosealicense.com/licenses/mit/)
