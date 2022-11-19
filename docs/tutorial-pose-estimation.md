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

This tutorial is divided into two separate sections : [Single-animal pose-estimation](#Single-animal pose-estimation)
& [Multi-animal pose-estimation](#Multi-animal pose-estimation). **OmniTrax** is primarily tailored to performing
analysis
of large groups of animals in discontinuous settings, utilising a two-step approach of (**1**) tracking all individuals
and
(**2**) performing pose-estimation on each returned ROI (region of interest).

To benefit from **OmniTrax** maximally, both, a [trained detector](https://github.com/AlexeyAB/darknet) and
[pose-estimator](https://github.com/DeepLabCut/DeepLabCut), are required.
For the provided [trained networks](trained_networks.md) we minimise manual labelling effort by training these networks
on
[synthetically generated data](https://github.com/FabianPlum/FARTS)

![](../images/single_ant_1080p_POSE_track_0.gif) ![](../images/single_ant_1080p_POSE_track_0_skeleton.gif)

_Pose estimation and skeleton overlay example (trained on [synthetic data](https://github.com/FabianPlum/FARTS))_

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

Enter the folder path to the exported model as the **DLC network path** in the **OmniTrax/Pose Estimation (DLC)** panel.

_**NOTE** : Curating the required datasets and training a pose-estimator is beyond the scope of this introduction.
For further information, refer to the official documentation of [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)._

## System Console

We recommend keeping the Blender **System Console** open while using **OmniTrax** to monitor the tracking progression
and make spotting potential issues easier.

Simply cick on **Window** > **Toggle System Console** to open it in a separate window (repeat the process to close in
again)

<a name="Single-animal pose-estimation"></a>

## Single-animal pose-estimation

### Requirements

* trained [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) model for full-frame pose-estimation
* optionally: hand-annotated (or auto-tracked) ROI to run inference on a subsection of the video

![](../images/VID_20220201_160304_50%25_POSE_fullframe.gif)

<a name="Multi-animal pose-estimation"></a>

## Multi-animal pose-estimation

### Requirements

* trained [YOLO](https://github.com/AlexeyAB/darknet) model for automated buffer-and-recover tracking and producing ROIs
    * _alternatively_: hand-annotated (or blender-tracked) ROI(s) to run inference on subsection(s) of the video
* trained [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) model for (cropped frame) pose-estimation

![](../images/multi_ants_online_tracking_&_pose_estimation.gif)

***

## License

© Fabian Plum, 2021
[MIT License](https://choosealicense.com/licenses/mit/)