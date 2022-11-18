<img src=../images/omnitrax_logo.svg#gh-dark-mode-only height="60">
<img src=../images/omnitrax_logo_light.svg#gh-light-mode-only height="60">

***

# Tutorial : Tracking

## 1. Open [Blender](https://www.blender.org/download/lts/3-3/) and start a new project

Let's create a new **Workspace** from the **VFX  >  Motion_Tracking** tab.
If you don't already have suitable videos at hand, grab some from the list of [example footage](docs/example_footage.md)
.
In this tutorial, we are going to use a recording of [herbivorous leafcutter ants](..images/example_ant_recording.mp4).

<img src=../images/use_01.jpg height="200">

### System Console

We recommend keeping the Blender **System Console** open while using **OmniTrax** to monitor the tracking progression
and make spotting potential issues easier.
Simply cick on **Window** > **Toggle System Console** to open it in a separate window (repeat the process to close in
again)

## 2. Select your compute device

If you have a **CUDA supported GPU** *(and the CUDA installation went as planned...)*, make sure your **GPU is
selected** here, **before** running any of the inference functions, as the compute device cannot be changed at runtime.

<img src=../images/use_02.jpg height="200">

## 3. Load trained [YOLO](https://github.com/AlexeyAB/darknet) network

Alternatively, use one of [our pre-trained YOLO networks](example_footage.md). Next, load a video you wish to analyse
from your drive by clicking on **Open**.

<img src=../images/use_03.jpg height="200">

**ATTENTION:** Double check the *"names="* filepath in the **obj.data** file points to the **ABSOLUTE** location of
the **obj.names** file. Otherwise, the program may crash when running tracking without telling you why.

**EXAMPLE:  *obj.data***
from [single class ant detector (trained on synthetic data)](https://drive.google.com/drive/folders/1PSseMeClcYIe9dcYG-JaOD2CzYceiWdl?usp=sharing)

```bash
classes = 1
train = data/train.txt
test = data/test.txt
names = C:/Users/Fabi/Downloads/atta_single_class/obj.names
backup = backup/
```

## 4. Configure the Detector

## 5. Configure the Tracker

## 6. Run the Tracker

Click on **RESTART Track** (or **TRACK** to continue tracking from a later point in the video).
Click on the video (which will open in a separate window) to monitor the tracking progress.
If desired, press **q** to terminate the process early.

![](../images/use_04.gif)

*NOTE: The ideal settings will always depend on your footage, especially on the relative animal size and movement speed.
Remember, **GIGO** (Garbage In Garbage Out) so ensuring your recordings are evenly-lit, free from noise, flickering, and
motion blur, will go a long way to improve inference quality.*

## 7. Manual Tracking _(OPTIONAL)_

## 8. Exporting Tracks

## 9. Visualising & Analysing Exported Tracks

![](../images/example_ant_tracked.gif)
_tracking output of OmniTrax for the first 1000 frames of [example_ant_recording.mp4](..images/example_ant_recording.mp4)
,
using
a [single class ant detector (trained on synthetic data)](https://drive.google.com/drive/folders/1PSseMeClcYIe9dcYG-JaOD2CzYceiWdl?usp=sharing)_

![](../example_scripts/_heatmap_of_ground_truth_tracks.svg)
_generated heatmap from the [tracked outputs](../example_scripts/example_ant_recording)_

***

## License

© Fabian Plum, 2021
[MIT License](https://choosealicense.com/licenses/mit/)
