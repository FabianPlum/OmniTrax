<img src=../images/omnitrax_logo.svg#gh-dark-mode-only height="60">
<img src=../images/omnitrax_logo_light.svg#gh-light-mode-only height="60">

***
# Tutorial : Tracking

## System Console
We recommend keeping the Blender **System Console** open while using **OmniTrax** to monitor the tracking progression and make spotting potential issues easier.
Simply cick on **Window** > **Toggle System Console** to open it in a separate window (repeat the process to close in again)

## 1. Open [Blender](https://www.blender.org/download/lts/3-3/) and start a new project
Let's create a new **Workspace** from the **VFX  >  Motion_Tracking** tab.
If you don't already have suitbale videos at hand, grab some from the list of [example footage](docs/example_footage.md)

<img src=../images/use_01.jpg height="200">

### System Console
We recommend keeping the Blender **System Console** open while using **OmniTrax** to monitor the tracking progression and make spotting potential issues easier.
Simply cick on **Window** > **Toggle System Console** to open it in a separate window (repeat the process to close in again)


## 2. Select your compute device
If you have a **CUDA supported GPU** *(and the CUDA installation went as planned...)*, make sure your **GPU is selected** here, **before** running any of the inference functions, as the compute device cannot be changed at runtime.

<img src=../images/use_02.jpg height="200">

## 3. Load trained [YOLO](https://github.com/AlexeyAB/darknet) network
Alternatively, use one of [our pre-trained YOLO networks](). Next, load a video you wish to analyse from your drive by clicking on **Open**.

<img src=../images/use_03.jpg height="200">

**ATTENTION:** Double check the *"names="* filepath in the **obj.data** file points to the **ABSOLUTE** location of the **obj.names** file. Otherwise, the program may crash when running tracking without telling you why. 

**EXAMPLE:  *obj.data*** from [single class ant detector (trained on synthetic data)](https://drive.google.com/drive/folders/1PSseMeClcYIe9dcYG-JaOD2CzYceiWdl?usp=sharing)

```bash
classes = 1
train = data/train.txt
test = data/test.txt
names = C:/Users/Legos/Downloads/atta_single_class/obj.names
backup = backup/
```

4. Now, configure the **Detector** and **Tracker** to your liking. When you are done, click on **RESTART Track** (or **TRACK** to continue tracking from a later point in the video). Click on the video (which will open in a separate window) and press **q** to terminate the tracking process early.  

![](../images/use_04.gif)
 
*NOTE: The ideal settings will always depend on your footage, especially on the relative animal size and movement speed. Remember, **GIGO** (Garbage In Garbage Out) so ensuring your recordings are evenly-lit, free from noise, flickering, and motion blur, will go a long way to improve inference quality.*

***
## License
© Fabian Plum, 2021
[MIT License](https://choosealicense.com/licenses/mit/)
