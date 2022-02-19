# OmniTrax
It will track many things. Eventually. In Blender.

![](images/preview_tracking.gif)

Oh, and pose-estimate as well

![](images/single_ant_1080p_POSE_track_0.gif) ![](images/single_ant_1080p_POSE_track_0_skeleton.gif)

## Updates:
* 19/02/2021 - Added [**release** version 0.1.1](https://github.com/FabianPlum/OmniTrax/releases/tag/V_0.1.1)! Things run a lot faster now and I have added support for devices without dedicated GPUs. 
* 06/12/2021 - Added the first [**release** version 0.1](https://github.com/FabianPlum/OmniTrax/releases/tag/V_0.1)! Lots of small improvements and mammal fixes. Now, it no longer feels like a pre-release and we can all give this a try. Happy Tracking!
* 29/11/2021 - Added [**pre-release** version 0.0.2](https://github.com/FabianPlum/OmniTrax/releases/tag/V_0.0.2), with [DeepLabCut-Live](https://github.com/DeepLabCut/DeepLabCut-live) support, tested for **Blender 2.9.2** only
* 20/11/2021 - Added [**pre-release** version 0.0.1](https://github.com/FabianPlum/OmniTrax/releases/tag/V_0.0.1), tested for **Blender 2.9.2** only

## Installation notes
### OmniTrax is currently only supported on Windows 10 / 11
* **YOU MUST** use [**Blender version 2.9.2**](https://download.blender.org/release/Blender2.92/) to match dependencies! I mean, you can try to use a different version too but then you will have to endure the pain of installing all sorts of python packages in Blender yourself... 
* As we are using **tensorflow 2.7**, to run inference on your GPU, you will need to install [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) and [cudNN 8.1](https://developer.nvidia.com/rdp/cudnn-archive). Refer to [this](https://www.tensorflow.org/install/source#gpu) official guide for version matching and installation instructions.
* When installing the **OmniTrax** package, you need to run **Blender** in **admnistrator mode** (on **Windows**). Otherwise, the additional required python packages may not be installable.

## Download trained networks and config files here:

**YOLO Networks**

* [single class ant detector (trained on synthetic data)](https://drive.google.com/drive/folders/1PSseMeClcYIe9dcYG-JaOD2CzYceiWdl?usp=sharing)
* [3 class ant detector (trained on synthetic data)](https://drive.google.com/drive/folders/1wQcfLlDUvnWthyzbvyVy9oqyTZ2F-JFo?usp=sharing)

**DLC Networks**
* [ResNet101 (single) ant pose-estimator (trained on synthetic data)](https://drive.google.com/file/d/1VnGXy_KyPHUIbFMx5n_ncN6z0H6qg5uB/view?usp=sharing)


***
## License
© Fabian Plum, 2021
[MIT License](https://choosealicense.com/licenses/mit/)
