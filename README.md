# OmniTrax
It will track many things. Eventually. In Blender.

![](images/preview_tracking.gif)

Oh, and pose-estimate as well

![](images/single_ant_1080p_POSE_track_0.gif) ![](images/single_ant_1080p_POSE_track_0_skeleton.gif)

## Updates:

* 20/11/2021 - Added **pre-release** version, tested for **Blender 2.9.2** only

## Installation notes
* As we are using **tensorflow 2.7**, in order to run inference on your GPU, you will need to install [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) and [cudNN 8.1](https://developer.nvidia.com/rdp/cudnn-archive). Refer to [this](https://www.tensorflow.org/install/source#gpu) official guide for version matching and installation instructions.
* When installing the **OmniTrax** package, you may need to run **Blender** in **admnistrator mode** (on **Windows**). Otherwise, the additional required python packages may not be installable.

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
