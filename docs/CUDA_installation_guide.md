<img src=../images/omnitrax_logo.svg#gh-dark-mode-only height="60">
<img src=../images/omnitrax_logo_light.svg#gh-light-mode-only height="60">

**OmniTrax - CUDA installation guide**
***


Installing [**CUDA** (Compute Unified Device Architecture)](https://en.wikipedia.org/wiki/CUDA) is widely regarded as one of the least fun but sadly most essential endeavours when diving into machine
learning. Here, we will keep things as straight forward as possible and only install what is necessary to run
**OmniTrax** with GPU support (which will result in an inference speed increase of at least one order of magnitude).

## Downloading required files

To match the dependencies of [darknet](https://github.com/AlexeyAB/darknet) and [DeepLabCut-Live](https://github.com/DeepLabCut/DeepLabCut-live) 
we are going to install [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) 
and [cuDNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-archive).

Download [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) 
and make sure you select the version matching your system.

<img src=CUDA_installation_images/CUDA_01.PNG width="600">

Once you ensured you have picked the correct version for your system, download the **Base Installer**

<img src=CUDA_installation_images/CUDA_02.PNG width="600">

Next, download [cuDNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-archive). Pay close attention to the selected version.
In this case we need **v8.1.0 for CUDA 11.0, 11.1, 11.2**, specifically the **cuDNN Library for Windows (x86)**.

<img src=CUDA_installation_images/CUDA_03.PNG width="700">

If this is your first time using **CUDA**, you will need to first create an NVIDIA account. (Please don't ask me why.)

<img src=CUDA_installation_images/CUDA_04.PNG width="400">

Once you have created an account, sign in and continue the download.

<img src=CUDA_installation_images/CUDA_05.PNG width="400">

The files are relatively large (even more so, when unpacked), so make sure to select a drive with sufficient space.
You can delete them, once the installation is completed.

<img src=CUDA_installation_images/CUDA_06.PNG width="500">

## Installing CUDA

Open the downloaded **CUDA** installer and chose a location to temporarily store the unpacked files (~5 GB).
These temporary files will be removed once the installation is completed.

<img src=CUDA_installation_images/CUDA_07.PNG width="400">

Unless you really know what you are doing, choosing the express installation option should be sufficient.

<img src=CUDA_installation_images/CUDA_08.PNG width="500">

If you installation does not work as expected, you may need to install Visual Studio for your system. However, in most
cases this is not strictly necessary.

<img src=CUDA_installation_images/CUDA_09.PNG width="500">

Wait for the installation to complete.

<img src=CUDA_installation_images/CUDA_10.PNG width="500">

When the installation has completed, doublecheck that **CUDA 11.2** has been added to your **Environment variables**.
Type ***Edit the system environment variables*** into the windows search bar and open the first suggested application.

Click on **Environment Variables...** and check whether **CUDA_PATH** as well as **CUDA_PATH_V11_2** are listed.

<img src=CUDA_installation_images/CUDA_11.PNG width="700">

All that is left to do now, is move the files contained in the downloaded **cuDNN**_###.zip into the appropriate CUDA directory.
Unpack the **cuDNN**_###.zip and check its contents:

<img src=CUDA_installation_images/CUDA_12.PNG width="600">

Simply drag these contents into your **CUDA/v11.2** directory, which by default is located in 
*C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2*

<img src=CUDA_installation_images/CUDA_13.PNG width="700">

**And that's (hopefully) it.**

### Check your installation
Restart your system before continuing the **OmniTrax** installation. If all has gone according to plan, you should now
be able to select your **GPU**(s) as compute devices within **OmniTrax**.

<img src=CUDA_installation_images/CUDA_14.PNG width="400">
