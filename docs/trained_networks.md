<img src=../images/omnitrax_logo.svg#gh-dark-mode-only height="60">
<img src=../images/omnitrax_logo_light.svg#gh-light-mode-only height="60">

**OmniTrax - trained networks**
***

## Download trained networks and config files here:

**[YOLO](https://github.com/AlexeyAB/darknet) Networks**

* [single class ant detector (trained on synthetic data)](https://drive.google.com/drive/folders/1PSseMeClcYIe9dcYG-JaOD2CzYceiWdl?usp=sharing)
* [3 class ant detector (trained on synthetic data)](https://drive.google.com/drive/folders/1wQcfLlDUvnWthyzbvyVy9oqyTZ2F-JFo?usp=sharing)
* [single class termite detector (trained on synthetic data)](https://drive.google.com/drive/folders/1U9jzOpjCcu6wDfTEH3uQqGKPxW_QzHGz?usp=sharing)
* [80 Class model trained on COCO](https://drive.google.com/drive/folders/1eXAowtyBsqGEjvmQE1YlSeHJ6AGBwpUs?usp=share_link)

You can also use any pre-trained network from the official [YOLO-model Zoo](https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo).
Just make sure to include the respective **.obj** as well as **.names** and update the respective folder paths.

**[DLC](https://github.com/DeepLabCut/DeepLabCut) Networks**
* [ResNet50 ant pose-estimator (trained on mixed synthetic/real data, 10:1 ratio)](https://drive.google.com/drive/folders/1or1TF3tvi1iIzldEAia3G2RNKY5J7Qz4?usp=sharing)
* [ResNet101 ant pose-estimator (trained on mixed synthetic/real data, 10:1 ratio)](https://drive.google.com/drive/folders/1FY3lAkAisOG_RIUBuaynz1OjBkzjH5LL?usp=sharing)
* [ResNet152 ant pose-estimator (trained on mixed synthetic/real data, 10:1 ratio)](https://drive.google.com/drive/folders/1or1TF3tvi1iIzldEAia3G2RNKY5J7Qz4?usp=sharing)
* [ResNet101 (single) ant pose-estimator (trained on synthetic data)](https://drive.google.com/file/d/1IH9R9PgJMYteigsrMi-bZnz4IMcydtWU/view?usp=sharing)
* [ResNet101 (single) stick-insect pose-estimator [full-frame] (trained on synthetic data, refined with real samples)](https://drive.google.com/drive/folders/1-DHkegHiTkWbO7YboXxDC5tU4Aa71-9z?usp=share_link)
* [ResNet101 "full_human" pose estimation](https://drive.google.com/drive/folders/1BLulUYkwww7SfzXgSSVM71GLI4dQysP5?usp=share_link)
  converted for **DeepLabCut** and **OmniTrax** from the original [DeeperCut publication](https://arxiv.org/abs/1605.03170)

You can also make use of [DeepLabCuts official Model Zoo](https://www.mackenziemathislab.org/dlc-modelzoo). To use these
models within **OmniTrax**, you will need to run the **deeplabcut.export_model** command. Refer to the [Pose-estimation
tutorial](tutorial-pose-estimation.md).

***
## License
© Fabian Plum, 2021
[MIT License](https://choosealicense.com/licenses/mit/)
