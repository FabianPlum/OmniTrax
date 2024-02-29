---
name: Trained model submission
about: Submitting a trained deep neural network to be included in the list of plug-and-play models.
title: ''
labels: ''
assignees: ''

---

**Describe the model you'd like to submit**
A clear and concise description of the model, parameter choices, and training data used.

**Link to your trained model and related files**
You can upload your model in any publicly available repository 
(e.g. google drive, Dropbox, or citable repos such as Zenodo. If your model is added, with your permission we will
re-upload all files to ensure perpetual access for other users.)

For a YOLO model, upload the following files:
- *.cfg
- *.weights
- optionally:
  - *.data
  - *.names

For a DLC model, upload the following files:
- snapshot-######.pb
- snapshot-######.index
- snapshot-######.meta
- snapshot-######.pbtxt
- pose_cfg.yaml
- config.ymal

**Additional context**
Add any other context or example footage to test the model.
