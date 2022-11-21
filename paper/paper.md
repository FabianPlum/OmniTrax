---
title: 'OmniTrax: A deep learning-based multi-animal tracking add-on for Blender'
tags:
  - Python
  - Blender
  - multi-object tracking
  - pose-estimation
  - deep learning
authors:
  - name: Fabian Plum
    orcid: 0000-0003-1012-6646
    affiliation: 1

affiliations:
  - name: Imperial College London, Department of Bioengineering, United Kingdom
    index: 1
date: 19 November 2022
bibliography: paper.bib

---

# Summary


`OmniTrax` is a deep learning-based multi-animal tracking and pose-estimation Blender Add-on [@Blender].
`OmniTrax` provides an intuitive high-throughput tracking solution for arbitrarily large groups of subjects through 
recent advancements in deep-learning based detection [@YOLOv3; @YOLOv4] and computationally inexpensive buffer-and-recover 
tracking approaches. Combining automated tracking with the Blender-internal motion tracking pipeline allows to streamline 
the annotation and analysis process of large video files with thousands of freely moving individuals. Additionally, 
`OmniTrax` integrates DeepLabCut-Live [@Kane2020dlclive] to enable running marker-less pose-estimation on arbitrary 
numbers of animals, leveraging the existing DeepLabCut Model Zoo [@Mathisetal2018] as well as our own custom trained 
detector and pose-estimator networks to facilitate large-scale behavioural studies of social animals. 

![OmniTrax user-interface.\label{fig:demo}](../images/omnitrax_demo_screen.jpg)

# Statement of need

Deep learning-based computer vision approaches promise to transform the landscape of large-scale human and animal 
behavioural research. The goal of `OmniTrax` is to provide a complete inference pipeline that decreases the entry barrier 
for researchers who wish to make use of the emerging libraries of machine learning driven tools.
`OmniTrax` is designed to track and infer the pose of large numbers of freely moving animals. Unlike background
subtraction and blob-detector based approaches, common in multi-animal tracking, the use of deep neural networks 
allows for robust buffer-and-recover tracking in constantly changing environments. A key advantage of integrating such a
pipeline into Blender is the seamless transition between automated tracking and iterative user-refinement. Additionally, 
Blender offers a number of video editing and compositing functions which make it possible to perform any required 
pre-processing, such as cropping, masking, or exposure adjustments, prior to running inference on the video footage within 
the same environment, without relying on external software packages.

`OmniTrax` offers marker-less pose-estimation through **DeepLabCut-Live**[@Kane2020dlclive] which enables 
extracting kinematic parameters from arbitrarily large groups of individuals. We are using `OmniTrax` in ongoing research 
monitoring foraging activities of various species of leafcutter ants, tracking the movements of thousands of 
individuals to extract path choice and changes to gait patterns.

Through a library of neural networks trained on synthetically generated samples of a number of study organisms [@Plum2021], 
we provide range of robust out-of-the-box inference solutions and encourage the community to contribute to this emerging 
collection. Pre-trained detection and pose-estimation networks can be used within `OmniTrax` to accelerate the annotation 
and analysis process of large video data sets. The ease of use and focus on extendability of `OmniTrax` will aid researchers 
in performing complex behavioural studies of social animals under laboratory as well as challenging field conditions.

# Acknowledgements

This work was in part funded by the Imperial College London President's PhD Scholarships.

# References