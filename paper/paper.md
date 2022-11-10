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
date: 10 November 2022
bibliography: paper.bib

---

# Summary

Multi-animal tracking and pose-estimation with thousands of individuals is difficult,
but that does not have to be the case.

![OmniTrax user-interface.\label{fig:demo}](../images/omnitrax_demo_screen.jpg)

# Statement of need

`OmniTrax` is a deep learning-based buffer and recover tracking Add-on for the
popular open-source software Blender [@Blender]. Combining recent advancements
in deep-learning based detection [@Yolov3; @Yolov4] with computationally inexpensive
buffer and recover tracking approaches, `OmniTrax` provides an intuitive high-throughput 
tracking solution for arbitrarily large groups of subjects. `OmniTrax` is being used
in ongoing research projects to monitor the path choices of thousands of freely moving 
herbivorous leaf-cutter ants. Including these tools into the Blender tracking pipeline
enables easily user editable tracking results for further refinement, export, and analysis.
`OmniTrax` additionally offers deep-learning based pose estimation [@Mathisetal2018], 
leveraging the deeplabcut-live package [@Kane2020dlclive] to provide key-point information
for subjects across large time frames. 

Through a library of neural networks trained on synthetically generated samples of
the aforementioned study organisms [@Plum2021], we provide range of robust out-of-the-box 
inference solutions and encourage the community to contribute to this emerging collection.
Pre-trained detection and pose-estimation networks can be used within `OmniTrax` to accelerate
the annotation and analysis process of large (video-)data sets. The ease of use and focus on
extendability of `OmniTrax` will aid researchers in performing complex behavioural studies
of social animals under laboratory as well as challenging field conditions.

# Acknowledgements

This work was in part funded by the Imperial College London President's PhD Scholarships.

# References