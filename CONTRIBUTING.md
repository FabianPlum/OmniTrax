# OmniTrax Contributing Guidelines

Thank you for considering contributing to OmniTrax! This open-source project aims to provide a deep learning-driven multi-animal tracking and pose-estimation solution within Blender. Your contributions are essential to the growth and improvement of this project. Please take a moment to review this guide to understand how you can contribute effectively.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Contributing](#contributing)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Code Contributions](#code-contributions)
4. [Contact](#contact)

## Introduction

OmniTrax is an open-source Blender Add-on designed for deep learning-driven multi-animal tracking and pose-estimation. It leverages recent advancements in deep-learning-based detection (YOLOv3, YOLOv4) and computationally inexpensive buffer-and-recover tracking techniques. OmniTrax integrates with Blender's internal motion tracking pipeline, making it an excellent tool for annotating and analyzing large video files containing numerous freely moving subjects. Additionally, it integrates DeepLabCut-Live for marker-less pose estimation on arbitrary numbers of animals, using both the DeepLabCut Model Zoo and custom-trained detector and pose estimator networks.

If you have any questions or need assistance, please don't hesitate to contact us at [Fabian.Plum18@Imperial.ac.uk](mailto:fabian.plum18@imperial.ac.uk).

## Getting Started

Before you start contributing to OmniTrax, make sure you have the following prerequisites:

- A working knowledge of Blender.
- Familiarity with deep learning concepts, especially YOLO and DeepLabCut.
- Python programming skills.
- An understanding of the project's goals and features by reviewing the [OmniTrax documentation](https://github.com/FabianPlum/OmniTrax/tree/main/docs).

## Contributing

We welcome contributions from the community. Whether you want to report a bug, suggest an enhancement, or contribute code, your help is appreciated.

### Reporting Bugs

If you encounter a bug while using OmniTrax, please follow these steps to report it:

1. Check the [existing issues](https://github.com/FabianPlum/OmniTrax/issues) to see if the bug has already been reported.
2. If not, [create a new issue](https://github.com/FabianPlum/OmniTrax/issues/new/choose) with a clear title and detailed description of the bug, including steps to reproduce it.
3. Add relevant labels and milestones to help categorize the issue appropriately.

### Suggesting Enhancements

If you have an idea for an enhancement or a new feature, please follow these steps:

1. Check the [existing issues](https://github.com/FabianPlum/OmniTrax/issues) to see if your suggestion has already been discussed.
2. If not, [create a new issue](https://github.com/FabianPlum/OmniTrax/issues/new/choose) with a clear title and a detailed description of your enhancement request.
3. Discuss your proposal with the community to gather feedback and reach a consensus on the implementation.

### Code Contributions

We welcome code contributions from the community. To contribute code to OmniTrax, follow these steps:

1. Fork the [OmniTrax repository](https://github.com/FabianPlum/OmniTrax) to your own GitHub account.
2. Clone your forked repository to your local machine.
3. Create a new branch for your work: `git checkout -b feature/your-feature-name`.
4. Make your changes, following the project's coding standards.
5. Write clear commit messages.
6. Push your changes to your forked repository.
7. Create a pull request (PR) against the `main` branch of the OmniTrax repository.
8. Provide a detailed description of your changes in the PR.

The maintainers will review your PR, and if everything looks good, it will be merged into the project.

### Trained model contributions

If you want to contribute trained detector or pose estimator models to our 
[model collection](https://github.com/FabianPlum/OmniTrax/blob/main/docs/trained_networks.md),
please open a [new issue](https://github.com/FabianPlum/OmniTrax/issues/new/choose).

Follow the file requirements listed in the `Trained model submission` issue template and, if possible, include example 
video footage for us to test your model.

## Contact

If you have any questions or need further assistance, please reach out to us at [Fabian.Plum18@Imperial.ac.uk](mailto:fabian.plum18@imperial.ac.uk).

Thank you for contributing to OmniTrax! Your support helps us improve multi-animal tracking and pose estimation in Blender.
Happy Tracking!
