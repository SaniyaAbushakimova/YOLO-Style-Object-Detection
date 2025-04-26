# YOLO-Style-Object-Detection

Project completed on April 21, 2025.

## Project Overview

This project focuses on building an object detection model inspired by the **YOLO framework**. A lightweight pre-trained backbone is used for efficiency, while a custom YOLO-style loss function is implemented to enable accurate object localization and classification on the **PASCAL VOC 2007** dataset.

The objective was to design a clear training and evaluation pipeline capable of achieving strong performance with minimal computational resources.

## What This Project Includes

1. **YOLO Loss Function Implementation**: Developed a custom *YOLO-style loss function* to train an object detector, combining localization, confidence, and classification losses.
2.	**Training and Evaluation Pipeline**: Built a full training loop and evaluation setup inside a Jupyter Notebook, allowing easy visualization of model progress and predictions.
3.	**Pre-Trained Backbone**: Integrated a lightweight pre-trained network inspired by *DetNet* to boost training efficiency and achieve higher accuracy with reduced computational cost.

## Results Summary

The model successfully reached a mean Average Precision (mAP) greater than 0.5 after training for 50 epochs, demonstrating effective object detection capabilities across multiple categories. Performance was tracked throughout training:
  * Epoch 5: mAP ≈ 0.02
  * Epoch 10: mAP ≈ 0.29
  * Epoch 20: mAP ≈ 0.42
  * Epoch 30: mAP ≈ 0.45
  * Epoch 40: mAP ≈ 0.48
  * Epoch 50: **mAP ≈ 0.51**

Below is an example of YOLO applied to a video:


## Repository Contents
* `data/` -- Folder containing the dataset files.
* `src/` -- Source code for dataset handling, model setup, and utilities.
* `Report.pdf` -- Report summarizing experiments and results.
* `download_data.sh` -- Shell script to automate dataset download and setup.
* `yolo_loss.py` -- Implementation of the YOLO-style custom loss function.
* `yolo_style_object_detection.ipynb` -- Jupyter Notebook for training, evaluation, and visualization of results.



