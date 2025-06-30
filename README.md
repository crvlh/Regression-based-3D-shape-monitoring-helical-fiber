# Regression-based 3D Shape Monitoring with Helical Macrobend Optical Fiber Sensor

## Project Description
Regression-based 3D shape monitoring using a helical macrobend optical fiber sensor and machine learning models, **Support Vector Regression (SVR)** and **Extreme Gradient Boosting Regression (XGBR)**. This project implements a robust pipeline for inferring three-dimensional deformations at multiple points along a flexible structure. By leveraging spectral signals, the system reconstructs the 3D shape of the structure.

## Introduction
This work focuses on monitoring the deformation of a flexible structure by tracking the 3D position of specific points. The sensing mechanism is based on spectral variations caused by macrobending, and these are processed using machine learning regression algorithms.
The workflow includes spectral filtering, dimensionality reduction via **Principal Component Analysis (PCA)**, hyperparameter optimization with **K-Fold Cross-Validation** and **Grid Search**, and final evaluation over test data, including **3D reconstructions** of the structure for multiple test cases.

![image](https://github.com/user-attachments/assets/300f4ba9-641a-4cf0-a27f-a1eb9d2fcc77)

## Repository Structure

| File/Folder              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `SVR_Kfold_xyz.py`       | Performs K-Fold cross-validation and grid search to optimize SVR hyperparameters for each point and axis (x, y, z). |
| `XGBR_Kfold_xyz.py`      | Same as above, but for the XGBR model.                                      |
| `SVR_test_xyz.py`        | Trains SVR models using optimal parameters, performs 3D reconstruction and evaluation on test data. |
| `XGBR_test_xyz.py`       | Same as above, but for the XGBR model.                  |
| `eval.py`                | Computes error metrics such as MAE and relative MAE for each point (color) and axis. Generates performance tables and summary reports. |
| `svr 1.mp4`, `svr 2.mp4`, `svr 3.mp4` | 3D reconstruction videos (test data) for 3 different days using SVR.        |
| `xgbr 1.mp4`, `xgbr 2.mp4`, `xgbr 3.mp4` | Equivalent videos for the XGBR-based model.                              |

## Purpose

This repository aims to:

- Support the reproducibility of the experimental results reported in the corresponding publication.
- Enable researchers to explore the use of SVR and XGBR for fiber-optic shape sensing.
- Provide a reference implementation for quasi-distributed 3D shape reconstruction using helical macrobend sensors.

## Publication
The codes and videos in this repository correspond to the findings presented in the article:  
📄 **“Regression-Based 3D Shape Monitoring in a Soft Structure with Helically Embedded Optical Fiber”**  
📅 Published in: ** JLT??**

## Citation

If you use this code or the results in your research, please cite the corresponding article once it is published.

## Contact

For questions or collaboration opportunities, please contact:

**Vinicius de Carvalho**  
📧 Email: crvlh.v@gmail.com


