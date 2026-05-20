# Architecture

This project implements a camera-to-BEV semantic segmentation pipeline. The code is organized around semantic image inputs, homography-based BEV alignment, and U-Net-style segmentation networks.

## Pipeline

1. Camera-view semantic images are loaded from one or more input folders.
2. Input colors are converted into one-hot semantic channels using `model/one_hot_conversion/convert_10.xml`.
3. A U-Net-style encoder extracts camera-view features.
4. SpatialTransformer layers apply precomputed homographies to align features into BEV space.
5. A decoder predicts BEV semantic classes.
6. Output channels are trained against BEV labels encoded with `model/one_hot_conversion/convert_3+occl.xml`.

## Preprocessing Utilities

- `preprocessing/ipm/ipm.py` performs inverse perspective mapping from camera images to the road plane.
- `preprocessing/occlusion/occlusion.py` creates occlusion-aware BEV labels from drone-view labels and camera visibility.
- `preprocessing/homography_converter/homography_converter.py` converts OpenCV homographies for the SpatialTransformer layer and different resolutions.

The default homography file is `preprocessing/homography_converter/uNetXST_homographies/2_F.py`.

## Model Variants

The architecture files under `model/architecture/` expose a `get_network(...)` entry point used by `train.py`, `evaluate.py`, and `predict.py`.

- `uNetXST.py`: compatibility wrapper for the default U-Net SpatialTransformer model.
- `uNet_front2bev.py`: baseline U-Net-style front-to-BEV model.
- `uNet_FPN_front2bev.py`: U-Net-style model with FPN-style feature fusion.
- `uNet_plus_FPN_front2bev.py`: U-Net++/nested-skip-inspired FPN-style variant.
- `uNet_dbplus_FPN_front2bev.py`: deeper/double-plus-style FPN variant.

The architecture code includes the BEV transformation inside the network through `third_party/spatial_transformer.py`.

## Classes

The active four-class BEV target is defined by `convert_3+occl.xml`:

- `road`
- `vehicle`
- `obstacle`
- `occluded`

The input palette `convert_10.xml` maps the camera-view semantic colors into ten input groups.
