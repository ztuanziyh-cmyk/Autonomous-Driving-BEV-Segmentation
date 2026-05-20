# Autonomous Driving BEV Segmentation

Camera-based bird's-eye-view (BEV) semantic segmentation for autonomous-driving research. The project follows a Cam2BEV-style pipeline: camera-view semantic images are transformed into a BEV representation and a U-Net-style segmentation model predicts BEV classes.

This repository contains the core training, evaluation, prediction, preprocessing, and model files for a camera-based BEV semantic segmentation project.

## Task Setting

The task is semantic segmentation in BEV space from one or more vehicle camera views. The default configuration targets the `2_F` front-camera setup and predicts four BEV label groups:

- `road`
- `vehicle`
- `obstacle`
- `occluded`

The label grouping is defined in `model/one_hot_conversion/convert_3+occl.xml`.

## Main Features

- Cam2BEV-style image-to-BEV transformation using homographies and a SpatialTransformer layer.
- BEV semantic segmentation for road, vehicle, obstacle, and occluded regions.
- U-Net-style and U-Net++/FPN-style architecture variants under `model/architecture/`.
- Training, evaluation, prediction, loss-weight calculation, and preprocessing utilities.
- Preprocessing scripts for inverse perspective mapping, occlusion masking, and homography conversion.

Attention module experiments such as CBAM/ECA are part of the broader project scope, but CBAM/ECA implementation files are not present in the current repository.

## Repository Structure

```text
.
├── data/                         # Dataset placeholder/notes; full datasets are not tracked
├── docs/                         # Project documentation
├── model/
│   ├── architecture/             # U-Net-style BEV segmentation architectures
│   ├── one_hot_conversion/       # Label color conversion XML files
│   ├── config.2_F.unetxst.yml    # Example experiment configuration
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── predict.py                # Prediction/inference script
│   ├── loss_weights.py           # Class-weight helper
│   └── utils.py                  # Dataset, image, metrics, and loss utilities
├── preprocessing/
│   ├── camera_configs/           # Camera calibration examples
│   ├── homography_converter/     # Homography conversion for SpatialTransformer
│   ├── ipm/                      # Inverse perspective mapping utility
│   └── occlusion/                # Occlusion mask generation utility
├── requirements.txt
└── README.md
```

## Installation

Use a Python environment compatible with TensorFlow 2.x.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Expectations

The full dataset is not included. The default scripts expect the front-camera `2_F` layout below:

```text
data/2_F/
├── train/
│   ├── front/
│   └── bev+occlusion/
└── val/
    ├── front/
    └── bev+occlusion/
```

Images are expected to be color-coded semantic PNGs compatible with the conversion palettes in `model/one_hot_conversion/`.

See [docs/dataset.md](docs/dataset.md) for more detail.

## Training

The scripts use paths relative to the `model/` directory, so run them from there or update the path variables at the top of each script.

```bash
cd model
python train.py
```

Default settings are mirrored in `model/config.2_F.unetxst.yml`, but the current script keeps its active parameters as Python variables near the top of `train.py`.

## Evaluation

Place a compatible checkpoint at `model/best_weights.hdf5` or update `model_weights` in `model/evaluate.py`.

```bash
cd model
python evaluate.py
```

Evaluation writes confusion matrix and per-class IoU artifacts next to the checkpoint directory.

## Prediction

Place a compatible checkpoint at `model/best_weights.hdf5` or update `model_weights` in `model/predict.py`.

```bash
cd model
python predict.py
```

Predictions are written to the configured `prediction_dir`.

## Experiment Notes

The codebase contains U-Net, U-Net+FPN, U-Net++/FPN-style, and double-plus/FPN-style architecture files. The resume/project description reports representative results including:

- UNet++ECA single-camera mIoU around 86.95%.
- Surround-view mIoU around 72.78%.

Those values are project-reported numbers, not independently reproduced from the current repository. The available project files do not include trained checkpoints, raw logs, or CBAM/ECA implementation files needed to independently reproduce every reported variant.

Model compression and sim-to-real transfer are part of the project scope notes, but pruning, clustering, and sim-to-real modules are not implemented in the current repository.

See [docs/experiments.md](docs/experiments.md).

## Known Limitations

- Full datasets and trained model weights are not tracked.
- Scripts currently use top-of-file Python variables rather than a complete CLI/config parser.
- Some project-reported experiments are not fully reproducible from the available project files alone.
- The repository is research/prototype code and has not been packaged as a reusable Python library.

## Acknowledgements

Parts of the preprocessing and SpatialTransformer-based BEV workflow are adapted from Cam2BEV-style research code. The included `model/utils.py` and `model/architecture/third_party/spatial_transformer.py` retain their original MIT license headers where present.

No repository-level license file is currently included. Add one before publishing if you want explicit reuse terms for the full project.
