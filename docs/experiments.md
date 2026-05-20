# Experiments

This page summarizes what is present in the current repository and what is available as project-reported context.

## Implemented Variants

The available architecture files include U-Net-style BEV segmentation models with SpatialTransformer-based feature warping:

| File | Notes |
| --- | --- |
| `model/architecture/uNet_front2bev.py` | Baseline U-Net-style front-to-BEV model. |
| `model/architecture/uNet_FPN_front2bev.py` | U-Net-style model with FPN-style feature fusion. |
| `model/architecture/uNet_plus_FPN_front2bev.py` | U-Net++/nested-skip-inspired FPN-style variant. |
| `model/architecture/uNet_dbplus_FPN_front2bev.py` | Deeper/double-plus-style FPN variant. |

The training script uses weighted categorical cross entropy and reports categorical accuracy plus mean IoU over one-hot labels.

## Reported Results

The resume/project description reports representative results:

| Setting | Reported result | Verification status |
| --- | ---: | --- |
| UNet++ECA, single-camera | mIoU around 86.95% | Project-reported; not independently reproduced from the current files. |
| Surround-view setup | mIoU around 72.78% | Project-reported; surround-view files/checkpoints are not included in the current repository. |

These numbers should be treated as historical project results. These results were not rerun during repository preparation, and the current repository does not include the checkpoints, logs, or full datasets needed to verify them.

## Attention Modules

The project description mentions CBAM and ECA comparisons. The available source files do not include CBAM/ECA implementation files or configs, so these docs do not present those as reproducible features.

## Compression And Sim-To-Real

The project scope included exploration of pruning, weight clustering, and sim-to-real transfer limitations. No pruning, clustering, or sim-to-real adaptation modules are present in the current repository, so they are documented as project context rather than runnable features.

## Reproducibility Notes

- The full dataset is not included.
- Trained checkpoints are intentionally ignored by `.gitignore`.
- Script defaults target the `2_F` front-camera setup.
- The active script parameters live near the top of `model/train.py`, `model/evaluate.py`, and `model/predict.py`.
