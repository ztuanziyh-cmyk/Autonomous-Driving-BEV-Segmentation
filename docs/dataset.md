# Dataset

The dataset is not included in this repository. The default paths target the `2_F` front-camera setup.

## Expected Layout

Run scripts from the `model/` directory, or adjust the relative paths in the scripts.

```text
data/2_F/
├── train/
│   ├── front/
│   └── bev+occlusion/
└── val/
    ├── front/
    └── bev+occlusion/
```

The default training paths in `model/train.py` are:

| Field | Default |
| --- | --- |
| `input_training` | `../data/2_F/train/front` |
| `label_training` | `../data/2_F/train/bev+occlusion` |
| `input_validation` | `../data/2_F/val/front` |
| `label_validation` | `../data/2_F/val/bev+occlusion` |
| `image_shape` | `[256, 512]` |
| `one_hot_palette_input` | `one_hot_conversion/convert_10.xml` |
| `one_hot_palette_label` | `one_hot_conversion/convert_3+occl.xml` |

## Config File

`model/config.2_F.unetxst.yml` records the intended experiment fields:

- `input-training`, `label-training`
- `input-validation`, `label-validation`
- `max-samples-training`, `max-samples-validation`
- `image-shape`
- `one-hot-palette-input`, `one-hot-palette-label`
- `model`
- `unetxst-homographies`
- `epochs`, `batch-size`, `learning-rate`
- `loss-weights`
- `early-stopping-patience`
- `save-interval`, `output-dir`
- `class-names`
- optional `model-weights`
- `input-testing`, `max-samples-testing`, optional `prediction-dir`

The current scripts do not fully parse this YAML file as a runtime config. Treat it as a documented experiment configuration and keep the top-of-file variables in the scripts synchronized when changing experiments.

## Label Palettes

`convert_10.xml` is used for camera-view semantic inputs. `convert_3+occl.xml` groups BEV labels into:

- `road`
- `vehicle`
- `obstacle`
- `occluded`

Images should be semantic color maps using the colors defined in the XML files.

## Checkpoints And Outputs

Training writes timestamped outputs under `model/output/` by default. Checkpoints and generated outputs are ignored by `.gitignore` and should not be committed to the repository.
