import importlib
import os
import sys
import tqdm
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf

import utils
# road 98.10
# vehicle  60.31
# obstacle 73.30
# occluded 86.62

input_validation = ['../data/2_F/val/front']
label_validation = '../data/2_F/val/bev+occlusion'
max_samples_validation = 10000
image_shape = [256, 512]
one_hot_palette_input = 'one_hot_conversion/convert_10.xml'
one_hot_palette_label = 'one_hot_conversion/convert_3+occl.xml'
#class_names = ['road', 'sidewalk', 'person', 'vehicle', 'truck', 'bus', 'bike', 'obstacle', 'vegetation', 'occluded']
class_names = ['road', 'vehicle', 'obstacle', 'occluded']
model = 'architecture/uNetXST.py'
unetxst_homographies = '../preprocessing/homography_converter/uNetXST_homographies/2_F.py'
model_weights = './best_weights.hdf5'

# determine absolute filepaths
input_validation = [utils.abspath(path) for path in input_validation]
label_validation = utils.abspath(label_validation)
one_hot_palette_input = utils.abspath(one_hot_palette_input)
one_hot_palette_label = utils.abspath(one_hot_palette_label)
model = utils.abspath(model)
unetxst_homographies = utils.abspath(unetxst_homographies) if unetxst_homographies is not None else unetxst_homographies
model_weights = utils.abspath(model_weights)

# load network architecture module
architecture = utils.load_module(model)

# get max_samples_validation random validation samples
files_input = [utils.get_files_in_folder(folder) for folder in input_validation]
files_label = utils.get_files_in_folder(label_validation)
_, idcs = utils.sample_list(files_label, n_samples=max_samples_validation)
files_input = [np.take(f, idcs) for f in files_input]
files_label = np.take(files_label, idcs)
n_inputs = len(input_validation)
n_samples = len(files_label)
image_shape_original_input = utils.load_image(files_input[0][0]).shape[0:2]
image_shape_original_label = utils.load_image(files_label[0]).shape[0:2]
print(f"Found {n_samples} samples")

# parse one-hot-conversion.xml
one_hot_palette_input = utils.parse_convert_xml(one_hot_palette_input)
one_hot_palette_label = utils.parse_convert_xml(one_hot_palette_label)
n_classes_input = len(one_hot_palette_input)
n_classes_label = len(one_hot_palette_label)

# build model
if unetxst_homographies is not None:
    uNetXSTHomographies = utils.load_module(unetxst_homographies)
    model = architecture.get_network((image_shape[0], image_shape[1], n_classes_input), n_classes_label,
                                     n_inputs=n_inputs, thetas=uNetXSTHomographies.H)
else:
    model = architecture.get_network((image_shape[0], image_shape[1], n_classes_input), n_classes_label)
model.load_weights(model_weights)
print(f"Reloaded model from {model_weights}")


# build data parsing function
def parse_sample(input_files, label_file):
    # parse and process input images
    inputs = []
    for inp in input_files:
        inp = utils.load_image_op(inp)
        inp = utils.resize_image_op(inp, image_shape_original_input, image_shape,
                                    interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        inp = utils.one_hot_encode_image_op(inp, one_hot_palette_input)
        inputs.append(inp)
    inputs = inputs[0] if n_inputs == 1 else tuple(inputs)
    # parse and process label image
    label = utils.load_image_op(label_file)
    label = utils.resize_image_op(label, image_shape_original_label, image_shape,
                                  interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = utils.one_hot_encode_image_op(label, one_hot_palette_label)
    return inputs, label


# evaluate confusion matrix
print("Evaluating confusion matrix ...")
confusion_matrix = np.zeros((n_classes_label, n_classes_label), dtype=np.int64)
for k in tqdm.tqdm(range(n_samples)):

    input_files = [files_input[i][k] for i in range(n_inputs)]
    label_file = files_label[k]

    # load sample
    inputs, label = parse_sample(input_files, label_file)

    # add batch dim
    if n_inputs > 1:
        inputs = [np.expand_dims(i, axis=0) for i in inputs]
    else:
        inputs = np.expand_dims(inputs, axis=0)

    # run prediction
    prediction = model.predict(inputs).squeeze()

    # compute confusion matrix
    label = np.argmax(label, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    sample_confusion_matrix = tf.math.confusion_matrix(label.flatten(), prediction.flatten(),
                                                       num_classes=n_classes_label).numpy()

    # sum confusion matrix over dataset
    confusion_matrix += sample_confusion_matrix

# normalize confusion matrix rows (What percentage of class X has been predicted to be class Y?)
confusion_matrix_norm = confusion_matrix / np.sum(confusion_matrix, axis=1)[:, np.newaxis]

# compute per-class IoU
row_sum = np.sum(confusion_matrix, axis=0)
col_sum = np.sum(confusion_matrix, axis=1)
diag = np.diag(confusion_matrix)
intersection = diag
union = row_sum + col_sum - diag
ious = intersection / union
iou = {}
for idx, v in enumerate(ious):
    iou[class_names[idx]] = v

# print metrics
print("\nPer-class IoU:")
for k, v in iou.items():
    print(f"  {k}: {100 * v:3.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix)
print("\nNormalized Confusion Matrix:")
print(confusion_matrix_norm)

# plot confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix_norm * 100, class_names, class_names)
plt.figure(figsize=(8, 8))
hm = sb.heatmap(confusion_matrix_df,
                annot=True,
                fmt=".2f",
                square=True,
                vmin=0,
                vmax=100,
                cbar_kws={"label": "%", "shrink": 0.8},
                cmap=plt.cm.Blues)
hm.set_xticklabels(hm.get_xticklabels(), rotation=30)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

# save confusion matrix and class ious to file and export plot
eval_folder = os.path.join(os.path.dirname(model_weights), os.pardir, "Evaluation")
if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)
filename = os.path.join(eval_folder, "confusion_matrix.txt")
np.savetxt(filename, confusion_matrix, fmt="%d")
filename = os.path.join(eval_folder, "class_iou.txt")
np.savetxt(filename, ious, fmt="%f")
filename = os.path.join(eval_folder, "confusion_matrix.pdf")
plt.savefig(filename, bbox_inches="tight")
