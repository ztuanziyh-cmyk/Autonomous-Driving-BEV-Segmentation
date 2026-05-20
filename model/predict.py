import importlib
import os
import sys
import tqdm
import numpy as np
import cv2
import tensorflow as tf

import utils


# parse parameters from config file or CLI
#input_testing = ['../data/1_FRLR/val/front', '../data/1_FRLR/val/rear', '../data/1_FRLR/val/left','../data/1_FRLR/val/right']
input_testing = ['../data/2_F/val/front']
max_samples_testing = 1000
image_shape = [256, 512]
one_hot_palette_input = 'one_hot_conversion/convert_10.xml'
one_hot_palette_label = 'one_hot_conversion/convert_3+occl.xml'
#class_names = ['road', 'sidewalk', 'person', 'vehicle', 'truck', 'bus', 'bike', 'obstacle', 'vegetation', 'occluded']
#class_names = ['road', 'vehicle', 'obstacle', 'occluded']
model = 'architecture/uNetXST.py'
unetxst_homographies = '../preprocessing/homography_converter/uNetXST_homographies/2_F.py'
model_weights = './best_weights.hdf5'
prediction_dir = '../prediction/f2b'

# determine absolute filepaths
input_testing          = [utils.abspath(path) for path in input_testing]
one_hot_palette_input  = utils.abspath(one_hot_palette_input)
one_hot_palette_label  = utils.abspath(one_hot_palette_label)
model                  = utils.abspath(model)
unetxst_homographies   = utils.abspath(unetxst_homographies) if unetxst_homographies is not None else unetxst_homographies
model_weights          = utils.abspath(model_weights)
prediction_dir         = utils.abspath(prediction_dir)


# load network architecture module
architecture = utils.load_module(model)


# get max_samples_testing samples
files_input = [utils.get_files_in_folder(folder) for folder in input_testing]
_, idcs = utils.sample_list(files_input[0], n_samples=max_samples_testing)
files_input = [np.take(f, idcs) for f in files_input]
n_inputs = len(input_testing)
n_samples = len(files_input[0])
image_shape_original = utils.load_image(files_input[0][0]).shape[0:2]
print(f"Found {n_samples} samples")


# parse one-hot-conversion.xml
one_hot_palette_input = utils.parse_convert_xml(one_hot_palette_input)
one_hot_palette_label = utils.parse_convert_xml(one_hot_palette_label)
n_classes_input = len(one_hot_palette_input)
n_classes_label = len(one_hot_palette_label)


# build model
if unetxst_homographies is not None:
    uNetXSTHomographies = utils.load_module(unetxst_homographies)
    model = architecture.get_network((image_shape[0], image_shape[1], n_classes_input), n_classes_label, n_inputs=n_inputs, thetas=uNetXSTHomographies.H)
else:
    model = architecture.get_network((image_shape[0], image_shape[1], n_classes_input), n_classes_label)
model.load_weights(model_weights)
print(f"Reloaded model from {model_weights}")


# build data parsing function
def parse_sample(input_files):
    # parse and process input images
    inputs = []
    for inp in input_files:
        inp = utils.load_image_op(inp)
        inp = utils.resize_image_op(inp, image_shape_original, image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        inp = utils.one_hot_encode_image_op(inp, one_hot_palette_input)
        inputs.append(inp)
    inputs = inputs[0] if n_inputs == 1 else tuple(inputs)
    return inputs


# create output directory
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)


# run predictions
print(f"Running predictions and writing to {prediction_dir} ...")
for k in tqdm.tqdm(range(n_samples)):

    input_files = [files_input[i][k] for i in range(n_inputs)]

    # load sample
    inputs = parse_sample(input_files)

    # add batch dim
    if n_inputs > 1:
        inputs = [np.expand_dims(i, axis=0) for i in inputs]
    else:
        inputs = np.expand_dims(inputs, axis=0)

    # run prediction
    prediction = model.predict(inputs).squeeze()

    # convert to output image
    prediction = utils.one_hot_decode_image(prediction, one_hot_palette_label)

    # write to disk
    output_file = os.path.join(prediction_dir, os.path.basename(files_input[0][k]))
    #cv2.imwrite(output_file, cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))
    cv2.imencode('.jpg', cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR), )[1].tofile(output_file)
