import importlib
import os
import sys
from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf

import utils

input_training       = ['../data/2_F/train/front']
label_training       ='../data/2_F/train/bev+occlusion'
max_samples_training = 100000
input_validation     = ['../data/2_F/val/front']
label_validation     ='../data/2_F/val/bev+occlusion'
max_samples_validation = 10000

image_shape          =[256, 512]
#one_hot_palette_input= 'one_hot_conversion/convert_10.xml'
#one_hot_palette_label ='one_hot_conversion/convert_9+occl.xml'
one_hot_palette_input = 'one_hot_conversion/convert_10.xml'
one_hot_palette_label ='one_hot_conversion/convert_3+occl.xml'

model      = 'architecture/uNetXST.py'
unetxst_homographies= '../preprocessing/homography_converter/uNetXST_homographies/2_F.py'
epochs= 50
batch_size= 4
learning_rate= 1e-4
#loss_weights= [0.98684351, 2.2481491, 10.47452063, 4.78351389, 7.01028204, 8.41360361, 10.91633349, 2.38571558, 1.02473193, 2.79359197]
loss_weights= [1.00752063, 5.06392476, 1.15378408, 1.16118375]
early_stopping_patience= 20

save_interval= 5
output_dir = 'output'

# for training continuation, evaluation and prediction only
#class_names= ['road', 'sidewalk', 'person', 'vehicle', 'truck', 'bus', 'bike', 'obstacle', 'vegetation', 'occluded']
class_names= ['road', 'vehicle', 'obstacle', 'occluded']
model_weights = None

# for predict.py only
input_testing= ['../data/2_F/val/front']
max_samples_testing= 10000
# prediction-dir:

# determine absolute filepaths
input_training         = [utils.abspath(path) for path in input_training]
label_training         = utils.abspath(label_training)
input_validation       = [utils.abspath(path) for path in input_validation]
label_validation       = utils.abspath(label_validation)
one_hot_palette_input  = utils.abspath(one_hot_palette_input)
one_hot_palette_label  = utils.abspath(one_hot_palette_label)
model                  = utils.abspath(model)
unetxst_homographies   = utils.abspath(unetxst_homographies) if unetxst_homographies is not None else unetxst_homographies
model_weights          = utils.abspath(model_weights) if model_weights is not None else model_weights
output_dir             = utils.abspath(output_dir)

# load network architecture module
architecture = utils.load_module(model)

# get max_samples_training random training samples
n_inputs = len(input_training)
files_train_input = [utils.get_files_in_folder(folder) for folder in input_training]
files_train_label = utils.get_files_in_folder(label_training)
_, idcs = utils.sample_list(files_train_label, n_samples=max_samples_training)
files_train_input = [np.take(f, idcs) for f in files_train_input]
files_train_label = np.take(files_train_label, idcs)
image_shape_original_input = utils.load_image(files_train_input[0][0]).shape[0:2]
image_shape_original_label = utils.load_image(files_train_label[0]).shape[0:2]
print(f"Found {len(files_train_label)} training samples")

# get max_samples_validation random validation samples
files_valid_input = [utils.get_files_in_folder(folder) for folder in input_validation]
files_valid_label = utils.get_files_in_folder(label_validation)
_, idcs = utils.sample_list(files_valid_label, n_samples= max_samples_validation)
files_valid_input = [np.take(f, idcs) for f in files_valid_input]
files_valid_label = np.take(files_valid_label, idcs)
print(f"Found {len(files_valid_label)} validation samples")

# parse one-hot-conversion.xml
one_hot_palette_input = utils.parse_convert_xml(one_hot_palette_input)
one_hot_palette_label = utils.parse_convert_xml(one_hot_palette_label)
n_classes_input = len(one_hot_palette_input)
n_classes_label = len(one_hot_palette_label)

# build dataset pipeline parsing functions
def parse_sample(input_files, label_file):
    # parse and process input images
    inputs = []
    for inp in input_files:
        inp = utils.load_image_op(inp)
        inp = utils.resize_image_op(inp, image_shape_original_input, image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        inp = utils.one_hot_encode_image_op(inp, one_hot_palette_input)
        inputs.append(inp)
    inputs = inputs[0] if n_inputs == 1 else tuple(inputs)
    # parse and process label image
    label = utils.load_image_op(label_file)
    label = utils.resize_image_op(label, image_shape_original_label, image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = utils.one_hot_encode_image_op(label,one_hot_palette_label)
    return inputs, label

# build training data pipeline
dataTrain = tf.data.Dataset.from_tensor_slices((tuple(files_train_input), files_train_label))
dataTrain = dataTrain.shuffle(buffer_size=max_samples_training, reshuffle_each_iteration=True)
dataTrain = dataTrain.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataTrain = dataTrain.batch(batch_size, drop_remainder=True)
dataTrain = dataTrain.repeat(epochs)
dataTrain = dataTrain.prefetch(1)
print("Built data pipeline for training")

# build validation data pipeline
dataValid = tf.data.Dataset.from_tensor_slices((tuple(files_valid_input), files_valid_label))
dataValid = dataValid.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataValid = dataValid.batch(1)
dataValid = dataValid.repeat(epochs)
dataValid = dataValid.prefetch(1)
print("Built data pipeline for validation")

# build model
if unetxst_homographies is not None:
  uNetXSTHomographies = utils.load_module(unetxst_homographies)
  model = architecture.get_network((image_shape[0], image_shape[1], n_classes_input), n_classes_label, n_inputs=n_inputs, thetas=uNetXSTHomographies.H)
else:
  model = architecture.get_network((image_shape[0], image_shape[1], n_classes_input), n_classes_label)
if model_weights is not None:
  model.load_weights(model_weights)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
if loss_weights is not None:
    loss = utils.weighted_categorical_crossentropy(loss_weights)
else:
    loss = tf.keras.losses.CategoricalCrossentropy()
metrics = [tf.keras.metrics.CategoricalAccuracy(), utils.MeanIoUWithOneHotLabels(num_classes=n_classes_label)]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#print(f"Compiled model {os.path.basename(model)}")
print("Compiled model")

# create output directories
model_output_dir = os.path.join(output_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
tensorboard_dir = os.path.join(model_output_dir, "TensorBoard")
checkpoint_dir  = os.path.join(model_output_dir, "Checkpoints")
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# create callbacks to be called after each epoch
n_batches_train = len(files_train_label) // batch_size
n_batches_valid = len(files_valid_label)
tensorboard_cb      = tf.keras.callbacks.TensorBoard(tensorboard_dir, update_freq="epoch", profile_batch=0)
checkpoint_cb       = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, "e{epoch:03d}_weights.hdf5"), save_freq=n_batches_train*save_interval, save_weights_only=True)
best_checkpoint_cb  = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, "best_weights.hdf5"), save_best_only=True, monitor="val_mean_io_u_with_one_hot_labels", mode="max", save_weights_only=True)
early_stopping_cb   = tf.keras.callbacks.EarlyStopping(monitor="val_mean_io_u_with_one_hot_labels", mode="max", patience=early_stopping_patience, verbose=1)
callbacks = [tensorboard_cb, checkpoint_cb, best_checkpoint_cb, early_stopping_cb]


# start training
print("Starting training...")
model.fit(dataTrain,
          epochs=epochs, steps_per_epoch=n_batches_train,
          validation_data=dataValid, validation_freq=1, validation_steps=n_batches_valid,
          callbacks=callbacks)
