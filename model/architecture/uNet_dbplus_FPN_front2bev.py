import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate

from third_party.spatial_transformer import SpatialTransformer


def encoder(input, kernel_size=(3,3), activation=tf.nn.relu, batch_norm=True, dropout=0.1):

    t = input
    # common parameters
    pool_size = (2,2)
    padding = "same"
    encoder_layers = 5 * [None]
    filters1 = 16
    # layer 1
    filters1 = (2**0)*filters1
    t = Conv2D(filters=filters1, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters1, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = encoder_layers[0] = BatchNormalization()(t) if batch_norm else t
    t = MaxPooling2D(pool_size=pool_size, padding=padding)(t)
    t = Dropout(rate=dropout)(t) if dropout > 0 else t
    # layer 2
    filters2 = (2**1)*filters1
    t = Conv2D(filters=filters2, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters2, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = encoder_layers[1] = BatchNormalization()(t) if batch_norm else t
    t = MaxPooling2D(pool_size=pool_size, padding=padding)(t)
    t = Dropout(rate=dropout)(t) if dropout > 0 else t
    # layer 3
    filters3 = (2**2)*filters1
    t = Conv2D(filters=filters3, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters3, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = encoder_layers[2] = BatchNormalization()(t) if batch_norm else t
    t = MaxPooling2D(pool_size=pool_size, padding=padding)(t)
    t = Dropout(rate=dropout)(t) if dropout > 0 else t
    # layer 4
    filters4 = (2**3)*filters1
    t = Conv2D(filters=filters4, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters4, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = encoder_layers[3] = BatchNormalization()(t) if batch_norm else t
    t = MaxPooling2D(pool_size=pool_size, padding=padding)(t)
    t = Dropout(rate=dropout)(t) if dropout > 0 else t
    # layer 5
    filters5 = (2**4)*filters1
    t = Conv2D(filters=filters5, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters5, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = encoder_layers[4] = BatchNormalization()(t) if batch_norm else t
    # layer creation with successive pooling

    return encoder_layers


def joiner(encoder_layers, thetas, kernel_size=(3,3), activation=tf.nn.relu, batch_norm=True):

    warped_layers = 5 * [None]
    nonwarped_layers = encoder_layers
    strides = (2,2)
    padding = "same"
    filters1 = (2**0) * 16
    filters2 = (2**1) * 16
    filters3 = (2**2) * 16
    filters4 = (2**3) * 16
    filters5 = (2**4) * 16

    shape = encoder_layers[0].shape[1:]
    x00 = SpatialTransformer(shape, shape, theta_init=thetas, theta_const=True)(encoder_layers[0])
    shape = encoder_layers[1].shape[1:]
    x10 = SpatialTransformer(shape, shape, theta_init=thetas, theta_const=True)(encoder_layers[1])
    shape = encoder_layers[2].shape[1:]
    x20 = SpatialTransformer(shape, shape, theta_init=thetas, theta_const=True)(encoder_layers[2])
    shape = encoder_layers[3].shape[1:]
    x30 = SpatialTransformer(shape, shape, theta_init=thetas, theta_const=True)(encoder_layers[3])
    shape = encoder_layers[4].shape[1:]
    x40 = SpatialTransformer(shape, shape, theta_init=thetas, theta_const=True)(encoder_layers[4])

    t = Conv2DTranspose(filters=filters1, kernel_size=kernel_size, strides=strides, padding=padding)(x10)
    x01 = Concatenate()([x00, t])
    t = Conv2DTranspose(filters=filters2, kernel_size=kernel_size, strides=strides, padding=padding)(x20)
    x11 = Concatenate()([x10, t])
    t = Conv2DTranspose(filters=filters3, kernel_size=kernel_size, strides=strides, padding=padding)(x30)
    x21 = Concatenate()([x20, t])

    t = Conv2DTranspose(filters=filters1, kernel_size=kernel_size, strides=strides, padding=padding)(x11)
    x02 = Concatenate()([x00, x01, t])
    t = Conv2DTranspose(filters=filters2, kernel_size=kernel_size, strides=strides, padding=padding)(x21)
    x12 = Concatenate()([x10, x11, t])

    t = Conv2DTranspose(filters=filters1, kernel_size=kernel_size, strides=strides, padding=padding)(x12)
    x03 = Concatenate()([x00, x01, x02, t])

    # layer 1
    t = Concatenate()([nonwarped_layers[0], x03])
    t = Conv2D(filters=filters1, kernel_size=kernel_size, padding="same", activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters1, kernel_size=kernel_size, padding="same", activation=activation)(t)
    t = warped_layers[0] = BatchNormalization()(t) if batch_norm else t

    # layer 2
    t = Concatenate()([nonwarped_layers[1], x12])
    t = Conv2D(filters=filters2, kernel_size=kernel_size, padding="same", activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters2, kernel_size=kernel_size, padding="same", activation=activation)(t)
    t = warped_layers[1] = BatchNormalization()(t) if batch_norm else t

    # layer 3
    t = Concatenate()([nonwarped_layers[2], x21])
    t = Conv2D(filters=filters3, kernel_size=kernel_size, padding="same", activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters3, kernel_size=kernel_size, padding="same", activation=activation)(t)
    t = warped_layers[2] = BatchNormalization()(t) if batch_norm else t

    # layer 4
    t = Concatenate()([nonwarped_layers[3], x30])
    t = Conv2D(filters=filters4, kernel_size=kernel_size, padding="same", activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters4, kernel_size=kernel_size, padding="same", activation=activation)(t)
    t = warped_layers[3] = BatchNormalization()(t) if batch_norm else t

    # layer 5
    t = Concatenate()([nonwarped_layers[4], x40])
    t = Conv2D(filters=filters5, kernel_size=kernel_size, padding="same", activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters5, kernel_size=kernel_size, padding="same", activation=activation)(t)
    t = warped_layers[4] = BatchNormalization()(t) if batch_norm else t

    # # layer 1
    # t = Conv2DTranspose(filters=filters4, kernel_size=kernel_size, strides=strides, padding=padding)(t)
    # t = Concatenate()([warped_layers[3], t])
    # filters1 = (2**0) * filters1
    # shape = encoder_layers[0].shape[1:]
    # t = SpatialTransformer(shape, shape, theta_init=thetas, theta_const=True)(encoder_layers[0])
    # #t = Conv2D(filters=filters1, kernel_size=kernel_size, padding="same", activation=activation)(t)
    # #t = BatchNormalization()(t) if batch_norm else t
    # t = Conv2D(filters=filters1, kernel_size=kernel_size, padding="same", activation=activation)(t)
    # #t = warped_layers[0] = BatchNormalization()(t) if batch_norm else t
    # warped = t = BatchNormalization()(t) if batch_norm else t
    #
    # t = encoder_layers[0]
    # #t = Conv2D(filters=filters1, kernel_size=kernel_size, padding="same", activation=activation)(t)
    # #t = BatchNormalization()(t) if batch_norm else t
    # t = Conv2D(filters=filters1, kernel_size=kernel_size, padding="same", activation=activation)(t)
    # nonwarped = t = BatchNormalization()(t) if batch_norm else t
    #
    # t = Concatenate()([warped, nonwarped])
    # #t = Conv2D(filters=filters1, kernel_size=kernel_size, padding="same", activation=activation)(t)
    # #t = BatchNormalization()(t) if batch_norm else t
    # t = Conv2D(filters=filters1, kernel_size=kernel_size, padding="same", activation=activation)(t)
    # t = warped_layers[0] = BatchNormalization()(t) if batch_norm else t

    return warped_layers


def decoder(warped_layers, kernel_size=(3,3), activation=tf.nn.relu, batch_norm=True, dropout=0.1):

    # start at lowest encoder layer
    t = warped_layers[4]
    # common parameters
    strides = (2,2)
    padding = "same"
    filters1 = 16
    # layer 5 ---> 4
    filters4 = (2**3) * filters1
    t = Conv2DTranspose(filters=filters4, kernel_size=kernel_size, strides=strides, padding=padding)(t)
    t = Concatenate()([warped_layers[3], t])
    t = Dropout(rate=dropout)(t) if dropout > 0 else t
    t = Conv2D(filters=filters4, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters4, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t

    # layer 4 ---> 3
    filters3 = (2**2) * filters1
    t = Conv2DTranspose(filters=filters3, kernel_size=kernel_size, strides=strides, padding=padding)(t)
    t = Concatenate()([warped_layers[2], t])
    t = Dropout(rate=dropout)(t) if dropout > 0 else t
    t = Conv2D(filters=filters3, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters3, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t

    # layer 3 ---> 2
    filters2 = (2**1) * filters1
    t = Conv2DTranspose(filters=filters2, kernel_size=kernel_size, strides=strides, padding=padding)(t)
    t = Concatenate()([warped_layers[1], t])
    t = Dropout(rate=dropout)(t) if dropout > 0 else t
    t = Conv2D(filters=filters2, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters2, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t

    # layer 2 ---> 1
    filters1 = (2**0) * filters1
    t = Conv2DTranspose(filters=filters1, kernel_size=kernel_size, strides=strides, padding=padding)(t)
    t = Concatenate()([warped_layers[0], t])
    t = Dropout(rate=dropout)(t) if dropout > 0 else t
    t = Conv2D(filters=filters1, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t
    t = Conv2D(filters=filters1, kernel_size=kernel_size, padding=padding, activation=activation)(t)
    t = BatchNormalization()(t) if batch_norm else t

    return t


def uNetXST_f2b(input_shape, n_output_channels,thetas,n_inputs = 1,
                kernel_size = (3,3), 
                activation = tf.nn.relu, 
                batch_norm = True, 
                dropout = 0.1):

    # build inputs
    inputs = [Input(input_shape) for i in range(n_inputs)]

    # encode all inputs separately
    for i in inputs:
        encoder_layers = encoder(i, kernel_size, activation, batch_norm, dropout)

    # fuse encodings of all inputs at all layers
    warped_layers = joiner(encoder_layers, thetas, kernel_size, activation, batch_norm)

    # decode from bottom to top layer
    reconstruction = decoder(warped_layers, kernel_size, activation, batch_norm, dropout)
    # build final prediction layer
    prediction = Conv2D(filters=n_output_channels, kernel_size=kernel_size, padding="same", activation=activation)(reconstruction)
    prediction = Activation("softmax")(prediction)

    return Model(inputs, prediction)

def get_network(input_shape, n_output_channels, thetas=None, n_inputs=1):
    if thetas is None:
        thetas = np.eye(3)
    return uNetXST_f2b(input_shape, n_output_channels, thetas=thetas, n_inputs=n_inputs)


if __name__ == "__main__":
    image_shape = [256, 512]
    model = get_network((image_shape[0], image_shape[1], 10), 4)
    model.summary()
