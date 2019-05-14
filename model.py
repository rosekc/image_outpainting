import keras
import keras.backend as K
from keras.layers import (Concatenate, Conv2D, Conv2DTranspose, Dense, Dropout,
                          Flatten, Input, Lambda, LeakyReLU, ReLU, BatchNormalization)
from keras.models import Sequential

activation_map = {
    'leaky_relu': LeakyReLU(0.02)
}


class conv2d:
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_dropout=False,
                 use_dropout_at_prediction=True,
                 use_batch_norm=False,
                 use_bias=True,
                 layers=Conv2D,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.build_in_activation = None
        if not activation in activation_map:
            self.build_in_activation = activation
        self.activation = activation
        self.use_dropout = use_dropout
        self.use_dropout_at_prediction = use_dropout_at_prediction
        self.use_batch_norm = use_batch_norm
        self.prepared_conv2d = layers(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      data_format=data_format,
                                      dilation_rate=dilation_rate,
                                      activation=self.build_in_activation,
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer,
                                      kernel_constraint=kernel_constraint,
                                      bias_constraint=bias_constraint,
                                      **kwargs)

    def __call__(self, x):
        x = self.prepared_conv2d(x)
        if self.build_in_activation is None:
            x = activation_map[self.activation](x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        if self.use_dropout:
            # this is a trick to use dropout at prediction
            if self.use_dropout_at_prediction:
                x = Dropout(0.5)(x, training=True)
            else:
                x = Dropout(0.5)(x)
        return x


class conv2dtranspose(conv2d):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_dropout=False,
                 use_dropout_at_prediction=False,
                 use_batch_norm=False,
                 use_bias=True,
                 layers=Conv2DTranspose,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        return super().__init__(filters, kernel_size, strides=strides,
                                padding=padding, data_format=data_format, dilation_rate=dilation_rate,
                                activation=activation, use_dropout=use_dropout, use_batch_norm=use_batch_norm, use_bias=use_bias,
                                layers=layers, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)


dcgan_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)


def bulid_generator(inputs):
    x = conv2d(64, (5, 5), strides=(1, 1), padding='same',
               activation='relu')(inputs)
    x = conv2d(128, (3, 3), strides=(2, 2), padding='same',
               activation='relu')(x)
    x = conv2d(256, (3, 3), strides=(1, 1), padding='same',
               activation='relu')(x)
    x = conv2d(256, (3, 3), strides=(1, 1), padding='same',
               dilation_rate=2, activation='relu')(x)
    x = conv2d(256, (3, 3), strides=(1, 1), padding='same',
               dilation_rate=4, activation='relu')(x)
    x = conv2d(256, (3, 3), strides=(1, 1), padding='same',
               dilation_rate=8, activation='relu')(x)
    if inputs.shape[2] == 256:
        x = conv2d(256, (3, 3), strides=(1, 1), padding='same',
                   dilation_rate=16, activation='relu')(x)
    x = conv2d(256, (3, 3), strides=(1, 1), padding='same',
               activation='relu')(x)
    x = conv2dtranspose(
        128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = conv2d(64, (3, 3), strides=(1, 1),
               padding='same', activation='relu')(x)
    x = conv2d(3, (3, 3), strides=(1, 1), padding='same',
               activation='tanh')(x)
    return x


def build_discriminator_proxy(combined_images, padding_width):
    padding_left = keras.layers.Lambda(
        lambda x: x[:, :, :padding_width, :])(combined_images)
    padding_right = keras.layers.Lambda(
        lambda x: x[:, :, -padding_width:, :])(combined_images)
    return padding_left, padding_right


def build_global_discriminator(inputs):
    x = conv2d(32, (5, 5),  strides=(2, 2), padding='same',
               activation='leaky_relu', use_dropout_at_prediction=False)(inputs)
    x = conv2d(64, (5, 5), strides=(
        2, 2), padding='same', activation='leaky_relu', use_dropout_at_prediction=False)(x)
    x = conv2d(64, (5, 5), strides=(
        2, 2), padding='same', activation='leaky_relu', use_dropout_at_prediction=False)(x)
    x = conv2d(64, (5, 5), strides=(
        2, 2), padding='same', activation='leaky_relu', use_dropout_at_prediction=False)(x)
    x = conv2d(64, (5, 5), strides=(
        2, 2), padding='same', activation='leaky_relu', use_dropout_at_prediction=False)(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU()(x)
    return x


def build_local_discriminator(inputs):
    x = conv2d(32, (5, 5), strides=(2, 2), padding='same',
               activation='relu', use_dropout_at_prediction=False)(inputs)
    x = conv2d(64, (5, 5), strides=(2, 2),
               padding='same', activation='leaky_relu', use_dropout_at_prediction=False)(x)
    x = conv2d(64, (5, 5), strides=(2, 2),
               padding='same', activation='leaky_relu', use_dropout_at_prediction=False)(x)
    x = conv2d(64, (5, 5), strides=(2, 2),
               padding='same', activation='leaky_relu', use_dropout_at_prediction=False)(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU()(x)
    return x

def build_model(input_shape=(256, 256, 3), padding_width=64, alpha=0.0004, gpus=None):
    def multi_gpu_wrapper(model):
        if gpus:
            from keras.utils import multi_gpu_model
            return multi_gpu_model(model, gpus=gpus)
        return model
    
    padded_iamges_inputs = Input(input_shape)

    G_layer = bulid_generator(padded_iamges_inputs)
    G = multi_gpu_wrapper(keras.Model(padded_iamges_inputs, G_layer))   
    G.compile(loss='mse', optimizer=dcgan_optimizer)

    discriminator_inputs = keras.layers.Input(input_shape)
    DiscriminatorProxy = build_discriminator_proxy(
        discriminator_inputs, padding_width)

    # local discriminator left and right
    DLL = build_local_discriminator(DiscriminatorProxy[0])
    DLR = build_local_discriminator(DiscriminatorProxy[1])
    # global discriminator
    DG = build_global_discriminator(discriminator_inputs)
    x = Concatenate()([DLL, DLR, DG])
    x = Dense(1, activation='sigmoid')(x)
    D =multi_gpu_wrapper(keras.Model(discriminator_inputs, x))

    D.compile(loss='binary_crossentropy', optimizer=dcgan_optimizer)

    D.trainable = False
    # combined model
    C = D(G(padded_iamges_inputs))
    C = multi_gpu_wrapper(keras.Model(padded_iamges_inputs, [G_layer, C]))
    C.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[
              1, alpha], optimizer=dcgan_optimizer)

    return G, D, C
