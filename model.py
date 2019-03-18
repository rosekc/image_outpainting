import keras
import keras.backend as K
from keras.layers import Conv2D, Conv2DTranspose, Dense
from keras.models import Sequential


def bulid_generator(inputs):
    x = Conv2D(64, (5, 5), strides=(1, 1), padding='same',
               activation='relu')(inputs)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same',
               activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
               activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
               dilation_rate=4, activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
               dilation_rate=8, activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
               dilation_rate=16, activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
               activation='relu')(x)
    x = Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1),
               padding='same', activation='relu')(x)
    x = Conv2D(3, (3, 3), strides=(1, 1), padding='same',
               activation='sigmoid')(x)
    return x


def build_discriminator_proxy(combined_images, padding_width):
    padding_left = keras.layers.Lambda(
        lambda x: x[:, :, :padding_width, :])(combined_images)
    padding_right = keras.layers.Lambda(
        lambda x: x[:, :, -padding_width:, :])(combined_images)
    return padding_left, padding_right


def build_global_discriminator(input_shape=(256, 256, 3)):
    return keras.models.Sequential([
        keras.layers.Conv2D(32, (5, 5),  strides=(2, 2), padding='same',
                            activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', activation='relu'),
        keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', activation='relu'),
        keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', activation='relu'),
        keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu')
    ])


def build_local_discriminator(input_shape=(256, 64, 3)):
    return keras.models.Sequential([
        keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                            activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', activation='relu'),
        keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', activation='relu'),
        keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu')
    ])


def direct(y_true, y_pred):
    return y_pred


def build_model(input_shape=(256, 256, 3), alpha=0.004, eps=1e-9):
    padded_iamges_inputs = keras.layers.Input(input_shape)

    G_layer = bulid_generator(padded_iamges_inputs)
    G = keras.Model(padded_iamges_inputs, G_layer)
    G.compile(loss='mse', optimizer=keras.optimizers.Adam())

    discriminator_inputs = keras.layers.Input(input_shape)
    DiscriminatorProxy = build_discriminator_proxy(discriminator_inputs, 64)

    # local discriminator left and right
    DLL = build_local_discriminator()(DiscriminatorProxy[0])
    DLR = build_local_discriminator()(DiscriminatorProxy[1])
    # global discriminator
    DG = build_global_discriminator()(discriminator_inputs)
    x = keras.layers.Concatenate()([DLL, DLR, DG])
    x = keras.layers.Dense(1)(x)
    D = keras.Model(discriminator_inputs, x)
    D.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())

    D.trainable = False
    # combined model
    C = D(G(padded_iamges_inputs))
    C = keras.layers.Lambda(lambda x: -alpha * K.log(K.maximum(x, eps)))(C)
    C = keras.Model(padded_iamges_inputs, [G_layer, C])
    C.compile(loss=['mse', direct], optimizer=keras.optimizers.Adam())

    return G, D, C
