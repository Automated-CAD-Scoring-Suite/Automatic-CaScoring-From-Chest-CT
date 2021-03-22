#
# U-NET TF Implementation
#

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model


def u_net(levels, convs, input_shape, kernel_size=(3, 3), out_channels=1, activation='relu',
          batch_norm=False, drop_out=None, initial_features=32):
    """
        Implementation of the Unet Arch, using tensorflow Layers
        and Functional API
    :param levels: Number of Down sampling Levels
    :param convs: Number of consecutive Convolutions
    :param input_shape: Data Input Shape
    :param kernel_size: Convolution kernel Size
    :param out_channels: Number of channels in the Final Layer
    :param activation: Default Relu
    :param batch_norm: Boolean used to apply Batch normalization
    :param drop_out: if Float Apply Drop out
    :param initial_features: Number of Initial Convolutional Features
    :return: tf Model() instance
    """
    # Parameters that are constant for all Layers
    parameters = dict(kernel_size=kernel_size, padding='same', activation=activation)

    # Model Input
    inputs = Input(input_shape)
    x = inputs

    # Encoder Part
    connects = {}
    for level in range(levels):
        for _ in range(convs):
            x = Conv2D(initial_features*2**level, **parameters)(x)
            x = BatchNormalization()(x) if batch_norm else x
            x = Dropout(drop_out)(x) if drop_out is not None else x
        if level < levels-1:
            connects[level] = x
            x = MaxPooling2D()(x)

    # Decoder Part
    for level in reversed(range(levels-1)):
        x = Conv2DTranspose(initial_features*2**level, strides=2, **parameters)(x)
        x = Concatenate()([x, connects[level]])
        for _ in range(convs):
            x = Conv2D(initial_features*2**level, **parameters)(x)
            x = BatchNormalization()(x) if batch_norm else x
            x = Dropout(drop_out)(x) if drop_out is not None else x

    # Model Output
    x = Conv2D(out_channels, kernel_size=1, activation='sigmoid', padding='same')(x)

    return Model(inputs=[inputs], outputs=[x], name=f'Unet_{levels}x{convs}_{initial_features}F')
