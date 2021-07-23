#
# U-NET TF Implementation
#
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Input, BatchNormalization, Dropout
from Unet.modes import Conv, Transpose, Pooling, UpSample
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Input, Activation, UpSampling3D, concatenate
from tensorflow.keras.models import Model


class UNet:
    def __init__(self, conv: str = "conv2D", up_sample: str = None, transpose: str = None, pool: str = "max2D"):
        """
        Implementation of the Unet Architecture, using Tensorflow's Keras API

        Added Functionality:
            - Convolutional and Up Sampling layers are now in both 2D and 3D
            - Added Up sampling using tf`s UpSampling2D
            - Added Average Pooling using tf`s AveragePooling2D

        :param conv:
        :param up_sample:
        :param transpose:
        :param pool:
        """
        print("Initializing Unet")
        self.transpose = False
        self.up_sample = False

        # Catch the Entered Mode from User
        if transpose:
            self.expand = Transpose[transpose].value
            self.transpose = True
        if up_sample:
            self.expand = UpSample[up_sample].value
            self.up_sample = True

        self.contract = Conv[conv].value
        self.Pool = Pooling[pool].value

    def __call__(self, levels, convolutions, input_shape, kernel_size, out_channels=1, activation='relu',
                 batch_norm=False, drop_out=None, initial_features=32):
        """
            Implementation of the Unet Arch, using tensorflow Layers
            and Functional API
        :param levels: Number of Down sampling Levels
        :param convolutions: Number of consecutive Convolutions
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
            for conv in range(convolutions):
                if level < levels -1:
                    x = self.contract(initial_features * (2 ** level) * (conv + 1), **parameters)(x)
                else:
                    # Preventing escalation of Layer Filter
                    x = self.contract(initial_features * (2 ** (level - 1)) * (conv + 1), **parameters)(x)

            if batch_norm:
                x = BatchNormalization()(x)
            if drop_out is not None:
                x = Dropout(drop_out)(x)

            if level < levels - 1:
                connects[level] = x
                x = self.Pool()(x)

        print(f"Connections for a {levels} is {len(connects)} ... ")
        # Decoder Part
        for level in reversed(range(levels - 1)):
            if self.transpose:
                x = self.expand(initial_features * (2 ** level), strides=2, **parameters)(x)
            if self.up_sample:
                x = self.expand()(x)

            x = Concatenate()([x, connects[level]])

            for conv in range(convolutions):
                if level != 0:
                    x = self.contract(initial_features * 2 ** level, **parameters)(x)
                else:
                    x = self.contract(initial_features * 2 ** (level+1), **parameters)(x)

        # Model Output
        x = self.contract(out_channels, kernel_size=1, activation='sigmoid', padding='same')(x)
        return Model(inputs=[inputs], outputs=[x], name=f'Unet_{levels}x{convolutions}_{initial_features}F')


def getUnet3d_3_MGPU(input_shape, pool_size=(2, 2, 2), conv_size=(3, 3, 3), drop_out=0.5):
    # inputs = Input(input_shape, name='model_input')
    # conv1 = Conv3D(32, conv_size, activation='relu', padding='same', name='conv_1_1')(inputs)
    # norm1 = BatchNormalization(axis=4, name='norm_1_1')(conv1)
    # conv1 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_1_2')(norm1)
    # norm1 = BatchNormalization(axis=4, name='norm_1_2')(conv1)
    # pool1 = MaxPooling3D(pool_size=pool_size, name='pool_1')(norm1)
    #
    # conv2 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_2_1')(pool1)
    # norm2 = BatchNormalization(axis=4, name='norm_2_1')(conv2)
    # conv2 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_2_2')(norm2)
    # norm2 = BatchNormalization(axis=4, name='norm_2_2')(conv2)
    # pool2 = MaxPooling3D(pool_size=pool_size, name='pool_2')(norm2)
    #
    # conv3 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_3_1')(pool2)
    # norm3 = BatchNormalization(axis=4, name='norm_3_1')(conv3)
    # conv3 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_3_2')(norm3)
    # norm3 = BatchNormalization(axis=4, name='norm_3_2')(conv3)
    # pool3 = MaxPooling3D(pool_size=pool_size, name='pool_3')(norm3)
    #
    # conv4 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_4_1')(pool3)
    # norm4 = BatchNormalization(axis=4, name='norm_4_1')(conv4)
    # conv4 = Conv3D(512, conv_size, activation='relu', padding='same', name='conv_4_2')(norm4)
    # norm4 = BatchNormalization(axis=4, name='norm_4_2')(conv4)
    #
    # up5 = UpSampling3D(size=pool_size, name='up_5')(norm4)
    # up5 = concatenate([up5, norm3], axis=4, name='conc_5')
    # drop5 = Dropout(rate=drop_out, name='drop_5')(up5)
    # conv5 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_5_1')(drop5)
    # conv5 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_5_2')(conv5)
    #
    # up6 = UpSampling3D(size=pool_size, name='up_6')(conv5)
    # up6 = concatenate([up6, norm2], axis=4, name='conc_6')
    # drop6 = Dropout(rate=drop_out, name='drop_6')(up6)
    # conv6 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_6_1')(drop6)
    # conv6 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_6_2')(conv6)
    #
    # up7 = UpSampling3D(size=pool_size, name='up_7')(conv6)
    # up7 = concatenate([up7, norm1], axis=4, name='conc_7')
    # drop7 = Dropout(rate=drop_out, name='drop_7')(up7)
    # conv7 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_7_1')(drop7)
    # conv7 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_7_2')(conv7)
    #
    # conv8 = Conv3D(1, (1, 1, 1), name='conv_8')(conv7)
    # act = Activation('sigmoid', name='act')(conv8)
    inputs = Input(input_shape, name='model_input')
    conv1 = Conv3D(32, conv_size, activation='relu', padding='same', name='conv_1_1')(inputs)
    conv1 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_1_2')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size, name='pool_1')(conv1)

    conv2 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_2_1')(pool1)
    conv2 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_2_2')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size, name='pool_2')(conv2)

    conv3 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_3_1')(pool2)
    conv3 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_3_2')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size, name='pool_3')(conv3)

    conv4 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_4_1')(pool3)
    conv4 = Conv3D(512, conv_size, activation='relu', padding='same', name='conv_4_2')(conv4)

    up5 = UpSampling3D(size=pool_size, name='up_5')(conv4)
    up5 = concatenate([up5, conv3], axis=4, name='conc_5')
    conv5 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_5_1')(up5)
    conv5 = Conv3D(256, conv_size, activation='relu', padding='same', name='conv_5_2')(conv5)

    up6 = UpSampling3D(size=pool_size, name='up_6')(conv5)
    up6 = concatenate([up6, conv2], axis=4, name='conc_6')
    conv6 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_6_1')(up6)
    conv6 = Conv3D(128, conv_size, activation='relu', padding='same', name='conv_6_2')(conv6)

    up7 = UpSampling3D(size=pool_size, name='up_7')(conv6)
    up7 = concatenate([up7, conv1], axis=4, name='conc_7')
    conv7 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_7_1')(up7)
    conv7 = Conv3D(64, conv_size, activation='relu', padding='same', name='conv_7_2')(conv7)

    conv8 = Conv3D(1, (1, 1, 1), name='conv_8')(conv7)
    act = Activation('sigmoid', name='act')(conv8)
    model = Model(inputs=inputs, outputs=act, name='UNET_CPD')
    return model


# TESTING
if __name__ == '__main__':
    import visualkeras
    from PIL import ImageFont
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, concatenate, Activation
    uNet3D = UNet("conv3D", up_sample="upSample3D", transpose=None, pool="max3D")
    model = uNet3D(5, 2, (112, 112, 112, 1), (3, 3, 3), batch_norm=False, drop_out=None)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 28, encoding="unic")
    print(model.summary())
    visualkeras.layered_view(model, to_file='heart_seg.png', legend=True, spacing=20, max_z=20, min_xy=70, one_dim_orientation='z',
                             draw_funnel=True, shade_step=2, font=font)
