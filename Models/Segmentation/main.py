#
# Main Segmentation
#
import numpy as np

import tensorflow as tf
import nibabel as nib
from unet import UNet
import callbacks as cb
from functions import Dice, Dice_Loss
import matplotlib.pyplot as plt
from Generator import NiftyGen, NiftyAugmentor

Testing = False
training = 'Data/Training'
model_path = 'Model_Weights'
validation = 'Data/Validation'
log_dir = './logs/'

# Adding Implementation from Challenge Paper

# TF Configurations
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Train Consts
down_factor = 8
input_shape = (512 // down_factor, 512 // down_factor, 1)

levels = 4
kernel_size = (3, 3, 3)
convolutions = 2
initial_features = 16

batch_norm = True
drop_out = 0.5
activation = 'elu'

start = 0
batch_size = 50

# Instances
# TODO: Change Augmentation Class Structure
aug = None

gen = NiftyGen(training, batch_size=batch_size, batch_start=start,
               augmenter=aug, down_factor=down_factor, save=True)

gen_val = NiftyGen(validation, batch_size=batch_size, batch_start=start, augmenter=None, down_factor=down_factor)

# UNet2D = UNet()
UNet3D = UNet(conv="conv3D", up_sample="transpose3D", pool="max3D")
# HeatMap3DReg = UNet(conv="conv3d", up_sample="upSample3D", pool="average3D")

model = UNet3D(levels, convolutions, input_shape, kernel_size, activation=activation,
               batch_norm=batch_norm, drop_out=drop_out, initial_features=initial_features)

print(model.summary())

callbacks = [
    cb.GarbageCollect(),
    tf.keras.callbacks.ModelCheckpoint(filepath=f'{model_path}/{model.name}_checkpoint.h5', save_freq='epoch'),
    tf.keras.callbacks.EarlyStopping(monitor='Dice', min_delta=0.001, patience=10),
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch='10, 15'),
    cb.DisplayCallback()
]

if not Testing:
    print("Training on ", gpus[0].name)
    # Training
    model.compile(optimizer='adam', loss=[Dice_Loss], metrics=['accuracy', Dice])

    with tf.device("/" + gpus[0].name[10:]):
        model.fit(gen, epochs=30, callbacks=callbacks, validation_data=gen_val)
    model.save(f'{model_path}/Model_{model.name}.h5')

if Testing:
    # Load Last Saved Model
    model.load_weights(f'{model_path}/Model_{model.name}.h5')

    # Load an Image Example
    num = np.random.randint(0, 3)
    test_img, test_seg = gen_val[num]
    test_pred = model.predict(test_img)
    print(test_pred.shape)
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))

    ax[0].imshow(test_pred[0, :, :, :], 'gray')
    ax[1].imshow(test_img[0, :, :, :], 'gray')
    ax[2].imshow(test_seg[0, :, :, :], 'gray')
    plt.show()
