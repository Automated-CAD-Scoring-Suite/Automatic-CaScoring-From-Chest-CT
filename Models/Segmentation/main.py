#
# Main Segmentation
#
import os

import tensorflow as tf
from Unet.unet import UNet
import callbacks as cb
from functions import dice_coef_loss, dice_coef
from Generator import NiftyGen, NiftyAugmentor

# Script Variables
Testing = False
training = 'Data/Training/'
model_path = 'Model_Weights/'
validation = 'Data/Validation/'
log_dir = './logs/'

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
################################################################################################################
# MODEL PARAMETERS
down_factor = 8
input_shape = (512//down_factor, 512//down_factor, 1)
# input_shape = (128, 128, 1)
levels = 4
kernel_size = (3, 3)
convolutions = 2
initial_features = 32
batch_norm = False
drop_out = 0.5
activation = 'relu'
lr = 0.001
optimizer = tf.keras.optimizers.Adam(lr)
####################################################################################################################
# Augmentation Parameters
aug = NiftyAugmentor([-10, 10, 0], [0.95, 1, 1.1], [0, 1])

# Dataset Loader, NIFTY Loader
gen = NiftyGen(training, augmenter=aug, down_factor=down_factor, save=False)
gen_val = NiftyGen(validation, augmenter=None, down_factor=down_factor)

# MODEL INSTANTIATION
UNet2D = UNet(transpose='transpose2D')
model = UNet2D(levels, convolutions, input_shape, kernel_size, activation=activation,
               batch_norm=batch_norm, drop_out=drop_out, initial_features=initial_features)

print(model.summary())
###################################################################################################################
# TRAINING
callbacks = [
    cb.GarbageCollect(),
    tf.keras.callbacks.ModelCheckpoint(filepath=f'{model_path}/{model.name}_checkpoint.h5', save_freq='epoch'),
    tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=20),
    tf.keras.callbacks.ReduceLROnPlateau(),
    cb.DisplayCallback()
]

if not Testing:
    print("Training on ", gpus[0].name)
    # Training
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=['accuracy', dice_coef])

    with tf.device("/" + gpus[0].name[10:]):
        # Load Saved Checkpoint if Exits for the Same model.
        if os.path.exists(os.path.join(model_path, f'{model.name}_checkpoint.h5')):
            print(f"Loading {model.name} CHECKPOINT !!!")
            model.load_weights(os.path.join(model_path, f'{model.name}_checkpoint.h5'))

        # Start Training
        model.fit(gen, epochs=100, callbacks=callbacks, validation_data=gen_val)
    model.save(f'{model_path}/Model_{model.name}.h5')

# TRAINING
if Testing:
    # TODO: Check Sanity of this part
    # Load Last Saved Model
    model.load_weights(os.path.join(model_path, f'{model.name}_checkpoint.h5'))

    # Predict Using loaded Parameters
