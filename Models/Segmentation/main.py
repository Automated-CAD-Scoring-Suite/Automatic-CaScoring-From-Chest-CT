#
# Main Segmentation
##
import numpy as np

from Generator import NiftyGen, NiftyAugmentor
from unet import u_net
import CustomCallbacks as cc
import tensorflow as tf
from functions import Dice, Dice_Loss
import nibabel as nib
import matplotlib.pyplot as plt


Testing = False
training = 'Data/Training'
model_path = 'Model_Weights'
validation = 'Data/Validation'
log_dir = './logs/'


gpu_name = tf.test.gpu_device_name()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Train Consts
down_factor = 4
input_shape = (512//down_factor, 512//down_factor, 1)

levels = 4
kernel_size = (3, 3)
convs = 3
initial_features = 16

batch_norm = True
drop_out = None
activation = 'elu'

start = 100
batch_size = 50


# Instances
aug = NiftyAugmentor()

gen = NiftyGen(training, batch_size=batch_size, batch_start=start,
               augmenter=aug, down_factor=down_factor)

gen_val = NiftyGen(validation, batch_size=100, batch_start=0, augmenter=None, down_factor=down_factor)

model = u_net(levels, convs, input_shape, kernel_size, activation=activation,
              batch_norm=batch_norm, drop_out=drop_out,
              initial_features=initial_features)

print(model.summary())

callbacks = [
    cc.GarbageCollect(),
    tf.keras.callbacks.ModelCheckpoint(filepath=f'{model.name}_checkpoint.h5', save_freq='epoch'),
    tf.keras.callbacks.EarlyStopping(monitor='Dice', min_delta=0.001, patience=10),
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch='10, 15')
]


if not Testing:
    print("Training on ", gpu_name)
    # Training
    model.compile(optimizer='adam', loss=[Dice_Loss], metrics=['accuracy', Dice])

    with tf.device(gpu_name):
        model.fit(gen, epochs=30, callbacks=callbacks, validation_data=gen_val)
    model.save(f'Model_{model.name}.h5')

if Testing:
    # Load Last Saved Model
    model.load_weights(f'{model_path}/Model_{model.name}.h5')

    # Load an Image Example
    test_img = nib.load(f'./{training}/ct_train_1001/imaging.nii.gz').get_fdata()
    test_seg = nib.load(f'./{training}/ct_train_1001/segmentation.nii.gz').get_fdata()

    # Down Sample
    test_img = test_img[::down_factor, ::down_factor, :]
    test_seg = test_seg[::down_factor, ::down_factor, :]

    test_img = test_img[:, :, 200]
    test_seg = test_seg[:, :, 200]

    test_img = np.expand_dims(test_img, axis=[0, -1])
    test_seg = np.expand_dims(test_seg, axis=[0, -1])
    test_pred = model.predict(test_img)

    fig, ax = plt.subplots(1, 3, figsize=(20, 20))

    ax[0].imshow(test_pred[0, :, :, 0], 'gray')
    ax[1].imshow(test_img[0, :, :, 0], 'gray')
    ax[2].imshow(test_seg[0, :, :, 0], 'gray')
    plt.show()
