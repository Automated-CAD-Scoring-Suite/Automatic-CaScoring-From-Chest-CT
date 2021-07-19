#
# Main Segmentation
#
import os

import tensorflow as tf
from Unet.unet import UNet
import callbacks as cb
from functions import dice_coef_loss, dice_coef
from Generator import NiftyGen, NiftyAugmentor, CACGen
import nibabel as nib
from skimage.transform import resize
import numpy as np
import SimpleITK as sITK

# Script Variables
Testing = False
Deployment = False
training = 'Data/Training/'
validation = 'Data/Validation/'
CA_training = './CaData/Training/'
CA_validation = './CaData/Validation/'

weights_path = 'Model_Weights/'
models_path = './Models_Saved/'
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
mode = '3D'
down_factor = True
output_shape = (112, 112, 112)
input_shape = (112, 112, 112, 1)
levels = 4
kernel_size = (3, 3, 3)
convolutions = 2
initial_features = 32

batch_norm = True
drop_out = None
activation = 'relu'
lr = 0.001
optimizer = tf.keras.optimizers.Adam(lr)
####################################################################################################################
# Augmentation Parameters
aug = NiftyAugmentor([-10, 10, 0], [0.95, 1, 1.1], [0, 1])

# Dataset Loader, NIFTY Loader
gen = NiftyGen(training, augmenter=aug, mode=mode, output_shape=output_shape, down_factor=down_factor, save=False,
               shuffle=False)
gen_val = NiftyGen(validation, augmenter=None, mode=mode, output_shape=output_shape, down_factor=down_factor,
                   shuffle=False)
# Ca Generators
ca_gen = CACGen(CA_training, augmenter=aug, mode=mode, output_shape=output_shape, down_factor=down_factor, save=False,
                shuffle=False)
ca_val = CACGen(CA_validation, augmenter=None, mode=mode, output_shape=output_shape, down_factor=down_factor,
                  shuffle=False)

# MODEL INSTANTIATION
uNet3D = UNet("conv3D", up_sample="upSample3D", transpose=None, pool="max3D")
model = uNet3D(levels=levels, convolutions=convolutions, input_shape=input_shape, kernel_size=kernel_size,
               activation=activation, batch_norm=batch_norm, drop_out=drop_out, initial_features=initial_features)
# UNet2D = UNet(transpose='transpose2D')
# model = UNet2D(levels, convolutions, input_shape, kernel_size, activation=activation,
#                batch_norm=batch_norm, drop_out=drop_out, initial_features=initial_features)

print(model.summary())
###################################################################################################################
# TRAINING
callbacks = [
    cb.GarbageCollect(),
    tf.keras.callbacks.ModelCheckpoint(filepath=f'{weights_path}/{model.name}_checkpoint.h5', save_freq='epoch'),
    tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=20),
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.CSVLogger(f'{model.name}_logs_CAC.csv'),
    cb.DisplayCallback()
]

if not Testing:
    print("Training on ", gpus[0].name)
    # Training
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=['accuracy', dice_coef])
    print(f"Training {model.name}")

    with tf.device("/" + gpus[0].name[10:]):
        # Load Saved Checkpoint if Exits for the Same model.
        if os.path.exists(os.path.join(weights_path, f'{model.name}_checkpoint.h5')):
            print(f"Loading {model.name} CHECKPOINT !!!")
            model.load_weights(os.path.join(weights_path, f'{model.name}_checkpoint.h5'))

        # Start Training
        model.fit(ca_gen, epochs=1000, callbacks=callbacks, validation_data=ca_val)
    model.save(f'{weights_path}/Model_{model.name}')

# Testing Model
if Testing:
    import cv2
    from skimage.transform import rotate

    def range_scale(img) -> np.ndarray:
        """
        Scale Given Image Array using HF range
        :param img: Input 3d Array
        :return: Scaled Array
        """
        src = np.copy(img)
        min_bound = -1024.0
        max_bound = 1354.0
        src = (src - min_bound) / (max_bound - min_bound)
        src[src > 1] = 1.
        src[src < 0] = 0.
        return src

    # Load Last Saved Model
    model.load_weights(os.path.join(weights_path, f'{model.name}_checkpoint.h5'))

    # Load Test Image
    # Image_nii = nib.load('./Data/Validation/ct_train_1019/imaging.nii.gz')
    # Segmented_nii = nib.load('./Data/Validation/ct_train_1019/seg_norm.nii.gz')

    # Image = Image_nii.get_fdata()
    # Segmented = Segmented_nii.get_fdata()

    # Load Patient's Scans
    sITK_image = sITK.ReadImage('./Test/trv1p1ctai.mhd')
    Image = sITK.GetArrayFromImage(sITK_image)

    # Test rotating Input
    # Image = rotate(Image, 90)

    Image = np.moveaxis(Image, 0, -1)

    print(Image.min(), Image.max())

    # Reshape Both Tensors
    Image = resize(Image, output_shape=output_shape, preserve_range=True)
    print(f"Data Shape , {Image.shape}")

    Image = np.expand_dims(Image, 0)
    Image = np.expand_dims(Image, -1)

    # Range Scale
    Image = range_scale(Image)

    # # Predict Using loaded Parameters
    pred = model.predict(Image)

    Image = np.squeeze(Image, 0)
    Image = np.squeeze(Image, -1)

    pred = np.squeeze(pred, 0)
    pred = np.squeeze(pred, -1)

    # print(pred.shape)
    pred[pred < 0.9] = 0.0
    pred[pred >= 0.9] = 1.0

    print(np.unique(pred, return_counts=True))
    cv2.imwrite('test_pred.png', pred[:, 90, :] * 255)
    cv2.imwrite('test_img.png', Image[:, 90, :] * 255)

    pred_mhd = sITK.GetImageFromArray(pred)

    writer = sITK.ImageFileWriter()
    writer.SetFileName('test_pred_mhd_rot.mhd')
    writer.Execute(pred_mhd)

    # img_nii_test = nib.Nifti1Image(Image, Image_nii.affine, Image_nii.header)
    # seg_nii_test = nib.Nifti1Image(pred, Segmented_nii.affine, Segmented_nii.header)
    # nib.save(seg_nii_test, './test_pred.nii.gz')
    # nib.save(img_nii_test, './test_Image.nii.gz')

if Deployment:
    # Compile Created Model
    # model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=['accuracy', dice_coef])

    print(f"Saving Model {model.name} ... ")

    # Load Model Weights
    model.load_weights(os.path.join(weights_path, f'{model.name}_checkpoint.h5'))

    # Save the Model Architecture and Weights
    model.save(os.path.join(models_path, 'Heart_Localization'))
    print("Done !! ")
