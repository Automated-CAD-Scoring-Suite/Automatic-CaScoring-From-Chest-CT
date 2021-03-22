
#
# Implementations of all custom callbacks
#
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import gc
import os

training = 'Training/'
threshold = 0.999
value = 0.0


class GarbageCollect(tf.keras.callbacks.Callback):
    """
    Call Garbage Collector on each epoch end to preserve memory
    """
    def on_epoch_end(self, epoch, logs=None):
        # Collect unreferenced objects in memory
        gc.collect()


class DisplayCallback(tf.keras.callbacks.Callback):
    """
    Plot a sample of predictions each epoch
    """

    def on_epoch_end(self, epoch, logs=None):
        num = np.random.randint(1, 9)

        # Load Image
        test_img = nib.load(training + f'/ct_train_100{num}/imaging.nii.gz').get_fdata()
        test_seg = nib.load(training + f'/ct_train_100{num}/seg_norm.nii.gz').get_fdata()

        # Down Sample
        test_img = test_img[0::4, 0::4, :]
        test_seg = test_seg[0::4, 0::4, :]

        # Prepare for Prediction
        idx = np.random.randint(100, test_img.shape[-1])
        test_img = test_img[:, :, idx]
        test_img = np.expand_dims(test_img, axis=[0, -1])

        if os.path.exists("model_weights.h5"):
            fig, ax = plt.subplots(1, 3, figsize=(5, 5))

            model.load_weights("model_weights.h5")
            # Make Prediction
            test_pred = self.model.predict(test_img)

            # Thresholding
            test_pred[test_pred < threshold] = value

            unique, counts = np.unique(test_pred, return_counts=True)
            print("\n", dict(zip(unique, counts)))
            print(test_pred.shape)

            test_pred = np.squeeze(test_pred, axis=0)

            # Plot Prediction
            ax[0].imshow(np.squeeze(test_img, 0), cmap='gray')
            ax[0].set_title('Image')

            ax[1].imshow(test_seg[:, :, 190], cmap='gray', vmin=0.0, vmax=1.0)
            ax[1].set_title('True Segmentation')

            ax[2].imshow(test_pred[:, :, 0], cmap='gray', vmin=0.0, vmax=1.0)
            ax[2].set_title('Model Prediciton')

            plt.show()
