
#
# Implementations of all custom callbacks
#
import random

import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import os
import Generator as Gen
import nibabel as nib
import numpy as np

validation = 'Data/Validation'
model_path = "Model_Weights/"
validation_figs = model_path+"Figs"

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
        pass

    # NIFTY Callback on epoch end
    def _Nifty_EpochEnd(self, epoch):
        # Load Image
        num = random.choice(range(0, 3))
        gen = Gen.NiftyGen(validation, augmenter=None, down_factor=2)
        img, seg = gen[num]
        model_weights = os.path.join(model_path, f"{self.model.name}_checkpoint.h5")

        if os.path.exists(model_weights):
            print("Saving Prediction")
            self.model.load_weights(os.path.join(model_path, f"{self.model.name}_checkpoint.h5"))

            # 2D Mode, Predict first 20 Images
            test_pred = self.model.predict(img[::10, :, :, :])
            test_pred = np.squeeze(test_pred, -1)
            print(f"Validating on {img.shape}, result {test_pred.shape}", flush=True)

            test_pred_nifty = nib.Nifti1Image(test_pred, np.eye(4))
            nib.save(test_pred_nifty, os.path.join(validation_figs, f"Val_{epoch}.nii.gz"))

            print(f"Validation Image {num} output with values {test_pred.min()}, {test_pred.max()}")

            # Thresholding
            thresh = 0.5
            test_pred[test_pred < thresh] = 0.
            test_pred[test_pred >= thresh] = 1.

            unique, counts = np.unique(test_pred, return_counts=True)
            print(f"Validation Image {num} output with values {dict(zip(unique, counts))}")

            fig, ax = plt.subplots(1, 3, figsize=(20, 20))

            for image in range(test_pred.shape[0]):
                ax[0].imshow(test_pred[image, :, :], 'gray')
                ax[0].set_title("Model Prediction")

                ax[1].imshow(seg[image, :, :], 'gray')
                ax[1].set_title("True Prediction")

                ax[2].imshow(img[image, :, :], 'gray')
                ax[2].set_title("CT Scan")

                plt.title(f"Validation Image {num}")
                if not os.path.isdir(validation_figs):
                    os.makedirs(validation_figs)
                plt.savefig(os.path.join(validation_figs, f"Val_Fig_{epoch}_{image}"))
        else:
            print("No Checkpoint Saved")
