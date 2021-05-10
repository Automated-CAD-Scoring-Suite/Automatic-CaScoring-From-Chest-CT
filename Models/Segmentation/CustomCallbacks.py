
#
# Implementations of all custom callbacks
#
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import gc
import os
import Generator as Gen

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
        # Load Image
        gen = Gen.NiftyGen(validation, batch_size=5, batch_start=0, down_factor=4)
        img, seg = gen[0]
        print(f"Validating Volume with Shape {img.shape}")

        self.model.load_weights(os.path.join(model_path, self.model.name))
        test_pred = self.model.predict(img)

        fig, ax = plt.subplots(5, 3, figsize=(30, 30))

        for i in range(5):
            ax[i][0].plot(test_pred[i, :, :, :], 'gray')
            ax[i][0].set_title("Model Prediction")
            ax[i][1].plot(seg[i, :, :, :], 'gray')
            ax[i][1].set_title("True Prediction")
            ax[i][2].plot(img[i, :, :, :], 'gray')
            ax[i][2].set_title("CT Scan")

        if not os.path.isdir(validation_figs):
            os.makedirs(validation_figs)
        plt.savefig(os.path.join(validation_figs, f"Val_Fig_{epoch}"))
