
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
        gen = Gen.NiftyGen(validation, batch_size=5, batch_start=0)
        img, seg = gen[0]
        print(f"Validating Volume with Shape {img.shape}")

        self.model.load_weights(os.path.join(model_path, self.model.name))
        test_pred = self.model.predict(img)
        print(test_pred.shape)
        fig, ax = plt.subplots(1, 3, figsize=(30, 30))


