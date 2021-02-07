
#
# Implementations of all custom callbacks
#


from tensorflow.keras.callbacks import Callback
import gc


class GarbageCollect(Callback):
    """
    Call Garbage Collector on each epoch end
    """
    def on_epoch_end(self, epoch, logs=None):
        # Collect unreferenced objects in memory
        gc.collect()
