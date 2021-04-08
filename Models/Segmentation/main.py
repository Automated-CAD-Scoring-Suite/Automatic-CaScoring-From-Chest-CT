##
# Main Segmentation
##

from Generator import NiftyGen, NiftyAugmentor
from unet import u_net
import CustomCallbacks as cc
import tensorflow as tf
from functions import focal_tversky_loss, dice_coef, tversky

training = 'Data/Training/'
model_path = 'Model_Weights'
gpu_name = tf.test.gpu_device_name()

# Train Consts
down_factor = 4
input_shape = (512//down_factor, 512//down_factor, 1)

levels = 5
kernel_size = (3, 3)
convs = 3
initial_features = 32

batch_norm = False
drop_out = None
activation = 'elu'

start = 0
batch_size = 50


# Instances
aug = NiftyAugmentor()

gen = NiftyGen(training, batch_size=batch_size, batch_start=start,
               augmenter=aug, down_factor=down_factor)

model = u_net(levels, convs, input_shape, kernel_size, activation=activation,
              batch_norm=batch_norm, drop_out=drop_out,
              initial_features=initial_features)

print(model.summary())

callbacks = [
    cc.GarbageCollect(),
    tf.keras.callbacks.ModelCheckpoint(filepath=f'{model.name}_checkpoint.h5', save_freq='epoch'),
    tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=10),
    tf.keras.callbacks.ReduceLROnPlateau()
]


print("Training on ", gpu_name)
# Training
model.compile(optimizer='adam', loss=[focal_tversky_loss], metrics=['accuracy', dice_coef, tversky])


with tf.device(gpu_name):
    model.fit(gen, epochs=50, callbacks=callbacks)

model.save(f'Model_{model.name}.h5')
