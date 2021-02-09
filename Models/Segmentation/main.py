##
# Main Segmentation
##
from Generator import NiftyGen
from unet import u_net


TRAINING = 'Data/Training/'

gen = NiftyGen(TRAINING, 100, 20, down_factor=8)
model = u_net(5, 3, (512//8, 512//8, 1), 1)
print(model.summary())

model.compile('adam', 'binary_crossentropy', ['accuracy'])

model.fit(gen, epochs=10)
