##
# Main Segmentation
##

from unet import u_net

model = u_net(levels=5, convs=3, input_shape=(512, 512, 1), out_channels=1)

print(model.summary())
