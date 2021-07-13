import nrrd

# Reading NRRD Files
image, image_header = nrrd.read("./0187/img.nrrd")
mask, mask_header = nrrd.read("./0187/img.nrrd")
print(image.shape)
print(mask.shape)
print(image_header)