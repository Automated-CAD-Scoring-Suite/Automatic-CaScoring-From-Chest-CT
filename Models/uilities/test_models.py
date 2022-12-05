# Import Packages
from helpers import load_nib, plot_slice
from Models.uilities.crop_roi import find_roi, crop_roi

# itkimage, img_data, origin, spacing = load_itk('../Dataset/trv1p1cti.nii.gz')
img_data = load_nib('../Dataset/trv1p1cti.nii.gz')
img_label = load_nib('../Dataset/trv1p1cti-heart_label.nii.gz')

plot_slice(data=img_data, x=321, y=231, z=23)

x1,x2,y1,y2,z1,z2 = find_roi(img_label)
print(x1, x2, y1, y2, z1, z2)
cropped_data = crop_roi(img_data, x1,x2,y1,y2,z1,z2)

print("Original Shape:", img_data.shape)
print("Label Shape:", img_label.shape)
print("New Shape:", cropped_data.shape)

