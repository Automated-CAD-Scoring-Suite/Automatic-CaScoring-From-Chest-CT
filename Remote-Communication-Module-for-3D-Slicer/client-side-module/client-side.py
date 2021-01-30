# Import Packages
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pylab as plt
import numpy as np


def find_roi_2D(s):
    # rotate -90
    s_rotated = np.rot90(s, k=3)

    # flip slice
    s_fliped = np.flip(s, axis=0)
    s_rotated_fliped = np.flip(s_rotated, axis=0)

    # Get up and down coordiates
    y1 = np.unravel_index(np.argmax(s, axis=None), s.shape)
    y2 = np.unravel_index(np.argmax(s_fliped, axis=None), s.shape)

    x1 = np.unravel_index(np.argmax(s_rotated, axis=None), s.shape)
    x2 = np.unravel_index(np.argmax(s_rotated_fliped, axis=None), s.shape)

    # return x1, x2, y1, y2 of image
    return x1[0], s.shape[1] - x2[0], y1[0], s.shape[0] - y2[0]


def find_roi(sample):
    X1, X2, Y1, Y2, Z1, Z2 = sample.shape[1], 0, sample.shape[0], 0, sample.shape[2], 0

    for index in range(sample.shape[2]):  # around Z (axial)
        # Take slice from sample
        s = sample[:, :, index]

        # find points
        x1, x2, y1, y2 = find_roi_2D(s)

        # check for min x1,y1 and max x2,y2
        X1 = min(x1, X1)
        Y1 = min(y1, Y1)
        X2 = max(x2, X2)
        Y2 = max(y2, Y2)

    for index in range(sample.shape[1]):  # around X (sagital)
        # Take slice from sample
        s = sample[:, index, :]

        # find points
        z1, z2, y1, y2 = find_roi_2D(s)

        # check for min z1,y1 and max z2,y2
        Z1 = min(z1, Z1)
        Y1 = min(y1, Y1)
        Z2 = max(z2, Z2)
        Y2 = max(y2, Y2)

    for index in range(sample.shape[0]):  # around Y (coronal)
        # Take slice from sample
        s = sample[index, :, :]

        # find points
        x1, x2, z1, z2 = find_roi_2D(s)

        # check for min x1,z1 and max x2,z2
        X1 = min(x1, X1)
        Z1 = min(z1, Z1)
        X2 = max(x2, X2)
        Z2 = max(z2, Z2)

    return X1, X2, Y1, Y2, Z1, Z2


def crop_roi(sample, x1, x2, y1, y2, z1, z2):
    y = (y2 - y1 + 1) if (y1 != 0) else (y2 - y1)
    x = (x2 - x1 + 1) if (x1 != 0) else (x2 - x1)
    z = (z2 - z1 + 1) if (z1 != 0) else (z2 - z1)

    sample_croped = np.empty((y, x, z, 1))

    # for index in range(sample_croped.shape[2]):
    #    # Take slice from sample
    #    s = sample[:,:, index]
    #
    #    # Crop
    #    croped_slice = np.copy(s[y1:y2+1 , x1:x2+1])
    #
    #    sample_croped[:,:, index] = croped_slice
    sample_croped = sample[y1:y2 + 1, x1:x2 + 1, z1:z2 + 1].copy()

    return sample_croped

def load_itk(filename: str):
    """
    This function reads a '.mhd' file using SimpleITK
    :param filename: Path of .mhd file
    :return: The image array, origin and spacing of the image.
    """

    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return itkimage, ct_scan, origin, spacing


def plot_view(data):
    plt.figure(figsize=(50, 50))
    plt.gray()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(data.shape[1]):
        plt.subplot(8, 8, i+1)
        plt.imshow(data[i])
        plt.axis('off')
        # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
    plt.show()


# itkimage, ct_scan_data, origin, spacing = load_itk('../data/trv1p1cti.mhd')
itkimage, ct_scan_data, origin, spacing = load_itk('../data/trv1p1cti.nii')
fig, ax = plt.subplots(1, 3, figsize=(20, 20))
ax[0].imshow(ct_scan_data[50, :, :], cmap='gray')
ax[1].imshow(ct_scan_data[:, 50, :], cmap='gray')
ax[2].imshow(ct_scan_data[:, :, 50], cmap='gray')

# print(itkimage)
# ct_scan_data = nib.load('../data/trv1p1cti.nii').get_fdata()
# ct_scan_data = np.swapaxes(ct_scan_data, 0, 2)

# itkimage2, ct_scan_label, origin2, spacing2 = load_itk('../data/trv1p1cti-heart.nii')
# itkimage2, ct_scan_label, origin2, spacing2 = load_itk('../data/trv1p1cti-heart_4.nii.gz')
# ct_scan_label = np.swapaxes(ct_scan_label, 0, 2)

ct_scan_label = nib.load('../data/trv1p1cti-heart_4.nii').get_fdata()
sagital_image = ct_scan_label[213, :, :] # Axis 0
coronal_image = ct_scan_label[:, 154, :] # Axis 1
axial_image = ct_scan_label[:, :, 32]    # Axis 2


plt.figure(figsize=(20, 10))
plt.style.use('grayscale')

plt.subplot(141)
plt.imshow(np.rot90(sagital_image))
plt.title('Sagital Plane')
plt.axis('off')

plt.subplot(142)
plt.imshow(np.rot90(axial_image))
plt.title('Axial Plane')
plt.axis('off')

plt.subplot(143)
plt.imshow(np.rot90(coronal_image))
plt.title('Coronal Plane')
plt.axis('off')
plt.show()

# x1,x2,y1,y2,z1,z2 = find_roi(ct_scan_label)
# print(x1, x2, y1, y2, z1, z2)

# print(ct_scan_label[32].shape)
# x1,x2,y1,y2 = find_roi_2D(ct_scan_label[32])
# print(x1, x2, y1, y2)
# croped = ct_scan_label[32][x1:x2+1, y1:y2+1]

# cropped_data = crop_roi(ct_scan_data, x1,x2,y1,y2,z1,z2)

# print("New Shape:", cropped_data.shape)
print("Original Shape:", ct_scan_data.shape)
print("Label Shape:", ct_scan_label.shape)


# plot_view(sagital_image)
# plot_view(axial_image)
# plot_view(coronal_image)


