# Import Required Packages
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pylab as plt
import numpy as np


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


def load_nib(filename):
    """

    :param filename:
    :return:
    """
    return nib.load(filename).get_fdata()


def plot_view(data):
    """

    :param data:
    :return:
    """
    plt.figure(figsize=(50, 50))
    plt.gray()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(data.shape[1]):
        plt.subplot(8, 8, i+1)
        plt.imshow(data[i])
        plt.axis('off')
        # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
    plt.show()

def plot_slice(data, x, y, z):
    """

    :param x:
    :param y:
    :param z:
    :return:
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    ax[0].imshow(np.rot90(data[x, :, :]), cmap='gray')
    ax[1].imshow(np.rot90(data[:, y, :]), cmap='gray')
    ax[2].imshow(np.rot90(data[:, :, z]), cmap='gray')
    fig.show()

    # sagital_image = data[213, :, :] # Axis 0
    # coronal_image = data[:, 154, :] # Axis 1
    # axial_image = data[:, :, 32]    # Axis 2

def show_image(image, title: str, cmap: str = 'gray'):
    """
    :param image: Image array "2D array"
    :param title: Figure Title "String"
    :param cmap: image color mapping "string"
    """
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap = cmap)
    plt.xticks([]), plt.yticks([])

