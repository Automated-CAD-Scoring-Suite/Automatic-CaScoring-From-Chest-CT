# Import Required Packages
import numpy as np

def find_roi_2D(s):
    """

    :param s:
    :return:
    """

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
    return x1[0], s.shape[1]-x2[0], y1[0], s.shape[0]-y2[0]


def find_roi(sample):
    """

    :param sample:
    :return:
    """

    X1,X2,Y1,Y2,Z1,Z2 = sample.shape[1], 0, sample.shape[0] ,0 , sample.shape[2], 0

    for index in range(sample.shape[2]): # around Z (axial)
        # Take slice from sample
        s = sample[:, :, index]

        # find points
        x1, x2, y1, y2 = find_roi_2D(s)

        # check for min x1,y1 and max x2,y2
        X1 = min(x1, X1)
        Y1 = min(y1, Y1)
        X2 = max(x2, X2)
        Y2 = max(y2, Y2)

    for index in range(sample.shape[1]): # around X (sagital)
        # Take slice from sample
        s = sample[:, index, :]

        # find points
        z1, z2, y1, y2 = find_roi_2D(s)

        # check for min z1,y1 and max z2,y2
        Z1 = min(z1, Z1)
        Y1 = min(y1, Y1)
        Z2 = max(z2, Z2)
        Y2 = max(y2, Y2)

    for index in range(sample.shape[0]): # around Y (coronal)
        # Take slice from sample
        s = sample[index, :, :]

        # find points
        x1, x2, z1, z2 = find_roi_2D(s)

        # check for min x1,z1 and max x2,z2
        X1 = min(x1, X1)
        Z1 = min(z1, Z1)
        X2 = max(x2, X2)
        Z2 = max(z2, Z2)

    return X1,X2,Y1,Y2,Z1,Z2


def crop_roi(sample, x1, x2, y1, y2, z1, z2):
    """

    :param sample:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param z1:
    :param z2:
    :return:
    """

    y = (y2 - y1 + 1) if (y1 != 0) else (y2 - y1)
    x = (x2 - x1 + 1) if (x1 != 0) else (x2 - x1)
    z = (z2 - z1 + 1) if (z1 != 0) else (z2 - z1)

    sample_croped = np.empty((y, x, z, 1))

    #for index in range(sample_croped.shape[2]):
    #    # Take slice from sample
    #    s = sample[:,:, index]
#
    #    # Crop
    #    croped_slice = np.copy(s[y1:y2+1 , x1:x2+1])
#
    #    sample_croped[:,:, index] = croped_slice
    sample_croped = sample[y1:y2+1, x1:x2+1, z1:z2+1].copy()

    return sample_croped